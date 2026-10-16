# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Real completions HTTP route and fixed-session splitting on OneRec/CUDA."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.st.test_onerec_v1_scheduler_st import (
    _BEAM_WIDTH,
    _engine_kwargs,
    _inputs,
    _worker_snapshot,
)


class CompletionBatchSTWorkerExtension:
    def completion_batch_observe(self) -> None:
        from vllm_gr.v1.worker.gpu_beam_stage_runner import GPUBeamStageRunner

        events: list[dict[str, Any]] = []
        self._completion_batch_events = events
        original = GPUBeamStageRunner.execute

        def execute(runner: Any, output: Any, *args: Any, **kwargs: Any) -> Any:
            metadata = next(iter(output.gr_stage_metadata.values()))
            events.append(
                {
                    "session": metadata.session_id,
                    "stage": metadata.stage_index,
                    "size": len(metadata.request_ids),
                }
            )
            return original(runner, output, *args, **kwargs)

        GPUBeamStageRunner.execute = execute

    def completion_batch_events(self) -> list[dict[str, Any]]:
        return self._completion_batch_events

    def completion_batch_snapshot(self) -> dict[str, Any]:
        return _worker_snapshot(self)


async def _run(work_dir: Path) -> None:
    import httpx
    from fastapi import FastAPI
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
    from vllm.entrypoints.openai.completion.api_router import attach_router
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

    from vllm_gr.plugin import initialize_runtime

    initialize_runtime()
    model, tokenizer, config, source, newline, _ = _inputs(work_dir)
    kwargs = _engine_kwargs(model, config)
    kwargs.update(
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        worker_extension_cls=(
            "tests.st.test_onerec_v1_completion_batch_st.CompletionBatchSTWorkerExtension"
        ),
    )
    args = AsyncEngineArgs(additional_config={"vllm_gr": config}, **kwargs)
    async with build_async_engine_client_from_engine_args(args) as engine:
        models = OpenAIServingModels(
            engine_client=engine,
            base_model_paths=[BaseModelPath(name=model, model_path=model)],
            lora_modules=None,
        )
        render = OpenAIServingRender(
            model_config=engine.model_config,
            renderer=engine.renderer,
            model_registry=models.registry,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        serving = OpenAIServingCompletion(
            engine_client=engine,
            models=models,
            openai_serving_render=render,
            request_logger=None,
        )
        native_beam_search = serving.beam_search
        materialized: list[Any] = []

        async def record_outputs(**call: Any) -> Any:
            async for output in native_beam_search(**call):
                materialized.append(
                    [
                        {"tokens": list(choice.token_ids), "score": choice.cumulative_logprob}
                        for choice in output.outputs
                    ]
                )
                yield output

        serving.beam_search = record_outputs
        await engine.collective_rpc("completion_batch_observe")
        app = FastAPI()
        app.state.openai_serving_completion = serving
        attach_router(app)
        prompts = [
            tokenizer.encode(f"slot {i}", add_special_tokens=False)
            + newline * (i * 16)
            + source[-256:]
            for i in range(6)
        ]
        body = {
            "model": model,
            "use_beam_search": True,
            "n": _BEAM_WIDTH,
            "temperature": 0.0,
            "max_tokens": 5,
            "return_token_ids": True,
            "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
        }

        async def released() -> None:
            snapshot = (await engine.collective_rpc("completion_batch_snapshot"))[0]
            assert snapshot["retired"] and snapshot["context_released"]
            assert snapshot["workspace_released"] and snapshot["registry_size"] == 0
            assert not snapshot["active_sessions"]
            assert not engine.output_processor.has_unfinished_requests()

        rows = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            reference = []
            for prompt in prompts:
                response = await client.post("/v1/completions", json={**body, "prompt": prompt})
                assert response.status_code == 200, response.text
                reference.append([choice["token_ids"] for choice in response.json()["choices"]])
            reference_scores = list(materialized)
            for count, sizes, cold in [(3, [2, 1], True), (5, [4, 1], True), (6, [4, 2], False)]:
                # B1 is diagnostic, not an equality oracle across GPU batch
                # shapes. Compare against the existing beam API with the same
                # explicit partition, independently of the new HTTP planner.
                assert await engine.reset_prefix_cache()
                # Warm every member in the same batch shape before comparing
                # hit paths; a preceding 4+1 request does not warm a 4+2 tail.
                for repeat in range(1 if cold else 2):
                    manual: list[Any] = []
                    offset = 0
                    for size in sizes:
                        request = CompletionRequest(**body, prompt=prompts[offset : offset + size])
                        inputs = await serving.render_completion_request(request)
                        assert not isinstance(inputs, ErrorResponse)
                        params = request.to_beam_search_params(5, serving.default_sampling_params)
                        async for output in native_beam_search(
                            prompt=inputs,
                            request_id=f"manual-{count}-{repeat}-{offset}",
                            params=params,
                            lora_request=None,
                            trace_headers=None,
                        ):
                            manual.append(
                                [
                                    {"tokens": list(c.token_ids), "score": c.cumulative_logprob}
                                    for c in output.outputs
                                ]
                            )
                        offset += size
                if cold:
                    assert await engine.reset_prefix_cache()
                before = len((await engine.collective_rpc("completion_batch_events"))[0])
                materialized.clear()
                response = await client.post(
                    "/v1/completions",
                    json={**body, "prompt": prompts[:count]},
                )
                assert response.status_code == 200, response.text
                data = response.json()
                assert len(data["choices"]) == count * _BEAM_WIDTH
                assert [c["index"] for c in data["choices"]] == list(range(count * _BEAM_WIDTH))
                assert data["usage"]["prompt_tokens"] == sum(map(len, prompts[:count]))
                assert data["usage"]["completion_tokens"] == count * _BEAM_WIDTH * 5
                (work_dir / f"comparison-{count}.json").write_text(
                    json.dumps(
                        {
                            "reference": reference_scores[:count],
                            "manual": manual,
                            "batched": materialized,
                        }
                    )
                )
                for index in range(count):
                    actual = data["choices"][index * _BEAM_WIDTH : (index + 1) * _BEAM_WIDTH]
                    assert [c["token_ids"] for c in actual] == [c["tokens"] for c in manual[index]]
                assert materialized == manual
                events = (await engine.collective_rpc("completion_batch_events"))[0][before:]
                assert [e["size"] for e in events if e["stage"] == 0] == sizes
                assert [e["stage"] for e in events] == [0, 1, 2] * len(sizes)
                assert len({e["session"] for e in events}) == len(sizes)
                await released()
                rows.append(
                    {
                        "count": count,
                        "cold": cold,
                        "sizes": sizes,
                        "usage": data["usage"],
                        "manual_partition_equal": True,
                        "b1_equal": [
                            [c["tokens"] for c in result] == reference[index]
                            for index, result in enumerate(materialized)
                        ],
                    }
                )

            # Pause the second child before its real submission, then cancel
            # while it waits in EngineCore. The first child has already run.
            original = serving.beam_search
            second_started = asyncio.Event()
            calls: list[str] = []

            async def observed(**call: Any) -> Any:
                calls.append(call["request_id"])
                if len(calls) == 2:
                    await engine.pause_generation(mode="keep", clear_cache=False)
                    second_started.set()
                async for result in original(**call):
                    yield result

            serving.beam_search = observed
            task = asyncio.create_task(
                serving.create_completion(
                    CompletionRequest(**body, prompt=prompts + prompts[:3]),
                )
            )
            try:
                await asyncio.wait_for(second_started.wait(), 30)
                async with asyncio.timeout(30):
                    while not engine.output_processor.has_unfinished_requests():
                        await asyncio.sleep(0.001)
                task.cancel()
                assert isinstance(await asyncio.wait_for(task, 30), ErrorResponse)
                assert len(calls) == 2
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                serving.beam_search = original
                await engine.resume_generation()
            response = await client.post("/v1/completions", json={**body, "prompt": prompts[:3]})
            assert response.status_code == 200, response.text
            await released()
            rows.append({"cancel_second_child": True, "recovery": True})
        (work_dir / "completion-batch-evidence.json").write_text(json.dumps(rows, indent=2))


@pytest.mark.slow
@pytest.mark.timeout(660)
def test_onerec_v1_completion_batches(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("OneRec completion batching ST requires CUDA")
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    command = [sys.executable, "-m", "tests.st.test_onerec_v1_completion_batch_st", str(tmp_path)]
    with subprocess.Popen(
        command,
        cwd=Path(__file__).parents[2],
        env=env,
        start_new_session=True,
    ) as process:
        try:
            assert process.wait(timeout=600) == 0
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    asyncio.run(_run(Path(sys.argv[1])))
