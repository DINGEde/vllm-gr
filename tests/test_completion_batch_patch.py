# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin route compatibility and serial sub-batch execution/response assembly."""

from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

import vllm_gr.entrypoints.openai.completion_batch_patch as patch_module
from vllm_gr.config import BeamExecutionMode
from vllm_gr.entrypoints.openai.completion_batch_patch import (
    _batch_size,
    _create_completion_batched,
    create_completion_batch_dispatch,
    patch_completion_batch,
)


class _ModeStub:
    """Stand-in for ``GRConfig`` that reports whatever mode a test sets."""

    beam_execution_mode = BeamExecutionMode.V1

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> _ModeStub:
        del vllm_config
        return cls()


def _request(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "prompt": ["a", "b", "c", "d"],
        "use_beam_search": True,
        "stream": False,
        "prompt_embeds": None,
        "max_tokens": 5,
        "truncate_prompt_tokens": None,
        "request_id": "request-0",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _serving(*, max_num_seqs: int = 8) -> Any:
    vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
    return SimpleNamespace(engine_client=SimpleNamespace(vllm_config=vllm_config))


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch_module, "GRConfig", _ModeStub)
    monkeypatch.setattr(_ModeStub, "beam_execution_mode", BeamExecutionMode.V1)


def test_a_four_prompt_beam_request_is_claimed() -> None:
    assert _batch_size(_serving(), _request()) == 4


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides",
    [
        {"use_beam_search": False},
        {"stream": True},
        {"prompt_embeds": b"embedding"},
        {"prompt": ["only-one"]},
        {"prompt": [1, 2, 3]},
        {"prompt": []},
    ],
)
def test_everything_else_is_left_to_the_original_handler(overrides: dict[str, Any]) -> None:
    """Only V1 multi-prompt completions use the planner."""
    assert _batch_size(_serving(), _request(**overrides)) is None


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 9])  # type: ignore[misc]
def test_prompt_lists_are_claimed_independently_of_execution_capacity(count: int) -> None:
    assert _batch_size(_serving(max_num_seqs=2), _request(prompt=["a"] * count)) == count
    assert _batch_size(_serving(), _request(prompt=[[1, 2]] * count)) == count


def test_a_legacy_engine_is_left_alone() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_ModeStub, "beam_execution_mode", BeamExecutionMode.LEGACY)
        assert _batch_size(_serving(), _request()) is None


def test_an_unclaimed_request_runs_the_original_function_object() -> None:
    """Delegation must run the captured original, not the installed attribute."""
    seen: list[Any] = []

    async def original(self: Any, request: Any, raw_request: Any = None) -> str:
        seen.append((self, request, raw_request))
        return "original"

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(patch_module, "_original_create_completion", original)
        request = _request(use_beam_search=False)
        result = asyncio.run(create_completion_batch_dispatch(_serving(), request, None))

    assert result == "original"
    assert seen and seen[0][1] is request


class _BatchSelf:
    """The slice of ``OpenAIServingCompletion`` the batch path borrows."""

    def __init__(
        self,
        engine_inputs: list[int],
        prompt_lengths: dict[int, int],
        max_num_batched_tokens: int = 4096,
    ) -> None:
        self.engine_inputs = engine_inputs
        self.prompt_lengths = prompt_lengths
        self.render_calls = 0
        self.beam_calls: list[dict[str, Any]] = []
        self.response_calls: list[list[Any]] = []
        self.logged: list[str] = []
        self.engine_client = SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens)
            )
        )
        self.model_config = SimpleNamespace(max_model_len=4096)
        self.default_sampling_params: dict[str, Any] = {}
        self.override_max_tokens = None
        self.models = SimpleNamespace(model_name=lambda lora: "a-model")
        self.renderer = SimpleNamespace(tokenizer=None)

    async def render_completion_request(self, request: Any) -> list[int]:
        del request
        self.render_calls += 1
        return self.engine_inputs

    def _base_request_id(self, raw_request: Any, default: Any) -> str:
        del raw_request
        return str(default)

    def _maybe_get_adapters(self, request: Any) -> None:
        del request

    def _extract_prompt_len(self, engine_input: int) -> int:
        return self.prompt_lengths[engine_input]

    def _extract_prompt_text(self, engine_input: int) -> str:
        return f"text-{engine_input}"

    def _log_inputs(self, request_id: str, engine_input: Any, **kwargs: Any) -> None:
        del engine_input, kwargs
        self.logged.append(request_id)

    def beam_search(self, **kwargs: Any) -> Any:
        self.beam_calls.append(kwargs)
        outputs = [SimpleNamespace(prompt=None) for _ in kwargs["prompt"]]

        async def generator() -> Any:
            for output in outputs:
                yield output

        return generator()

    def request_output_to_completion_response(self, final_res_batch: list[Any], *args: Any) -> Any:
        self.response_calls.append(final_res_batch)
        return f"response-for-{len(final_res_batch)}"

    def create_error_response(self, message: Any, **kwargs: Any) -> Any:
        self.error = (message, kwargs)
        return "error-response"


def _batched_self(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> _BatchSelf:
    serving = _BatchSelf(
        kwargs.pop("engine_inputs", [0, 1, 2, 3]),
        kwargs.pop("prompt_lengths", {}),
        kwargs.pop("max_num_batched_tokens", 4096),
    )
    for name, value in kwargs.items():
        setattr(serving, name, value)
    # The budget formula belongs to vLLM; what this patch owes is the choice
    # among the per-prompt results, so the input length is the only variable.
    monkeypatch.setattr(
        patch_module,
        "get_max_tokens",
        lambda _len, _max, input_length, _default, _override, **_: 1000 - input_length,
    )
    return serving


def test_a_batch_renders_once_and_submits_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Four prompts are one render, one session, and one response."""
    serving = _batched_self(
        monkeypatch,
        engine_inputs=[0, 1, 2, 3],
        prompt_lengths={0: 10, 1: 10, 2: 10, 3: 10},
    )
    request = _request(to_beam_search_params=lambda tokens, _defaults: {"max_tokens": tokens})

    result = asyncio.run(_create_completion_batched(serving, request, None, 4))

    assert serving.render_calls == 1
    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1, 2, 3]]
    assert result == "response-for-4"
    assert serving.logged == [
        "cmpl-request-0-0",
        "cmpl-request-0-1",
        "cmpl-request-0-2",
        "cmpl-request-0-3",
    ]
    assert [res.prompt for res in serving.response_calls[0]] == [
        "text-0",
        "text-1",
        "text-2",
        "text-3",
    ]


def _delegates_to_the_original(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture the upstream handler and record what reaches it."""
    calls: list[Any] = []

    async def original(self: Any, request: Any, raw_request: Any = None) -> str:
        del self, raw_request
        calls.append(request)
        return "original"

    monkeypatch.setattr(patch_module, "_original_create_completion", original)
    return calls


def test_the_batch_runs_the_shared_per_prompt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """One session runs one budget, so the batch carries the agreed one."""
    serving = _batched_self(
        monkeypatch,
        engine_inputs=[0, 1, 2, 3],
        prompt_lengths={0: 10, 1: 10, 2: 10, 3: 10},
    )
    seen: list[int] = []

    def to_beam_search_params(tokens: int, _defaults: Any) -> dict[str, Any]:
        seen.append(tokens)
        return {"max_tokens": tokens}

    request = _request(to_beam_search_params=to_beam_search_params)

    asyncio.run(_create_completion_batched(serving, request, None, 4))

    assert seen == [990]


def test_prompts_that_disagree_on_their_budget_are_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each sub-batch retains its inputs' original generation budget."""
    serving = _batched_self(
        monkeypatch,
        engine_inputs=[0, 1, 2, 3],
        prompt_lengths={0: 10, 1: 20, 2: 30, 3: 40},
    )
    calls = _delegates_to_the_original(monkeypatch)
    request = _request(to_beam_search_params=lambda tokens, _defaults: {"max_tokens": tokens})

    result = asyncio.run(_create_completion_batched(serving, request, None, 4))

    assert result == "response-for-4"
    assert calls == []
    assert [call["params"]["max_tokens"] for call in serving.beam_calls] == [990, 980, 970, 960]
    assert [call["prompt"] for call in serving.beam_calls] == [[0], [1], [2], [3]]
    assert serving.render_calls == 1


def test_a_batch_that_fits_is_still_a_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control for the delegations below: the same shape with headroom."""
    serving = _batched_self(
        monkeypatch,
        engine_inputs=[0, 1, 2, 3],
        prompt_lengths={0: 10, 1: 10, 2: 10, 3: 10},
        max_num_batched_tokens=40,
    )
    request = _request(to_beam_search_params=lambda tokens, _defaults: {"max_tokens": tokens})

    result = asyncio.run(_create_completion_batched(serving, request, None, 4))

    assert result == "response-for-4"
    assert len(serving.beam_calls) == 1


def test_different_lengths_can_share_the_real_generation_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.entrypoints.utils import get_max_tokens

    serving = _batched_self(monkeypatch, prompt_lengths={0: 10, 1: 20, 2: 30, 3: 40})
    monkeypatch.setattr(patch_module, "get_max_tokens", get_max_tokens)
    request = _request(to_beam_search_params=lambda tokens, _: {"max_tokens": tokens})
    asyncio.run(_create_completion_batched(serving, request, None, 4))
    assert len(serving.beam_calls) == 1
    assert serving.beam_calls[0]["params"]["max_tokens"] == 5


def test_a_batch_past_the_token_budget_is_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity constrains each execution batch, not the HTTP prompt list."""
    serving = _batched_self(
        monkeypatch,
        engine_inputs=[0, 1, 2, 3],
        prompt_lengths={0: 10, 1: 10, 2: 10, 3: 10},
        max_num_batched_tokens=39,
    )
    calls = _delegates_to_the_original(monkeypatch)
    request = _request(to_beam_search_params=lambda tokens, _defaults: {})

    result = asyncio.run(_create_completion_batched(serving, request, None, 4))

    assert result == "response-for-4"
    assert calls == []
    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1], [2, 3]]


def test_the_begin_token_counts_towards_the_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine measures the Prefill after appending the begin token.

    The renderer never sees that token, so a budget that fits the rendered
    lengths exactly is one token per prompt short of what the engine checks.
    """
    serving = _batched_self(
        monkeypatch,
        engine_inputs=[0, 1, 2, 3],
        prompt_lengths={0: 10, 1: 10, 2: 10, 3: 10},
        max_num_batched_tokens=40,
    )
    calls = _delegates_to_the_original(monkeypatch)
    request = _request(
        vllm_xargs={"begin_token": "<|sid_begin|>"},
        to_beam_search_params=lambda tokens, _defaults: {},
    )

    result = asyncio.run(_create_completion_batched(serving, request, None, 4))

    assert result == "response-for-4"
    assert calls == []
    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1], [2, 3]]


def test_a_cancelled_client_is_answered_rather_than_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    serving = _batched_self(monkeypatch, engine_inputs=[0], prompt_lengths={0: 10})
    request = _request(prompt=["a"], to_beam_search_params=lambda tokens, _defaults: {})

    def cancelled(**kwargs: Any) -> Any:
        async def generator() -> Any:
            raise asyncio.CancelledError
            yield  # pragma: no cover - unreachable, makes this an async generator

        return generator()

    serving.beam_search = cancelled  # type: ignore[method-assign]

    result = asyncio.run(_create_completion_batched(serving, request, None, 1))

    assert result == "error-response"
    assert serving.error[0] == "Client disconnected"


def test_a_render_that_disagrees_with_the_count_is_a_server_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing downstream holds if the renderer and the request disagree.

    Checking it before the budget is chosen also keeps an empty render out of
    ``min``, where it would surface as an unexplained ValueError.
    """
    serving = _batched_self(monkeypatch, engine_inputs=[0, 1], prompt_lengths={0: 10, 1: 20})
    request = _request(to_beam_search_params=lambda tokens, _defaults: {})

    result = asyncio.run(_create_completion_batched(serving, request, None, 4))

    assert result == "error-response"
    assert serving.error[1]["err_type"] == "InternalServerError"
    assert serving.beam_calls == []


def test_a_short_result_batch_is_a_server_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """A result that does not match the request is a defect, not a caller error."""
    serving = _batched_self(
        monkeypatch,
        engine_inputs=[0, 1, 2, 3],
        prompt_lengths={0: 10, 1: 10, 2: 10, 3: 10},
    )
    request = _request(to_beam_search_params=lambda tokens, _defaults: {})

    def short(**kwargs: Any) -> Any:
        async def generator() -> Any:
            yield SimpleNamespace(prompt=None)

        return generator()

    serving.beam_search = short  # type: ignore[method-assign]

    result = asyncio.run(_create_completion_batched(serving, request, None, 4))

    assert result == "error-response"
    assert serving.error[1]["err_type"] == "InternalServerError"


@pytest.mark.parametrize("count", [3, 5, 6, 7, 9])  # type: ignore[misc]
def test_sub_batches_preserve_order_and_use_distinct_ids(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    serving = _batched_self(
        monkeypatch,
        engine_inputs=list(range(count)),
        prompt_lengths=dict.fromkeys(range(count), 10),
    )
    request = _request(to_beam_search_params=lambda tokens, _: {})
    assert asyncio.run(_create_completion_batched(serving, request, None, count)) == (
        f"response-for-{count}"
    )
    assert [i for call in serving.beam_calls for i in call["prompt"]] == list(range(count))
    assert len({call["request_id"] for call in serving.beam_calls}) == len(serving.beam_calls)
    assert serving.render_calls == 1
    assert len(serving.response_calls) == 1
    assert [res.prompt for res in serving.response_calls[0]] == [f"text-{i}" for i in range(count)]


@pytest.mark.parametrize("limit_name", ["max_num_seqs", "max_num_scheduled_tokens"])  # type: ignore[misc]
def test_effective_capacity_splits_before_submission(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    serving = _batched_self(monkeypatch, prompt_lengths=dict.fromkeys(range(4), 10))
    setattr(
        serving.engine_client.vllm_config.scheduler_config,
        limit_name,
        2 if limit_name == "max_num_seqs" else 20,
    )
    request = _request(to_beam_search_params=lambda tokens, _: {})
    asyncio.run(_create_completion_batched(serving, request, None, 4))
    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1], [2, 3]]


def test_an_invalid_later_prompt_prevents_all_submissions(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.exceptions import VLLMValidationError

    serving = _batched_self(
        monkeypatch,
        prompt_lengths={0: 10, 1: 10, 2: 10, 3: 41},
        max_num_batched_tokens=40,
    )
    with pytest.raises(VLLMValidationError, match="Prompt 3 Prefill length"):
        asyncio.run(
            _create_completion_batched(
                serving,
                _request(to_beam_search_params=lambda tokens, _: {}),
                None,
                4,
            )
        )
    assert serving.beam_calls == []


@pytest.mark.parametrize("failure", ["execution", "cleanup", "cancel", "short"])  # type: ignore[misc]
def test_second_batch_failure_never_submits_the_third(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    serving = _batched_self(
        monkeypatch,
        engine_inputs=list(range(9)),
        prompt_lengths=dict.fromkeys(range(9), 10),
    )
    delegated = _delegates_to_the_original(monkeypatch)
    events: list[tuple[str, int]] = []

    def beam_search(**kwargs: Any) -> Any:
        number = len(serving.beam_calls)
        serving.beam_calls.append(kwargs)

        async def generator() -> Any:
            events.append(("start", number))
            try:
                await asyncio.sleep(0)
                if number == 1:
                    if failure == "execution":
                        raise RuntimeError("execution failed")
                    if failure == "cancel":
                        raise asyncio.CancelledError
                    if failure == "short":
                        return
                for _ in kwargs["prompt"]:
                    yield SimpleNamespace(prompt=None)
            finally:
                events.append(("close", number))
                if number == 1 and failure == "cleanup":
                    raise RuntimeError("cleanup failed")

        return generator()

    serving.beam_search = beam_search  # type: ignore[method-assign]
    call = _create_completion_batched(
        serving,
        _request(to_beam_search_params=lambda tokens, _: {}),
        None,
        9,
    )
    if failure in ("execution", "cleanup"):
        with pytest.raises(RuntimeError, match=failure):
            asyncio.run(call)
    else:
        assert asyncio.run(call) == "error-response"
    assert events == [("start", 0), ("close", 0), ("start", 1), ("close", 1)]
    assert len(serving.beam_calls) == 2
    assert not serving.response_calls
    assert not delegated


def test_cancellation_between_batches_prevents_next_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serving = _batched_self(
        monkeypatch,
        engine_inputs=list(range(5)),
        prompt_lengths=dict.fromkeys(range(5), 10),
    )

    def beam_search(**kwargs: Any) -> Any:
        serving.beam_calls.append(kwargs)

        async def generator() -> Any:
            for _ in kwargs["prompt"]:
                yield SimpleNamespace(prompt=None)
            task = asyncio.current_task()
            assert task is not None
            task.cancel()

        return generator()

    serving.beam_search = beam_search  # type: ignore[method-assign]
    assert (
        asyncio.run(
            _create_completion_batched(
                serving,
                _request(to_beam_search_params=lambda tokens, _: {}),
                None,
                5,
            )
        )
        == "error-response"
    )
    assert len(serving.beam_calls) == 1
    assert not serving.response_calls


def test_task_cancellation_waits_for_current_iterator_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serving = _batched_self(
        monkeypatch,
        engine_inputs=list(range(5)),
        prompt_lengths=dict.fromkeys(range(5), 10),
    )

    async def run() -> None:
        started, closed = asyncio.Event(), asyncio.Event()

        def beam_search(**kwargs: Any) -> Any:
            serving.beam_calls.append(kwargs)

            async def generator() -> Any:
                try:
                    started.set()
                    await asyncio.Event().wait()
                    yield SimpleNamespace(prompt=None)
                finally:
                    await asyncio.sleep(0)
                    closed.set()

            return generator()

        serving.beam_search = beam_search  # type: ignore[method-assign]
        task = asyncio.create_task(
            _create_completion_batched(
                serving,
                _request(to_beam_search_params=lambda tokens, _: {}),
                None,
                5,
            )
        )
        await started.wait()
        task.cancel()
        assert await task == "error-response"
        assert closed.is_set()
        assert len(serving.beam_calls) == 1

    asyncio.run(run())


def test_real_response_conversion_preserves_indices_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.outputs import CompletionOutput, RequestOutput

    serving = _batched_self(
        monkeypatch,
        engine_inputs=list(range(6)),
        prompt_lengths=dict.fromkeys(range(6), 10),
        enable_prompt_tokens_details=False,
        system_fingerprint=None,
        _raise_if_error=lambda *_: None,
    )
    serving.request_output_to_completion_response = MethodType(  # type: ignore[method-assign]
        OpenAIServingCompletion.request_output_to_completion_response,
        serving,
    )

    def beam_search(**kwargs: Any) -> Any:
        serving.beam_calls.append(kwargs)

        async def generator() -> Any:
            for index in kwargs["prompt"]:
                yield RequestOutput(
                    request_id=kwargs["request_id"],
                    prompt=None,
                    prompt_token_ids=[index] * 10,
                    prompt_logprobs=None,
                    finished=True,
                    outputs=[
                        CompletionOutput(
                            index=beam,
                            text=f"{index}:{beam}",
                            token_ids=[index, beam],
                            cumulative_logprob=0.0,
                            logprobs=None,
                            finish_reason="length",
                        )
                        for beam in range(2)
                    ],
                )

        return generator()

    serving.beam_search = beam_search  # type: ignore[method-assign]
    request = CompletionRequest(
        model="a-model",
        prompt=["prompt"] * 6,
        n=2,
        max_tokens=5,
        use_beam_search=True,
        request_id="request-0",
        return_token_ids=True,
    )
    result = asyncio.run(_create_completion_batched(serving, request, None, 6))
    assert result.id == "cmpl-request-0"
    assert [c.index for c in result.choices] == list(range(12))
    assert [c.text for c in result.choices] == [f"{i}:{b}" for i in range(6) for b in range(2)]
    assert result.usage.prompt_tokens == 60
    assert result.usage.completion_tokens == 24
    assert result.usage.total_tokens == 84
    assert [len(call["prompt"]) for call in serving.beam_calls] == [4, 2]


def test_the_patch_installs_and_verifies() -> None:
    patch_completion_batch()

    assert OpenAIServingCompletion._create_completion is create_completion_batch_dispatch
    assert patch_module._validate_completion_batch_patch() is None


def test_the_patch_is_idempotent() -> None:
    patch_completion_batch()
    original = patch_module._original_create_completion
    patch_completion_batch()

    assert patch_module._original_create_completion is original
