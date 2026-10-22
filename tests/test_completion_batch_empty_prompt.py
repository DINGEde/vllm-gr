# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin what the empty prompt renders to with the real completion tokenizer.

The batch planner refuses a zero-length prompt, and the engine refuses a
zero-token one. Whether "" reaches the engine as zero tokens is a tokenizer
property -- whether it adds a BOS -- not a property of the empty string, so
this test renders it with the same tokenizer and parameters the completion
route uses instead of assuming either answer.

The render runs in a child process. ``HF_HUB_OFFLINE`` is read when
``huggingface_hub`` is imported, so setting it inside this process would both
miss that import and outlive the fixture; passing it in the child's environment
scopes it to the probe and is the only way it takes effect.

Requires the model in the local cache; skipped otherwise (CI runs both tests
with ``--run-slow`` only when a model is available).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest

from vllm_gr.entrypoints.openai.completion_batch_plan import (
    CompletionBatch,
    plan_completion_batches,
)

MODEL = os.environ.get("VLLM_GR_EMPTY_PROMPT_MODEL", "OpenOneRec/OneRec-1.7B")
# Captured before the probe: the child owns the offline flag, not this process.
_OFFLINE_SETTING = os.environ.get("HF_HUB_OFFLINE")
_PROBE_MARKER = "__VLLM_GR_RENDER__"
_PROBE_TIMEOUT_S = 600

# Runs with the same renderer the completion route tokenizes prompts with.
_PROBE = """
import json
import sys

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config

model_config = ModelConfig(model=sys.argv[1])
# The renderer never runs the model; a CPU device avoids GPU probing.
config = VllmConfig(model_config=model_config, device_config=DeviceConfig(device="cpu"))
tokenizer = cached_tokenizer_from_config(model_config)
if tokenizer is None:
    print("__PROBE__" + json.dumps({"tokenizer": False}))
    raise SystemExit(0)
renderer = RENDERER_REGISTRY.load_renderer("hf", config, tokenizer)
params = renderer.default_cmpl_tok_params


def render(prompt):
    return list(renderer.tokenize_prompt({"prompt": prompt}, params)["prompt_token_ids"])


print(
    "__PROBE__"
    + json.dumps(
        {
            "tokenizer": True,
            "add_special_tokens": params.add_special_tokens,
            "bos_token_id": getattr(renderer.get_tokenizer(), "bos_token_id", None),
            "empty": render(""),
            "a": render("a"),
        }
    )
)
""".replace("__PROBE__", _PROBE_MARKER)


@pytest.fixture(scope="module")  # type: ignore[misc]
def render() -> dict[str, Any]:
    """What the completion renderer does with "" and with a one-character prompt."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, MODEL],
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_S,
        # Offline only for this child: the caller's environment is untouched.
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith(_PROBE_MARKER):
            result: dict[str, Any] = json.loads(line[len(_PROBE_MARKER) :])
            break
    else:
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()[-1:]
            pytest.skip(f"{MODEL} is not available locally: {' '.join(detail)}")
        raise AssertionError(f"the render probe printed no result: {completed.stdout!r}")
    if not result["tokenizer"]:
        pytest.skip("the model skips tokenizer initialization")
    return result


@pytest.mark.slow  # type: ignore[misc]
def test_the_render_probe_leaves_this_process_environment_alone(render: dict[str, Any]) -> None:
    """The offline flag is the child's, so it cannot reach later test subprocesses."""
    del render
    assert os.environ.get("HF_HUB_OFFLINE") == _OFFLINE_SETTING


@pytest.mark.slow  # type: ignore[misc]
def test_the_completion_render_adds_special_tokens_and_reports_the_bos(
    render: dict[str, Any],
) -> None:
    """The render, not the string, decides whether an empty prompt is empty."""
    assert render["add_special_tokens"] is True

    bos_token_id = render["bos_token_id"]
    assert render["empty"] == ([] if bos_token_id is None else [bos_token_id])


@pytest.mark.slow  # type: ignore[misc]
def test_a_zero_token_render_is_only_schedulable_with_the_begin_token(
    render: dict[str, Any],
) -> None:
    """The planner's empty-render decision matches the real tokenization."""
    lengths = [len(render["empty"]), len(render["a"])]
    if lengths[0] > 0:
        pytest.skip("this tokenizer renders the empty prompt as a token")

    plan_options = dict(configured_sizes=(1, 2, 4), max_num_seqs=2, token_budget=2048)
    with pytest.raises(ValueError, match="Prompt 0 Prefill length 0"):
        plan_completion_batches(lengths, [2, 2], **plan_options)

    # The frontend adds the begin token to its length and the engine appends it
    # to the prompt, so the same render becomes one prompt token.
    assert plan_completion_batches(
        [length + 1 for length in lengths], [2, 2], reserved_tokens=1, **plan_options
    ).batches == (CompletionBatch(0, 2, 2),)
