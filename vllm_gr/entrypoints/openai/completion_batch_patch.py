# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Render a prompt list once, execute fixed beam sub-batches, and merge once.

V1 multi-prompt completions use contiguous batches supported by the engine.
Each session finishes before the next is submitted; input order and individual
generation budgets are preserved. Other routes use the original handler.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from http import HTTPStatus
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.renderers.inputs.preprocess import prompt_to_seq

from vllm_gr.config import BeamExecutionMode, GRConfig
from vllm_gr.entrypoints.openai.completion_batch_plan import (
    CompletionBatch,
    plan_completion_batches,
)
from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES

if TYPE_CHECKING:
    from fastapi import Request
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.outputs import RequestOutput

_COMPLETION_BATCH_PATCH_MARKER = "_vllm_gr_completion_batch_patch"

# Filled in by ``patch_completion_batch``. Kept module-level so the wrapper has
# exactly one referent: the function object the route resolved when it was
# installed, not whatever the class attribute holds later.
_original_create_completion = None


def _batch_size(serving, request: CompletionRequest) -> int | None:
    """Return the claimed prompt count, or None to leave the request alone.

    None is the common answer. Delegating is how this patch stays inert on
    every request it does not exist to serve, so each condition below is a
    reason to run the upstream body, not an error to report: a caller who asks
    for something this patch cannot batch keeps the behaviour they had before.
    """
    if not request.use_beam_search or request.stream:
        return None
    # Bytes prompts carry their own embeddings and are rejected deeper in as
    # unsupported; count them as unclaimed rather than as a batch.
    if request.prompt_embeds is not None:
        return None
    vllm_config = getattr(serving.engine_client, "vllm_config", None)
    # A missing or LEGACY config means this engine was not built for the
    # persistent V1 sessions used by this route.
    if GRConfig.from_vllm_config(vllm_config).beam_execution_mode is not BeamExecutionMode.V1:
        return None
    try:
        count = len(prompt_to_seq(request.prompt))
    except (AttributeError, TypeError):
        return None
    # One prompt is one session either way, so the common single-prompt request
    # keeps running the upstream body rather than an equivalent copy of it.
    if count <= 1:
        return None
    return count


def _internal_error_response(serving, message: str) -> ErrorResponse:
    # Reached only when the frontend contradicts itself; a server-side defect
    # must not be answered as a caller mistake.
    return serving.create_error_response(
        message, err_type="InternalServerError", status_code=HTTPStatus.INTERNAL_SERVER_ERROR
    )


def _reserved_prompt_tokens(request: CompletionRequest) -> int:
    """Tokens the engine appends to each prompt before it measures the Prefill.

    ``_prepare_prompt`` appends the begin token, so a request carrying one is
    one token longer per prompt than the renderer reports. The end token is a
    decode-side stop, not prompt length.
    """
    extra_args = getattr(request, "vllm_xargs", None) or {}
    return 1 if extra_args.get("begin_token") is not None else 0


def _completion_batch_plan(
    serving, request: CompletionRequest, engine_inputs: list
) -> tuple[CompletionBatch, ...]:
    """Resolve per-input budgets before any session is submitted."""
    max_model_len = serving.model_config.max_model_len
    lengths = [serving._extract_prompt_len(engine_input) for engine_input in engine_inputs]
    budgets = [
        get_max_tokens(
            max_model_len,
            request.max_tokens,
            length,
            serving.default_sampling_params,
            serving.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )
        for length in lengths
    ]
    scheduler = serving.engine_client.vllm_config.scheduler_config
    token_budget = scheduler.max_num_batched_tokens
    scheduled_limit = getattr(scheduler, "max_num_scheduled_tokens", None)
    if scheduled_limit is not None:
        token_budget = min(token_budget, scheduled_limit)
    reserved = _reserved_prompt_tokens(request)
    try:
        return plan_completion_batches(
            [length + reserved for length in lengths],
            budgets,
            supported_sizes=SUPPORTED_BEAM_BATCH_SIZES,
            max_num_seqs=getattr(scheduler, "max_num_seqs", max(SUPPORTED_BEAM_BATCH_SIZES)),
            token_budget=token_budget,
        )
    except ValueError as exc:
        raise VLLMValidationError(str(exc)) from exc


async def _create_completion_batched(
    self, request: CompletionRequest, raw_request: Request | None, claimed: int
) -> object:
    """Run a complete plan serially, preserving one response per HTTP request."""
    result = await self.render_completion_request(request)
    if isinstance(result, ErrorResponse):
        return result
    engine_inputs = result
    if len(engine_inputs) != claimed:
        # Everything below -- the partition and one output per prompt --
        # rests on the renderer agreeing with the count taken off the request.
        return _internal_error_response(
            self,
            f"render produced {len(engine_inputs)} inputs for {claimed} prompts",
        )

    plan = _completion_batch_plan(self, request, engine_inputs)
    batch_params = [
        request.to_beam_search_params(batch.max_tokens, self.default_sampling_params)
        for batch in plan
    ]

    request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
    created_time = int(time.time())

    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    lora_request = self._maybe_get_adapters(request)

    trace_headers = (
        None if raw_request is None else await self._get_trace_headers(raw_request.headers)
    )
    final_res_batch: list[RequestOutput] = []
    try:
        for batch_index, (batch, sampling_params) in enumerate(
            zip(plan, batch_params, strict=True)
        ):
            # Make cancellation observable even if the previous iterator did
            # not suspend. The HTTP route's with_cancellation wrapper owns the
            # disconnect listener; do not compete for ASGI receive messages.
            await asyncio.sleep(0)
            for index in range(batch.start, batch.stop):
                self._log_inputs(
                    f"{request_id}-{index}",
                    engine_inputs[index],
                    params=sampling_params,
                    lora_request=lora_request,
                )
            child_id = request_id if len(plan) == 1 else f"{request_id}-batch-{batch_index}"
            generator = self.beam_search(
                prompt=engine_inputs[batch.start : batch.stop],
                request_id=child_id,
                params=sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )
            async with aclosing(generator):
                results = [res async for res in generator]
            expected = batch.stop - batch.start
            if len(results) != expected:
                return _internal_error_response(
                    self,
                    f"beam batch {batch_index} returned {len(results)} outputs for {expected} prompts",
                )
            final_res_batch.extend(results)
    except asyncio.CancelledError:
        return self.create_error_response("Client disconnected")

    for index, final_res in enumerate(final_res_batch):
        if final_res.prompt is None:
            final_res.prompt = self._extract_prompt_text(engine_inputs[index])

    return self.request_output_to_completion_response(
        final_res_batch,
        request,
        request_id,
        created_time,
        self.models.model_name(lora_request),
        self.renderer.tokenizer,
        request_metadata,
    )


async def create_completion_batch_dispatch(
    self, request: CompletionRequest, raw_request: Request | None = None
) -> object:
    """Route a completion request to the batched or the original handler."""
    claimed = _batch_size(self, request)
    if claimed is None:
        return await _original_create_completion(self, request, raw_request)
    return await _create_completion_batched(self, request, raw_request, claimed)


def patch_completion_batch() -> None:
    """Claim ``_create_completion`` for supported beam prompt lists.

    The route method calls the attribute, so replacing it is enough; the
    kv-transfer cleanup wrapped around it upstream keeps applying. Idempotent,
    and re-applied in every process that runs the patch plan.
    """
    global _original_create_completion

    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    current = OpenAIServingCompletion._create_completion
    if getattr(current, _COMPLETION_BATCH_PATCH_MARKER, False):
        return
    _original_create_completion = current
    setattr(create_completion_batch_dispatch, _COMPLETION_BATCH_PATCH_MARKER, True)
    OpenAIServingCompletion._create_completion = create_completion_batch_dispatch


def _validate_completion_batch_patch() -> str | None:
    """Verify the batch route is installed and the upstream seam still exists.

    The check is fail-closed on the surface the batch path calls. Instance
    attributes set by ``__init__`` (``models``, ``renderer``, ``model_config``,
    the sampling defaults) cannot be read off the class and are left to the
    attribute lookups themselves.
    """
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    if OpenAIServingCompletion._create_completion is not create_completion_batch_dispatch:
        return "OpenAIServingCompletion._create_completion is not the batch dispatch"
    missing = tuple(
        name
        for name in (
            "render_completion_request",
            "create_error_response",
            "request_output_to_completion_response",
            "_base_request_id",
            "_maybe_get_adapters",
            "_extract_prompt_len",
            "_extract_prompt_text",
            "_log_inputs",
            "_get_trace_headers",
            "beam_search",
        )
        if not hasattr(OpenAIServingCompletion, name)
    )
    if missing:
        return f"OpenAIServingCompletion is missing {', '.join(missing)}"
    return None
