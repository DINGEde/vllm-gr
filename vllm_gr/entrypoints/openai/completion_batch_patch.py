# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Split a prompt list into contiguous beam batches served by one HTTP request.

The completions route expands a prompt list into one engine request per prompt,
so k prompts become k beam sessions the engine runs one after another, even
though the V1 engine runs the members of one logical batch in a single forward.
This module renders the list once and then runs the largest legal batches in
the configured online policy, one after another, under the caller's request id.

A request this route does not exist to serve -- non-beam, streaming, embedded
prompts, a single prompt, a LEGACY engine -- delegates to the original handler.
The decision is taken from the request alone, before the render, so a delegated
request is never rendered twice. The plan is logged whatever it turns out to be,
including a plan of all-singletons, because a list that ran unbatched is
otherwise indistinguishable from one that did not: the response shape is
identical either way.
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
from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchAdmissionError
from vllm_gr.entrypoints.openai.completion_batch_plan import (
    CompletionBatchPlan,
    completion_batch_plan_reasons,
    plan_completion_batches,
)
from vllm_gr.logger import init_logger
from vllm_gr.v1.beam.batch_contracts import select_supported_batch_sizes

if TYPE_CHECKING:
    from fastapi import Request
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.outputs import RequestOutput

logger = init_logger(__name__)

_COMPLETION_BATCH_PATCH_MARKER = "_vllm_gr_completion_batch_patch"

# An all-singleton plan that configuration and capacity did not force is
# usually a budget-shape surprise. Five minutes between warnings keeps a
# pathological list from writing one line per request while still making the
# degradation visible; the suppressed count rides on the next warning.
_SINGLETON_WARNING_INTERVAL_S = 300.0
_singleton_warning_at: float | None = None
_singleton_warnings_suppressed = 0

# Filled in by ``patch_completion_batch``. Kept module-level so the wrapper has
# exactly one referent: the function object the route resolved when it was
# installed, not whatever the class attribute holds later.
_original_create_completion = None


def _batch_size(serving, request: CompletionRequest) -> int | None:
    """Return the prompt count the caller asked to batch, or None to ignore.

    None is the common answer: delegating is how this patch stays inert on
    every request it does not exist to serve, so each condition below is a
    reason to run the upstream body rather than an error to report. The gates
    need nothing but the request, so they run before the renderer, which is the
    expensive step this patch exists to do once instead of once per prompt.
    """
    if not request.use_beam_search or request.stream:
        return None
    # Bytes prompts carry their own embeddings and are rejected deeper in as
    # unsupported; count them as unclaimed rather than as a batch.
    if request.prompt_embeds is not None:
        return None
    vllm_config = getattr(serving.engine_client, "vllm_config", None)
    # A LEGACY config means this engine was not built for the persistent V1
    # sessions this route runs on.
    if GRConfig.from_vllm_config(vllm_config).beam_execution_mode is not BeamExecutionMode.V1:
        return None
    try:
        count = len(prompt_to_seq(request.prompt))
    except (AttributeError, TypeError):
        return None
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

    ``_prepare_prompt`` appends the begin token, so a request carrying one is a
    token longer per prompt than the renderer reports; the end token is a
    decode-side stop, not prompt length.
    """
    extra_args = getattr(request, "vllm_xargs", None) or {}
    return 1 if extra_args.get("begin_token") is not None else 0


def _output_boundary_tokens(request: CompletionRequest) -> int:
    """Public budget positions the engine produces outside the Decode steps.

    ``_create_request_params`` subtracts the begin and end tokens from
    ``max_tokens`` before the engine sees the budget, so neither position costs
    a Decode step. Both are read from the same ``vllm_xargs`` mapping the beam
    adapter uses.
    """
    extra_args = getattr(request, "vllm_xargs", None) or {}
    return int(extra_args.get("begin_token") is not None) + int(
        extra_args.get("end_token") is not None
    )


def _needs_decode_stage(request: CompletionRequest, budget: int) -> bool:
    """Whether a budget leaves work after the Prefill forward.

    The planner admits an effective budget of 1..3 Decode steps. An effective
    budget of one is Prefill-only and never occupies a Decode row.
    """
    return budget - _output_boundary_tokens(request) > 1


def _decode_batch_capacity(serving, request: CompletionRequest) -> int | None:
    """Bound NPU B*W rows by native input storage, not the Prefill budget."""
    config = serving.engine_client.vllm_config
    device = getattr(getattr(config, "device_config", None), "device", "cuda")
    if str(device).split(":")[0] != "npu":
        return None
    # The OpenAI beam adapter maps n to beam_width (None means one).
    width = request.n if request.n is not None else 1
    return config.scheduler_config.max_num_batched_tokens // width


def _completion_batch_plan(
    serving, request: CompletionRequest, engine_inputs: list
) -> CompletionBatchPlan:
    """Resolve every budget and partition the list before any session runs.

    Budgets stay per input: a sub-batch carries the budget of the prompts it
    holds, so no prompt is silently shortened to match a neighbour. The plan is
    complete before the first submission, so an input the planner can reject
    statically cannot leave earlier prompts computed but unanswered. Device/KV
    admission remains the engine's responsibility.
    """
    max_model_len = serving.model_config.max_model_len
    lengths = [
        serving._extract_prompt_len(engine_input) + _reserved_prompt_tokens(request)
        for engine_input in engine_inputs
    ]
    budgets = [
        get_max_tokens(
            max_model_len,
            request.max_tokens,
            serving._extract_prompt_len(engine_input),
            serving.default_sampling_params,
            serving.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )
        for engine_input in engine_inputs
    ]
    vllm_config = serving.engine_client.vllm_config
    gr_config = GRConfig.from_vllm_config(vllm_config)
    sizes = gr_config.beam_online_batch_sizes
    scheduler = vllm_config.scheduler_config
    token_budget = scheduler.max_num_batched_tokens
    scheduled_limit = getattr(scheduler, "max_num_scheduled_tokens", None)
    if scheduled_limit is not None:
        token_budget = min(token_budget, scheduled_limit)
    max_num_seqs = getattr(scheduler, "max_num_seqs", max(sizes))
    decode_capacity = _decode_batch_capacity(serving, request)
    if decode_capacity is not None and any(
        _needs_decode_stage(request, budget) for budget in budgets
    ):
        # One logical batch occupies B*W native Decode rows on NPU. Cap each
        # session instead of declining the whole list, so the planner can still
        # split it into legal sub-batches.
        max_num_seqs = max(1, min(max_num_seqs, decode_capacity))
    configured_sizes = tuple(sorted({size for size in sizes if size > 0}, reverse=True))
    reserved_tokens = _output_boundary_tokens(request)
    try:
        usable_sizes = select_supported_batch_sizes(sizes, max_num_seqs=max_num_seqs)
        batches = plan_completion_batches(
            lengths,
            budgets,
            supported_sizes=sizes,
            max_num_seqs=max_num_seqs,
            token_budget=token_budget,
            reserved_tokens=reserved_tokens,
            max_decode_steps=gr_config.beam_max_decode_steps,
        )
    except ValueError as exc:
        raise VLLMValidationError(str(exc)) from exc
    return CompletionBatchPlan(
        batches=batches,
        configured_sizes=configured_sizes,
        usable_sizes=usable_sizes,
        reasons=completion_batch_plan_reasons(
            batches,
            lengths,
            [budget - reserved_tokens for budget in budgets],
            configured_sizes=configured_sizes,
            usable_sizes=usable_sizes,
            token_budget=token_budget,
        ),
    )


def _warn_if_unexpected_singletons(
    plan: CompletionBatchPlan,
    request_id: str,
    prompt_count: int,
) -> None:
    """Rate-limit the warning for singletons nothing in the configuration asked for."""
    global _singleton_warning_at, _singleton_warnings_suppressed

    if not plan.all_singletons or plan.singletons_are_intentional:
        return
    now = time.monotonic()
    if (
        _singleton_warning_at is not None
        and now - _singleton_warning_at < _SINGLETON_WARNING_INTERVAL_S
    ):
        _singleton_warnings_suppressed += 1
        return
    logger.warning(
        "Beam batch plan ran unbatched request_id=%s prompts=%d "
        "configured_sizes=%s usable_sizes=%s reasons=%s suppressed_since_last=%d",
        request_id,
        prompt_count,
        plan.configured_sizes,
        plan.usable_sizes,
        plan.reasons,
        _singleton_warnings_suppressed,
    )
    _singleton_warning_at = now
    _singleton_warnings_suppressed = 0


def _admission_rejection_response(
    serving,
    message: str,
    batch_index: int,
    plan_size: int,
) -> ErrorResponse:
    """Report an admission rejection as a server-side, retryable failure.

    The caller asked for k prompts and upstream would have run them as k
    independent requests; the plan that grouped them is this frontend's
    decision, so a group the engine refuses is not a caller mistake and must
    not be answered as 400. Earlier sub-batches completed but their outputs are
    not sent: retrying them would recompute work and ordering the caller never
    asked for, and retrying only the refused group would still answer one HTTP
    request from two different scheduling decisions.
    """
    return serving.create_error_response(
        f"Beam sub-batch {batch_index} of {plan_size} was rejected by engine "
        f"admission after {batch_index} completed sub-batches; the request was "
        f"not retried: {message}",
        err_type="ServiceUnavailableError",
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )


async def _create_completion_batched(
    self,
    request: CompletionRequest,
    raw_request: Request | None,
    engine_inputs: list,
    plan: CompletionBatchPlan,
) -> object:
    """Run a complete plan serially, preserving one response per HTTP request.

    Sessions do not overlap: each finishes -- iterator closed, engine slot
    released -- before the next is submitted, so the engine still owns one
    session at a time and the caller still gets one response.
    """
    batch_params = [
        request.to_beam_search_params(batch.max_tokens, self.default_sampling_params)
        for batch in plan.batches
    ]
    request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
    logger.info(
        "Beam batch plan request_id=%s prompts=%d sub_batches=%d sizes=%s "
        "configured_sizes=%s usable_sizes=%s reasons=%s",
        request_id,
        len(engine_inputs),
        len(plan.batches),
        [batch.stop - batch.start for batch in plan.batches],
        plan.configured_sizes,
        plan.usable_sizes,
        plan.reasons,
    )
    _warn_if_unexpected_singletons(plan, request_id, len(engine_inputs))
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
        for batch_index, (batch, params) in enumerate(zip(plan.batches, batch_params, strict=True)):
            # Yield so a cancellation that arrived while the previous iterator
            # was closing is delivered before the next session is submitted.
            # The HTTP route's with_cancellation wrapper owns the disconnect
            # listener; this must not compete for ASGI receive messages.
            await asyncio.sleep(0)
            for index in range(batch.start, batch.stop):
                self._log_inputs(
                    f"{request_id}-{index}",
                    engine_inputs[index],
                    params=params,
                    lora_request=lora_request,
                )
            # A one-session plan keeps the identifier the caller already sees;
            # a split list names each session so the engine's logs, the
            # frontend's, and any cancellation can be traced to one of them.
            child_id = request_id if len(plan.batches) == 1 else f"{request_id}-batch-{batch_index}"
            generator = self.beam_search(
                prompt=engine_inputs[batch.start : batch.stop],
                request_id=child_id,
                params=params,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )
            try:
                async with aclosing(generator):
                    results = [res async for res in generator]
            except BeamSearchAdmissionError as error:
                # The V1 frontend submits the whole session before yielding,
                # so a refusal arrives as this exception and no output carries
                # it. Only the typed rejection is converted: any other
                # VLLMValidationError is a caller mistake and must stay a 400.
                return _admission_rejection_response(
                    self, str(error), batch_index, len(plan.batches)
                )
            # The frontend above never yields an error output; reaching this
            # means an unexpected one did, and it must not be counted as a
            # completed prompt.
            if any(
                output.finish_reason == "error" for result in results for output in result.outputs
            ):
                return _internal_error_response(
                    self, f"beam batch {batch_index} returned an error output"
                )
            expected = batch.stop - batch.start
            if len(results) != expected:
                return _internal_error_response(
                    self,
                    f"beam batch {batch_index} returned {len(results)} outputs for {expected} prompts",
                )
            final_res_batch.extend(results)
    except asyncio.CancelledError:
        # Match upstream non-streaming completions at the HTTP adapter boundary:
        # iterator cleanup precedes this error response, not a successful result.
        # Engine/client cancellation propagation remains unchanged.
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
    count = _batch_size(self, request)
    if count is None:
        return await _original_create_completion(self, request, raw_request)

    result = await self.render_completion_request(request)
    if isinstance(result, ErrorResponse):
        return result
    engine_inputs = result
    if len(engine_inputs) != count:
        # Everything below -- the partition and one output per prompt -- rests
        # on the renderer agreeing with the count taken off the request.
        return _internal_error_response(
            self, f"render produced {len(engine_inputs)} inputs for {count} prompts"
        )

    plan = _completion_batch_plan(self, request, engine_inputs)
    # A plan whose batches all hold one prompt is not a failure: it is the
    # answer for a list whose prompts agree on nothing the plan can join. It
    # runs here rather than delegating, because delegating would render the
    # list a second time to reach the same sessions. The plan log below names
    # the sizes either way, so a list that ran unbatched is visible in it.
    return await _create_completion_batched(self, request, raw_request, engine_inputs, plan)


def patch_completion_batch() -> None:
    """Claim ``_create_completion`` for supported beam prompt lists.

    The route calls this attribute, so replacing it is enough and the
    kv-transfer cleanup wrapped around it upstream keeps applying.
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

    Fail-closed on the surface the batch path calls. Instance attributes set by
    ``__init__`` (``models``, ``renderer``, ``model_config``, the sampling
    defaults) cannot be read off the class and are left to lookup.
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
