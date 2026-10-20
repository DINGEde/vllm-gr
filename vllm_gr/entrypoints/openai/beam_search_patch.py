# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.inputs import EngineInput
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import BeamSearchParams

_ADMISSION_REJECTION_PATCH_MARKER = "_vllm_gr_admission_rejection_patch"
_CLIENT_REQUEST_ID_PATCH_MARKER = "_vllm_gr_client_request_id_patch"
_SUBMIT_REQUEST_ID_PATCH_MARKER = "_vllm_gr_submit_request_id_patch"


async def beam_search_dispatch(
    self,
    prompt: EngineInput | Sequence[EngineInput],
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    """Forward a beam request to the interface this engine was built for.

    The OpenAI servers only ever call ``self.beam_search``, so that attribute
    has to select the implementation itself: vLLM has no call site for the V1
    interface. The engine, not the request, owns the choice. Both interfaces
    share this signature and each one rejects a mismatched engine, so a
    disagreement between this dispatch and the engine is reported by the
    implementation instead of being silently served. A missing config parses
    as LEGACY -- the interface this attribute installed before the dispatch
    existed -- so non-GR engines are unaffected.

    ``prompt`` is a sequence when the caller submits one batch, which only the
    V1 interface accepts; the Legacy one owns a single request per call.
    """
    from vllm_gr.config import BeamExecutionMode, GRConfig
    from vllm_gr.entrypoints.openai.serving_engine import beam_search, beam_search_v1

    mode = GRConfig.from_vllm_config(
        getattr(self.engine_client, "vllm_config", None)
    ).beam_execution_mode
    impl = beam_search_v1 if mode is BeamExecutionMode.V1 else beam_search
    async for output in impl(
        self,
        prompt=prompt,
        request_id=request_id,
        params=params,
        lora_request=lora_request,
        trace_headers=trace_headers,
    ):
        yield output


def patch_beam_search():
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    from vllm_gr.entrypoints.openai.completion_batch_patch import patch_completion_batch
    from vllm_gr.entrypoints.openai.serving_engine import beam_search_v1

    OpenAIServing.beam_search = beam_search_dispatch
    OpenAIServing.beam_search_v1 = beam_search_v1
    patch_admission_rejection_reporting()
    patch_completion_batch()


def patch_admission_rejection_reporting():
    """Report an engine admission rejection to its caller as a client error.

    The engine answers a request it refuses -- an ordinary one arriving while a
    beam session owns it, say -- with a bare error output, and the serving loops
    turn any error finish reason into an internal server error. The reason is
    read off the output on its way through the frontend, so restoring it here
    restores the 400 the caller's own mistake deserves, for every endpoint that
    shares ``_raise_if_error`` rather than one call site at a time. The name the
    reason is parked under is the engine's, so the caller's own name is passed
    alongside it; see ``patch_client_request_id`` for how the two are paired.
    """
    from functools import wraps

    from vllm.entrypoints.openai.engine.serving import OpenAIServing
    from vllm.exceptions import VLLMValidationError

    from vllm_gr.v1.engine.async_llm import (
        _CLIENT_REQUEST_ID,
        take_admission_rejection,
    )

    patch_client_request_id()

    original = OpenAIServing._raise_if_error
    if getattr(original, _ADMISSION_REJECTION_PATCH_MARKER, False):
        return

    @wraps(original)
    def patched_raise_if_error(self, finish_reason, request_id):
        if finish_reason == "error":
            message = take_admission_rejection(
                getattr(self, "engine_client", None),
                request_id,
                _CLIENT_REQUEST_ID.get(),
            )
            if message is not None:
                raise VLLMValidationError(message)
        return original(self, finish_reason, request_id)

    setattr(patched_raise_if_error, _ADMISSION_REJECTION_PATCH_MARKER, True)
    OpenAIServing._raise_if_error = patched_raise_if_error


def patch_client_request_id():
    """Pair the name a caller is answered under with the engine's name for it.

    The frontends take a caller-supplied id, report the request under it, and
    hand the engine an extension of it -- ``f"{request_id}-{i}"`` for the i-th
    of an ``n``-way completion, extended again by the engine's own
    ``assign_request_id``. A refusal is filed under the engine's name, which no
    serving loop ever sees, so the two names have to be paired where the
    frontend still holds both.

    Two wrapper sites, and each is the only one of its kind: upstream resolves
    a caller's name through ``OpenAIServing._base_request_id`` and nowhere
    else, and every request -- completing, beam session, or batch item --
    reaches the frontend's request table through ``OutputProcessor.add_request``.
    The first publishes the resolved name for the request being handled in that
    task, the second records the engine request it became.
    """
    from functools import wraps

    from vllm.entrypoints.openai.engine.serving import OpenAIServing
    from vllm.v1.engine.output_processor import OutputProcessor

    from vllm_gr.v1.engine.async_llm import (
        _CLIENT_REQUEST_ID,
        remember_client_request_id,
    )

    original_base_request_id = OpenAIServing._base_request_id
    if not getattr(original_base_request_id, _CLIENT_REQUEST_ID_PATCH_MARKER, False):

        @wraps(original_base_request_id)
        def patched_base_request_id(raw_request, default=None):
            request_id = original_base_request_id(raw_request, default)
            _CLIENT_REQUEST_ID.set(request_id)
            return request_id

        setattr(patched_base_request_id, _CLIENT_REQUEST_ID_PATCH_MARKER, True)
        OpenAIServing._base_request_id = staticmethod(patched_base_request_id)

    original_add_request = OutputProcessor.add_request
    if getattr(original_add_request, _SUBMIT_REQUEST_ID_PATCH_MARKER, False):
        return

    @wraps(original_add_request)
    def patched_add_request(self, request, prompt, parent_req=None, request_index=0, queue=None):
        client_request_id = _CLIENT_REQUEST_ID.get()
        if client_request_id is not None:
            remember_client_request_id(self, client_request_id, request.request_id)
        return original_add_request(self, request, prompt, parent_req, request_index, queue)

    setattr(patched_add_request, _SUBMIT_REQUEST_ID_PATCH_MARKER, True)
    OutputProcessor.add_request = patched_add_request


def patch_sampling():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    from vllm_gr.entrypoints.openai.protocol import to_beam_search_params

    ChatCompletionRequest.to_beam_search_params = to_beam_search_params
    CompletionRequest.to_beam_search_params = to_beam_search_params


def patch_engine_core_process_entrypoint() -> None:
    """Wrap the EngineCore process entrypoint for spawn-based runtimes."""
    from vllm.logger import init_logger
    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.v1.engine.engine_core_patch import run_engine_core

    init_logger(__name__).info("vllm-gr: patch_engine_core_process_entrypoint called")

    # Save the original so the wrapper can delegate without recursion
    # even if run_patch() is called again in a spawned child process.
    current_run_engine_core = EngineCoreProc.run_engine_core
    if current_run_engine_core is run_engine_core:
        return
    original_run_engine_core = getattr(
        run_engine_core,
        "_original_run_engine_core",
        current_run_engine_core,
    )
    if original_run_engine_core is run_engine_core:
        original_run_engine_core = current_run_engine_core
    run_engine_core._original_run_engine_core = original_run_engine_core
    EngineCoreProc.run_engine_core = run_engine_core


def patch_batch_and_fork() -> None:
    """Compatibility helper that installs the complete grouped-beam pipeline."""
    from vllm_gr.v1.engine.core_client_patch import apply_batch_fork_patches

    apply_batch_fork_patches()
    patch_engine_core_process_entrypoint()
