# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the beam interface selection installed on ``OpenAIServing``.

vLLM's OpenAI servers only ever call ``self.beam_search``; there is no call
site for ``beam_search_v1``. That single attribute therefore has to select the
implementation from the engine's own configuration, and a wrong selection is
otherwise visible only when a request is refused by the implementation's mode
guard -- two minutes of engine startup away. These tests decide it without a
device.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai.engine.serving import OpenAIServing

import vllm_gr.config as gr_config
import vllm_gr.entrypoints.openai.serving_engine as serving_engine
from vllm_gr.config import BeamExecutionMode, GRConfig
from vllm_gr.entrypoints.openai.beam_search_patch import (
    beam_search_dispatch,
    patch_beam_search,
)
from vllm_gr.patch import _validate_openai_beam_search_patch


class _ModeStub:
    """Stand-in for ``GRConfig`` that reports whatever mode a test sets."""

    beam_execution_mode = BeamExecutionMode.LEGACY

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> _ModeStub:
        del vllm_config
        return cls()


def _recorder(name: str, picked: list[str]) -> Callable[..., AsyncGenerator[str, None]]:
    """Return an async-generator stub that records the name it was reached by."""

    async def impl(
        self: object,
        prompt: object,
        request_id: str,
        params: object,
        lora_request: object = None,
        trace_headers: object = None,
    ) -> AsyncGenerator[str, None]:
        del self, prompt, request_id, params, lora_request, trace_headers
        picked.append(name)
        yield name

    return impl


async def _drain(serving: object) -> list[str]:
    return [
        output async for output in beam_search_dispatch(serving, "prompt", "request-id", "params")
    ]


def _picked(mode: BeamExecutionMode) -> list[str]:
    """Record which implementation a dispatch against ``mode`` reaches."""
    picked: list[str] = []
    serving = SimpleNamespace(engine_client=SimpleNamespace(vllm_config=object()))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(gr_config, "GRConfig", _ModeStub)
        monkeypatch.setattr(_ModeStub, "beam_execution_mode", mode)
        monkeypatch.setattr(serving_engine, "beam_search", _recorder("legacy", picked))
        monkeypatch.setattr(serving_engine, "beam_search_v1", _recorder("v1", picked))
        asyncio.run(_drain(serving))
    return picked


def test_patch_installs_the_dispatcher() -> None:
    patch_beam_search()

    assert OpenAIServing.beam_search is beam_search_dispatch
    assert _validate_openai_beam_search_patch() is None


def test_engine_without_gr_config_stays_on_legacy() -> None:
    # Read through the real parser: the interface this attribute installed
    # before the dispatch existed.
    config = GRConfig.from_vllm_config(None)

    assert config.beam_execution_mode is BeamExecutionMode.LEGACY


def test_dispatch_reaches_legacy_for_a_legacy_engine() -> None:
    assert _picked(BeamExecutionMode.LEGACY) == ["legacy"]


def test_dispatch_reaches_v1_for_a_v1_engine() -> None:
    assert _picked(BeamExecutionMode.V1) == ["v1"]
