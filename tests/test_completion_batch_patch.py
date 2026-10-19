# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin route selection, decline reporting, and serial sub-batch execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

import vllm_gr.entrypoints.openai.completion_batch_patch as patch_module
from vllm_gr.config import DEFAULT_ONLINE_BATCH_SIZES, BeamExecutionMode, GRConfig
from vllm_gr.entrypoints.openai.completion_batch_plan import CompletionBatch
from vllm_gr.entrypoints.openai.completion_batch_patch import (
    _batch_size,
    _create_completion_batched,
    create_completion_batch_dispatch,
    patch_completion_batch,
)

# The one input shape every execution test shares unless it says otherwise:
# four prompts of equal rendered length, so nothing declines for capacity.
_EQUAL_LENGTHS = dict.fromkeys(range(4), 10)


class _ModeStub:
    """Stand-in for ``GRConfig`` that reports whatever mode a test sets."""

    beam_execution_mode = BeamExecutionMode.V1
    beam_online_batch_sizes = DEFAULT_ONLINE_BATCH_SIZES

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> _ModeStub:
        del vllm_config
        return cls()


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch_module, "GRConfig", _ModeStub)
    monkeypatch.setattr(_ModeStub, "beam_execution_mode", BeamExecutionMode.V1)


class _BatchSelf:
    """The slice of ``OpenAIServingCompletion`` the batch path borrows."""

    def __init__(
        self,
        engine_inputs: list[int],
        prompt_lengths: dict[int, int],
        max_num_batched_tokens: int = 4096,
        max_num_seqs: int = 8,
    ) -> None:
        self.engine_inputs = engine_inputs
        self.prompt_lengths = prompt_lengths
        self.render_calls = 0
        self.beam_calls: list[dict[str, Any]] = []
        self.response_calls: list[list[Any]] = []
        self.logged: list[str] = []
        self.engine_client = SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                )
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


@pytest.fixture  # type: ignore[misc]
def make_serving(monkeypatch: pytest.MonkeyPatch) -> Callable[..., _BatchSelf]:
    """Build the borrowed serving surface with a stubbed budget formula.

    The budget formula belongs to vLLM; what this patch owes is the choice
    among the per-prompt results, so the input length is the only variable.
    """

    def _make(**kwargs: Any) -> _BatchSelf:
        serving = _BatchSelf(
            kwargs.pop("engine_inputs", [0, 1, 2, 3]),
            kwargs.pop("prompt_lengths", dict(_EQUAL_LENGTHS)),
            kwargs.pop("max_num_batched_tokens", 4096),
            kwargs.pop("max_num_seqs", 8),
        )
        for name, value in kwargs.items():
            setattr(serving, name, value)
        monkeypatch.setattr(
            patch_module,
            "get_max_tokens",
            lambda _len, _max, input_length, _default, _override, **_: 1000 - input_length,
        )
        return serving

    return _make


@pytest.fixture  # type: ignore[misc]
def serving(make_serving: Callable[..., _BatchSelf]) -> _BatchSelf:
    return make_serving()


@pytest.fixture  # type: ignore[misc]
def make_request() -> Callable[..., Any]:
    """Build a completion request whose beam parameters echo their budget."""

    def _make(**overrides: Any) -> Any:
        fields: dict[str, Any] = {
            "prompt": ["a", "b", "c", "d"],
            "use_beam_search": True,
            "stream": False,
            "prompt_embeds": None,
            "max_tokens": 5,
            "truncate_prompt_tokens": None,
            "request_id": "request-0",
            "to_beam_search_params": lambda tokens, _defaults: {"max_tokens": tokens},
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    return _make


@pytest.fixture  # type: ignore[misc]
def logged(monkeypatch: pytest.MonkeyPatch) -> Mock:
    info = Mock()
    monkeypatch.setattr(patch_module.logger, "info", info)
    return info


@pytest.fixture  # type: ignore[misc]
def delegate(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture the upstream handler and record what reaches it."""
    calls: list[Any] = []

    async def original(self: Any, request: Any, raw_request: Any = None) -> str:
        del self, raw_request
        calls.append(request)
        return "original"

    monkeypatch.setattr(patch_module, "_original_create_completion", original)
    return calls


# --------------------------------------------------------------------------- #
# Route selection
# --------------------------------------------------------------------------- #


def test_a_four_prompt_beam_request_is_claimed(
    serving: _BatchSelf, make_request: Callable[..., Any]
) -> None:
    assert _batch_size(serving, make_request()) == 4


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
def test_everything_else_is_left_to_the_original_handler(
    serving: _BatchSelf, make_request: Callable[..., Any], overrides: dict[str, Any]
) -> None:
    assert _batch_size(serving, make_request(**overrides)) is None


@pytest.mark.parametrize("count", [2, 4, 3, 5, 6, 9])  # type: ignore[misc]
def test_a_prompt_list_is_counted_before_capacity_is_consulted(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any], count: int
) -> None:
    """Claiming depends on the request alone; the plan decides the rest.

    Any count above one comes back, not just the sizes a batch can hold, so an
    arbitrary list is partitioned rather than declined for its length.
    """
    capacity = make_serving(max_num_seqs=2)
    assert _batch_size(capacity, make_request(prompt=["a"] * count)) == count
    assert _batch_size(capacity, make_request(prompt=[[1, 2]] * count)) == count


def test_a_legacy_engine_is_left_alone(
    serving: _BatchSelf, make_request: Callable[..., Any]
) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_ModeStub, "beam_execution_mode", BeamExecutionMode.LEGACY)
        assert _batch_size(serving, make_request()) is None


# --------------------------------------------------------------------------- #
# Delegation: the original handler runs, and the render is not repeated
# --------------------------------------------------------------------------- #


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
    ids=["not-beam", "streaming", "embeddings", "one-prompt", "token-ids", "empty"],
)
def test_a_request_this_route_does_not_serve_reaches_the_original_handler(
    serving: _BatchSelf,
    make_request: Callable[..., Any],
    delegate: list[Any],
    overrides: dict[str, Any],
) -> None:
    """The gates need nothing but the request, so nothing is rendered for them."""
    request = make_request(**overrides)

    assert asyncio.run(create_completion_batch_dispatch(serving, request)) == "original"

    assert delegate == [request]
    assert serving.render_calls == 0
    assert serving.beam_calls == []


# --------------------------------------------------------------------------- #
# One session: one render, one submit, one response
# --------------------------------------------------------------------------- #


def test_a_batch_renders_once_and_submits_once(
    serving: _BatchSelf, make_request: Callable[..., Any], logged: Mock
) -> None:
    """Four prompts are one render, one session, and one response."""
    assert asyncio.run(create_completion_batch_dispatch(serving, make_request())) == (
        "response-for-4"
    )
    assert serving.render_calls == 1
    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1, 2, 3]]
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
    assert logged.call_args.args == (
        "Beam batch plan request_id=%s prompts=%d sub_batches=%d sizes=%s",
        "cmpl-request-0",
        4,
        1,
        [4],
    )


def test_the_batch_runs_the_shared_per_prompt_budget(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    """One session runs one budget, so the batch carries the agreed one."""
    serving = make_serving()
    seen: list[int] = []

    def to_beam_search_params(tokens: int, _defaults: Any) -> dict[str, Any]:
        seen.append(tokens)
        return {"max_tokens": tokens}

    asyncio.run(
        create_completion_batch_dispatch(
            serving, make_request(to_beam_search_params=to_beam_search_params)
        )
    )

    assert seen == [990]


def test_different_lengths_can_share_the_real_generation_budget(
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the real formula the budget is the request's, not the length's."""
    from vllm.entrypoints.utils import get_max_tokens

    serving = make_serving(prompt_lengths={0: 10, 1: 20, 2: 30, 3: 40})
    monkeypatch.setattr(patch_module, "get_max_tokens", get_max_tokens)

    asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert len(serving.beam_calls) == 1
    assert serving.beam_calls[0]["params"]["max_tokens"] == 5


# --------------------------------------------------------------------------- #
# Many sessions: the list is split, and the split is safe
# --------------------------------------------------------------------------- #


def _input_lengths(count: int, length: int = 10) -> dict[int, int]:
    return dict.fromkeys(range(count), length)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("count", "sizes"),
    [(2, [2]), (3, [2, 1]), (5, [4, 1]), (6, [4, 2]), (7, [4, 2, 1]), (8, [4, 4]), (9, [4, 4, 1])],
)
def test_an_arbitrary_prompt_list_is_split_into_contiguous_batches(
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    logged: Mock,
    count: int,
    sizes: list[int],
) -> None:
    """Any length is served: the plan picks the largest batch that still fits."""
    serving = make_serving(engine_inputs=list(range(count)), prompt_lengths=_input_lengths(count))

    result = asyncio.run(
        create_completion_batch_dispatch(serving, make_request(prompt=["a"] * count))
    )

    assert result == f"response-for-{count}"

    assert [len(call["prompt"]) for call in serving.beam_calls] == sizes
    assert [i for call in serving.beam_calls for i in call["prompt"]] == list(range(count))
    assert serving.render_calls == 1
    assert len(serving.response_calls) == 1
    assert logged.call_args.args[3:] == (len(sizes), sizes)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("sizes", "count", "seqs", "tokens", "expected"),
    [(list(range(1, 9)), n, 8, 100, [n]) for n in range(2, 9)]
    + [
        ([1, 3, 7], 10, 8, 100, [7, 3]),
        ([1, 3, 7], 8, 8, 100, [7, 1]),
        ([1, 3], 7, 8, 100, [3, 3, 1]),
        ([1, 3, 7], 7, 4, 100, [3, 3, 1]),
        ([1, 3, 7], 7, 8, 30, [3, 3, 1]),
        ([1], 3, 8, 100, [1, 1, 1]),
    ],
)
def test_online_batch_policy_reaches_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    sizes: list[int],
    count: int,
    seqs: int,
    tokens: int,
    expected: list[int],
) -> None:
    monkeypatch.setattr(patch_module, "GRConfig", GRConfig)
    serving = make_serving(
        engine_inputs=list(range(count)),
        prompt_lengths=_input_lengths(count),
        max_num_seqs=seqs,
        max_num_batched_tokens=tokens,
    )
    serving.engine_client.vllm_config.additional_config = {
        "vllm_gr_beam_execution_mode": "v1",
        "vllm_gr_beam_online_batch_sizes": sizes,
    }
    result = asyncio.run(
        create_completion_batch_dispatch(serving, make_request(prompt=["a"] * count))
    )
    assert result == f"response-for-{count}"
    assert [len(call["prompt"]) for call in serving.beam_calls] == expected
    assert [i for call in serving.beam_calls for i in call["prompt"]] == list(range(count))
    assert [output.prompt for output in serving.response_calls[0]] == [
        f"text-{i}" for i in range(count)
    ]
    assert serving.render_calls == len(serving.response_calls) == 1


def test_sub_batches_use_distinct_ids_and_keep_the_first_batch_unchanged(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    """One session keeps the caller's id; a split names each session after it."""
    single = make_serving(engine_inputs=[0, 1], prompt_lengths={0: 10, 1: 10})
    asyncio.run(create_completion_batch_dispatch(single, make_request(prompt=["a", "b"])))

    split = make_serving(
        engine_inputs=list(range(6)), prompt_lengths=_input_lengths(6), max_num_seqs=2
    )
    asyncio.run(create_completion_batch_dispatch(split, make_request(prompt=["a"] * 6)))

    assert [call["request_id"] for call in single.beam_calls] == ["cmpl-request-0"]
    assert [call["request_id"] for call in split.beam_calls] == [
        "cmpl-request-0-batch-0",
        "cmpl-request-0-batch-1",
        "cmpl-request-0-batch-2",
    ]


def test_a_prompt_list_is_validated_before_the_first_submission(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any], delegate: list[Any]
) -> None:
    """An input the engine would reject stops the list instead of truncating it.

    Submitting prompt by prompt would compute the earlier ones and then have
    nowhere to put the answer, so the whole plan is settled first.
    """
    from vllm.exceptions import VLLMValidationError

    serving = make_serving(prompt_lengths={0: 10, 1: 10, 2: 10, 3: 41}, max_num_batched_tokens=40)

    with pytest.raises(VLLMValidationError, match="Prompt 3 Prefill length"):
        asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert serving.beam_calls == []
    assert delegate == []


def test_prompts_that_disagree_on_their_budget_keep_their_own(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any], delegate: list[Any]
) -> None:
    """Budgets stay per input: no prompt is shortened to match a neighbour.

    Four prompts that agree on nothing the plan can join still come back as one
    response from the route that claimed them -- the split is just trivial.
    """
    serving = make_serving(prompt_lengths={0: 10, 1: 20, 2: 30, 3: 40})
    request = make_request(to_beam_search_params=lambda tokens, _: {"max_tokens": tokens})

    assert asyncio.run(create_completion_batch_dispatch(serving, request)) == "response-for-4"

    assert delegate == []
    assert [call["params"]["max_tokens"] for call in serving.beam_calls] == [990, 980, 970, 960]
    assert [call["prompt"] for call in serving.beam_calls] == [[0], [1], [2], [3]]
    assert serving.render_calls == 1


def test_a_list_past_the_token_budget_is_split_rather_than_refused(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    """The token budget constrains each session, not the HTTP list."""
    serving = make_serving(prompt_lengths=_input_lengths(4), max_num_batched_tokens=39)

    asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1], [2, 3]]


def test_the_begin_token_counts_towards_the_token_budget(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    """The engine measures the Prefill after appending the begin token.

    The renderer never sees that token, so a budget that fits the rendered
    lengths exactly is one token per prompt short of what the engine checks.
    """
    serving = make_serving(prompt_lengths=_input_lengths(4), max_num_batched_tokens=40)
    request = make_request(vllm_xargs={"begin_token": "<|sid_begin|>"})

    asyncio.run(create_completion_batch_dispatch(serving, request))

    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1], [2, 3]]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("limit_name", "limit"),
    [("max_num_seqs", 2), ("max_num_scheduled_tokens", 20)],
)
def test_each_execution_limit_splits_the_plan_before_submission(
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    limit_name: str,
    limit: int,
) -> None:
    """Both caps the scheduler applies are honoured by the plan, not by luck."""
    serving = make_serving(prompt_lengths=_input_lengths(4))
    setattr(serving.engine_client.vllm_config.scheduler_config, limit_name, limit)

    asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1], [2, 3]]


@pytest.mark.parametrize("failure", ["execution", "cleanup", "cancel", "short"])  # type: ignore[misc]
def test_a_failure_mid_list_never_submits_another_session(
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    delegate: list[Any],
    failure: str,
) -> None:
    """Whether the second session raises, is cancelled, or comes up short, the
    third is not submitted and no partial response is assembled."""
    serving = make_serving(engine_inputs=list(range(9)), prompt_lengths=_input_lengths(9))
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
    call = create_completion_batch_dispatch(serving, make_request(prompt=["a"] * 9))

    if failure in ("execution", "cleanup"):
        # An engine fault is the engine's, and propagates as it would without
        # this patch: cleanup still runs, and nothing is retried or replayed.
        with pytest.raises(RuntimeError, match=failure):
            asyncio.run(call)
    else:
        assert asyncio.run(call) == "error-response"
    assert events == [("start", 0), ("close", 0), ("start", 1), ("close", 1)]
    assert len(serving.beam_calls) == 2
    assert serving.response_calls == []
    assert delegate == []


def test_cancellation_between_sessions_prevents_the_next_submission(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    serving = make_serving(engine_inputs=list(range(5)), prompt_lengths=_input_lengths(5))

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
    result = asyncio.run(create_completion_batch_dispatch(serving, make_request(prompt=["a"] * 5)))

    assert result == "error-response"
    assert len(serving.beam_calls) == 1
    assert serving.response_calls == []


def test_cancellation_waits_for_the_running_iterator_to_close(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    """The engine's session is released before the error response is built."""
    serving = make_serving(engine_inputs=list(range(5)), prompt_lengths=_input_lengths(5))

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
            create_completion_batch_dispatch(serving, make_request(prompt=["a"] * 5))
        )
        await started.wait()
        task.cancel()
        assert await task == "error-response"
        assert closed.is_set()
        assert len(serving.beam_calls) == 1

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Server faults and cancellation
# --------------------------------------------------------------------------- #


def test_a_render_that_disagrees_with_the_count_is_a_server_fault(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    """Nothing downstream holds if the renderer and the request disagree."""
    serving = make_serving(engine_inputs=[0, 1], prompt_lengths={0: 10, 1: 20})

    result = asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert result == "error-response"
    assert serving.error[1]["err_type"] == "InternalServerError"
    assert serving.beam_calls == []


def test_a_short_result_batch_is_a_server_fault(
    serving: _BatchSelf, make_request: Callable[..., Any]
) -> None:
    """A result that does not match the request is a defect, not a caller error."""

    def short(**kwargs: Any) -> Any:
        async def generator() -> Any:
            yield SimpleNamespace(prompt=None)

        return generator()

    serving.beam_search = short  # type: ignore[method-assign]

    result = asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert result == "error-response"
    assert serving.error[1]["err_type"] == "InternalServerError"


def test_a_cancelled_client_is_answered_rather_than_raised(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    serving = make_serving(engine_inputs=[0], prompt_lengths={0: 10})
    request = make_request(prompt=["a"])

    def cancelled(**kwargs: Any) -> Any:
        async def generator() -> Any:
            raise asyncio.CancelledError
            yield  # pragma: no cover - unreachable, makes this an async generator

        return generator()

    serving.beam_search = cancelled  # type: ignore[method-assign]

    plan = (CompletionBatch(start=0, stop=1, max_tokens=5),)
    result = asyncio.run(_create_completion_batched(serving, request, None, [0], plan))

    assert result == "error-response"
    assert serving.error[0] == "Client disconnected"


# --------------------------------------------------------------------------- #
# The real response converter and the patch itself
# --------------------------------------------------------------------------- #


def test_real_response_conversion_preserves_indices_and_usage(
    make_serving: Callable[..., _BatchSelf], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.outputs import CompletionOutput, RequestOutput

    from vllm_gr.entrypoints.openai.protocol import to_beam_search_params

    monkeypatch.setattr(
        CompletionRequest, "to_beam_search_params", to_beam_search_params, raising=False
    )
    serving = make_serving(
        engine_inputs=list(range(6)),
        prompt_lengths=_input_lengths(6),
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

    result = asyncio.run(create_completion_batch_dispatch(serving, request))

    # Six prompts become two sessions, and one response covering both: the
    # choice indices run across the whole list, in order, as they do upstream.
    assert [len(call["prompt"]) for call in serving.beam_calls] == [4, 2]
    assert result.id == "cmpl-request-0"
    assert [c.index for c in result.choices] == list(range(12))
    assert [c.text for c in result.choices] == [f"{i}:{b}" for i in range(6) for b in range(2)]
    assert result.usage.prompt_tokens == 60
    assert result.usage.completion_tokens == 24
    assert result.usage.total_tokens == 84


@pytest.fixture  # type: ignore[misc]
def restore_route() -> Any:
    """Installing the patch touches the class itself, so put it back after."""
    before = OpenAIServingCompletion._create_completion
    yield
    OpenAIServingCompletion._create_completion = before
    patch_module._original_create_completion = None


def test_the_patch_installs_and_verifies(restore_route: Any) -> None:  # noqa: ARG001
    patch_completion_batch()

    assert OpenAIServingCompletion._create_completion is create_completion_batch_dispatch
    assert patch_module._validate_completion_batch_patch() is None


def test_the_patch_is_idempotent(restore_route: Any) -> None:  # noqa: ARG001
    patch_completion_batch()
    original = patch_module._original_create_completion
    patch_completion_batch()

    assert patch_module._original_create_completion is original
