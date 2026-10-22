# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the partition of a prompt list into contiguous beam batches."""

import pytest

import vllm_gr.entrypoints.openai.completion_batch_plan as plan_module
from vllm_gr.entrypoints.openai.completion_batch_plan import plan_completion_batches
from vllm_gr.v1.beam.batch_contracts import effective_generation_budget_error


def _plan(
    lengths: list[int],
    budgets: list[int],
    *,
    seqs: int = 4,
    tokens: int = 40,
    reserved: int = 0,
) -> list[tuple[int, int, int]]:
    plan = plan_completion_batches(
        lengths,
        budgets,
        configured_sizes=(1, 2, 4),
        max_num_seqs=seqs,
        token_budget=tokens,
        reserved_tokens=reserved,
    )
    return [(batch.start, batch.stop, batch.max_tokens) for batch in plan.batches]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("count", "expected"),
    [
        (1, [1]),
        (2, [2]),
        (3, [2, 1]),
        (4, [4]),
        (5, [4, 1]),
        (6, [4, 2]),
        (7, [4, 2, 1]),
        (8, [4, 4]),
        (9, [4, 4, 1]),
    ],
)
def test_every_length_is_partitioned_largest_first(count: int, expected: list[int]) -> None:
    plan = _plan([10] * count, [3] * count)

    assert [stop - start for start, stop, _ in plan] == expected
    # Contiguous and complete: the caller's order is the response's order.
    assert [i for start, stop, _ in plan for i in range(start, stop)] == list(range(count))


@pytest.mark.parametrize(  # type: ignore[misc]
    ("lengths", "budgets", "seqs", "tokens", "expected"),
    [
        ([10] * 4, [3] * 4, 2, 40, [(0, 2, 3), (2, 4, 3)]),
        ([10] * 4, [3] * 4, 1, 40, [(i, i + 1, 3) for i in range(4)]),
        ([10] * 4, [3] * 4, 4, 39, [(0, 2, 3), (2, 4, 3)]),
        ([10, 20, 5, 5], [3] * 4, 4, 40, [(0, 4, 3)]),
        ([10, 31, 5, 5], [3] * 4, 4, 40, [(0, 1, 3), (1, 3, 3), (3, 4, 3)]),
        ([10] * 4, [3, 3, 2, 2], 4, 40, [(0, 2, 3), (2, 4, 2)]),
        ([10] * 3, [3, 2, 3], 4, 40, [(0, 1, 3), (1, 2, 2), (2, 3, 3)]),
    ],
)
def test_capacity_and_budget_shape_the_boundaries(
    lengths: list[int],
    budgets: list[int],
    seqs: int,
    tokens: int,
    expected: list[tuple[int, int, int]],
) -> None:
    assert _plan(lengths, budgets, seqs=seqs, tokens=tokens) == expected


@pytest.mark.parametrize(  # type: ignore[misc]
    ("lengths", "budgets", "reserved"),
    [
        ([], [], 0),
        ([10], [], 0),
        ([10, 41], [3, 3], 0),
        ([10, 0], [3, 3], 0),
        ([10, 10], [3, 0], 0),
        # The last input is judged with every other input before any batch is
        # returned, on both the lower and the upper effective-budget bound.
        ([10, 10, 10, 10], [3, 3, 3, 1], 1),
        ([10, 10, 10, 10], [3, 3, 3, 4], 0),
        ([10, 10, 10, 10], [3, 3, 3, 0], 0),
    ],
)
def test_an_invalid_input_rejects_the_whole_list(
    lengths: list[int], budgets: list[int], reserved: int
) -> None:
    """Checked before any batch is returned, so nothing runs partially."""
    with pytest.raises(ValueError):
        _plan(lengths, budgets, reserved=reserved)


def test_an_empty_render_is_only_legal_with_a_boundary_token() -> None:
    """A tokenizer without BOS renders "" to nothing; the begin token fills it.

    The planner sees prompt lengths after the frontend adds the begin token, so
    the same empty render is one token long exactly when that token is reserved.
    """
    with pytest.raises(ValueError, match="Prompt 0 Prefill length 0 must be between 1"):
        _plan([0, 10], [2, 2])

    assert _plan([0 + 1, 10], [2, 2], reserved=1) == [(0, 2, 2)]


def test_reserved_tokens_bound_the_effective_budget_not_the_carried_budget() -> None:
    # The batch carries the public budget; the engine removes the reserved
    # begin/end positions, leaving a legal N=3 or N=1 Decode budget.
    assert _plan([10] * 4, [4] * 4, reserved=1) == [(0, 4, 4)]
    assert _plan([10] * 4, [2] * 4, reserved=1) == [(0, 4, 2)]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("budget", "reserved", "legal"),
    [
        (1, 0, True),
        (3, 0, True),
        (4, 0, False),
        (1, 1, False),
        (2, 1, True),
        (3, 1, True),
        (4, 1, True),
        (6, 2, False),
        (5, 2, True),
        (3, 2, True),
    ],
)
def test_effective_generation_budget_matches_the_engine_contract(
    budget: int, reserved: int, legal: bool
) -> None:
    assert (effective_generation_budget_error(budget, reserved) is None) is legal


@pytest.mark.parametrize(  # type: ignore[misc]
    ("budget", "reserved", "cap", "legal"),
    [
        (3, 0, 3, True),
        (3, 0, 2, False),
        (2, 0, 1, False),
        # The engine's 1..3 range is checked before the configured cap.
        (4, 0, 3, False),
        (4, 1, 3, True),
        (2, 1, 3, True),
        (3, 1, 3, True),
        (4, 1, 2, False),
    ],
)
def test_the_configured_decode_cap_narrows_the_engine_contract(
    budget: int, reserved: int, cap: int, legal: bool
) -> None:
    error = effective_generation_budget_error(budget, reserved, max_decode_steps=cap)
    assert (error is None) is legal
    if not legal:
        assert "beam_max_decode_steps" in error or "asynchronous scheduling" in error


def test_a_last_input_over_the_configured_cap_rejects_the_whole_list() -> None:
    """The engine's configured cap is static too, and the last input is judged too."""
    with pytest.raises(
        ValueError,
        match="Prompt 2 generation budget 3 leaves 3 Decode steps.*limit is 2",
    ):
        plan_completion_batches(
            [10, 10, 10],
            [1, 1, 3],
            configured_sizes=(1, 2),
            max_num_seqs=2,
            token_budget=40,
            max_decode_steps=2,
        )

    # Within the cap the same list plans, so the cap is what rejected it.
    plan = plan_completion_batches(
        [10, 10, 10],
        [1, 1, 2],
        configured_sizes=(1, 2),
        max_num_seqs=2,
        token_budget=40,
        max_decode_steps=2,
    )
    assert [(batch.start, batch.stop, batch.max_tokens) for batch in plan.batches] == [
        (0, 2, 1),
        (2, 3, 2),
    ]


def test_custom_sizes_respect_generation_budget_boundaries() -> None:
    plan = plan_completion_batches(
        [10] * 8,
        [3] * 4 + [2] * 4,
        configured_sizes=(1, 4, 7),
        max_num_seqs=8,
        token_budget=100,
    )
    assert [(b.start, b.stop, b.max_tokens) for b in plan.batches] == [
        (0, 4, 3),
        (4, 8, 2),
    ]


def test_capacity_never_introduces_a_size_the_policy_omits() -> None:
    """A documented `[1, 3]` row: capacity only removes sizes, never adds one."""
    plan = plan_completion_batches(
        [10] * 7,
        [3] * 7,
        configured_sizes=(1, 3),
        max_num_seqs=4,
        token_budget=100,
    )

    # Seven inputs under max_num_seqs=4 stay 3 + 3 + 1: 4 is not configured,
    # so neither capacity nor the leftover input can produce it.
    assert [(b.start, b.stop) for b in plan.batches] == [(0, 3), (3, 6), (6, 7)]
    assert plan.configured_sizes == (3, 1)
    assert plan.usable_sizes == (3, 1)
    assert plan.capacity_source is None
    assert plan.reasons == ()


def test_a_broken_progress_invariant_fails_instead_of_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject a rejected sum after valid singletons passed prevalidation.
    monkeypatch.setattr(plan_module, "sum", lambda _: 41, raising=False)
    with pytest.raises(RuntimeError, match="made no progress at prompt 0"):
        _plan([10], [3])


# --------------------------------------------------------------------------- #
# Per batch: name the limit that stopped this boundary, not the limits in play
# --------------------------------------------------------------------------- #


def test_every_batch_names_the_limit_that_stopped_its_boundary() -> None:
    # A pair costs 20 tokens and a quartet 40, over a 25-token budget: the two
    # pairs were refused a wider batch by the budget, the last pair is the end
    # of the list and nothing larger was ever a candidate.
    plan = plan_completion_batches(
        [10] * 6,
        [3] * 6,
        configured_sizes=(1, 2, 4),
        max_num_seqs=4,
        token_budget=25,
    )

    assert [(b.start, b.stop, b.boundary_reason) for b in plan.batches] == [
        (0, 2, "token_budget"),
        (2, 4, "token_budget"),
        (4, 6, None),
    ]
    assert plan.reasons == ("token_budget",)


def test_a_batch_split_by_its_neighbour_names_the_budget_shape() -> None:
    plan = plan_completion_batches(
        [10] * 4,
        [3, 3, 2, 2],
        configured_sizes=(1, 2, 4),
        max_num_seqs=4,
        token_budget=40,
    )

    # The second batch is on the boundary, not over a budget: the plan as a
    # whole is narrow because two prompts disagree, and only the first batch
    # can say so.
    assert [(b.start, b.stop, b.boundary_reason) for b in plan.batches] == [
        (0, 2, "unequal_effective_budgets"),
        (2, 4, None),
    ]
    assert plan.reasons == ("unequal_effective_budgets",)


# --------------------------------------------------------------------------- #
# Capacity: which cap, and only when it actually held the plan back
# --------------------------------------------------------------------------- #


def test_capacity_below_the_configured_width_is_labelled_by_its_source() -> None:
    plan = plan_completion_batches(
        [10] * 8,
        [3] * 8,
        configured_sizes=(1, 2, 4, 8),
        max_num_seqs=4,
        token_budget=100,
        capacity_source="max_num_seqs",
    )

    assert [b.stop - b.start for b in plan.batches] == [4, 4]
    assert plan.configured_sizes == (8, 4, 2, 1)
    assert plan.usable_sizes == (4, 2, 1)
    assert plan.capacity_source == "max_num_seqs"
    # Every batch reached the usable width, so nothing degraded and no reason
    # may claim otherwise.
    assert plan.reasons == ()
    assert (plan.capacity_source == "decode_rows") is False


def test_the_decode_row_cap_keeps_its_own_label() -> None:
    """The two caps are told apart: an NPU row limit is not ``max_num_seqs``."""
    plan = plan_completion_batches(
        [10] * 4,
        [3] * 4,
        configured_sizes=(1, 2, 4, 8),
        max_num_seqs=4,
        token_budget=100,
        capacity_source="decode_rows",
    )

    assert plan.usable_sizes == (4, 2, 1)
    assert plan.capacity_source == "decode_rows"
    assert plan.reasons == ()


def test_a_deployment_that_runs_the_configured_width_reports_no_cap() -> None:
    plan = plan_completion_batches(
        [10] * 4,
        [3] * 4,
        configured_sizes=(1, 2, 4),
        max_num_seqs=8,
        token_budget=100,
        capacity_source="max_num_seqs",
    )

    assert plan.usable_sizes == (4, 2, 1)
    # The label names a cap nobody hit, so it is dropped rather than reported.
    assert plan.capacity_source is None


def test_a_singleton_configuration_is_named_where_capacity_is_not() -> None:
    plan = plan_completion_batches(
        [10] * 4,
        [3] * 4,
        configured_sizes=(1,),
        max_num_seqs=8,
        token_budget=100,
        capacity_source="max_num_seqs",
    )

    assert plan.usable_sizes == (1,)
    assert plan.capacity_source is None
    assert plan.reasons == ("configured_singletons",)
    assert plan.singletons_are_intentional


# --------------------------------------------------------------------------- #
# The Prefill budget is the scheduler's, and no chunked path can beat it
# --------------------------------------------------------------------------- #


def test_the_prefill_budget_is_the_scheduling_budget_of_the_replaced_path() -> None:
    """A prompt over the budget is refused here, not scheduled in pieces.

    Beam search cannot chunk a Prefill: the V1 runtime rejects
    ``enable_chunked_prefill`` at startup and every input is measured against
    the same ``max_num_batched_tokens`` the non-batched renderer already
    applies to the list total. The planner refusing one prompt over that
    budget is therefore the same limit, decided before any sub-batch runs.
    """
    with pytest.raises(ValueError, match="Prompt 0 Prefill length 20480"):
        _plan([20480, 10], [3, 3], tokens=8192)

    # The same input at the budget the scheduler was built with plans normally.
    assert _plan([20480, 10], [3, 3], tokens=32768, seqs=2) == [(0, 2, 3)]
