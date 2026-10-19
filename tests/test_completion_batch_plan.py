# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the partition of a prompt list into contiguous beam batches."""

import pytest

import vllm_gr.entrypoints.openai.completion_batch_plan as plan_module
from vllm_gr.entrypoints.openai.completion_batch_plan import plan_completion_batches


def _plan(
    lengths: list[int],
    budgets: list[int],
    *,
    seqs: int = 4,
    tokens: int = 40,
) -> list[tuple[int, int, int]]:
    plan = plan_completion_batches(
        lengths,
        budgets,
        supported_sizes=(1, 2, 4),
        max_num_seqs=seqs,
        token_budget=tokens,
    )
    return [(batch.start, batch.stop, batch.max_tokens) for batch in plan]


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
    plan = _plan([10] * count, [5] * count)

    assert [stop - start for start, stop, _ in plan] == expected
    # Contiguous and complete: the caller's order is the response's order.
    assert [i for start, stop, _ in plan for i in range(start, stop)] == list(range(count))


@pytest.mark.parametrize(  # type: ignore[misc]
    ("lengths", "budgets", "seqs", "tokens", "expected"),
    [
        ([10] * 4, [5] * 4, 2, 40, [(0, 2, 5), (2, 4, 5)]),
        ([10] * 4, [5] * 4, 1, 40, [(i, i + 1, 5) for i in range(4)]),
        ([10] * 4, [5] * 4, 4, 39, [(0, 2, 5), (2, 4, 5)]),
        ([10, 20, 5, 5], [5] * 4, 4, 40, [(0, 4, 5)]),
        ([10, 31, 5, 5], [5] * 4, 4, 40, [(0, 1, 5), (1, 3, 5), (3, 4, 5)]),
        ([10] * 4, [5, 5, 4, 4], 4, 40, [(0, 2, 5), (2, 4, 4)]),
        ([10] * 3, [5, 4, 5], 4, 40, [(0, 1, 5), (1, 2, 4), (2, 3, 5)]),
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
    ("lengths", "budgets"),
    [([], []), ([10], []), ([10, 41], [5, 5]), ([10, 0], [5, 5]), ([10, 10], [5, 0])],
)
def test_an_invalid_input_rejects_the_whole_list(lengths: list[int], budgets: list[int]) -> None:
    """Checked before any batch is returned, so nothing runs partially."""
    with pytest.raises(ValueError):
        _plan(lengths, budgets)


def test_custom_sizes_respect_generation_budget_boundaries() -> None:
    plan = plan_completion_batches(
        [10] * 8,
        [5] * 3 + [4] * 5,
        supported_sizes=(1, 3, 7),
        max_num_seqs=8,
        token_budget=100,
    )
    assert [(b.start, b.stop, b.max_tokens) for b in plan] == [
        (0, 3, 5),
        (3, 6, 4),
        (6, 7, 4),
        (7, 8, 4),
    ]


def test_a_broken_progress_invariant_fails_instead_of_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject a rejected sum after valid singletons passed prevalidation.
    monkeypatch.setattr(plan_module, "sum", lambda _: 41, raising=False)
    with pytest.raises(RuntimeError, match="made no progress at prompt 0"):
        _plan([10], [5])
