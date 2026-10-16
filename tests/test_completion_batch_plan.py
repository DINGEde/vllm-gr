# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm_gr.entrypoints.openai.completion_batch_plan as plan_module
from vllm_gr.entrypoints.openai.completion_batch_plan import plan_completion_batches


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
def test_contiguous_partition(count: int, expected: list[int]) -> None:
    plan = plan_completion_batches(
        [10] * count,
        [5] * count,
        supported_sizes=(1, 2, 4),
        max_num_seqs=4,
        token_budget=40,
    )
    assert [batch.stop - batch.start for batch in plan] == expected
    assert [i for batch in plan for i in range(batch.start, batch.stop)] == list(range(count))


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
def test_capacity_and_budget_boundaries(
    lengths: list[int],
    budgets: list[int],
    seqs: int,
    tokens: int,
    expected: list[tuple[int, int, int]],
) -> None:
    plan = plan_completion_batches(
        lengths,
        budgets,
        supported_sizes=(1, 2, 4),
        max_num_seqs=seqs,
        token_budget=tokens,
    )
    assert [(batch.start, batch.stop, batch.max_tokens) for batch in plan] == expected


@pytest.mark.parametrize(  # type: ignore[misc]
    ("lengths", "budgets"),
    [([], []), ([10], []), ([10, 41], [5, 5]), ([10, 0], [5, 5]), ([10, 10], [5, 0])],
)
def test_invalid_later_input_rejects_the_whole_plan(
    lengths: list[int],
    budgets: list[int],
) -> None:
    with pytest.raises(ValueError):
        plan_completion_batches(
            lengths,
            budgets,
            supported_sizes=(1, 2, 4),
            max_num_seqs=4,
            token_budget=40,
        )


def test_broken_progress_invariant_fails_instead_of_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject a rejected sum after valid singletons passed prevalidation.
    monkeypatch.setattr(plan_module, "sum", lambda _: 41, raising=False)
    with pytest.raises(RuntimeError, match="made no progress at prompt 0"):
        plan_completion_batches(
            [10], [5], supported_sizes=(1, 2, 4), max_num_seqs=4, token_budget=40
        )
