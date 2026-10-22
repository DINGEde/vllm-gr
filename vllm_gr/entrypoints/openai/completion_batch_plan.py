# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Plan contiguous beam batches without submitting or mutating requests."""

from collections.abc import Sequence
from dataclasses import dataclass

from vllm_gr.v1.beam.batch_contracts import (
    effective_generation_budget_error,
    select_supported_batch_sizes,
    validate_prompt_lengths,
)


@dataclass(frozen=True, slots=True)
class CompletionBatch:
    start: int
    stop: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class CompletionBatchPlan:
    """Batches plus the static facts that explain why they are no larger."""

    batches: tuple[CompletionBatch, ...]
    configured_sizes: tuple[int, ...]
    usable_sizes: tuple[int, ...]
    reasons: tuple[str, ...]

    @property
    def all_singletons(self) -> bool:
        return all(batch.stop - batch.start == 1 for batch in self.batches)

    @property
    def singletons_are_intentional(self) -> bool:
        """True when configuration or capacity leaves no larger batch to use."""
        return self.usable_sizes == (1,)


def completion_batch_plan_reasons(
    batches: Sequence[CompletionBatch],
    prompt_lengths: Sequence[int],
    effective_budgets: Sequence[int],
    *,
    configured_sizes: Sequence[int],
    usable_sizes: Sequence[int],
    token_budget: int,
) -> tuple[str, ...]:
    """Name the static limits that kept this plan from using a larger batch."""
    if not batches:
        return ()
    reasons = []
    if len(set(effective_budgets)) > 1:
        reasons.append("unequal_effective_budgets")
    configured_max = max(configured_sizes, default=1)
    if configured_max > max(usable_sizes, default=1):
        reasons.append("sequence_capacity")
    if configured_max == 1:
        reasons.append("configured_singletons")
    if len(batches) > 1 and sum(prompt_lengths) > token_budget:
        reasons.append("token_budget")
    return tuple(reasons)


def plan_completion_batches(
    prompt_lengths: Sequence[int],
    generation_budgets: Sequence[int],
    *,
    supported_sizes: Sequence[int],
    max_num_seqs: int,
    token_budget: int,
    reserved_tokens: int = 0,
    max_decode_steps: int | None = None,
) -> tuple[CompletionBatch, ...]:
    """Choose the largest legal contiguous batch at each input position.

    ``prompt_lengths`` include any tokens appended by the beam frontend, and
    ``reserved_tokens`` counts the begin/end positions the engine removes from
    each public generation budget before scheduling. ``max_decode_steps`` is the
    deployment's configured Decode cap, which the engine enforces on top of its
    supported range. Every input is validated before any batch is chosen, so a
    rejection the planner can decide statically cannot leave earlier prompts
    computed but unanswered. Device/KV admission remains the engine's
    responsibility.
    """
    if not prompt_lengths or len(prompt_lengths) != len(generation_budgets):
        raise ValueError("Expected nonempty, equally sized prompt lengths and budgets")
    sizes = select_supported_batch_sizes(supported_sizes, max_num_seqs=max_num_seqs)
    validate_prompt_lengths(prompt_lengths, token_budget=token_budget)
    for index, budget in enumerate(generation_budgets):
        failure = effective_generation_budget_error(
            budget,
            reserved_tokens,
            name=f"Prompt {index} generation budget",
            max_decode_steps=max_decode_steps,
        )
        if failure is not None:
            raise ValueError(failure)

    batches = []
    start = 0
    while start < len(prompt_lengths):
        budget = generation_budgets[start]
        for size in sizes:
            stop = start + size
            if stop > len(prompt_lengths):
                continue
            if any(value != budget for value in generation_budgets[start:stop]):
                continue
            if sum(prompt_lengths[start:stop]) > token_budget:
                continue
            batches.append(CompletionBatch(start, stop, budget))
            start = stop
            break
        else:
            # Singleton validation guarantees progress; fail if that invariant breaks.
            raise RuntimeError(f"Beam batch planner made no progress at prompt {start}")
    return tuple(batches)
