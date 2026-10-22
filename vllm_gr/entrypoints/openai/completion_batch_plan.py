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

# Why a batch could not be widened, per batch and per plan. Each name is one
# static limit the planner measured, so a reader of the log never has to guess
# which of several limits a single word stood for.
UNEQUAL_BUDGETS_REASON = "unequal_effective_budgets"
TOKEN_BUDGET_REASON = "token_budget"
CONFIGURED_SINGLETONS_REASON = "configured_singletons"
_PLAN_REASON_ORDER = (
    CONFIGURED_SINGLETONS_REASON,
    UNEQUAL_BUDGETS_REASON,
    TOKEN_BUDGET_REASON,
)

# Cap that lowered the usable width below the configured width. ``None`` means
# the configuration is what the deployment can run.
MAX_NUM_SEQS_CAPACITY = "max_num_seqs"
DECODE_ROWS_CAPACITY = "decode_rows"


@dataclass(frozen=True, slots=True)
class CompletionBatch:
    """One contiguous sub-batch, and the limit that kept it from being wider."""

    start: int
    stop: int
    max_tokens: int
    # ``None`` when no wider legal batch exists: the batch already covers the
    # rest of the list, or no larger supported size fits in it.
    boundary_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionBatchPlan:
    """Batches plus the static facts that explain why they are no larger."""

    batches: tuple[CompletionBatch, ...]
    configured_sizes: tuple[int, ...]
    usable_sizes: tuple[int, ...]
    # Which deployment cap holds the usable width below the configured width,
    # or ``None`` when the configuration is what the deployment can run. This
    # is a fact about the deployment, not a claim that the plan degraded: a
    # plan whose batches all reach ``usable_sizes`` lost no width.
    capacity_source: str | None
    # The limits that actually held this plan below ``usable_sizes``.
    reasons: tuple[str, ...]

    @property
    def all_singletons(self) -> bool:
        return all(batch.stop - batch.start == 1 for batch in self.batches)

    @property
    def singletons_are_intentional(self) -> bool:
        """True when configuration or capacity leaves no larger batch to use."""
        return self.usable_sizes == (1,)


def batch_boundary_reason(
    start: int,
    size: int,
    prompt_lengths: Sequence[int],
    generation_budgets: Sequence[int],
    *,
    usable_sizes: Sequence[int],
    token_budget: int,
) -> str | None:
    """Name the first static limit that refused a batch wider than ``size``.

    The partition is greedy and ``usable_sizes`` is descending, so every size
    above ``size`` was tried at this position and refused; this reports the
    refusal for the widest such size -- the one the batch lost the most from.
    A size that would run past the last input was never a candidate, so a batch
    at the end of the list legitimately reports no limit at all.
    """
    for candidate in usable_sizes:
        if candidate <= size:
            continue
        stop = start + candidate
        if stop > len(prompt_lengths):
            continue
        budget = generation_budgets[start]
        if any(value != budget for value in generation_budgets[start:stop]):
            return UNEQUAL_BUDGETS_REASON
        if sum(prompt_lengths[start:stop]) > token_budget:
            return TOKEN_BUDGET_REASON
        # Unreachable while the partition stays greedy: a wider candidate that
        # passes every check is the candidate the loop would have taken.
    return None


def plan_reasons(
    batches: Sequence[CompletionBatch],
    *,
    configured_sizes: Sequence[int],
) -> tuple[str, ...]:
    """Name the limits that held this plan below the width it could have run."""
    present = {batch.boundary_reason for batch in batches}
    present.discard(None)
    if max(configured_sizes, default=1) == 1:
        present.add(CONFIGURED_SINGLETONS_REASON)
    return tuple(name for name in _PLAN_REASON_ORDER if name in present)


def plan_completion_batches(
    prompt_lengths: Sequence[int],
    generation_budgets: Sequence[int],
    *,
    configured_sizes: Sequence[int],
    max_num_seqs: int,
    token_budget: int,
    reserved_tokens: int = 0,
    max_decode_steps: int | None = None,
    capacity_source: str | None = None,
) -> CompletionBatchPlan:
    """Choose the largest legal contiguous batch at each input position.

    ``prompt_lengths`` include any tokens appended by the beam frontend, and
    ``reserved_tokens`` counts the begin/end positions the engine removes from
    each public generation budget before scheduling. ``max_decode_steps`` is the
    deployment's configured Decode cap, which the engine enforces on top of its
    supported range. ``capacity_source`` labels the cap the caller applied to
    reach ``max_num_seqs``; it is reported only when it is what keeps the usable
    width below the configured width.

    The configured sizes are narrowed to the usable ones exactly once, here, and
    the returned plan carries that decision along with why each batch stopped
    growing. Every input is validated before any batch is chosen, so a rejection
    the planner can decide statically cannot leave earlier prompts computed but
    unanswered. Device/KV admission remains the engine's responsibility.
    """
    if not prompt_lengths or len(prompt_lengths) != len(generation_budgets):
        raise ValueError("Expected nonempty, equally sized prompt lengths and budgets")
    configured = tuple(sorted({size for size in configured_sizes if size > 0}, reverse=True))
    sizes = select_supported_batch_sizes(configured, max_num_seqs=max_num_seqs)
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
            batches.append(
                CompletionBatch(
                    start,
                    stop,
                    budget,
                    batch_boundary_reason(
                        start,
                        size,
                        prompt_lengths,
                        generation_budgets,
                        usable_sizes=sizes,
                        token_budget=token_budget,
                    ),
                )
            )
            start = stop
            break
        else:
            # Singleton validation guarantees progress; fail if that invariant breaks.
            raise RuntimeError(f"Beam batch planner made no progress at prompt {start}")
    return CompletionBatchPlan(
        batches=tuple(batches),
        configured_sizes=configured,
        usable_sizes=sizes,
        # Only a deployment that cannot run the configured width is worth
        # labelling; at equal width the label would name a limit nobody hit.
        capacity_source=(capacity_source if max(sizes) < max(configured, default=1) else None),
        reasons=plan_reasons(batches, configured_sizes=configured),
    )
