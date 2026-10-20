# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Plan contiguous beam batches without submitting or mutating requests."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompletionBatch:
    start: int
    stop: int
    max_tokens: int


def plan_completion_batches(
    prompt_lengths: Sequence[int],
    generation_budgets: Sequence[int],
    *,
    supported_sizes: Sequence[int],
    max_num_seqs: int,
    token_budget: int,
    max_decode_seqs: int | None = None,
) -> tuple[CompletionBatch, ...]:
    """Choose the largest legal contiguous batch at each input position.

    Lengths include any tokens appended by the beam frontend. Validate every
    singleton first so an invalid later input cannot cause partial execution.
    An optional physical Decode capacity applies only beyond the Prefill token.
    Device/KV admission remains the engine's responsibility.
    """
    if not prompt_lengths or len(prompt_lengths) != len(generation_budgets):
        raise ValueError("Expected nonempty, equally sized prompt lengths and budgets")
    if max_num_seqs < 1 or token_budget < 1:
        raise ValueError("Beam batch capacity must be positive")
    sizes = sorted({size for size in supported_sizes if 0 < size <= max_num_seqs}, reverse=True)
    if 1 not in sizes:
        raise ValueError("Beam batching requires singleton support")
    for index, (length, budget) in enumerate(zip(prompt_lengths, generation_budgets, strict=True)):
        if length < 1 or length > token_budget:
            raise ValueError(
                f"Prompt {index} Prefill length {length} must be between 1 and "
                f"the scheduling token budget {token_budget}"
            )
        if budget < 1:
            raise ValueError(f"Prompt {index} has no generation budget")
        if budget > 1 and max_decode_seqs is not None and max_decode_seqs < 1:
            raise ValueError(f"Prompt {index} exceeds the physical Decode capacity")

    batches = []
    start = 0
    while start < len(prompt_lengths):
        budget = generation_budgets[start]
        for size in sizes:
            if budget > 1 and max_decode_seqs is not None and size > max_decode_seqs:
                continue
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
