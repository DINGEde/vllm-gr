# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Statistics shared by benchmark runners."""

from collections.abc import Sequence


def linear_percentile_sorted(values: Sequence[float], quantile: float) -> float:
    """Interpolate a quantile in [0, 1] from non-empty, ascending values."""
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
