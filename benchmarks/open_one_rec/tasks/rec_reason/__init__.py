# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/Kuaishou-OneRec/OpenOneRec

"""
Recommendation Reason Task Module
"""

from . import utils
from .config import REC_REASON_CONFIG
from .evaluator import RecoReasonEvaluator

__all__ = [
    "REC_REASON_CONFIG",
    "RecoReasonEvaluator",
    "utils",
]
