"""Continuous Core driving score, separate from FDQ's native DS/RC/IP.

The Core trace's private ``progress`` is a [0, 1] projection on the arm's
reference path. The boolean Progress gate remains an independent verdict.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def driving_score_percent(rows: Iterable[Mapping[str, Any]], *, hold_target: bool = False) -> float | None:
    """Return 100 × maximum valid route progress, or N/A for a hold/no evidence.

    A hold target's ``progress`` encodes elapsed wait time, not route distance.
    It must not be represented as a driving/route-completion percentage.
    """

    if hold_target:
        return None
    maximum: float | None = None
    for row in rows:
        value = row.get("progress")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            continue
        maximum = value if maximum is None else max(maximum, value)
    return None if maximum is None else 100.0 * maximum


def valid_driving_score(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 100.0
