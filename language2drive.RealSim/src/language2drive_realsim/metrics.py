"""PCU-level paired metrics; seeds and arms never inflate the PCU count."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .constants import ARMS, COUPLED_SEEDS


def aggregate_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["pcu_id"])].append(row)
    pcu_rows = []
    for pcu_id, rows in sorted(grouped.items()):
        identities = {(row["arm_id"], int(row["seed"])) for row in rows}
        expected = {(arm, seed) for arm in ARMS for seed in COUPLED_SEEDS}
        complete = identities == expected and len(rows) == len(expected)
        all_passed = complete and all(row.get("passed") is True for row in rows)
        arm_success = {
            arm: (
                sum(row.get("passed") is True for row in rows if row["arm_id"] == arm)
                / max(1, sum(row["arm_id"] == arm for row in rows))
            )
            for arm in ARMS
        }
        pcu_rows.append(
            {
                "pcu_id": pcu_id,
                "complete": complete,
                "paired_success": all_passed,
                "arm_success": arm_success,
                "contrast_min": min(arm_success.values()),
            }
        )
    denominator = len(pcu_rows)
    return {
        "schema_version": "l2d.realsim.metrics.v0.1",
        "pcu_denominator": denominator,
        "paired_pcu_success_count": sum(row["paired_success"] for row in pcu_rows),
        "paired_pcu_success_rate": (
            sum(row["paired_success"] for row in pcu_rows) / denominator if denominator else 0.0
        ),
        "macro_contrast_min": (
            sum(row["contrast_min"] for row in pcu_rows) / denominator if denominator else 0.0
        ),
        "pcus": pcu_rows,
    }
