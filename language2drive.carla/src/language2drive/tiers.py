"""Deterministic PCU subsets for practical Language2Drive benchmark sizes."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any


TIER_PCU_COUNTS = {"small": 32, "base": 64, "large": 119}
_MAP_POLICY_PATH = Path(__file__).resolve().parents[2] / "config/benchmark_map_policy.json"


def loadable_pcu_maps() -> frozenset[str]:
    value = json.loads(_MAP_POLICY_PATH.read_text(encoding="utf-8"))
    maps = value.get("loadable_pcu_maps") if isinstance(value, dict) else None
    if not isinstance(maps, list) or not maps or not all(isinstance(item, str) for item in maps):
        raise ValueError("benchmark map policy is invalid")
    return frozenset(maps)


def tier_pcu_ids(pcus: list[dict[str, Any]], tier: str) -> tuple[str, ...]:
    """Return a nested, layer-balanced PCU subset.

    Unloadable-map PCUs are excluded without replacement.  Small and base are
    nested, layer-balanced samples from the remaining PCUs; large retains all
    119 currently loadable PCUs.  Source order makes the tiers reproducible
    without introducing a random sampling seed.
    """

    if tier not in TIER_PCU_COUNTS:
        raise ValueError(f"unknown benchmark tier {tier!r}")
    allowed_maps = loadable_pcu_maps()
    by_layer: dict[str, list[str]] = defaultdict(list)
    for pcu in pcus:
        layer, pcu_id, town = pcu.get("layer"), pcu.get("id"), pcu.get("town")
        if not isinstance(layer, str) or not isinstance(pcu_id, str) or not isinstance(town, str):
            raise ValueError("PCU tiering requires string layer, id, and town fields")
        if town not in allowed_maps:
            continue
        by_layer[layer].append(pcu_id)
    if set(by_layer) != {"A", "B", "C", "D"} or any(len(by_layer[layer]) < 16 for layer in by_layer):
        raise ValueError("benchmark tiering requires at least 16 loadable PCUs in each A-D layer")

    if tier == "large":
        selected = tuple(pcu_id for layer in "ABCD" for pcu_id in by_layer[layer])
    else:
        per_layer = TIER_PCU_COUNTS[tier] // 4
        base_per_layer = TIER_PCU_COUNTS["base"] // 4
        base_indices = {
            layer: [index * len(by_layer[layer]) // base_per_layer for index in range(base_per_layer)]
            for layer in "ABCD"
        }
        selected = tuple(
            by_layer[layer][index]
            for layer in "ABCD"
            for index in (
                base_indices[layer]
                if tier == "base"
                else base_indices[layer][::2]
            )
        )
    if len(selected) != TIER_PCU_COUNTS[tier]:
        raise ValueError(f"{tier} tier cardinality differs")
    return selected
