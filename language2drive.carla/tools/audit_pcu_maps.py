#!/usr/bin/env python3
"""List all 256 core arms and verify their CARLA maps can be loaded."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from language2drive.io import atomic_json  # noqa: E402


def _map_name(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "sources/core_pcus.json")
    parser.add_argument("--output", type=Path, default=ROOT / "generated/audits/pcu_map_loadability.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    pcus = source.get("pcus")
    if not isinstance(pcus, list) or len(pcus) != 128:
        raise ValueError("source must contain exactly 128 PCUs")
    arms = []
    by_map: dict[str, list[str]] = defaultdict(list)
    for pcu in pcus:
        pcu_id, town, pcu_arms = pcu.get("id"), pcu.get("town"), pcu.get("arms")
        if not isinstance(pcu_id, str) or not isinstance(town, str) or not isinstance(pcu_arms, list):
            raise ValueError("PCU map/arm shape differs")
        for arm in pcu_arms:
            arm_id = arm.get("id") if isinstance(arm, dict) else None
            if not isinstance(arm_id, str):
                raise ValueError(f"{pcu_id}: invalid arm")
            arms.append({"pcu_id": pcu_id, "arm_id": arm_id, "map": town})
            by_map[town].append(f"{pcu_id}/{arm_id}")
    if len(arms) != 256:
        raise ValueError("source does not contain exactly 256 arms")

    import carla  # imported only when an explicit live map audit is requested

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    available = {_map_name(item) for item in client.get_available_maps()}
    maps = []
    for town in sorted(by_map):
        row = {"map": town, "arms": sorted(by_map[town]), "arm_count": len(by_map[town])}
        if town not in available:
            row["status"] = "unavailable"
        else:
            try:
                world = client.load_world(town, reset_settings=False)
                row.update({"status": "loaded", "world_map": world.get_map().name})
            except Exception as error:  # CARLA exceptions vary between versions
                row.update({"status": "failed", "error": str(error)})
        maps.append(row)

    loaded = [row["map"] for row in maps if row["status"] == "loaded"]
    payload = {
        "schema_version": "language2drive.pcu_map_loadability.v1",
        "source": str(args.source.resolve()),
        "server": {"host": args.host, "port": args.port, "version": client.get_server_version()},
        "arms": arms,
        "maps": maps,
        "summary": {
            "pcu_count": len(pcus), "arm_count": len(arms), "map_count": len(maps),
            "loaded_maps": loaded,
            "unavailable_maps": [row["map"] for row in maps if row["status"] == "unavailable"],
            "failed_maps": [row["map"] for row in maps if row["status"] == "failed"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if not payload["summary"]["failed_maps"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
