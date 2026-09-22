#!/usr/bin/env python3
"""Freeze CARLA collision occupancy for every confirmed parking-space label.

The XODR parking-space inventory describes semantic bays, but the cooked map
also contains unlabelled, collision-enabled parked vehicles.  Level-BBox
queries deliberately omit those ``NONE`` meshes.  This capture therefore uses
the release ego blueprint and CARLA's own spawn collision test at the exact
terminal transform.  Two complete passes must agree and no pre-existing
vehicle may be present in the controlled world.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import carla

from language2drive.io import atomic_json, sha256_file, stable_hash
from language2drive import runner


SCHEMA_VERSION = "language2drive.parking_slot_collision_occupancy.v1"
METHOD = "carla_try_spawn_release_ego_at_xodr_slot_terminal_v1"
VEHICLE_BLUEPRINT = "vehicle.lincoln.mkz_2020"
SPAWN_Z_M = 0.5
REPEAT_COUNT = 2


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _slot_rows(path: Path, town: str) -> list[dict[str, Any]]:
    expected_fields = [
        "entity_id", "physical_space_id", "town", "slot_number", "region",
        "polygon_json", "heading_deg", "status", "notes",
    ]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_fields:
            raise RuntimeError("slot registry columns differ from the reviewed schema")
        for raw in reader:
            if raw["town"] != town or not raw["status"].startswith("confirmed"):
                continue
            polygon = json.loads(raw["polygon_json"])
            if (
                not isinstance(polygon, list) or len(polygon) != 5
                or polygon[0] != polygon[-1]
                or not all(
                    isinstance(point, list) and len(point) == 2
                    and all(_finite_number(value) for value in point)
                    for point in polygon
                )
            ):
                raise RuntimeError(f"slot {raw['slot_number']} polygon is not canonical")
            heading = float(raw["heading_deg"])
            if not math.isfinite(heading):
                raise RuntimeError(f"slot {raw['slot_number']} heading is not finite")
            vertices = polygon[:-1]
            rows.append({
                "slot_number": raw["slot_number"],
                "physical_space_id": raw["physical_space_id"],
                "source_entity_id": raw["entity_id"],
                "terminal_pose": {
                    "x": sum(float(point[0]) for point in vertices) / len(vertices),
                    "y": sum(float(point[1]) for point in vertices) / len(vertices),
                    "z": 0.0,
                    "yaw_deg": heading,
                },
            })
    if not rows or [row["slot_number"] for row in rows] != sorted(
        {row["slot_number"] for row in rows}
    ):
        raise RuntimeError("confirmed slot inventory is empty, duplicated, or unsorted")
    return rows


@contextmanager
def _endpoint_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _probe_pass(
    client: carla.Client,
    world: carla.World,
    blueprint: carla.ActorBlueprint,
    slots: list[dict[str, Any]],
) -> list[bool]:
    result: list[bool] = []
    for slot in slots:
        pose = slot["terminal_pose"]
        actor = world.try_spawn_actor(
            blueprint,
            carla.Transform(
                carla.Location(
                    x=float(pose["x"]), y=float(pose["y"]), z=SPAWN_Z_M,
                ),
                carla.Rotation(yaw=float(pose["yaw_deg"])),
            ),
        )
        clear = actor is not None
        result.append(clear)
        if actor is not None:
            response = client.apply_batch_sync(
                [carla.command.DestroyActor(actor.id)], True,
            )
            if len(response) != 1 or response[0].has_error():
                raise RuntimeError("probe actor destruction failed")
            if any(
                value.attributes.get("role_name") == "l2d_slot_occupancy_probe"
                for value in world.get_actors().filter("vehicle.*")
            ):
                raise RuntimeError("probe actor survived its terminal collision test")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--slot-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--town", default="parkinglotfish-new3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve(strict=True)
    slot_registry = arguments.slot_registry.resolve(strict=True)
    output = arguments.output.resolve()
    slots = _slot_rows(slot_registry, arguments.town)
    lock_path = project_root / (
        f".runtime/locks/carla-{arguments.host}-{arguments.port}"
        f"-tm{arguments.traffic_manager_port}.lock"
    )

    with _endpoint_lock(lock_path):
        runtime_assets = runner._current_runtime_asset_authority()
        listener = runner._attest_carla_listener_process(
            arguments.host,
            arguments.port,
            expected_sha256=runtime_assets["shipping_binary_sha256"],
        )
        client = carla.Client(arguments.host, arguments.port)
        client.set_timeout(20.0)
        world = client.get_world()
        map_name = world.get_map().name
        if map_name.rsplit("/", 1)[-1] != arguments.town:
            raise RuntimeError(f"active CARLA map is not {arguments.town}: {map_name}")
        if client.get_server_version() != runtime_assets["server_version"]:
            raise RuntimeError("active CARLA server version differs from runtime authority")
        existing = list(world.get_actors().filter("vehicle.*"))
        if existing:
            raise RuntimeError("slot occupancy capture requires a world with zero vehicles")

        original_settings = world.get_settings()
        try:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            world.tick()
            blueprint = world.get_blueprint_library().find(VEHICLE_BLUEPRINT)
            blueprint.set_attribute("role_name", "l2d_slot_occupancy_probe")
            passes = [
                _probe_pass(client, world, blueprint, slots)
                for _ in range(REPEAT_COUNT)
            ]
            if any(value != passes[0] for value in passes[1:]):
                differences = [
                    slot["slot_number"]
                    for index, slot in enumerate(slots)
                    if any(value[index] != passes[0][index] for value in passes[1:])
                ]
                raise RuntimeError(
                    "terminal spawn collision results differ across repeats: "
                    + ",".join(differences)
                )
            if list(world.get_actors().filter("vehicle.*")):
                raise RuntimeError("slot occupancy capture leaked a vehicle actor")
        finally:
            leaked = [
                actor
                for actor in world.get_actors().filter("vehicle.*")
                if actor.attributes.get("role_name") == "l2d_slot_occupancy_probe"
            ]
            if leaked:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(actor.id) for actor in leaked], True,
                )
            world.apply_settings(original_settings)

        listener_after = runner._attest_carla_listener_process(
            arguments.host,
            arguments.port,
            expected_sha256=runtime_assets["shipping_binary_sha256"],
        )
        if runner._listener_static_receipt(listener) != runner._listener_static_receipt(
            listener_after
        ):
            raise RuntimeError("CARLA listener changed during slot occupancy capture")

    rows = [
        {**slot, "terminal_spawn_clear": clear}
        for slot, clear in zip(slots, passes[0])
    ]
    clear_slots = [row["slot_number"] for row in rows if row["terminal_spawn_clear"]]
    blocked_slots = [row["slot_number"] for row in rows if not row["terminal_spawn_clear"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "town": arguments.town,
        "map_name": map_name,
        "vehicle_blueprint": VEHICLE_BLUEPRINT,
        "spawn_z_m": SPAWN_Z_M,
        "repeat_count": REPEAT_COUNT,
        "slots": rows,
        "slots_sha256": stable_hash(rows),
        "summary": {
            "slot_count": len(rows),
            "clear_count": len(clear_slots),
            "blocked_count": len(blocked_slots),
            "clear_slots_sha256": stable_hash(clear_slots),
            "blocked_slots_sha256": stable_hash(blocked_slots),
        },
        "source_files": {
            "slot_registry": {
                "path": str(slot_registry), "sha256": sha256_file(slot_registry),
            },
            "runtime_manifest": {
                "path": str(runner.CARLA_RUNTIME_MANIFEST),
                "sha256": sha256_file(runner.CARLA_RUNTIME_MANIFEST),
            },
            "runtime_provenance": {
                "path": str(runner.CARLA_RUNTIME_PROVENANCE),
                "sha256": sha256_file(runner.CARLA_RUNTIME_PROVENANCE),
            },
            "available_maps_witness": {
                "path": str(runner.CARLA_RUNTIME_MAP_WITNESS),
                "sha256": sha256_file(runner.CARLA_RUNTIME_MAP_WITNESS),
            },
            "capture_tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "runtime_authority": {
            key: runtime_assets[key]
            for key in (
                "resolved_carla_root", "manifest_sha256", "inventory_sha256",
                "files_sha256", "file_count", "total_size_bytes",
                "shipping_binary_sha256", "provenance_sha256",
                "available_maps_sha256", "client_version", "server_version",
                "available_map_count", "server_uid", "server_gids",
            )
        },
        "listener_process_receipt_sha256": stable_hash(listener),
        "scope_statement": (
            "terminal_spawn_clear is the deterministic result of CARLA's own "
            "collision-aware try_spawn_actor call for the release Lincoln at the "
            "XODR parking-space terminal transform, repeated twice in an otherwise "
            "vehicle-free controlled world. It proves terminal collision occupancy "
            "for this frozen cooked map; complete ingress remains a live-rollout gate."
        ),
    }
    atomic_json(output, payload)


if __name__ == "__main__":
    main()
