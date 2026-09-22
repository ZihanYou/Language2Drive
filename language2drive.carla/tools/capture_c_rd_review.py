#!/usr/bin/env python3
"""Capture the eight C-RD semantic review witnesses from a running CARLA.

The core generator can establish lane connectivity and the road-name-to-exit
binding from the existing annotations, but a top-down plot cannot establish
that a passenger-camera user can actually read the relevant sign.  This tool
turns the remaining review into eight contact sheets.  It deliberately reads
the evaluator-only sidecar, runs outside the agent process, and writes only to
``generated/review/rgb`` under the v1 project.

Start the existing CARLA server with rendering enabled, then run this script in
the ``simlingo-py311`` environment.  The script never spawns an ego vehicle or
changes benchmark data; it places a temporary RGB sensor at the registered
approach lane and restores the original world settings when it exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReviewCase:
    pcu_id: str
    town: str
    point: dict[str, float]
    names: tuple[str, str]
    gate_ids: tuple[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate_id(contract: Any) -> str:
    if isinstance(contract, dict):
        gate = contract.get("enter_gate")
        if isinstance(gate, dict) and isinstance(gate.get("id"), str):
            return gate["id"]
        for value in contract.values():
            try:
                return _gate_id(value)
            except ValueError:
                pass
    elif isinstance(contract, list):
        for value in contract:
            try:
                return _gate_id(value)
            except ValueError:
                pass
    raise ValueError("contract contains no concrete enter_gate id")


def _instruction_name(arm: dict[str, Any]) -> str:
    metadata_name = arm.get("metadata", {}).get("road_name")
    if isinstance(metadata_name, str) and metadata_name.strip():
        return metadata_name.strip()
    instruction = str(arm.get("instructions", [""])[0]).strip()
    for prefix in ("Take ", "Drive onto ", "At the sign, continue toward "):
        if instruction.startswith(prefix):
            instruction = instruction[len(prefix):]
            break
    return instruction.rstrip(".")


def load_review_cases(sidecar_path: Path) -> list[ReviewCase]:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    cases: list[ReviewCase] = []
    for pcu in payload.get("pcus", []):
        if pcu.get("layer") != "C" or pcu.get("category") != "road_name":
            continue
        arms = pcu.get("arms", [])
        envelope = pcu.get("coarse_envelope", [])
        if len(arms) != 2 or len(envelope) < 2:
            raise ValueError(f"{pcu.get('id')}: malformed C-RD review case")
        point = {
            axis: float(envelope[0].get(axis, 0.0))
            for axis in ("x", "y", "z")
        }
        cases.append(
            ReviewCase(
                pcu_id=str(pcu["id"]),
                town=str(pcu["town"]),
                point=point,
                names=tuple(_instruction_name(arm) for arm in arms),
                gate_ids=tuple(_gate_id(arm["contract"]) for arm in arms),
            )
        )
    cases.sort(key=lambda value: value.pcu_id)
    if len(cases) != 8 or len({case.pcu_id for case in cases}) != 8:
        raise ValueError(f"expected exactly eight unique C-RD cases, found {len(cases)}")
    towns = {case.town for case in cases}
    if towns != {"American-city0428"}:
        raise ValueError(f"C-RD capture currently expects American-city0428, got {sorted(towns)}")
    return cases


def _camera_transforms(carla: Any, world_map: Any, case: ReviewCase) -> list[tuple[str, Any]]:
    location = carla.Location(**case.point)
    waypoint = world_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None or waypoint.transform.location.distance(location) > 8.0:
        raise ValueError(f"{case.pcu_id}: review point does not project to a driving lane within 8 m")

    approaches: list[tuple[str, Any]] = [("decision", waypoint)]
    previous = waypoint.previous(12.0)
    if previous:
        # Prefer the predecessor whose heading is closest to the registered
        # decision-lane heading.  This is deterministic at merges.
        previous.sort(
            key=lambda item: abs(
                ((item.transform.rotation.yaw - waypoint.transform.rotation.yaw + 180.0) % 360.0) - 180.0
            )
        )
        approaches.insert(0, ("approach", previous[0]))

    output: list[tuple[str, Any]] = []
    for position_name, lane_waypoint in approaches:
        lane_transform = lane_waypoint.transform
        base = lane_transform.location
        # Exact AutoVLA evaluation camera rig from team_code/config_autovla.py.
        # CARLA sensor offsets are expressed in the ego coordinate frame.
        rig = (
            ("front_left", 0.27, -0.55, -55.0),
            ("front", 0.80, 0.0, 0.0),
            ("front_right", 0.27, 0.55, 55.0),
        )
        yaw_radians = math.radians(lane_transform.rotation.yaw)
        cos_yaw, sin_yaw = math.cos(yaw_radians), math.sin(yaw_radians)
        for view_name, forward, lateral, yaw_offset in rig:
            offset_x = forward * cos_yaw - lateral * sin_yaw
            offset_y = forward * sin_yaw + lateral * cos_yaw
            transform = carla.Transform(
                carla.Location(x=base.x + offset_x, y=base.y + offset_y, z=base.z + 1.60),
                carla.Rotation(
                    pitch=0.0,
                    yaw=lane_transform.rotation.yaw + yaw_offset,
                    roll=0.0,
                ),
            )
            output.append((f"{position_name}_{view_name}", transform))
    return output


def _receive_image(sensor: Any, world: Any, transform: Any, timeout: float) -> Any:
    inbox: queue.Queue[Any] = queue.Queue()
    sensor.set_transform(transform)
    sensor.listen(inbox.put)
    minimum_frame = world.tick()
    while True:
        image = inbox.get(timeout=timeout)
        if int(image.frame) >= int(minimum_frame):
            sensor.stop()
            return image


def _contact_sheet(case: ReviewCase, images: list[tuple[str, Path]], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    width, height = 640, 320
    header = 92
    canvas = Image.new("RGB", (width * 3, header + height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{case.pcu_id}: passenger-camera semantic witness", fill="black")
    draw.text((12, 30), f"A: {case.names[0]} -> {case.gate_ids[0]}", fill="black")
    draw.text((12, 52), f"B: {case.names[1]} -> {case.gate_ids[1]}", fill="black")
    draw.text((12, 72), "Review only sign readability and arrow/name-to-exit consistency.", fill="black")
    for index, (label, path) in enumerate(images):
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        x, y = (index % 3) * width, header + (index // 3) * height
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + 190, y + 24), fill=(0, 0, 0))
        draw.text((x + 6, y + 5), label, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def capture(
    project_root: Path,
    output_dir: Path,
    host: str,
    port: int,
    timeout: float,
) -> dict[str, Any]:
    sidecar = project_root / "build/suite/sidecar.private.json"
    cases = load_review_cases(sidecar)

    # libcarla otherwise creates ``carlaCache`` under the invoking account's
    # home directory.  Keep this potentially growing OpenDRIVE cache on the
    # data volume with the rest of the benchmark artifacts.
    carla_cache = project_root / ".runtime/carla-cache"
    carla_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CARLA_CACHE_DIR", str(carla_cache))

    try:
        import carla
    except ImportError as error:
        raise RuntimeError(
            "CARLA Python API is unavailable; activate simlingo-py311 and set the environment from environment.lock.yml"
        ) from error

    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    current_name = world.get_map().name.rsplit("/", 1)[-1]
    if current_name != cases[0].town:
        world = client.load_world(cases[0].town)
    world_map = world.get_map()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", "1024")
    blueprint.set_attribute("image_size_y", "512")
    blueprint.set_attribute("fov", "110")
    blueprint.set_attribute("sensor_tick", "0.0")
    sensor = None
    records: list[dict[str, Any]] = []
    try:
        first_transform = _camera_transforms(carla, world_map, cases[0])[0][1]
        sensor = world.spawn_actor(blueprint, first_transform)
        for case in cases:
            case_dir = output_dir / case.pcu_id
            case_dir.mkdir(parents=True, exist_ok=True)
            captured: list[tuple[str, Path]] = []
            camera_rows: list[dict[str, Any]] = []
            for label, transform in _camera_transforms(carla, world_map, case):
                image = _receive_image(sensor, world, transform, timeout)
                path = case_dir / f"{label}.png"
                image.save_to_disk(str(path))
                captured.append((label, path))
                camera_rows.append(
                    {
                        "label": label,
                        "relative_path": str(path.relative_to(project_root)),
                        "sha256": _sha256(path),
                        "frame": int(image.frame),
                        "transform": {
                            "x": transform.location.x,
                            "y": transform.location.y,
                            "z": transform.location.z,
                            "pitch": transform.rotation.pitch,
                            "yaw": transform.rotation.yaw,
                            "roll": transform.rotation.roll,
                        },
                    }
                )
            contact = output_dir / f"{case.pcu_id}_contact.png"
            _contact_sheet(case, captured, contact)
            records.append(
                {
                    "pcu_id": case.pcu_id,
                    "town": case.town,
                    "names": list(case.names),
                    "gate_ids": list(case.gate_ids),
                    "contact_sheet": str(contact.relative_to(project_root)),
                    "contact_sheet_sha256": _sha256(contact),
                    "camera_images": camera_rows,
                }
            )
    finally:
        if sensor is not None:
            try:
                sensor.stop()
            finally:
                sensor.destroy()
        world.apply_settings(original)

    payload = {
        "schema_version": "1.0",
        "purpose": "C-RD passenger-camera sign readability and arrow-to-exit confirmation",
        "sidecar": str(sidecar.relative_to(project_root)),
        "sidecar_sha256": _sha256(sidecar),
        "town": cases[0].town,
        "case_count": len(records),
        "required_human_decision": "one confirm/reject checkbox per contact sheet",
        "cases": records,
    }
    metadata_path = output_dir / "capture_manifest.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index = ["# C-RD RGB review", "", "Choose only **confirm** or **reject** for each row.", ""]
    for row in records:
        index.append(
            f"- {row['pcu_id']}: [{row['names'][0]} / {row['names'][1]}]"
            f"(./{Path(row['contact_sheet']).name}) — ☐ confirm ☐ reject"
        )
    (output_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    output = (args.output or project_root / "generated/review/rgb").resolve()
    if "/root/autodl-tmp/" not in str(project_root) + "/":
        parser.error("project-root must be under /root/autodl-tmp to protect the system disk")
    if not output.is_relative_to(project_root):
        parser.error("output must remain under the v1 project root")
    try:
        payload = capture(project_root, output, args.host, args.port, args.timeout)
    except (OSError, RuntimeError, ValueError, queue.Empty) as error:
        print(f"capture_c_rd_review: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"case_count": payload["case_count"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
