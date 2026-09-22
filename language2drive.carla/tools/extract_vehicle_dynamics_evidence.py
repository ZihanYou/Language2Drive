#!/usr/bin/env python3
"""Freeze CARLA wheel geometry and normalized-steer response evidence.

The capture is deliberately performed on a stationary, hand-braked Lincoln
under a temporary synchronous 20 Hz world.  Both ascending and descending
command sweeps must converge to the same front-wheel angles.  World settings
are restored even when a capture fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import carla


COMMANDS = tuple(round(-0.8 + 0.05 * index, 2) for index in range(33))
WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
WHEEL_LOCATIONS = (
    carla.VehicleWheelLocation.FL_Wheel,
    carla.VehicleWheelLocation.FR_Wheel,
    carla.VehicleWheelLocation.BL_Wheel,
    carla.VehicleWheelLocation.BR_Wheel,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_source_path(path: Path) -> Path:
    """Prefer the stable lwq workspace spelling for shared backing files."""

    lexical = path.absolute()
    alternate_root = Path("/root/autodl-tmp/simlingo_fzh")
    canonical_root = Path("/root/autodl-tmp/lwq/simlingo")
    try:
        lexical.relative_to(canonical_root)
    except ValueError:
        pass
    else:
        if lexical.is_file():
            return lexical
    resolved = path.resolve()
    try:
        alternate_relative = resolved.relative_to(alternate_root)
    except ValueError:
        return resolved
    candidate = canonical_root / alternate_relative
    if candidate.is_file() and _sha256(candidate) == _sha256(resolved):
        return candidate
    return resolved


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2,
    ) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _round(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def _vector(value: Any, *, scale: float = 1.0) -> dict[str, float]:
    return {
        axis: _round(float(getattr(value, axis)) * scale)
        for axis in ("x", "y", "z")
    }


def _rotation(value: Any) -> dict[str, float]:
    return {
        axis: _round(float(getattr(value, axis)))
        for axis in ("pitch", "yaw", "roll")
    }


def _world_centimetres_to_actor_local_metres(
    actor_transform: carla.Transform, position: Any,
) -> dict[str, float]:
    world_metres = carla.Location(
        x=float(position.x) / 100.0,
        y=float(position.y) / 100.0,
        z=float(position.z) / 100.0,
    )
    local = actor_transform.inverse_transform(world_metres)
    return _vector(local)


def _wheel_angles(vehicle: carla.Vehicle) -> dict[str, float]:
    return {
        name: _round(vehicle.get_wheel_steer_angle(location))
        for name, location in zip(WHEEL_NAMES, WHEEL_LOCATIONS)
    }


def _center_curvature(
    command: float,
    angles: dict[str, float],
    wheelbase_m: float,
    front_track_m: float,
) -> tuple[float, float]:
    if abs(command) <= 1e-12:
        return 0.0, 0.0
    front = [abs(float(angles["front_left"])), abs(float(angles["front_right"]))]
    inner_deg, outer_deg = max(front), min(front)
    if inner_deg <= 1e-9:
        raise RuntimeError("non-zero steer command produced zero front-wheel angle")
    radius = wheelbase_m / math.tan(math.radians(inner_deg)) + front_track_m / 2.0
    curvature = math.copysign(1.0 / radius, command)
    expected_outer = math.degrees(
        math.atan(wheelbase_m / (radius + front_track_m / 2.0))
    )
    return _round(curvature, 12), _round(outer_deg - expected_outer, 9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="parkinglotfish-new3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xodr", type=Path, required=True)
    parser.add_argument("--carla-binary", type=Path, required=True)
    parser.add_argument("--settle-ticks", type=int, default=40)
    arguments = parser.parse_args()
    if arguments.settle_ticks < 20:
        raise ValueError("settle-ticks must be at least 20")

    client = carla.Client(arguments.host, arguments.port)
    client.set_timeout(20.0)
    world = client.get_world()
    current_name = world.get_map().name.rsplit("/", 1)[-1]
    if current_name != arguments.town:
        world = client.load_world(arguments.town)
    original = world.get_settings()
    vehicle = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = True
        world.apply_settings(settings)
        world.tick()

        blueprint = world.get_blueprint_library().find(
            "vehicle.lincoln.mkz_2020"
        )
        for raw_transform in world.get_map().get_spawn_points():
            transform = carla.Transform(
                raw_transform.location + carla.Location(z=0.5),
                raw_transform.rotation,
            )
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("no collision-free spawn for dynamics capture")
        for _ in range(40):
            vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            world.tick()

        actor_transform = vehicle.get_transform()
        physics = vehicle.get_physics_control()
        if len(physics.wheels) != 4:
            raise RuntimeError("Lincoln physics control does not expose four wheels")
        wheels = []
        for name, wheel in zip(WHEEL_NAMES, physics.wheels):
            wheels.append({
                "name": name,
                "position_world_cm_raw": _vector(wheel.position),
                "position_actor_local_m": _world_centimetres_to_actor_local_metres(
                    actor_transform, wheel.position,
                ),
                "max_steer_angle_deg": _round(wheel.max_steer_angle),
                "radius_cm": _round(wheel.radius),
            })
        by_name = {item["name"]: item for item in wheels}
        front_x = sum(
            by_name[name]["position_actor_local_m"]["x"]
            for name in ("front_left", "front_right")
        ) / 2.0
        rear_x = sum(
            by_name[name]["position_actor_local_m"]["x"]
            for name in ("rear_left", "rear_right")
        ) / 2.0
        front_track = abs(
            by_name["front_right"]["position_actor_local_m"]["y"]
            - by_name["front_left"]["position_actor_local_m"]["y"]
        )
        rear_track = abs(
            by_name["rear_right"]["position_actor_local_m"]["y"]
            - by_name["rear_left"]["position_actor_local_m"]["y"]
        )
        wheelbase = front_x - rear_x
        if not 2.0 < wheelbase < 4.0:
            raise RuntimeError(f"implausible captured wheelbase: {wheelbase}")

        sweeps: dict[str, list[dict[str, Any]]] = {}
        for sweep_name, commands in (
            ("ascending", COMMANDS), ("descending", tuple(reversed(COMMANDS))),
        ):
            samples = []
            for command in commands:
                control = carla.VehicleControl(
                    throttle=0.0, steer=float(command), brake=1.0,
                    hand_brake=True, reverse=False,
                )
                history = []
                for tick in range(arguments.settle_ticks):
                    vehicle.apply_control(control)
                    world.tick()
                    if tick >= arguments.settle_ticks - 5:
                        history.append(_wheel_angles(vehicle))
                spread = max(
                    max(row[name] for row in history)
                    - min(row[name] for row in history)
                    for name in WHEEL_NAMES
                )
                if spread > 1e-4:
                    raise RuntimeError(
                        f"steer {command:+.2f} did not settle: spread={spread}"
                    )
                angles = history[-1]
                curvature, ackermann_residual = _center_curvature(
                    command, angles, wheelbase, front_track,
                )
                samples.append({
                    "normalized_steer": command,
                    "settled_wheel_angles_deg": angles,
                    "last_five_angle_spread_max_deg": _round(spread),
                    "ackermann_center_curvature_1pm": curvature,
                    "ackermann_outer_angle_residual_deg": ackermann_residual,
                })
            sweeps[sweep_name] = samples

        ascending = {
            row["normalized_steer"]: row for row in sweeps["ascending"]
        }
        descending = {
            row["normalized_steer"]: row for row in sweeps["descending"]
        }
        hysteresis = max(
            abs(
                ascending[command]["settled_wheel_angles_deg"][name]
                - descending[command]["settled_wheel_angles_deg"][name]
            )
            for command in COMMANDS for name in WHEEL_NAMES
        )
        if hysteresis > 1e-3:
            raise RuntimeError(
                f"steer sweep hysteresis exceeds 0.001deg: {hysteresis}"
            )
        canonical_samples = []
        for command in COMMANDS:
            first, second = ascending[command], descending[command]
            angles = {
                name: _round(
                    0.5 * (
                        first["settled_wheel_angles_deg"][name]
                        + second["settled_wheel_angles_deg"][name]
                    )
                ) for name in WHEEL_NAMES
            }
            curvature, ackermann_residual = _center_curvature(
                command, angles, wheelbase, front_track,
            )
            canonical_samples.append({
                "normalized_steer": command,
                "front_left_deg": angles["front_left"],
                "front_right_deg": angles["front_right"],
                "ackermann_center_curvature_1pm": curvature,
                "ackermann_outer_angle_residual_deg": ackermann_residual,
            })

        script_path = Path(__file__).resolve()
        carla_module = Path(carla.__file__).resolve()
        carla_native_module = Path(carla.libcarla.__file__).resolve()
        binary = _canonical_source_path(arguments.carla_binary)
        xodr = _canonical_source_path(arguments.xodr)
        payload = {
            "schema_version": "language2drive.vehicle_dynamics_evidence.v1",
            "vehicle_blueprint": "vehicle.lincoln.mkz_2020",
            "coordinate_frames": {
                "actor": "carla_actor_local_metres",
                "raw_wheel_positions": "carla_WheelPhysicsControl_position_world_centimetres",
            },
            "wheel_geometry": {
                "wheels": wheels,
                "front_axle_x_m": _round(front_x),
                "rear_axle_x_m": _round(rear_x),
                "wheelbase_m": _round(wheelbase),
                "front_track_m": _round(front_track),
                "rear_track_m": _round(rear_track),
            },
            "physics_control": {
                "mass_kg": _round(physics.mass),
                # This installed API does not attach a unit contract to the
                # curve's x axis.  Preserve the raw values instead of
                # inventing a speed unit in the release evidence.
                "steering_curve": [
                    {"x_axis_raw": _round(value.x), "scale": _round(value.y)}
                    for value in physics.steering_curve
                ],
            },
            "normalized_steer_response": {
                "method": "stationary_handbraked_bidirectional_sweep_v1",
                "fixed_delta_seconds": 0.05,
                "settle_ticks_per_command": arguments.settle_ticks,
                "command_min": -0.8,
                "command_max": 0.8,
                "command_step": 0.05,
                "hysteresis_max_deg": _round(hysteresis),
                "canonical_samples": canonical_samples,
                "canonical_samples_sha256": _stable_hash(canonical_samples),
                "raw_sweeps": sweeps,
                "raw_sweeps_sha256": _stable_hash(sweeps),
            },
            "capture": {
                "town": arguments.town,
                "map_name": world.get_map().name,
                "client_version": client.get_client_version(),
                "server_version": client.get_server_version(),
                "actor_transform": {
                    "location": _vector(actor_transform.location),
                    "rotation": _rotation(actor_transform.rotation),
                },
                "source_api": [
                    "carla.Vehicle.get_physics_control",
                    "carla.Vehicle.get_wheel_steer_angle",
                ],
            },
            "source_files": {
                str(script_path): _sha256(script_path),
                str(carla_module): _sha256(carla_module),
                str(carla_native_module): _sha256(carla_native_module),
                str(binary): _sha256(binary),
                str(xodr): _sha256(xodr),
            },
            "scope_statement": (
                "This immutable asset records one installed CARLA package's "
                "Lincoln wheel geometry and stationary normalized-steer response. "
                "Piecewise interpolation is valid only inside [-0.8,0.8]; live "
                "collision-free multi-seed parking witnesses remain authoritative."
            ),
        }
        payload["wheel_geometry_sha256"] = _stable_hash(
            payload["wheel_geometry"]
        )
        _atomic_json(arguments.output.resolve(), payload)
    finally:
        if vehicle is not None:
            vehicle.destroy()
            try:
                world.tick()
            except RuntimeError:
                pass
        restored = world.get_settings()
        restored.synchronous_mode = bool(original.synchronous_mode)
        restored.fixed_delta_seconds = original.fixed_delta_seconds
        restored.no_rendering_mode = bool(original.no_rendering_mode)
        world.apply_settings(restored)


if __name__ == "__main__":
    main()
