"""One model contract across all five backends; no evaluator labels cross it."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import importlib.util
import math
import hashlib
from pathlib import Path
import sys
from typing import Any

@dataclass(frozen=True)
class OutputSpec:
    timestamps_s: tuple[float, ...]
    reference_point: str
    coordinate_frame: str = "ego_at_anchor"
    axes: str = "+x forward, +y left; metres"

@dataclass(frozen=True)
class Request:
    track: str
    instruction: str
    observation: dict[str, Any]
    output_spec: OutputSpec
    seed: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        spec = dict(value.pop("output_spec"))
        spec["timestamps_s"] = tuple(spec["timestamps_s"])
        return cls(output_spec=OutputSpec(**spec), **value)

SPECS = {
    "nuscenes": OutputSpec(tuple(i * .25 for i in range(1, 13)), "ego_pose_origin"),
    "navsim": OutputSpec(tuple(i * .5 for i in range(1, 9)), "rear_axle"),
    "wod": OutputSpec(tuple(i * .25 for i in range(1, 21)), "rear_axle"),
    "carla": OutputSpec(tuple(i * .25 for i in range(1, 21)), "rear_axle"),
    "realsim": OutputSpec(tuple(i * .1 for i in range(1, 61)), "rig_origin"),
}

def load_adapter(path, config=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model adapter missing: {path}")
    name = "l2d_adapter_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    cached = sys.modules.get(name)
    if cached is not None:
        factory = getattr(cached, "create_adapter", None)
        return factory(config or {}) if callable(factory) else cached
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    factory = getattr(module, "create_adapter", None)
    adapter = factory(config or {}) if callable(factory) else module
    if not callable(getattr(adapter, "predict", None)):
        raise ValueError("Define create_adapter(config) returning an object with predict(request), or a module-level predict(request)")
    return adapter


def validate_trajectory(result, spec):
    if not isinstance(result, dict):
        raise ValueError("predict must return a dictionary")
    for key, expected in (("coordinate_frame", spec.coordinate_frame), ("reference_point", spec.reference_point)):
        if result.get(key) != expected:
            raise ValueError(f"{key} must be {expected}; convert the model frame explicitly")
    times, xy = result.get("timestamps_s"), result.get("xy_m")
    if not isinstance(times, (list, tuple)) or len(times) != len(spec.timestamps_s):
        raise ValueError("timestamps_s must match requested grid")
    if any(type(t) not in (int, float) or not math.isfinite(t) or abs(t - target) > 1e-7
           for t, target in zip(times, spec.timestamps_s)):
        raise ValueError("timestamps_s must match requested grid")
    if not isinstance(xy, (list, tuple)) or len(xy) != len(times):
        raise ValueError("xy_m must have one XY point per requested timestamp")
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or
           any(type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in xy):
        raise ValueError("xy_m must contain finite XY pairs")
    return {"coordinate_frame": spec.coordinate_frame, "reference_point": spec.reference_point,
            "timestamps_s": list(spec.timestamps_s), "xy_m": [[float(v) for v in p] for p in xy]}


def predict(adapter, request):
    return validate_trajectory(adapter.predict(request), request.output_spec)
