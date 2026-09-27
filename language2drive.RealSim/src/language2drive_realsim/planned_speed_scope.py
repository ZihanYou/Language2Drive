"""Sampled planned geometry prefilter; never actual safety or qualification."""
import numpy as np

from .events import SemanticEvent, point_in_polygon, segments_intersect
from .verifier import _speed_verifier


def audit_planned_speed_scope(times, centers_xy, segment, band_mps, *, start_s=1.7, deadline_s=7.,
                              activation_polygon=None):
    """Use the actual speed semantic checker, including the incoming interval.

    Positions must already be AABB centers, matching actual trace normalization.
    Zero epsilon makes this prefilter conservative. Passing does not certify
    tracking, signals, visibility, trigger grounding, rendering or traffic.
    """
    times, xy, band = (np.asarray(v, dtype=float) for v in (times, centers_xy, band_mps))
    if (times.ndim != 1 or len(times) < 2 or xy.shape != (len(times), 2)
        or band.shape != (2,) or not all(np.isfinite(v).all() for v in (times, xy, band))
        or np.any(np.diff(times) <= 0) or not 0 <= band[0] < band[1]
        or not np.isfinite([start_s, deadline_s]).all() or not times[0] <= start_s < deadline_s):
        raise ValueError("finite ordered planned samples, speed band and decision window required")
    speeds = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(times)]
    rows = [{"time_s": float(t), "speed_mps": float(speed),
             "scopes": ["planned-scope"] if point_in_polygon(p, segment["scope_polygon"]) else []}
            for t, p, speed in zip(times, xy, speeds)]
    if activation_polygon is not None:
        from shapely.geometry import Polygon, shape
        polygon = shape(activation_polygon) if isinstance(activation_polygon, dict) else Polygon(activation_polygon)
        if polygon.geom_type != "Polygon" or not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
            raise ValueError("valid nonempty planned activation polygon required")
        for row, point in zip(rows, xy):
            row["triggers"] = ["planned-trigger"] if point_in_polygon(point, activation_polygon) else []
    events = []
    for i in range(1, len(times)):
        for gate in ("entry_gate", "exit_gate"):
            if segments_intersect(xy[i-1], xy[i], *segment[gate]):
                events.append(SemanticEvent(float(times[i]), "CrossGate", gate))
    contract = {"kind": "speed_band", "scope": "planned-scope", "entry_gate": "entry_gate",
                "exit_gate": "exit_gate", "speed_min_mps": float(band[0]),
                "speed_max_mps": float(band[1]), "deadline_s": deadline_s}
    if activation_polygon is not None:
        contract.update(kind="triggered_speed_band", trigger_id="planned-trigger")
    result = _speed_verifier(contract, rows, 0., events, {"start_s": start_s, "deadline_s": deadline_s}).to_dict()
    result.update(entry_s=next((e.time_s for e in events if e.subject == "entry_gate" and e.time_s >= start_s), None),
                  exit_s=next((e.time_s for e in events if e.subject == "exit_gate" and e.time_s >= start_s), None),
                  speed_convention="incoming AABB-center XY interval, matching actual normalization",
                  actual_execution_verified=False, qualified=False)
    return result
