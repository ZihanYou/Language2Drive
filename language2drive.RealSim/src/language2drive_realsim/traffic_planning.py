"""CPU-only sampled traffic checks and stop hypotheses, not oracle qualification."""
from pathlib import Path

import numpy as np
from shapely import area, distance, intersection
from shapely.geometry import Polygon

from .compat import strict_zip
from .io import load_json


def smooth_speed_retime(times, positions, headings, speed_fraction, slowdown_start_s=1.7,
                        slowdown_duration_s=4., intervention_s=1.7):
    """Slow the reference clock to a positive rate, without stopping or truncating.

    Preserve all timestamps and the exact common prefix. This only changes the
    proposed timing; geometry, traffic, motion and actual rendering must be
    audited again. No reference extrapolation is permitted.
    """
    t, xyz, yaw = [np.asarray(a, dtype=float) for a in (times, positions, headings)]
    if (t.ndim != 1 or len(t) < 3 or xyz.shape != (len(t), 3) or yaw.shape != t.shape
        or not all(np.isfinite(a).all() for a in (t, xyz, yaw))
        or not np.isfinite([speed_fraction, slowdown_start_s, slowdown_duration_s, intervention_s]).all()
        or isinstance(speed_fraction, (bool, np.bool_)) or not 0 < speed_fraction <= 1
        or t[0] != 0 or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > .200001)
        or not np.any(t == intervention_s) or intervention_s < 0 or slowdown_start_s < intervention_s
        or slowdown_duration_s < 2 or slowdown_start_s + slowdown_duration_s > t[-1]):
        raise ValueError("positive bounded speed fraction and complete dense slowdown horizon required")
    if speed_fraction == 1:
        return xyz.copy(), yaw.copy()
    u = np.clip((t - slowdown_start_s) / slowdown_duration_s, 0., 1.)
    integral = u - 2.5*u**4 + 3*u**5 - u**6
    reference = slowdown_start_s + slowdown_duration_s * (speed_fraction*u + (1-speed_fraction)*integral)
    reference += speed_fraction * np.maximum(t - slowdown_start_s - slowdown_duration_s, 0.)
    prefix = t <= slowdown_start_s
    reference[prefix] = t[prefix]
    if np.any(np.diff(reference) <= 0) or reference[0] < t[0] or reference[-1] > t[-1]:
        raise ValueError("slowdown reference must stay within the confirmed path")
    result = np.column_stack([np.interp(reference, t, xyz[:, i]) for i in range(3)])
    headings = np.interp(reference, t, np.unwrap(yaw))
    result[prefix], headings[prefix] = xyz[prefix], yaw[prefix]
    return result, headings


def smooth_stop_retime(times, positions, headings, brake_start_s, brake_duration_s,
                       intervention_s=1.7):
    """Slow the reference clock smoothly to rest, keeping the same spatial path.

    Actual acceleration, traffic, completed exit and stop legality must be
    checked independently. Holding the endpoint is not horizon truncation.
    """
    t, xyz, yaw = [np.asarray(a, dtype=float) for a in (times, positions, headings)]
    if (t.ndim != 1 or len(t) < 3 or xyz.shape != (len(t), 3) or yaw.shape != t.shape
        or not all(np.isfinite(a).all() for a in (t, xyz, yaw))
        or not np.isfinite([brake_start_s, brake_duration_s, intervention_s]).all()
        or t[0] != 0 or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > .200001)
        or intervention_s < 0 or brake_start_s < intervention_s or brake_duration_s <= 0
        or brake_start_s + brake_duration_s > t[-1]):
        raise ValueError("invalid stop hypothesis or incomplete time horizon")
    u = np.clip((t - brake_start_s) / brake_duration_s, 0., 1.)
    # Integral of 1 - smoothstep5(u), hence zero terminal speed/acceleration.
    reference_time = brake_start_s + brake_duration_s * (u - 2.5*u**4 + 3*u**5 - u**6)
    prefix = t <= brake_start_s
    reference_time[prefix] = t[prefix]
    result = np.column_stack([np.interp(reference_time, t, xyz[:, i]) for i in range(3)])
    result_yaw = np.interp(reference_time, t, np.unwrap(yaw))
    result[prefix], result_yaw[prefix] = xyz[prefix], yaw[prefix]
    return result, result_yaw


class SampledTraffic:
    """Validated snapshot of official actor polygons on the original time grid."""
    def __init__(self, payload):
        if (payload.get("scope") != "actual_artifact_footprint_cache_not_integrity_or_support_attestation"
            or payload.get("complete") is not True or payload.get("qualified") is not False):
            raise ValueError("completed explicitly diagnostic traffic cache required")
        self.times = np.asarray(payload["time_s"], dtype=float)
        self.transform = np.asarray(payload["rig_to_aabb_center_matrix"], dtype=float)
        self.length, self.width = payload["vehicle_length_m"], payload["vehicle_width_m"]
        if (self.times.ndim != 1 or len(self.times) < 3 or self.times[0] != 0
            or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)
            or np.any(np.diff(self.times) > .200001) or self.transform.shape != (4, 4)
            or not np.isfinite(self.transform).all() or not np.allclose(self.transform[3], [0, 0, 0, 1])
            or not np.isfinite([self.length, self.width]).all() or min(self.length, self.width) <= 0):
            raise ValueError("invalid traffic grid or exact ego transform")
        self.polygons = [[] for _ in self.times]
        self.ids = [[] for _ in self.times]
        seen = set()
        for actor in payload["actors"]:
            actor_id = actor["actor_id"]
            indices = actor["sample_indices"]
            vertices = np.asarray(actor["footprints_xy"], dtype=float)
            if (not isinstance(actor_id, str) or not actor_id or actor_id in seen or not indices
                or any(type(i) is not int or not 0 <= i < len(self.times) for i in indices)
                or any(b <= a for a, b in zip(indices, indices[1:]))
                or vertices.shape != (len(indices), 4, 2) or not np.isfinite(vertices).all()):
                raise ValueError("invalid actor identity, coverage or quadrilateral samples")
            seen.add(actor_id)
            for i, points in strict_zip(indices, vertices):
                polygon = Polygon(points)
                if not polygon.is_valid or polygon.area <= 0:
                    raise ValueError("invalid actor footprint")
                self.polygons[i].append(polygon)
                self.ids[i].append(actor_id)
        if [len(p) for p in self.polygons] != payload["actor_count_per_sample"]:
            raise ValueError("actor coverage changed since official extraction")

    @classmethod
    def from_file(cls, path, candidate):
        data = load_json(path)
        binding = candidate["scene_execution_proposal_not_frozen"]
        if (any(data[k] != candidate[k] for k in ("proposal_key", "review_revision", "artifact_uuid"))
            or data["scene_id"] != candidate["scene_id_private"]
            or data["start_timestamp_us"] != binding["start_timestamp_us"]
            or Path(data["source_artifact"]).resolve() != Path(binding["artifact_path"]).resolve()):
            raise ValueError("traffic cache belongs to another scene or review")
        stat = Path(binding["artifact_path"]).stat()
        if stat.st_size != data["source_size_bytes"] or stat.st_mtime_ns != data["source_mtime_ns"]:
            raise ValueError("artifact changed since traffic extraction")
        return cls(data)

    def ego_footprints(self, xyz, yaw):
        xyz, yaw = np.asarray(xyz, dtype=float), np.asarray(yaw, dtype=float)
        if xyz.shape != (len(self.times), 3) or yaw.shape != self.times.shape or not np.isfinite(xyz).all() or not np.isfinite(yaw).all():
            raise ValueError("finite full-horizon ego path required")
        corners = np.array([[-self.length/2, -self.width/2, 0, 1],
                            [self.length/2, -self.width/2, 0, 1],
                            [self.length/2, self.width/2, 0, 1],
                            [-self.length/2, self.width/2, 0, 1]])
        polygons = []
        for position, heading in strict_zip(xyz, yaw):
            c, s = np.cos(heading), np.sin(heading)
            matrix = np.array([[c, -s, 0, position[0]], [s, c, 0, position[1]],
                               [0, 0, 1, position[2]], [0, 0, 0, 1]])
            polygons.append(Polygon((corners @ (matrix @ self.transform).T)[:, :2]))
        return polygons

    def audit(self, xyz, yaw):
        minimum, overlaps = float("inf"), []
        closest = None
        for i, ego in enumerate(self.ego_footprints(xyz, yaw)):
            if not self.polygons[i]:
                continue
            distances = distance(ego, self.polygons[i])
            nearest = int(np.argmin(distances))
            if distances[nearest] < minimum:
                minimum = float(distances[nearest])
                closest = {"time_s": float(self.times[i]), "actor_id": self.ids[i][nearest]}
            for j in np.flatnonzero(distances == 0):
                if area(intersection(ego, self.polygons[i][j])) > 0:
                    overlaps.append({"time_s": float(self.times[i]), "actor_id": self.ids[i][j]})
        return {"scope": "cached_sampled_traffic_not_continuous_safety_or_qualification", "qualified": False,
                "samples": len(self.times), "minimum_2d_clearance_m": minimum if np.isfinite(minimum) else None,
                "closest_sample": closest, "overlap_sample_count": len(overlaps), "overlap_events": overlaps,
                "passed": bool(np.isfinite(minimum) and minimum >= .5 and not overlaps)}
