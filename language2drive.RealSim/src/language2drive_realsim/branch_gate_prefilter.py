"""Cheap original-map commitment checks on plans, never actual witness evidence."""
import numpy as np

from .events import segments_intersect
from .speed_spatial_draft import central_lane_segment


def original_branch_gates(candidate, geometry):
    gates, basis = {}, {}
    for arm in candidate["selected_arm_pair"]["arms"]:
        matches = [v for v in geometry["arms"][arm["arm"]]["lanes"] if v["lane_id"] == arm["lane_id"]]
        if len(matches) != 1:
            raise ValueError("unique selected original branch section required")
        lane = matches[0]
        if lane["polygon"]["type"] != "Polygon" or len(lane["polygon"]["coordinates"]) != 1:
            raise ValueError("unambiguous original branch polygon required")
        section = central_lane_segment([v[:2] for v in lane["centerline_xyz"]], lane["polygon"]["coordinates"][0])
        gates["draft-gate-" + arm["arm"]] = section["exit_gate"]
        basis[arm["arm"]] = {"lane_id": arm["lane_id"], "section": section}
    if set(gates) != {"draft-gate-R0", "draft-gate-R1"}:
        raise ValueError("two distinct branch arms required")
    return gates, basis


def check_planned_gate_crossings(rows, rig_to_center, gates, deadline_s, intervention_s=1.7):
    """Use plan yaw and the exact rig/AABB transform; actual poses still required."""
    transform = np.asarray(rig_to_center, dtype=float)
    if (set(rows) != {"R0", "R1"} or set(gates) != {"draft-gate-R0", "draft-gate-R1"}
        or not np.isfinite([deadline_s, intervention_s]).all() or not 0 < intervention_s < deadline_s
        or transform.shape != (4, 4) or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
        raise ValueError("valid paired timing, gates and rigid AABB transform required")
    for gate in gates.values():
        points = np.asarray(gate, dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all() or np.linalg.norm(points[1]-points[0]) <= 0:
            raise ValueError("finite nondegenerate original gate required")
    results = {}
    for arm, values in rows.items():
        values = np.asarray(values, dtype=float)
        if (values.ndim != 2 or values.shape[1] != 5 or len(values) < 2
            or not np.isfinite(values).all() or values[0, 0] != 0
            or np.any(np.diff(values[:, 0]) <= 0) or np.any(np.diff(values[:, 0]) > .200001)
            or values[-1, 0] < deadline_s + 6):
            raise ValueError("finite complete plan plus six-second future required")
        c, s = np.cos(values[:, 4]), np.sin(values[:, 4])
        shift = transform[:3, 3]
        xy = values[:, 1:3] + np.column_stack((c*shift[0]-s*shift[1], s*shift[0]+c*shift[1]))
        crossings = {}
        for name, gate in gates.items():
            crossings[name] = next((float(values[i, 0]) for i in range(1, len(values))
                if segments_intersect(xy[i-1], xy[i], gate[0], gate[1])), None)
        own = crossings["draft-gate-" + arm]
        other = crossings["draft-gate-" + ("R1" if arm == "R0" else "R0")]
        passed = (own is not None and intervention_s < own <= deadline_s
                  and (other is None or other > deadline_s))
        results[arm] = {"own_first_crossing_s": own, "opposing_first_crossing_s": other,
                        "passed": bool(passed)}
    return {"scope": "planned_original_gate_prefilter_not_actual_contract_verification",
            "qualified": False, "human_reviewed": False, "deadline_s": deadline_s,
            "position_reference": "planned_yaw_only_ego_aabb_center",
            "arms": results, "passed": all(r["passed"] for r in results.values()),
            "limitations": ["planned yaw-only poses are not actual closed-loop poses",
                            "does not establish tracking, safety, legality or rendering validity"]}
