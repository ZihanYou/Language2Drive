"""Exploratory progress along selected branch geometry, never inferred legality."""
import numpy as np
from shapely.geometry import LineString, Point

from .lane_path_search import smoothstep5, lane_pair_semantic_prefilter


def retimed_branch_path(times, recorded_xyz, recorded_yaws, corridor_xyz,
                        speed_fraction=0.8, intervention_s=1.7,
                        slowdown_s=3.0, alignment_s=1.4):
    """Follow arc length instead of projecting a turning recording onto another exit.

    Preserve the exact recorded prefix. After intervention, both arms may use
    the same recorded speed profile, retimed by a fixed smooth slowdown. An
    initial XY offset decays smoothly onto the selected lane centerline; rig
    height above the map centerline is held at its intervention value. Every
    sample remains a hypothesis requiring corridor, dynamics, traffic and NRE
    checks. Neither endpoint extrapolation nor guessed successor links occur.
    """
    times = np.asarray(times, dtype=float)
    xyz = np.asarray(recorded_xyz, dtype=float)
    yaws = np.asarray(recorded_yaws, dtype=float)
    corridor = np.asarray(corridor_xyz, dtype=float)
    if (times.ndim != 1 or len(times) < 3 or xyz.shape != (len(times), 3)
        or yaws.shape != times.shape or corridor.ndim != 2 or corridor.shape[1] != 3
        or len(corridor) < 2 or not all(np.isfinite(v).all() for v in (times, xyz, yaws, corridor))
        or not np.isfinite([speed_fraction, intervention_s, slowdown_s, alignment_s]).all()
        or times[0] != 0 or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .200001)
        or not 0 < speed_fraction <= 1 or slowdown_s <= 0 or alignment_s <= 0
        or not 0 < intervention_s < times[-1] or not np.any(times == intervention_s)):
        raise ValueError("invalid branch path inputs or missing exact intervention sample")
    segment_lengths = np.linalg.norm(np.diff(corridor[:, :2], axis=0), axis=1)
    if np.any(segment_lengths <= 1e-8):
        raise ValueError("branch centerline contains duplicate or zero-length segments")
    arc = np.r_[0., np.cumsum(segment_lengths)]
    line = LineString(corridor[:, :2])
    if not line.is_simple:
        raise ValueError("branch centerline is self-intersecting")
    index = int(np.flatnonzero(times == intervention_s)[0])
    origin_s = line.project(Point(xyz[index, :2]))
    if not 1e-6 < origin_s < arc[-1] - 1e-6:
        raise ValueError("intervention projects onto branch endpoint")
    scale = 1 - (1 - speed_fraction) * smoothstep5((times - intervention_s) / slowdown_s)
    recorded_steps = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    if np.any(recorded_steps[index:] <= 1e-8):
        raise ValueError("stopped recording needs a separate stop-aware planner")
    progress = np.r_[0., np.cumsum(recorded_steps * (scale[:-1] + scale[1:]) / 2)]
    distances = origin_s + progress - progress[index]
    active = times > intervention_s
    if np.any(distances[active] >= arc[-1] - 1e-6):
        raise ValueError("selected branch would require endpoint extrapolation")
    center_at_intervention = np.array([np.interp(origin_s, arc, corridor[:, i]) for i in range(3)])
    offset = xyz[index] - center_at_intervention
    positions = np.column_stack([np.interp(distances, arc, corridor[:, i]) for i in range(3)])
    align = smoothstep5((times - intervention_s) / alignment_s)
    positions[:, :2] += (1 - align[:, None]) * offset[:2]
    positions[:, 2] += offset[2]
    positions[~active] = xyz[~active]
    velocity = np.gradient(positions[:, :2], times, axis=0)
    headings = np.arctan2(velocity[:, 1], velocity[:, 0])
    headings[~active] = yaws[~active]
    return positions, headings


def branch_pair_semantic_prefilter(times, r0_xyz, r1_xyz, r0_region, r1_region,
                                   rollout_end_s=7.0, dwell_s=1.0):
    """Both sampled paths must remain in their corridor and end exclusively."""
    forward = lane_pair_semantic_prefilter(times, r0_xyz, r1_xyz, r0_region, r1_region,
                                          rollout_end_s=rollout_end_s, dwell_s=dwell_s)
    reverse = lane_pair_semantic_prefilter(times, r1_xyz, r0_xyz, r1_region, r0_region,
                                          rollout_end_s=rollout_end_s, dwell_s=dwell_s)
    return {"scope": "sampled_branch_corridor_not_frozen_exit_contract",
            "passed": forward["passed"] and reverse["passed"],
            "both_in_own_corridor_throughout": forward["keep_in_source_throughout"] and reverse["keep_in_source_throughout"],
            "exclusive_final_dwell_both_arms": forward["exclusive_final_dwell_both_arms"],
            "active_interval_s": [1.7, rollout_end_s],
            "dwell_interval_s": [rollout_end_s - dwell_s, rollout_end_s],
            "boundary_margin_m": .15,
            "r0_outside_times_s": forward["keep_outside_source_times_s"],
            "r1_outside_times_s": reverse["keep_outside_source_times_s"],
            "r0_nonexclusive_final_times_s": forward["keep_nonexclusive_final_times_s"],
            "r1_nonexclusive_final_times_s": forward["change_nonexclusive_final_times_s"]}
