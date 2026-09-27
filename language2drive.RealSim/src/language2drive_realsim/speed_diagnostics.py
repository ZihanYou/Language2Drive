"""Speed semantics from stat-matched actual ASL pose diagnostics, not witnesses."""
from pathlib import Path

import numpy as np

from .constants import COUPLED_SEEDS
from .io import load_json
from .speed_path_search import speed_band_prefilter
from .speed_timing import validate_speed_runtime_timing


def inspect_speed_semantics(run_dir):
    run_dir = Path(run_dir).resolve()
    sidecar = load_json(run_dir / "speed-pair-sidecar.private.json")
    inspection = load_json(run_dir / "speed-pair-inspection.private.json")
    if (sidecar.get("hypothesis_kind") != "speed" or inspection.get("hypothesis_kind") != "speed"
        or sidecar.get("execution_scope") != "real_counterfactual_pair_integration_not_qualified_oracle"
        or inspection.get("scope") != "actual_paired_ASL_diagnostics_not_qualified_oracle"
        or inspection.get("qualified") is not False or inspection.get("errors")):
        raise ValueError("complete exploratory speed ASL inspection required")
    sessions = sidecar["sessions"]
    timing = validate_speed_runtime_timing(run_dir, sidecar)
    if (sorted(p["arm_id"] for p in sessions.values()) != ["R0", "R1"]
        or len({p["seed"] for p in sessions.values()}) != 1
        or any(p["seed"] not in COUPLED_SEEDS for p in sessions.values())):
        raise ValueError("one complete coupled speed pair required")
    bands = [p["diagnostic_speed_band_mps"] for p in sessions.values()]
    if (any(len(b) != 2 or not np.isfinite(b).all() or not 0 <= b[0] < b[1] for b in bands)
        or not (bands[0][1] < bands[1][0] or bands[1][1] < bands[0][0])):
        raise ValueError("speed bands must be finite and disjoint")
    results = {}
    for session, private in sessions.items():
        arm = private["arm_id"]
        cached = inspection["arms"][arm]
        asl = Path(cached["source_asl"]).resolve()
        if (cached["session_uuid"] != session or asl.parent.name != session
            or not asl.is_relative_to(run_dir / "rollouts")):
            raise ValueError("actual pose cache belongs to a different run/session")
        stat = asl.stat()
        if stat.st_size != cached["source_size_bytes"] or stat.st_mtime_ns != cached["source_mtime_ns"]:
            raise ValueError("ASL changed since actual pose inspection")
        poses = cached["rig_samples_for_diagnostics"]
        times = np.asarray([p["time_s"] for p in poses])
        xyz = np.asarray([p["position_xyz"] for p in poses])
        if len(times) < 2 or times[0] != 0 or times[-1] < timing["rollout_duration_s"]:
            raise ValueError("actual trace must span the entire declared speed run")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .200001):
            raise ValueError("actual speed trace must remain dense through its full horizon")
        lower, upper = private["diagnostic_speed_band_mps"]
        result = speed_band_prefilter(times, xyz, lower, upper)
        result["session_uuid"] = session
        result["opposing_band_passed"] = speed_band_prefilter(
            times, xyz, *next(p["diagnostic_speed_band_mps"] for p in sessions.values() if p["arm_id"] != arm))["passed"]
        results[arm] = result
    return {**({"speed_timing": timing} if "speed_timing" in sidecar else {}),
            "scope": "actual_speed_semantic_diagnostic_not_qualification", "qualified": False,
            "hypothesis_kind": "speed", "proposal_key": inspection["proposal_key"],
            "pose_source": "stat_matched_ASL_diagnostic_cache_not_integrity_attestation",
            "sampled_runtime_safety_passed": inspection["sampled_runtime_safety_passed"],
            "preintervention_non_language_equal": inspection["preintervention_non_language_equal"],
            "actual_speed_prefilter_passed": all(r["passed"] and not r["opposing_band_passed"] for r in results.values()),
            "shared_spatial_scope_frozen": False, "human_speed_pair_reviewed": False, "arms": results}
