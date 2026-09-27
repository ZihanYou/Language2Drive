"""Actual branch occupancy from stat-bound ASL caches, never qualification."""
from pathlib import Path

import numpy as np

from .branch_pair_smoke import validate_branch_search
from .branch_path_search import branch_pair_semantic_prefilter
from .constants import COUPLED_SEEDS
from .corridor_sections import linked_section_region
from .io import load_json


def _stat_matches(path, size, mtime):
    stat = path.stat()
    if stat.st_size != size or stat.st_mtime_ns != mtime:
        raise ValueError("source changed since diagnostic cache was produced")


def branch_geometry_cache_path(search, source_path):
    """Honor the bound duration-specific cache, with legacy fallback only if absent."""
    source = Path(source_path).resolve()
    path = Path(search.get("source_cache", source.parent / "branch-join-diagnostic.private.json")).resolve()
    if not path.is_relative_to(source.parent):
        raise ValueError("map cache must belong to the isolated search directory")
    return path


def inspect_branch_semantics(run_dir):
    run_dir = Path(run_dir).resolve()
    candidate = load_json(run_dir / "launch.private.json")["candidate"]
    sidecar = load_json(run_dir / "branch-pair-sidecar.private.json")
    inspection = load_json(run_dir / "branch-pair-inspection.private.json")
    if (sidecar.get("hypothesis_kind") != "branch" or inspection.get("hypothesis_kind") != "branch"
        or sidecar.get("execution_scope") != "real_counterfactual_pair_integration_not_qualified_oracle"
        or inspection.get("scope") != "actual_paired_ASL_diagnostics_not_qualified_oracle"
        or inspection.get("qualified") is not False or inspection.get("errors") != []
        or inspection.get("proposal_key") != candidate["proposal_key"]):
        raise ValueError("complete exploratory branch ASL inspection required")
    sessions = sidecar["sessions"]
    if (sorted(p["arm_id"] for p in sessions.values()) != ["R0", "R1"]
        or set(inspection["arms"]) != {"R0", "R1"}
        or len({p["seed"] for p in sessions.values()}) != 1
        or any(type(p["seed"]) is not int or p["seed"] not in COUPLED_SEEDS for p in sessions.values())):
        raise ValueError("one complete coupled branch pair required")
    plan = load_json(run_dir / "branch-pair-plan.private.json")
    reference = plan["diagnostic_search_reference"]
    source = Path(reference["path"]).resolve()
    _stat_matches(source, reference["size_bytes"], reference["mtime_ns"])
    search = load_json(source)
    validate_branch_search(search, candidate)
    duration = search["rollout_duration_s"]
    geometry_path = branch_geometry_cache_path(search, source)
    geometry = load_json(geometry_path)
    if (geometry.get("scope") != "actual_map_branch_join_diagnostic_not_qualification"
        or geometry.get("qualified") is not False
        or any(geometry[a] != candidate[b] for a, b in
               (("proposal_key", "proposal_key"), ("review_revision", "review_revision"),
                ("artifact_uuid", "artifact_uuid"), ("scene_id", "scene_id_private")))):
        raise ValueError("map cache and reviewed branch binding mismatch")
    binding = candidate["scene_execution_proposal_not_frozen"]
    artifact = Path(geometry["source_artifact"]).resolve()
    if artifact != Path(binding["artifact_path"]).resolve():
        raise ValueError("map cache belongs to another artifact path")
    _stat_matches(artifact, geometry["source_size_bytes"], geometry["source_mtime_ns"])
    traces, regions, joins, actual_sources = {}, {}, {}, {}
    for session, private in sessions.items():
        arm = private["arm_id"]
        cached = inspection["arms"][arm]
        if (type(private.get("rollout_duration_s")) not in (float, int)
            or private["rollout_duration_s"] != duration or cached.get("required_duration_s") != duration
            or private["start_timestamp_us"] != binding["start_timestamp_us"]
            or private["instruction_issued_at_us"] != binding["start_timestamp_us"] + 1700000
            or private["oracle_plan"] != search["selected_plans"][arm]):
            raise ValueError("actual run timing/plan differs from searched branch")
        asl = Path(cached["source_asl"]).resolve()
        if (cached["session_uuid"] != session or asl.parent.name != session
            or not asl.is_relative_to(run_dir / "rollouts")):
            raise ValueError("actual pose cache belongs to a different run/session")
        _stat_matches(asl, cached["source_size_bytes"], cached["source_mtime_ns"])
        poses = cached["rig_samples_for_diagnostics"]
        times = np.asarray([p["time_s"] for p in poses], dtype=float)
        xyz = np.asarray([p["position_xyz"] for p in poses], dtype=float)
        if (len(times) < 2 or xyz.shape != (len(times), 3)
            or not np.isfinite(times).all() or not np.isfinite(xyz).all()
            or times[0] != 0 or times[-1] < duration
            or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .200001)):
            raise ValueError("each actual trace must densely span the entire branch run")
        traces[arm] = times, xyz
        sections = geometry["arms"][arm]["lanes"]
        selected = next(a["lane_id"] for a in candidate["selected_arm_pair"]["arms"] if a["arm"] == arm)
        if selected not in [s["lane_id"] for s in sections]:
            raise ValueError("reviewed branch absent from map corridor")
        links = [(j["from_lane"], j["to_lane"]) for j in geometry["arms"][arm]["joins"]]
        regions[arm], joins[arm] = linked_section_region(sections, links)
        actual_sources[arm] = {k: cached[k] for k in
                              ("session_uuid", "source_asl", "source_size_bytes", "source_mtime_ns")}
    from .branch_shared_surface import recheck_shared_precommit_surface
    regions = recheck_shared_precommit_surface(search, candidate, geometry, regions)
    # Test every actual timestamp from either arm, plus exact contract bounds.
    # Validate each arm before merging so the other arm cannot conceal a gap.
    grid = np.unique(np.concatenate(([0., 1.7, duration - 1, duration],
                                    *[t[t <= duration] for t, _ in traces.values()])))
    positions = {arm: np.column_stack([np.interp(grid, t, xyz[:, i]) for i in range(3)])
                 for arm, (t, xyz) in traces.items()}
    semantic = branch_pair_semantic_prefilter(grid, positions["R0"], positions["R1"],
                                              regions["R0"], regions["R1"], rollout_end_s=duration)
    return {"scope": "actual_branch_semantic_diagnostic_not_qualification", "qualified": False,
            "hypothesis_kind": "branch", "proposal_key": candidate["proposal_key"],
            "review_revision": candidate["review_revision"], "seed": next(iter(sessions.values()))["seed"],
            "pose_source": "stat_matched_ASL_diagnostic_cache_not_integrity_attestation",
            "rollout_duration_s": duration, "actual_branch_prefilter_passed": semantic["passed"],
            "preintervention_non_language_equal": inspection["preintervention_non_language_equal"],
            "sampled_runtime_safety_passed": inspection["sampled_runtime_safety_passed"],
            "semantic_prefilter": semantic, "actual_sources": actual_sources,
            "geometry_policy": "bounded_original_cap_hull_v2", "derived_geometry_audit": joins,
            "shared_precommit_original_surface": search.get("shared_precommit_original_surface"),
            "position_reference": "rig_origin_in_nre_local_not_full_vehicle_footprint",
            "actual_future_plan_or_completed_stop_verified": False, "stop_legality_verified": False,
            "human_branch_pair_reviewed": False, "continuous_support_verified": False}
