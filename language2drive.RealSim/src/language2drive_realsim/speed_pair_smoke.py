"""Prepare explicitly unreviewed speed-pair integration, never formal PCUs."""
from pathlib import Path
import uuid

import numpy as np
import yaml

from .compat import strict_zip
from .constants import COUPLED_SEEDS, ORACLE_CONTROLLER_ID
from .io import atomic_json, load_json
from .speed_path_search import speed_band_prefilter, validate_speed_targets_kph, validate_terminal_stop_window
from .speed_timing import speed_timing


def prepare_speed_pair(run_dir, search_path, seed=1701):
    run_dir, search_path = Path(run_dir).resolve(), Path(search_path).resolve()
    if type(seed) is not int or seed not in COUPLED_SEEDS:
        raise ValueError("unsupported coupled seed")
    if any((run_dir / name).exists() for name in (
        "speed-pair-plan.private.json", "speed-pair-sidecar.private.json", "lane-pair-plan.private.json")):
        raise ValueError("preserve previous paired run evidence")
    launch, search = load_json(run_dir / "launch.private.json"), load_json(search_path)
    if (launch["scope"] != "exploratory_runtime_not_oracle_witness"
        or launch["driver_binding"] != "registered_static_for_asl_authority"
        or search["scope"] != "unreviewed_speed_pair_search_not_qualification"
        or search["qualified"] is not False
        or search["task_family_proposal"] != "B1_segment_speed_band"):
        raise ValueError("explicit unreviewed speed hypothesis and static oracle runtime required")
    candidate = launch["candidate"]
    if launch.get("curation_authority"):
        from .agent_curation import validate_agent_runs, load_agent_candidate
        root = Path(__file__).resolve().parents[2]
        if validate_agent_runs(root, [run_dir]) != candidate:
            raise ValueError("delegated speed authoring authority changed")
        if (not search.get("offline_source")
            or load_agent_candidate(root, search["offline_source"]) != candidate):
            raise ValueError("speed search differs from the delegated window/pair")
    for search_key, candidate_key in (("location_source_proposal_key", "proposal_key"),
                                      ("artifact_uuid", "artifact_uuid"), ("scene_id", "scene_id_private")):
        if search[search_key] != candidate[candidate_key]:
            raise ValueError("speed hypothesis and prepared scene mismatch")
    binding = candidate["scene_execution_proposal_not_frozen"]
    timing = speed_timing(search)
    duration, horizon = timing["rollout_duration_s"], timing["plan_horizon_s"]
    terminal_stop = validate_terminal_stop_window(search.get("terminal_stop_window_s"), duration, horizon)
    reference = search.get("recorded_spatial_reference")
    if reference is not None:
        artifact_path = Path(reference["source_artifact"]).resolve()
        stat = artifact_path.stat()
        if (artifact_path != Path(binding["artifact_path"]).resolve()
            or stat.st_size != reference["source_size_bytes"] or str(stat.st_mtime_ns) != reference["source_mtime_ns"]
            or reference["endpoint_extrapolation"] is not False
            or not 13 <= reference["horizon_s"] <= 30
            or binding["start_timestamp_us"] + round(reference["horizon_s"] * 1e6) >= binding["recording_end_timestamp_us"]):
            raise ValueError("recorded spatial reference changed or exceeds the real artifact")
    if search["start_timestamp_us"] != binding["start_timestamp_us"]:
        raise ValueError("speed hypothesis start time mismatch")
    if any(search[k] != value for k, value in timing.items()):
        raise ValueError("unsupported exploratory timing or speed-band policy")
    if "speed_timing" in search and (reference is None or reference["horizon_s"] < horizon
        or binding["start_timestamp_us"] + round(horizon * 1e6) >= binding["recording_end_timestamp_us"]):
        raise ValueError("explicit speed timing requires the complete bound recorded future")
    targets, half_width = validate_speed_targets_kph(search["selected_targets_kph"], search["fixed_band_half_width_kph"])
    attempts = [a for a in search["attempts"] if a["targets_kph"] == targets.tolist() and a["passed"] is True]
    if len(attempts) != 1 or set(attempts[0]["arms"]) != {"R0", "R1"}:
        raise ValueError("no unique passed speed-pair attempt")
    for arm in attempts[0]["arms"].values():
        audit = arm["sampled_audit"]
        if (arm["semantic_prefilter"]["passed"] is not True or audit["passed"] is not True
            or audit["overlap_sample_count"] != 0
            or not np.isfinite([audit["minimum_2d_clearance_m"], audit["max_planar_acceleration_mps2"],
                                   audit["max_lateral_acceleration_mps2"]]).all()
            or audit["minimum_2d_clearance_m"] < .5 or audit["max_planar_acceleration_mps2"] > 3.5
            or audit["max_lateral_acceleration_mps2"] > 3):
            raise ValueError("speed pair failed sampled traffic or dynamics")
    plans = search["selected_plans"]
    if not isinstance(plans, dict) or set(plans) != {"R0", "R1"}:
        raise ValueError("both selected plans are required")
    prefixes = []
    from .speed_scope_binding import selected_speed_scope, read_bound_speed_scope
    scope_key, scope_reference = selected_speed_scope(search)
    approach, transform = None, None
    if scope_reference is not None:
        if "speed_timing" not in search:
            raise ValueError("preselected landmark scope requires explicit speed timing")
        approach, transform = read_bound_speed_scope(scope_key, scope_reference, candidate)
    for arm, kph in strict_zip(("R0", "R1"), targets):
        plan = plans[arm]
        if plan["controller"] != ORACLE_CONTROLLER_ID or plan["coordinate_frame"] != "nre_local":
            raise ValueError("incorrect oracle controller or coordinate frame")
        rows = np.asarray([[w[k] for k in ("time_s", "x", "y", "z", "yaw_rad")]
                           for w in plan["waypoints"]], dtype=float)
        if (rows.ndim != 2 or rows.shape[1] != 5 or len(rows) < 2 or not np.isfinite(rows).all()
            or rows[0, 0] != 0 or rows[-1, 0] < horizon or np.any(np.diff(rows[:, 0]) <= 0)
            or np.any(np.diff(rows[:, 0]) > .200001) or not np.any(rows[:, 0] == 1.7)):
            raise ValueError("plan lacks exact prefix or complete dense horizon")
        if not speed_band_prefilter(rows[:, 0], rows[:, 1:4], (kph - half_width) / 3.6, (kph + half_width) / 3.6)["passed"]:
            raise ValueError("selected plan does not satisfy its speed prefilter")
        if terminal_stop is not None:
            tail = rows[rows[:, 0] >= terminal_stop[1], 1:]
            if len(tail) == 0 or not np.allclose(tail, tail[0], rtol=0, atol=1e-9):
                raise ValueError("selected plan does not retain its declared terminal stop")
        if approach is not None:
            from .planned_speed_scope import audit_planned_speed_scope
            offset, yaw = transform[:3, 3], rows[:, 4]
            centers = rows[:, 1:3] + np.c_[np.cos(yaw)*offset[0]-np.sin(yaw)*offset[1],
                                          np.sin(yaw)*offset[0]+np.cos(yaw)*offset[1]]
            if not audit_planned_speed_scope(rows[:, 0], centers, approach["segment"],
                [(kph-half_width)/3.6, (kph+half_width)/3.6], deadline_s=duration,
                activation_polygon=approach["segment"]["scope_polygon"] if scope_key == "sign_scope_reference" else None)["passed"]:
                label = "sign" if scope_key == "sign_scope_reference" else "crosswalk"
                raise ValueError("selected plan fails its preselected original " + label + " scope")
        prefixes.append(rows[rows[:, 0] <= 1.7].tolist())
    if prefixes[0] != prefixes[1]:
        raise ValueError("speed plans have different non-language prefixes")
    scope = "real_counterfactual_pair_integration_not_qualified_oracle"
    start = binding["start_timestamp_us"]
    sidecar = {"schema_version": "l2d.realsim.sidecar.v0.1", "execution_scope": scope,
               "hypothesis_kind": "speed", "task_family_proposal": "B1_segment_speed_band",
               "speed_pair_human_reviewed": False, "sessions": {}}
    if scope_reference is not None:
        sidecar[scope_key] = scope_reference
    if terminal_stop is not None:
        sidecar["terminal_stop_window_s"] = list(terminal_stop)
    stat = search_path.stat()
    plan = {"source_binding": scope, "hypothesis_kind": "speed", "run_count": 2, "runs": [],
            "diagnostic_search_reference": {"path": str(search_path), "size_bytes": stat.st_size,
                                            "mtime_ns": stat.st_mtime_ns, "integrity_attestation": False}}
    for arm, kph in strict_zip(("R0", "R1"), targets):
        session = str(uuid.uuid4())
        sidecar["sessions"][session] = {"arm_id": arm, "seed": seed, "start_timestamp_us": start,
            "instruction_issued_at_us": start + 1700000, "oracle_plan": plans[arm],
            "target_speed_kph": float(kph), "diagnostic_speed_band_mps": [(kph - half_width) / 3.6, (kph + half_width) / 3.6]}
        plan["runs"].append({"scenario_id": candidate["scene_id_private"], "session_uuid": session,
                             "start_time_offset_us": binding["start_time_offset_us"], "random_seed": seed})
    user_path = run_dir / "user-config.yaml"
    user = yaml.safe_load(user_path.read_text())
    if "speed_timing" in search:
        if (user["simulation_config"].get("control_timestep_us") != 100000
            or user["simulation_config"].get("force_gt_duration_us") != 1700000):
            raise ValueError("explicit speed timing requires 100ms steps and the exact prefix")
        sidecar["speed_timing"] = timing
    user["simulation_config"]["n_sim_steps"] = round(duration * 10)
    launch["execution_scope"] = scope
    launch["speed_hypothesis"] = {"targets_kph": targets.tolist(), "task_family_proposal": "B1_segment_speed_band",
                                   "band_half_width_kph": half_width,
                                   "human_reviewed": False, "qualified": False,
                                   "initial_lane_pair_review_does_not_cover_speed_pair": True}
    if "speed_timing" in search:
        launch["speed_hypothesis"]["speed_timing"] = timing
    if terminal_stop is not None:
        launch["speed_hypothesis"]["terminal_stop_window_s"] = list(terminal_stop)
    atomic_json(run_dir / "speed-pair-sidecar.private.json", sidecar)
    atomic_json(run_dir / "speed-pair-plan.private.json", plan)
    atomic_json(run_dir / "launch.private.json", launch)
    user_path.write_text(yaml.safe_dump(user, sort_keys=False))
    return {"scope": scope, "seed": seed, "sessions": 2, "qualified": False, "human_reviewed": False}
