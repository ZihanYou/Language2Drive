"""Delegated A4 pilot preparation; recompute plans from the bound real asset."""
from pathlib import Path
import os
import subprocess
import sys
import uuid

import numpy as np
import yaml

from .agent_curation import load_agent_candidate, validate_agent_runs
from .constants import COUPLED_SEEDS, ORACLE_CONTROLLER_ID
from .io import atomic_json, load_json
from .forward_choices import validate_grouped_forward_choices
from .path_turning import turning_budget
from .planned_lane_motion import sampled_lane_motion_passed
from .bilateral_protocol import bilateral_protocol, LEGACY_VERSION, EXPLICIT_VERSION


def validate_bilateral_search(search, candidate):
    """Fail closed before reading large assets or creating runtime sessions."""
    validate_grouped_forward_choices(search.get("explicit_forward_choices_privileged_only"), ("middle", "R0", "R1"))
    protocol = bilateral_protocol(search)
    pair = candidate["selected_arm_pair"]
    if (search.get("scope") != "unreviewed_offline_bilateral_lane_path_search"
        or search.get("semantic_prefilter_version") not in (LEGACY_VERSION, EXPLICIT_VERSION)
        or search.get("hypothesis_kind") != "bilateral_lane"
        or search.get("qualified") is not False or search.get("human_reviewed") is not False
        or search.get("final_human_review") != "pending" or search.get("complete") is not True
        or pair["kind"] != "left_vs_right"
        or [(a["arm"], a["role"]) for a in pair["arms"]] != [("R0", "change_left"), ("R1", "change_right")]
        or len({str(a["lane_id"]) for a in pair["arms"]}) != 2):
        raise ValueError("complete unreviewed A4 left/right search required")
    for sk, ck in (("proposal_key", "proposal_key"), ("artifact_uuid", "artifact_uuid"), ("scene_id", "scene_id_private")):
        if search.get(sk) != candidate[ck]:
            raise ValueError("bilateral search identity mismatch")
    if (search.get("start_timestamp_us") != candidate["scene_execution_proposal_not_frozen"]["start_timestamp_us"]
        or search.get("intervention_s") != 1.7 or search.get("planned_horizon_s") != 13.
        or search.get("rollout_duration_s_not_run") != 7.
        or search.get("three_lane_prefix", {}).get("passed") is not True):
        raise ValueError("bilateral initial binding or timing mismatch")
    chosen = [a for a in search["attempts"] if a["parameters"] == search["selected_parameters"]]
    if len(chosen) != 1 or chosen[0].get("passed") is not True:
        raise ValueError("one selected passing attempt required")
    chosen = chosen[0]
    if chosen.get("semantics", {}).get("passed") is not True:
        raise ValueError("bilateral semantics did not pass")
    if protocol is not None:
        start = protocol["dwell_start_s"]
        if chosen["semantics"].get("dwell_interval_s") != [start, start + 1.]:
            raise ValueError("chosen A4 dwell differs from preselected protocol")
        if (search.get("original_lane_use_prefilter", {}).get("passed") is not True
            or any(chosen.get("returned_to_original_middle_exclusively", {}).get(a) is not True
                   for a in protocol["post_task_return_arms"])):
            raise ValueError("explicit A4 lane-use or return check missing")
    for arm in ("R0", "R1"):
        audit = chosen.get("motion", {}).get(arm, {})
        if (chosen.get("turning", {}).get(arm, {}).get("passed") is not True
            or chosen.get("native_road", {}).get(arm, {}).get("passed") is not True
            or any(type(audit.get(k)) not in (int, float) or not np.isfinite(audit[k]) for k in (
                "minimum_2d_clearance_m", "overlap_sample_count", "max_planar_acceleration_mps2",
                "max_lateral_acceleration_mps2", "progress_at_7s_m"))
            or not sampled_lane_motion_passed(audit)):
            raise ValueError("bilateral sampled safety did not pass")
    plans = search.get("selected_plans")
    if not isinstance(plans, dict) or set(plans) != {"R0", "R1"}:
        raise ValueError("both bilateral plans required")
    prefixes = []
    for plan in plans.values():
        if plan["controller"] != ORACLE_CONTROLLER_ID or plan["coordinate_frame"] != "nre_local":
            raise ValueError("incorrect bilateral controller/frame")
        rows = np.asarray([[w[k] for k in ("time_s", "x", "y", "z", "yaw_rad")] for w in plan["waypoints"]])
        if (rows.shape != (131, 5) or not np.isfinite(rows).all()
            or not np.allclose(rows[:, 0], np.round(np.arange(131) / 10., 6), atol=1e-12, rtol=0)):
            raise ValueError("dense exact 13-second bilateral plan required")
        if not turning_budget(rows[:, 0], rows[:, 1:4], rows[:, 4])["passed"]:
            raise ValueError("bilateral steering budget exceeded")
        prefixes.append(rows[rows[:, 0] <= 1.7].tolist())
    if prefixes[0] != prefixes[1]:
        raise ValueError("bilateral non-language plan prefixes differ")
    return plans


def prepare_bilateral_pair(run_dir, search_path, seed=1701):
    root = Path(__file__).resolve().parents[2]
    run_dir, search_path = Path(run_dir).resolve(), Path(search_path).resolve()
    if type(seed) is not int or seed not in COUPLED_SEEDS:
        raise ValueError("unsupported coupled seed")
    if not run_dir.is_relative_to(root / ".runtime") or not search_path.is_relative_to(root / ".runtime"):
        raise ValueError("project-local exploratory inputs required")
    if any((run_dir / name).exists() for name in (
        "lane-pair-plan.private.json", "lane-pair-sidecar.private.json", "speed-pair-plan.private.json",
        "bilateral-preparation-recheck.private.json", "runtime-return.private.json")):
        raise ValueError("preserve existing preparation or actual evidence")
    launch, search = load_json(run_dir / "launch.private.json"), load_json(search_path)
    if (launch.get("scope") != "exploratory_runtime_not_oracle_witness"
        or launch.get("driver_binding") != "registered_static_for_asl_authority"):
        raise ValueError("prepared static exploratory runtime required")
    candidate = validate_agent_runs(root, [run_dir])
    source = Path(search["offline_source"]).resolve()
    stat = source.stat()
    if (load_agent_candidate(root, source) != candidate
        or search["offline_source_stat"] != {"size_bytes": stat.st_size, "mtime_ns": str(stat.st_mtime_ns)}):
        raise ValueError("bilateral search input changed")
    plans = validate_bilateral_search(search, candidate)
    protocol = bilateral_protocol(search)
    params = search["selected_parameters"]
    command = [sys.executable, str(root / "scripts/search_bilateral_lane_pair.py"), str(run_dir),
        "--offline-candidate", str(source), "--output-name", "bilateral-preparation-recheck.private.json",
        "--speed-fractions", str(params["speed_fraction"]), "--slowdown-s", str(params["slowdown_s"]),
        "--lane-change-starts-s", str(params["lane_change_start_s"]),
        "--lane-change-duration-s", str(params["lane_change_duration_s"]),
        "--minimum-corridor-length-m", str(search["minimum_corridor_length_m"]),
        "--target-progress-mode", params["target_progress_mode"]]
    if search["return_hypothesis"] is None:
        command.append("--no-return")
    else:
        command += ["--return-start-s", str(search["return_hypothesis"]["start_s"]),
                    "--return-duration-s", str(search["return_hypothesis"]["duration_s"])]
    if search["completed_stop_heading_hypothesis"]:
        command.append("--hold-completed-stop-heading")
    if protocol is not None:
        command += ["--dwell-start-s", str(protocol["dwell_start_s"])]
        if len(protocol["post_task_return_arms"]) == 1:
            command += ["--return-arm", protocol["post_task_return_arms"][0]]
    choices = validate_grouped_forward_choices(search.get("explicit_forward_choices_privileged_only"), ("middle", "R0", "R1"))
    for group, links in choices.items():
        for source_lane, target_lane in sorted(links.items()):
            command += [f"--{group.lower()}-forward-choice", source_lane+"="+target_lane]
    env = {k: v for k, v in os.environ.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    env.update(CUDA_VISIBLE_DEVICES="", POLARS_MAX_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (run_dir / "bilateral-preparation-recheck.log").open("x") as log:
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    recheck_path = run_dir / "bilateral-preparation-recheck.private.json"
    recheck = load_json(recheck_path)
    if bilateral_protocol(recheck) != protocol:
        raise ValueError("A4 protocol changed during real-asset regeneration")
    if validate_grouped_forward_choices(recheck.get("explicit_forward_choices_privileged_only"), ("middle", "R0", "R1")) != choices:
        raise ValueError("explicit forward choices changed during real-asset regeneration")
    if validate_bilateral_search(recheck, candidate) != plans:
        raise ValueError("plans differ from fresh real-asset regeneration")
    if validate_agent_runs(root, [run_dir]) != candidate or load_json(search_path) != search:
        raise ValueError("inputs changed during preparation")
    scope = "real_counterfactual_pair_integration_not_qualified_oracle"
    scene = candidate["scene_execution_proposal_not_frozen"]
    sidecar = {"schema_version": "l2d.realsim.sidecar.v0.1", "execution_scope": scope,
               "hypothesis_kind": "lane", "lane_pair_kind": "left_vs_right",
               "task_family_proposal": "A4_lane_change_left_right", "human_reviewed": False, "sessions": {}}
    stat = recheck_path.stat()
    plan = {"source_binding": scope, "run_count": 2, "runs": [], "hypothesis_kind": "lane",
            "diagnostic_search_reference": {"path": str(recheck_path), "size_bytes": stat.st_size,
                "mtime_ns": str(stat.st_mtime_ns), "integrity_attestation": False}}
    for arm in ("R0", "R1"):
        session = str(uuid.uuid4())
        sidecar["sessions"][session] = {"arm_id": arm, "seed": seed, "start_timestamp_us": scene["start_timestamp_us"],
            "instruction_issued_at_us": scene["start_timestamp_us"] + 1700000, "oracle_plan": plans[arm]}
        plan["runs"].append({"scenario_id": candidate["scene_id_private"], "session_uuid": session,
            "start_time_offset_us": scene["start_time_offset_us"], "random_seed": seed})
    user_path = run_dir / "user-config.yaml"
    user = yaml.safe_load(user_path.read_text())
    user["simulation_config"]["n_sim_steps"] = 70
    launch["commands"]["runtime"][2] = "language2drive_realsim.host_runtime"
    launch["execution_scope"] = scope
    launch["bilateral_hypothesis"] = {"qualified": False, "human_reviewed": False,
        "task_family_proposal": "A4_lane_change_left_right", "source_search": str(search_path),
        "regenerated_search": str(recheck_path)}
    if protocol is not None:
        sidecar["bilateral_protocol"] = protocol
        plan["bilateral_protocol"] = protocol
        launch["bilateral_hypothesis"]["bilateral_protocol"] = protocol
    atomic_json(run_dir / "lane-pair-sidecar.private.json", sidecar)
    atomic_json(run_dir / "lane-pair-plan.private.json", plan)
    atomic_json(run_dir / "launch.private.json", launch)
    user_path.write_text(yaml.safe_dump(user, sort_keys=False))
    return {"scope": scope, "sessions": 2, "seed": seed, "qualified": False, "human_reviewed": False}
