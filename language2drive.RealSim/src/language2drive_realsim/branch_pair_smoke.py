"""Prepare sampled-safe branch exploration while keeping every formal gate false."""
from pathlib import Path
import uuid

import numpy as np
import yaml
from shapely.geometry import shape

from .branch_path_search import branch_pair_semantic_prefilter
from .constants import COUPLED_SEEDS, ORACLE_CONTROLLER_ID
from .corridor_sections import linked_section_region
from .io import atomic_json, load_json
from .traffic_planning import SampledTraffic
from .path_turning import turning_budget


CURVATURE_SCOPE = "curvature_branch_geometry_dynamics_search_not_traffic_or_qualification"


def validate_continuing_branch_search(search, candidate):
    """Check continuing plans without inventing a stop or trusting audit flags alone."""
    if (search.get("scope") != CURVATURE_SCOPE or search.get("qualified") is not False
        or search.get("complete") is not True or search.get("human_reviewed") is not False
        or candidate["selected_arm_pair"]["kind"] != "recorded_vs_counterfactual_branch"):
        raise ValueError("completed exploratory continuing branch required")
    for a, b in (("proposal_key", "proposal_key"), ("review_revision", "review_revision"),
                 ("artifact_uuid", "artifact_uuid"), ("scene_id", "scene_id_private")):
        if search[a] != candidate[b]:
            raise ValueError("continuing branch review binding mismatch")
    duration, horizon = search["rollout_duration_s"], search["plan_horizon_s"]
    if (not np.isfinite([duration, horizon]).all() or not 7 <= duration <= 30
        or horizon != duration + 6 or search["intervention_s"] != 1.7
        or not np.isclose(duration * 10, round(duration * 10), atol=1e-8, rtol=0)):
        raise ValueError("complete continuing branch timing required")
    matches = [a for a in search["attempts"] if a["parameters"] == search["selected_parameters"]]
    if (len(matches) != 1 or matches[0].get("geometry_and_dynamics_passed") is not True
        or matches[0].get("semantics", {}).get("passed") is not True):
        raise ValueError("unique passed selected curvature attempt required")
    times = np.round(np.arange(round(horizon * 10) + 1) / 10, 6)
    plans = search["selected_plans"]
    if not isinstance(plans, dict) or set(plans) != {"R0", "R1"}:
        raise ValueError("both continuing branch plans required")
    rows = {}
    for arm, plan in plans.items():
        if plan["controller"] != ORACLE_CONTROLLER_ID or plan["coordinate_frame"] != "nre_local":
            raise ValueError("unexpected continuing controller or coordinates")
        row = np.asarray([[w[k] for k in ("time_s", "x", "y", "z", "yaw_rad")] for w in plan["waypoints"]])
        if row.shape != (len(times), 5) or not np.isfinite(row).all() or not np.array_equal(row[:, 0], times):
            raise ValueError("finite full continuing horizon required")
        velocity = np.gradient(row[:, 1:3], times, axis=0)
        acceleration = np.gradient(velocity, times, axis=0)
        active = (times > 1.8) & (times < horizon - .1)
        if (np.linalg.norm(acceleration[active], axis=1).max() > 3.5
            or np.abs(np.linalg.norm(velocity, axis=1) * np.gradient(np.unwrap(row[:, 4]), times))[active].max() > 3
            or not turning_budget(times, row[:, 1:4], row[:, 4])["passed"]):
            raise ValueError("continuing plan fails recomputed dynamics or turning")
        rows[arm] = row
    if not np.array_equal(rows["R0"][times <= 1.7], rows["R1"][times <= 1.7]):
        raise ValueError("continuing branch prefix mismatch")
    return times, rows


def validate_branch_search(search, candidate):
    if search.get("scope") == CURVATURE_SCOPE:
        return validate_continuing_branch_search(search, candidate)
    if (search.get("scope") != "sampled_traffic_yield_branch_search_not_qualification"
        or search.get("qualified") is not False or search.get("complete") is not True
        or search.get("pair_semantics", {}).get("passed") is not True
        or search.get("planned_preintervention_prefix_equal") is not True
        or candidate["selected_arm_pair"]["kind"] != "recorded_vs_counterfactual_branch"):
        raise ValueError("completed explicitly exploratory branch pair required")
    for a, b in (("proposal_key", "proposal_key"), ("review_revision", "review_revision"),
                 ("artifact_uuid", "artifact_uuid"), ("scene_id", "scene_id_private")):
        if search[a] != candidate[b]:
            raise ValueError("branch search and current candidate binding mismatch")
    duration, horizon = search["rollout_duration_s"], search["plan_horizon_s"]
    if (not np.isfinite([duration, horizon]).all() or duration < 3 or horizon != duration + 6
        or search["intervention_s"] != 1.7 or search["sampling_period_s"] != .1
        or not np.isclose(duration * 10, round(duration * 10), atol=1e-8, rtol=0)):
        raise ValueError("explicit duration and complete six-second future plan required")
    times = np.round(np.arange(round(horizon * 10) + 1) / 10, 6)
    plans = search["selected_plans"]
    if not isinstance(plans, dict) or set(plans) != {"R0", "R1"}:
        raise ValueError("both selected branch plans required")
    rows = {}
    for arm, plan in plans.items():
        matches = [a for a in search["attempts"][arm] if a["parameters"] == search["selected_parameters"][arm]]
        if (len(matches) != 1 or matches[0]["passed"] is not True
            or matches[0]["entire_stopped_footprint_beyond_selected_branch"] is not True
            or matches[0].get("traffic", {}).get("passed") is not True):
            raise ValueError("selected branch plan lacks a unique passed sampled audit")
        if plan["controller"] != ORACLE_CONTROLLER_ID or plan["coordinate_frame"] != "nre_local":
            raise ValueError("unexpected oracle controller or coordinate frame")
        row = np.asarray([[w[k] for k in ("time_s", "x", "y", "z", "yaw_rad")] for w in plan["waypoints"]])
        if row.shape != (len(times), 5) or not np.isfinite(row).all() or not np.array_equal(row[:, 0], times):
            raise ValueError("finite complete branch horizon required")
        parameters = search["selected_parameters"][arm]
        stop_time = parameters["brake_start_s"] + parameters["brake_duration_s"]
        if (not np.isfinite(stop_time) or parameters["brake_start_s"] < 1.7
            or parameters["brake_duration_s"] <= 0 or stop_time > horizon
            or not np.all(row[times >= stop_time, 1:] == row[-1, 1:])):
            raise ValueError("selected waiting plan does not retain its full stationary horizon")
        velocity = np.gradient(row[:, 1:3], times, axis=0)
        acceleration = np.gradient(velocity, times, axis=0)
        active = (times > 1.8) & (times < horizon - .1)
        if (np.linalg.norm(acceleration[active], axis=1).max() > 3.5
            or np.abs(np.linalg.norm(velocity, axis=1) * np.gradient(np.unwrap(row[:, 4]), times))[active].max() > 3):
            raise ValueError("selected path fails recomputed dynamics")
        rows[arm] = row
    if not np.array_equal(rows["R0"][times <= 1.7], rows["R1"][times <= 1.7]):
        raise ValueError("actual selected plans have unequal recorded prefixes")
    return times, rows


def prepare_branch_pair(run_dir, search_path, seed=1701, traffic_path=None):
    run_dir, search_path = Path(run_dir).resolve(), Path(search_path).resolve()
    if type(seed) is not int or seed not in COUPLED_SEEDS:
        raise ValueError("unsupported coupled seed")
    if any(run_dir.glob("*-pair-*.private.json")):
        raise ValueError("preserve existing paired run evidence")
    launch, search = load_json(run_dir / "launch.private.json"), load_json(search_path)
    if launch["scope"] != "exploratory_runtime_not_oracle_witness" or launch["driver_binding"] != "registered_static_for_asl_authority":
        raise ValueError("prepared static exploratory oracle runtime required")
    candidate = launch["candidate"]
    if launch.get("curation_authority"):
        from .agent_curation import validate_agent_runs
        root = Path(__file__).resolve().parents[2]
        if validate_agent_runs(root, [run_dir]) != candidate:
            raise ValueError("delegated branch authoring authority changed")
    times, rows = validate_branch_search(search, candidate)
    continuing = search["scope"] == CURVATURE_SCOPE
    if continuing:
        if traffic_path is None:
            raise ValueError("explicit bound traffic report required for continuing branch")
        report = load_json(traffic_path)
        stat = search_path.stat()
        if (report.get("scope") != "actual_artifact_sampled_traffic_not_live_oracle_witness"
            or report.get("qualified") is not False or report.get("complete") is not True
            or report.get("sampled_traffic_passed") is not True
            or Path(report["source_search"]).resolve() != search_path
            or report["source_search_size_bytes"] != stat.st_size
            or report["source_search_mtime_ns"] != stat.st_mtime_ns
            or any(report[k] != search[k] for k in ("proposal_key", "review_revision", "artifact_uuid", "scene_id", "rollout_duration_s"))
            or report["audited_plan_horizon_s"] != times[-1]
            or report["sampling_period_s"] != .1 or report["samples_per_arm"] != len(times)):
            raise ValueError("continuing traffic report binding or completion mismatch")
        traffic_cache = report["traffic_cache"]
    else:
        if traffic_path is not None:
            raise ValueError("yield branch already binds its own traffic cache")
        traffic_cache = search["traffic_cache"]
    traffic = SampledTraffic.from_file(traffic_cache, candidate)
    if not np.array_equal(traffic.times, times):
        raise ValueError("traffic cache does not cover the full selected horizon")
    geometry_path = Path(search.get("source_cache", search_path.parent / "branch-join-diagnostic.private.json")).resolve()
    if not geometry_path.is_relative_to(search_path.parent):
        raise ValueError("geometry cache must belong to this isolated search directory")
    geometry = load_json(geometry_path)
    if any(geometry[k] != candidate[k] for k in ("proposal_key", "review_revision", "artifact_uuid")):
        raise ValueError("geometry belongs to another reviewed branch")
    if continuing:
        artifact = Path(candidate["scene_execution_proposal_not_frozen"]["artifact_path"])
        stat = artifact.stat()
        expected_binding = {"start_timestamp_us": candidate["scene_execution_proposal_not_frozen"]["start_timestamp_us"],
                            "selected_arm_pair": candidate["selected_arm_pair"]}
        if "candidate_execution_binding" in geometry and geometry["candidate_execution_binding"] != expected_binding:
            raise ValueError("continuing geometry window or arm pair changed")
        if (Path(search["source_cache"]).resolve() != geometry_path.resolve()
            or geometry.get("scope") != "actual_map_branch_join_diagnostic_not_qualification"
            or geometry.get("qualified") is not False or geometry["scene_id"] != candidate["scene_id_private"]
            or Path(geometry["source_artifact"]).resolve() != artifact.resolve()
            or geometry["source_size_bytes"] != stat.st_size or geometry["source_mtime_ns"] != stat.st_mtime_ns
            or not np.array_equal(geometry["time_s"], times)):
            raise ValueError("continuing geometry binding or coverage mismatch")
        recorded = np.column_stack((geometry["recorded_xyz"], geometry["recorded_yaws"]))
        if recorded.shape != (len(times), 4) or not np.isfinite(recorded).all():
            raise ValueError("complete recorded prefix source required")
        if any(not np.array_equal(row[times <= 1.7, 1:], recorded[times <= 1.7]) for row in rows.values()):
            raise ValueError("plans differ from recorded non-language prefix")
        from .planned_approach import recheck_common_stop
        recheck_common_stop(search, geometry, times, rows)
    planned_gate_audit = None
    if launch.get("curation_authority"):
        from .branch_gate_prefilter import original_branch_gates, check_planned_gate_crossings
        gates, _ = original_branch_gates(candidate, geometry)
        planned_gate_audit = check_planned_gate_crossings(rows, traffic.transform, gates, search["rollout_duration_s"])
        if not planned_gate_audit["passed"]:
            raise ValueError("planned original exit gates fail before GPU preparation: " + str(planned_gate_audit["arms"]))
    regions, audits = {}, {}
    for arm, row in rows.items():
        audits[arm] = traffic.audit(row[:, 1:4], row[:, 4])
        if not audits[arm]["passed"]:
            raise ValueError("selected branch path fails repeated cached traffic audit")
        sections = geometry["arms"][arm]["lanes"]
        links = [(j["from_lane"], j["to_lane"]) for j in geometry["arms"][arm]["joins"]]
        regions[arm], _ = linked_section_region(sections, links)
        lane_id = next(a["lane_id"] for a in candidate["selected_arm_pair"]["arms"] if a["arm"] == arm)
        index = next(i for i, section in enumerate(sections) if section["lane_id"] == lane_id)
        if not continuing:
            exit_region, _ = linked_section_region(sections[index + 1:], links)
            final = traffic.ego_footprints(row[:, 1:4], row[:, 4])[-1]
            if (not exit_region.contains(final) or exit_region.boundary.distance(final) < .15
                or shape(sections[index]["polygon"]).distance(final) < .15):
                raise ValueError("planned wait would not clear the selected branch")
    from .branch_shared_surface import recheck_shared_precommit_surface
    regions = recheck_shared_precommit_surface(search, candidate, geometry, regions)
    semantics = branch_pair_semantic_prefilter(times, rows["R0"][:, 1:4], rows["R1"][:, 1:4],
                                              regions["R0"], regions["R1"], rollout_end_s=search["rollout_duration_s"])
    if not semantics["passed"]:
        raise ValueError("selected paths fail repeated branch semantic prefilter")
    user_path = run_dir / "user-config.yaml"
    user = yaml.safe_load(user_path.read_text())
    if user["simulation_config"]["control_timestep_us"] != 100000 or user["simulation_config"]["force_gt_duration_us"] != 1700000:
        raise ValueError("runtime timing differs from planned shared prefix")
    binding = candidate["scene_execution_proposal_not_frozen"]
    if binding["recording_end_timestamp_us"] - binding["start_timestamp_us"] <= round(times[-1] * 1e6):
        raise ValueError("recording does not cover complete selected plan")
    scope = "real_counterfactual_pair_integration_not_qualified_oracle"
    sidecar = {"schema_version": "l2d.realsim.sidecar.v0.1", "execution_scope": scope,
               "hypothesis_kind": "branch", "human_reviewed": False, "sessions": {}}
    stat = search_path.stat()
    plan = {"source_binding": scope, "hypothesis_kind": "branch", "run_count": 2, "runs": [],
            "diagnostic_search_reference": {"path": str(search_path), "size_bytes": stat.st_size,
                                           "mtime_ns": stat.st_mtime_ns, "integrity_attestation": False}}
    if continuing:
        traffic_path = Path(traffic_path).resolve()
        traffic_stat = traffic_path.stat()
        plan["diagnostic_traffic_reference"] = {"path": str(traffic_path), "size_bytes": traffic_stat.st_size,
                                               "mtime_ns": traffic_stat.st_mtime_ns, "integrity_attestation": False}
    for arm in ("R0", "R1"):
        session = str(uuid.uuid4())
        sidecar["sessions"][session] = {"arm_id": arm, "seed": seed, "start_timestamp_us": binding["start_timestamp_us"],
            "instruction_issued_at_us": binding["start_timestamp_us"] + 1700000,
            "rollout_duration_s": search["rollout_duration_s"], "oracle_plan": search["selected_plans"][arm]}
        plan["runs"].append({"session_uuid": session, "random_seed": seed, "scenario_id": candidate["scene_id_private"],
                             "start_time_offset_us": binding["start_time_offset_us"]})
    user["simulation_config"]["n_sim_steps"] = round(search["rollout_duration_s"] * 10)
    launch["execution_scope"] = scope
    launch["branch_hypothesis"] = {"rollout_duration_s": search["rollout_duration_s"], "qualified": False,
                                    "human_reviewed": False, "stop_legality_verified": False,
                                    "plan_mode": "continuing" if continuing else "yield_after_branch",
                                    "geometry_policy": "bounded_original_cap_hull_v2" if continuing else search["geometry_policy"]}
    atomic_json(run_dir / "branch-pair-sidecar.private.json", sidecar)
    atomic_json(run_dir / "branch-pair-plan.private.json", plan)
    atomic_json(run_dir / "branch-pair-preparation.private.json", {"qualified": False, "cached_traffic": audits,
                                                                  "shared_precommit_original_surface": search.get("shared_precommit_original_surface"),
                                                                  "semantic_prefilter": semantics,
                                                                  "planned_native_gates": planned_gate_audit})
    atomic_json(run_dir / "launch.private.json", launch)
    user_path.write_text(yaml.safe_dump(user, sort_keys=False))
    return {"scope": scope, "seed": seed, "sessions": 2, "qualified": False, "human_reviewed": False}
