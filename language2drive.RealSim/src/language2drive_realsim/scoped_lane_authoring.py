"""Source-shaped B2 drafts from authentic scope evidence, never qualification."""
from copy import deepcopy
from pathlib import Path
import uuid
import zipfile

import numpy as np
from shapely.geometry import Point, Polygon, shape

from .agent_curation import validate_agent_runs
from .agent_drafts import validate_draft_batch
from .constants import ALPASIM_COMMIT, COUPLED_SEEDS, SCHEMA_VERSION
from .io import load_json
from .pcu_authoring import relabel_actual_spatial_rows
from .planned_lane_motion import footprint
from .schema import SchemaError, validate_pcu
from .speed_authoring import _replace_unbound_opportunity
from .verifier import verify_trace
from .scoped_lane_scope import continuous_scope_coverage, scope_centerline_between_gates


def audit_actual_scoped_lane_coverage(pcu, arm, rows, own):
    """Check between-sample geometry, including the exact frozen entry point.

    A discrete own-contract pass alone cannot exclude an entry sliver or an
    interior map gap. Do not trim to the first in-lane sample or buffer the map.
    """
    if not own.passed:
        return dict(passed=False, reason="own_contract_failed")
    annotations = pcu["private_annotations"]
    contract = pcu["arms"][arm]["semantic_contract"]
    other = pcu["arms"]["R1" if arm == "R0" else "R0"]["semantic_contract"]
    if contract["kind"] != "scoped_lane" or other["kind"] != "scoped_lane":
        raise ValueError("both scoped-lane contracts required")
    line = scope_centerline_between_gates(rows, own.diagnostics["entry_time_s"],
        own.diagnostics["exit_time_s"], annotations["gates"][contract["entry_gate"]],
        annotations["gates"][contract["exit_gate"]])
    def geometry(value):
        return shape(value) if isinstance(value, dict) else Polygon(value)
    lane = continuous_scope_coverage(geometry(annotations["lanes"][contract["target_lane"]]), line)
    scope = continuous_scope_coverage(geometry(annotations["scopes"][contract["scope"]]), line)
    excludes = not geometry(annotations["lanes"][other["target_lane"]]).intersects(line)
    return dict(passed=lane["passed"] and scope["passed"] and excludes,
        original_lane_coverage=lane, shared_scope_coverage=scope,
        excludes_opposing_lane=excludes, original_geometry_modified=False)


def scoped_lane_utterances(side, marker_description):
    if side not in {"left", "right"} or not isinstance(marker_description, str) or not marker_description.strip():
        raise ValueError("explicit lane side and physical marker description required")
    return [f"Stay in the {side} lane through the upcoming stretch until you reach {marker_description}.",
            f"Use the {side} lane along the next road segment, up to {marker_description}.",
            f"Keep to the {side} lane through that stretch until reaching {marker_description}."]


def audit_scoped_lane_draft(pcu, native, profile):
    from .compiler import compile_source
    if pcu.get("task_family") != "B2_scoped_lane_preference" or pcu.get("primary_layer") != "B":
        raise ValueError("only explicit B2 scoped-lane sources may use this audit")
    validate_pcu(pcu, allow_pending_reviews=True)
    try:
        compile_source(dict(schema_version=SCHEMA_VERSION, pcus=[pcu]))
    except SchemaError as error:
        blocker = str(error)
    else:
        raise ValueError("pending B2 source must not enter the formal compiler")
    results = []
    for actual in native["arms"]:
        arm = actual["arm_id"]
        rows = relabel_actual_spatial_rows(actual["normalized_actual_samples"], pcu["private_annotations"])
        own = verify_trace(pcu, arm, rows, pcu["private_annotations"], profile)
        opposing = verify_trace(pcu, "R1" if arm == "R0" else "R0", rows, pcu["private_annotations"], profile)
        continuous = audit_actual_scoped_lane_coverage(pcu, arm, rows, own)
        results.append(dict(arm_id=arm, seed=actual["seed"], session_uuid=actual["session_uuid"],
                            own=own.to_dict(), opposing=opposing.to_dict(), continuous_scope=continuous))
    valid = (len(results) == 2 and {r["arm_id"] for r in results} == {"R0", "R1"}
             and len({r["session_uuid"] for r in results}) == 2
             and len({r["seed"] for r in results}) == 1
             and all(r["seed"] in COUPLED_SEEDS and r["own"]["passed"] and not r["opposing"]["passed"]
                     and r["continuous_scope"]["passed"] for r in results))
    return dict(scope="assembled_B2_actual_contract_audit_not_qualification", qualified=False,
        all_own_pass_opposing_fail=valid, results=results, formal_compile_blocker=blocker,
        actual_pilot_seeds=sorted({r["seed"] for r in results}), three_seed_pilot_complete=False,
        formal_witnesses_created=0)


def assemble_scoped_lane_pcu(root, run_dir, native_path, batch_path, draft_id, marker_description,
                             *, supplementary_acquisition=None):
    root, run, native_path, batch_path = [Path(p).resolve() for p in (root, run_dir, native_path, batch_path)]
    if (not native_path.is_relative_to(root / ".runtime/curation_review/spatial")
        or not batch_path.is_relative_to(root / "data/private/agent_drafts")):
        raise ValueError("project-local isolated B2 authoring evidence required")
    policy = load_json(root / "config/curation_workflow.private.json")
    if policy.get("agent_may_author_drafts") is not True:
        raise ValueError("explicit delegated source-authoring authority required")
    candidate = validate_agent_runs(root, [run])
    launch = load_json(run / "launch.private.json")
    plan = load_json(run / "scoped_lane-pair-plan.private.json")
    for ref in plan["diagnostic_references"]:
        stat = Path(ref["path"]).stat()
        if (stat.st_size, str(stat.st_mtime_ns)) != (ref["size_bytes"], ref["mtime_ns"]):
            raise ValueError("frozen B2 source changed")
    scope = load_json(run / "scoped-lane-pair-scope.private.json")
    native = load_json(native_path)
    inspected = load_json(run / "scoped_lane-pair-inspection.private.json")
    sidecar = load_json(run / "scoped_lane-pair-sidecar.private.json")
    if (native.get("scope") != "actual_B2_native_protocol_full_verifier_not_qualification"
        or native.get("complete") is not True or native.get("qualified") is not False
        or native.get("human_reviewed") is not False or native.get("all_own_pass_opposing_fail") is not True
        or native.get("formal_witnesses_created") != 0 or Path(native["source_run_dir"]) != run
        or native["candidate_binding"] != {k: candidate[k] for k in ("proposal_key", "artifact_uuid", "scene_id_private", "review_revision")}
        or native["contracts_draft"] != scope["semantic_contracts"]
        or inspected["errors"] or not inspected["preintervention_non_language_equal"]
        or not inspected["sampled_runtime_safety_passed"]
        or any(native["annotations_draft"][k] != scope["private_annotations"][k] for k in ("lanes", "scopes", "gates"))):
        raise ValueError("current complete authentic B2 scope and native evidence required")
    ledger = {(v["arm_id"], v["seed"], session) for session, v in sidecar["sessions"].items()}
    if (len(native["arms"]) != 2 or len(ledger) != 2
        or {(v["arm_id"], v["seed"], v["session_uuid"]) for v in native["arms"]} != ledger
        or any(v["invalid_frame_times_s"] for v in native["arms"])):
        raise ValueError("one complete actual coupled pair required")
    for evidence in inspected["arms"].values():
        asl = Path(evidence["source_asl"])
        if ((asl.stat().st_size, asl.stat().st_mtime_ns) != (evidence["source_size_bytes"], evidence["source_mtime_ns"])
            or (asl.with_name("metrics.parquet").stat().st_size, asl.with_name("metrics.parquet").stat().st_mtime_ns)
            != (evidence["metrics_size_bytes"], evidence["metrics_mtime_ns"])):
            raise ValueError("actual ASL or official metrics changed")
    triage = load_json(run / "agent-visual-triage.private.json")
    if (triage.get("qualified") is not False or triage.get("human_reviewed") is not False
        or triage.get("agent_recommendation") != "continue_source_authoring_pending_final_review"
        or triage.get("marker_description") != marker_description
        or triage.get("source_marker_id") != scope["source_sign"]["key"]["map_id"]):
        raise ValueError("explicit non-excluded actual-frame authoring triage required")
    waits = load_json(run / "actual-downstream-wait-lines.private.json")
    if (triage.get("downstream_red_signal_visible") is True
        and (set(waits["arms"]) != {"R0", "R1"}
             or any(not arm["wait_lines"] or any(w["intersecting_actual_footprint_times_s"]
                    or w["minimum_actual_footprint_distance_m"] <= 0 for w in arm["wait_lines"])
                    for arm in waits["arms"].values()))):
        raise ValueError("visible downstream signal needs retained actual wait-line separation evidence")
    batch = _replace_unbound_opportunity(root, load_json(batch_path), launch["agent_candidate_source"], draft_id,
        "New original shared-marker B2 pair passed actual complete-scope and native diagnostics; final review remains pending.",
        task_family="B2_scoped_lane_preference", supplementary_acquisition=supplementary_acquisition)
    slot = next(r for r in batch["drafts"] if r["draft_id"] == draft_id)
    ids = {k: k + "-" + uuid.uuid4().hex for k in ("pcu", "group", "recording", "support", "trigger", "scope", "entry", "exit")}
    lane_ids = {old: "lane-" + uuid.uuid4().hex for old in native["annotations_draft"]["lanes"]}
    annotations = deepcopy(native["annotations_draft"])
    annotations["lanes"] = {lane_ids[k]: v for k, v in annotations["lanes"].items()}
    annotations["scopes"] = {ids["scope"]: annotations["scopes"]["scope-main"]}
    annotations["gates"] = {ids[k]: annotations["gates"]["gate-" + k] for k in ("entry", "exit")}
    scene = candidate["scene_execution_proposal_not_frozen"]
    by_arm = {p["arm_id"]: p for p in sidecar["sessions"].values()}
    seed = by_arm["R0"]["seed"]
    if by_arm["R1"]["seed"] != seed or seed not in COUPLED_SEEDS:
        raise ValueError("one supported coupled seed required")
    prefixes = [[w for w in by_arm[a]["oracle_plan"]["waypoints"] if w["time_s"] <= 1.7] for a in ("R0", "R1")]
    if prefixes[0] != prefixes[1] or prefixes[0][-1]["time_s"] != 1.7:
        raise ValueError("identical recorded preintervention prefix required")
    prefix = prefixes[0]
    # A trigger is an activation region, not a drivable lane or render support
    # polygon. Use the actual shared recorded vehicle footprint at issue time;
    # do not fill original-map gaps to manufacture a containing road section.
    from alpasim_utils.scenario import Rig
    from alpasim_runtime.unbound_rollout import get_ds_rig_to_aabb_center_transform
    with zipfile.ZipFile(scene["artifact_path"]) as archive:
        rig, = Rig.load_from_json(archive.read("rig_trajectories.json").decode())
    issue = scene["start_timestamp_us"] + 1700000
    pose = rig.trajectory.interpolate_pose(issue)
    if not np.allclose(pose.vec3, [prefix[-1][k] for k in ("x", "y", "z")], atol=1e-6, rtol=0):
        raise ValueError("original recorded instruction pose differs")
    vehicle = rig.vehicle_config
    trigger = footprint(pose.as_se3() @ get_ds_rig_to_aabb_center_transform(vehicle).as_se3(),
                        vehicle.aabb_x_m, vehicle.aabb_y_m)
    for row in native["arms"]:
        actual = row["normalized_actual_samples"]
        point = [float(np.interp(1.7, [r["time_s"] for r in actual], [r["position"][i] for r in actual])) for i in (0, 1)]
        if not trigger.covers(Point(point)):
            raise ValueError("shared recorded trigger does not contain both actual issue positions")
    annotations["triggers"] = {ids["trigger"]: list(map(list, trigger.exterior.coords))}
    annotations["trigger_basis"] = dict(kind="shared_recorded_ego_footprint_at_instruction", timestamp_us=issue,
        is_drivable_or_reconstruction_support_geometry=False, original_map_modified=False)
    annotations["shared_lane_scope_basis"] = dict(source_sign=scope["source_sign"], source_lane_records=scope["source_lane_records"],
        marker_description=marker_description, same_major_route_reviewed=False, marker_identity_reviewed=False,
        marker_visibility_reviewed=False, source_scope=str(run / "scoped-lane-pair-scope.private.json"))
    if "source_lane_chains" in scope:
        annotations["shared_lane_scope_basis"]["source_lane_chains"] = deepcopy(scope["source_lane_chains"])
    if "entry_construction" in scope:
        annotations["shared_lane_scope_basis"]["entry_construction"] = deepcopy(scope["entry_construction"])
    contracts = deepcopy(native["contracts_draft"])
    sides = {"keep_vs_right": {"R0": "left", "R1": "right"}, "keep_vs_left": {"R0": "right", "R1": "left"}}[candidate["selected_arm_pair"]["kind"]]
    for arm, contract in contracts.items():
        contract.update(id="contract-" + uuid.uuid4().hex, signature_set=["signature-" + uuid.uuid4().hex],
            target_lane=lane_ids[contract["target_lane"]], scope=ids["scope"], entry_gate=ids["entry"], exit_gate=ids["exit"])
    duration = scope["decision_window"]["deadline_s"]
    plans = {str(s): {a: deepcopy(p["oracle_plan"]) for a, p in by_arm.items()} for s in COUPLED_SEEDS}
    prefix_evidence = inspected["arms"]["R0"]["prefix"]
    pcu = dict(schema_version=SCHEMA_VERSION, pcu_id=ids["pcu"], meta_group_id=ids["group"],
        backend=dict(type="alpasim", alpasim_commit=ALPASIM_COMMIT, renderer_type="nurec", dataset_version="26.01"),
        primary_layer="B", task_family="B2_scoped_lane_preference",
        scene=dict(artifact_id=candidate["artifact_uuid"], scene_id_private=candidate["scene_id_private"],
            recording_id_private=ids["recording"], **{k: scene[k] for k in ("recording_start_timestamp_us", "start_timestamp_us", "start_time_offset_us")},
            initial_pose={k: prefix[0][k] for k in ("x", "y", "z", "yaw_rad")},
            camera_configuration=dict(logical_ids=prefix_evidence["preintervention_camera_logical_ids"], height=320, width=576),
            reconstruction_support_id=ids["support"]),
        shared_non_language_input=dict(route_envelope=[[w["x"], w["y"]] for w in prefix], route_envelope_kind="COMMON_SUPPORT",
            route_semantic_target=None, route_neutrality_reviewed=False, command_policy="UNKNOWN_NEUTRAL",
            coupled_exogenous_seeds=list(COUPLED_SEEDS), preintervention_history_id=prefix_evidence["preintervention_non_language_sha256"],
            simulator_configuration=dict(renderer=dict(kind="sensorsim", implementation="NuRec"), route_use_waypoint_commands=False)),
        decision_window=dict(trigger=ids["trigger"], start_s=1.7, deadline_s=duration, commitment_gates=[]),
        hard_constraints=dict(collision=False, offroad=False, legality=True, minimum_progress_m=10., deadline_s=duration),
        arms={a: dict(instruction_schedule=[dict(issue_at_us=issue, utterances=scoped_lane_utterances(sides[a], marker_description))],
            semantic_contract=c, opposing_contract=deepcopy(contracts["R1" if a == "R0" else "R0"])) for a, c in contracts.items()},
        private_annotations=annotations, oracle_plans_by_seed=plans)
    audit = audit_scoped_lane_draft(pcu, native, load_json(root / "config/threshold_profile.json"))
    if not audit["all_own_pass_opposing_fail"]:
        raise ValueError("assembled actual B2 contracts failed")
    slot.update(family_binding_status="source_shaped_B2_draft_pending_final_review", arms=deepcopy(pcu["arms"]),
        authored_start_offset_us_not_frozen=scene["start_time_offset_us"])
    validate_draft_batch(batch)
    draft = dict(scope="source_shaped_agent_pcu_draft_pending_review_not_qualified", qualified=False,
        final_human_review="pending", formal_witnesses_created=0, pcu=pcu, source_draft_id=draft_id,
        proposal_key=candidate["proposal_key"], actual_pilot_seeds=[seed], three_seed_pilot_complete=False,
        structural_validation_passed_with_pending_reviews=True, assembled_actual_contract_audit=audit,
        plan_seed_status={str(s): "actual_exploratory_plan_not_formal_witness" if s == seed else
                          "deterministic_plan_hypothesis_not_seed_validation" for s in COUPLED_SEEDS},
        evidence_paths=[str(native_path), str(run / "agent-visual-triage.private.json"), str(run / "actual-downstream-wait-lines.private.json")],
        remaining=["shared physical marker identity, visibility and route review", "scope-language review", "initial merge and downstream signal legality review",
                   "full visual quality and route neutrality review", "three-seed formal qualification", "source freeze and model evaluation"])
    return batch, draft
