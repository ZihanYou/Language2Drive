"""Assemble source-shaped lane PCU drafts without minting reviews or witnesses."""
from copy import deepcopy
from pathlib import Path
import uuid
import json

from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
import numpy as np

from .agent_curation import validate_agent_runs
from .agent_drafts import validate_draft_batch
from .constants import ALPASIM_COMMIT, COUPLED_SEEDS, SCHEMA_VERSION
from .feasibility_summary import summarize_lane_seeds, summarize_lane_pilot
from .io import load_json
from .schema import validate_pcu
from .events import point_in_polygon, validate_trace


LANE_UTTERANCES = {
    "keep": ["Keep your current lane.", "Stay in the lane you are in.", "Continue in your present lane without changing lanes."],
    "change_left": ["Safely change to the adjacent lane on your left.", "Move one lane to the left when safe.",
                    "When it is safe, switch into the lane immediately to your left."],
    "change_right": ["Safely change to the adjacent lane on your right.", "Move one lane to the right when safe.",
                     "When it is safe, switch into the lane immediately to your right."],
}


def relabel_actual_spatial_rows(rows, annotations):
    """Recompute only geometry membership after assigning opaque draft IDs.

    Actual positions, timestamps, safety and render signals are unchanged.
    Reusing old scope labels would spuriously fail the new source contracts.
    """
    validate_trace(rows)
    result = deepcopy(rows)
    for row in result:
        for registry, output in (("lanes", "lane_ids"), ("regions", "regions"),
                                 ("scopes", "scopes"), ("triggers", "triggers")):
            row[output] = [key for key, geometry in annotations.get(registry, {}).items()
                           if point_in_polygon(row["position"], geometry)]
    return result


def bilateral_middle_trigger(spatial):
    """Bind A4 activation to the original middle lane, not either target lane."""
    binding = spatial["original_three_lane_binding"]
    rows = [r for r in binding["checks"] if r["time_s"] == 1.7]
    if binding.get("passed") is not True or binding.get("geometry_modified") is not False or len(rows) != 1:
        raise ValueError("unchanged original A4 three-lane instruction binding required")
    middle_id = rows[0]["middle_lane"]
    if middle_id not in spatial["middle_corridor_lane_ids"]:
        raise ValueError("instruction middle lane is not in the original middle corridor")
    polygon = Polygon(spatial["original_lane_polygons_xy"][middle_id])
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("cannot repair the original A4 trigger polygon")
    if not spatial["pairs"]:
        raise ValueError("actual A4 AABB poses required")
    for pair in spatial["pairs"]:
        if set(pair["actual_aabb_samples_txy"]) != {"R0", "R1"}:
            raise ValueError("both actual AABB arms required")
        for values in pair["actual_aabb_samples_txy"].values():
            trace = np.asarray(values, dtype=float)
            if (trace.ndim != 2 or trace.shape[1] != 3 or len(trace) < 2 or not np.isfinite(trace).all()
                or trace[0, 0] > 0 or trace[-1, 0] < 1.7 or np.any(np.diff(trace[:, 0]) <= 0)
                or np.any(np.diff(trace[:, 0]) > .200001)):
                raise ValueError("complete dense actual AABB history required")
            point = [float(np.interp(1.7, trace[:, 0], trace[:, i])) for i in (1, 2)]
            if not point_in_polygon(point, spatial["original_lane_polygons_xy"][middle_id]):
                raise ValueError("actual instruction AABB is outside the original middle trigger")
    return json.loads(json.dumps(mapping(polygon)))


def assemble_agent_lane_pcu(root, spatial_path, draft_id, batch_path):
    root, spatial_path = Path(root).resolve(), Path(spatial_path).resolve()
    if not spatial_path.is_relative_to(root / ".runtime/curation_review/spatial"):
        raise ValueError("project-local spatial evidence required")
    spatial = load_json(spatial_path)
    run_dirs = [Path(next(iter(pair["pose_bindings"].values()))["source_asl"]).parents[3]
                for pair in spatial["pairs"]]
    candidate = validate_agent_runs(root, run_dirs)
    selected = candidate["selected_arm_pair"]
    family = {"left_vs_right": "A4_lane_change_left_right", "keep_vs_left": "A5_lane_change_keep",
              "keep_vs_right": "A5_lane_change_keep"}.get(selected["kind"])
    if family is None:
        raise ValueError("only A4/A5 original lane pairs are supported")
    batch_path = Path(batch_path).resolve()
    if not batch_path.is_relative_to(root / "data/private/agent_drafts"):
        raise ValueError("isolated authoring batch required")
    batch = load_json(batch_path)
    validate_draft_batch(batch)
    slots = [r for r in batch["drafts"] if r["draft_id"] == draft_id]
    if (len(slots) != 1 or slots[0]["source_proposal_key"] != candidate["proposal_key"]
        or slots[0]["task_family"] != family):
        raise ValueError("PCU must bind the matching lane-family slot in the exact 50-item batch")
    summary = summarize_lane_pilot(run_dirs[0]) if len(run_dirs) == 1 else summarize_lane_seeds(run_dirs)
    probe_path = spatial_path.with_name("semantic-contract-probe.private.json")
    native_path = spatial_path.with_name("native-full-verifier.private.json")
    probe, native = load_json(probe_path), load_json(native_path)
    if any(r["candidate_binding"] != summary["candidate_binding"] for r in (spatial, probe, native)):
        raise ValueError("spatial/native evidence candidate differs")
    stat = spatial_path.stat()
    if (probe["source_draft"] != str(spatial_path) or probe["source_size_bytes"] != stat.st_size
        or probe["source_mtime_ns"] != str(stat.st_mtime_ns)
        or probe["all_own_pass_opposing_fail"] is not True
        or native["complete"] is not True or native["all_own_pass_opposing_fail"] is not True
        or native["qualified"] is not False or native["human_reviewed"] is not False):
        raise ValueError("complete unchanged real pilot evidence required")
    ledger = {(p["seed"], arm, p["arms"][arm]["session_uuid"]) for p in summary["pairs"] for arm in ("R0", "R1")}
    observed = [(r["seed"], r["arm_id"], r["session_uuid"]) for r in native["arms"]]
    if (len(observed) != 2 * len(summary["pairs"]) or set(observed) != ledger
        or len({r["session_uuid"] for r in native["arms"]}) != len(observed)
        or any(not r["own"]["passed"] or r["opposing"]["passed"] or r["invalid_frame_times_s"] for r in native["arms"])):
        raise ValueError("native verifier ledger incomplete or failed")
    ids = {kind: kind + "-" + uuid.uuid4().hex for kind in ("pcu", "group", "scope", "trigger", "support", "recording")}
    scene = candidate["scene_execution_proposal_not_frozen"]
    plans, prefix = {}, None
    for pair in summary["pairs"]:
        sidecar = load_json(Path(pair["run_dir"]) / "lane-pair-sidecar.private.json")
        if 'predeclared_original_lane_contract' in sidecar:
            from .planned_lane_contract import validate_predeclared_lane_geometry
            spatial_pair = next(p for p in spatial['pairs'] if p['seed'] == pair['seed'])
            validate_predeclared_lane_geometry(sidecar['predeclared_original_lane_contract'],
                spatial_pair['spatial_coverage'], spatial['original_lane_polygons_xy'],
                probe['draft_semantic_contracts'], probe['shared_scope_proposal'])
        plans[str(pair["seed"])] = {r["arm_id"]: deepcopy(r["oracle_plan"]) for r in sidecar["sessions"].values()}
        for row in sidecar["sessions"].values():
            if row["instruction_issued_at_us"] != scene["start_timestamp_us"] + 1700000:
                raise ValueError("unexpected real intervention timing")
            current = [w for w in row["oracle_plan"]["waypoints"] if w["time_s"] <= 1.7]
            if prefix is not None and prefix != current:
                raise ValueError("shared initial envelope differs")
            prefix = current
    actual_seeds = sorted(int(seed) for seed in plans)
    for seed in COUPLED_SEEDS:
        if str(seed) not in plans:
            plans[str(seed)] = deepcopy(plans[str(actual_seeds[0])])
    contracts = deepcopy(probe["draft_semantic_contracts"])
    for arm, contract in contracts.items():
        contract.update(id="contract-" + uuid.uuid4().hex, signature_set=["signature-" + uuid.uuid4().hex], scope=ids["scope"])
    upstream = [spatial["corridor_lane_ids_by_arm"][arm][0] for arm in ("R0", "R1")]
    trigger = json.loads(json.dumps(mapping(unary_union([
        Polygon(spatial["original_lane_polygons_xy"][key]) for key in upstream]))))
    if family == "A4_lane_change_left_right":
        trigger = bilateral_middle_trigger(spatial)
    annotations = {"lanes": {c["target_lane"]: spatial["original_lane_polygons_xy"][c["target_lane"]] for c in contracts.values()},
        "scopes": {ids["scope"]: probe["shared_scope_proposal"]}, "triggers": {ids["trigger"]: trigger},
        "gates": {}, "regions": {}, "actor_ids": [], "reconstruction_support_polygon": None,
        "render_validity_protocol": deepcopy(native["protocol_proposal"]),
        "legality_reviewed": False, "balance_reviewed": False, "render_reviewed": False}
    first_run = Path(summary["pairs"][0]["run_dir"])
    inspection = load_json(first_run / "lane-pair-inspection.private.json")
    camera_ids = inspection["arms"]["R0"]["prefix"]["preintervention_camera_logical_ids"]
    pcu = {"schema_version": SCHEMA_VERSION, "pcu_id": ids["pcu"], "meta_group_id": ids["group"],
        "backend": {"type": "alpasim", "alpasim_commit": ALPASIM_COMMIT, "renderer_type": "nurec", "dataset_version": "26.01"},
        "primary_layer": "A", "task_family": family,
        "scene": {"artifact_id": candidate["artifact_uuid"], "scene_id_private": candidate["scene_id_private"],
            "recording_id_private": ids["recording"], "recording_start_timestamp_us": scene["recording_start_timestamp_us"],
            "start_timestamp_us": scene["start_timestamp_us"], "start_time_offset_us": scene["start_time_offset_us"],
            "initial_pose": {k: prefix[0][k] for k in ("x", "y", "z", "yaw_rad")},
            "camera_configuration": {"logical_ids": camera_ids, "height": 320, "width": 576},
            "reconstruction_support_id": ids["support"]},
        "shared_non_language_input": {"route_envelope": [[w["x"], w["y"]] for w in prefix],
            "route_envelope_kind": "COMMON_SUPPORT", "route_semantic_target": None, "route_neutrality_reviewed": False,
            "route_envelope_basis": "identical_observed_preintervention_prefix_only_not_target_route",
            "command_policy": "UNKNOWN_NEUTRAL", "coupled_exogenous_seeds": list(COUPLED_SEEDS),
            "preintervention_history_id": inspection["arms"]["R0"]["prefix"]["preintervention_non_language_sha256"],
            "simulator_configuration": {"renderer": {"kind": "sensorsim", "implementation": "NuRec"},
                                        "route_use_waypoint_commands": False}},
        "decision_window": {"trigger": ids["trigger"], "start_s": 1.7, "deadline_s": 7., "commitment_gates": []},
        "hard_constraints": {"collision": False, "offroad": False, "legality": True,
                             "minimum_progress_m": native["pilot_minimum_progress_m"], "deadline_s": 7.},
        "arms": {}, "private_annotations": annotations, "oracle_plans_by_seed": plans}
    for arm in selected["arms"]:
        key = arm["arm"]
        pcu["arms"][key] = {"instruction_schedule": [{"issue_at_us": scene["start_timestamp_us"] + 1700000,
            "utterances": deepcopy(LANE_UTTERANCES[arm["role"]])}], "semantic_contract": contracts[key],
            "opposing_contract": deepcopy(contracts["R1" if key == "R0" else "R0"])}
    validate_pcu(pcu, allow_pending_reviews=True)
    return {"scope": "source_shaped_agent_pcu_draft_pending_review_not_qualified", "pcu": pcu,
        "source_draft_id": draft_id, "proposal_key": candidate["proposal_key"], "qualified": False,
        "batch_binding": {"path": str(batch_path), "size_bytes": batch_path.stat().st_size,
                          "mtime_ns": str(batch_path.stat().st_mtime_ns)},
        "structural_validation_passed_with_pending_reviews": True, "final_human_review": "pending",
        "formal_witnesses_created": 0, "evidence_paths": [str(spatial_path), str(probe_path), str(native_path)],
        "actual_pilot_seeds": actual_seeds, "three_seed_pilot_complete": len(actual_seeds) == 3,
        "plan_seed_status": {str(seed): ("actual_exploratory_pilot_not_formal_witness" if seed in actual_seeds
                                       else "deterministic_plan_hypothesis_not_run_or_validated") for seed in COUPLED_SEEDS},
        "remaining": ["final language/spatial/route/legality/balance/render review", "source and configuration freeze",
                      "fresh compiled oracle sessions and formal ASL receipts", "qualification and model evaluation"]}
