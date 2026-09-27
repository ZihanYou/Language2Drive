"""Source-shaped branch hypotheses from one real pilot, never seed completion."""
from copy import deepcopy
from pathlib import Path
import uuid

from .agent_curation import validate_agent_runs
from .agent_drafts import TEMPLATES, validate_draft_batch
from .branch_diagnostics import inspect_branch_semantics
from .constants import ALPASIM_COMMIT, COUPLED_SEEDS, SCHEMA_VERSION
from .io import load_json
from .schema import validate_pcu


def audit_branch_draft(pcu, native, profile):
    """Verify newly assigned geometry IDs on actual rows; never invent metrics."""
    from .compiler import compile_source
    from .pcu_authoring import relabel_actual_spatial_rows
    from .schema import SchemaError
    from .verifier import verify_trace
    validate_pcu(pcu, allow_pending_reviews=True)
    try:
        compile_source({"schema_version": SCHEMA_VERSION, "pcus": [pcu]})
    except SchemaError as error:
        blocker = str(error)
    else:
        raise ValueError("pending human-review draft unexpectedly compiled")
    results = []
    for row in native["arms"]:
        arm = row["arm_id"]
        samples = relabel_actual_spatial_rows(row["normalized_actual_samples"], pcu["private_annotations"])
        own = verify_trace(pcu, arm, samples, pcu["private_annotations"], profile)
        other = verify_trace(pcu, "R1" if arm == "R0" else "R0", samples, pcu["private_annotations"], profile)
        results.append({"arm_id": arm, "seed": row["seed"], "session_uuid": row["session_uuid"],
                        "own": own.to_dict(), "opposing": other.to_dict()})
    passed = len(results) == 2 and {r["arm_id"] for r in results} == {"R0", "R1"} and all(
        r["own"]["passed"] and not r["opposing"]["passed"] for r in results)
    return {"scope": "assembled_branch_draft_actual_audit_not_qualification", "qualified": False,
            "results": results, "all_own_pass_opposing_fail": passed, "formal_compile_blocker": blocker,
            "formal_witnesses_created": 0}


BRANCH_UTTERANCES = {
    "straight": ["Continue straight on the current road.", "Keep going straight along this road.",
                 "Follow the current road straight ahead."],
    "right": ["Take the upcoming right-hand entrance.", "Turn right into the next entrance.",
              "Use the next entrance on your right."],
    "left": ["Take the upcoming left-hand entrance.", "Turn left into the next entrance.",
             "Use the next entrance on your left."],
}


def branch_utterances(role, turn_context="entrance"):
    """Keep a street intersection distinct from an entrance in model language."""
    if role not in BRANCH_UTTERANCES or turn_context not in {"entrance", "intersection"}:
        raise ValueError("known branch role and explicit entrance/intersection context required")
    if role == "straight" or turn_context == "entrance":
        return deepcopy(BRANCH_UTTERANCES[role])
    return [f"Turn {role} at the upcoming intersection.",
            f"At the next intersection, take the road to the {role}.",
            f"Make a {role} turn at the intersection ahead."]


def paired_branch_utterances(roles, turn_context, task_family="A2_straight_turn"):
    """Require explicit family semantics; topology alone never assigns A1."""
    if not isinstance(roles, dict) or set(roles) != {"R0", "R1"}:
        raise ValueError("two explicit branch roles required")
    values = tuple(roles.values())
    if any(not isinstance(v, str) for v in values):
        raise ValueError("branch roles must be named directions")
    if task_family == "A1_intersection_left_right":
        if set(values) != {"left", "right"} or turn_context != "intersection":
            raise ValueError("A1 requires left/right at an explicit intersection")
    elif task_family == "A2_straight_turn":
        if set(values) not in ({"straight", "left"}, {"straight", "right"}):
            raise ValueError("A2 requires explicit straight/turn mapping")
    else:
        raise ValueError("only explicit A1 or A2 branch authoring is supported")
    return {a: branch_utterances(role, turn_context) for a, role in roles.items()}


def allocate_unbound_opportunity(batch, proposal_key, target_draft_id):
    """Exchange only allocation labels of two unbound slots, with explicit history.

    The displaced opportunity stays an unverified hypothesis, not an inferred
    A1 scene. No successful/pilot-bound or previously authored row can move.
    """
    validate_draft_batch(batch)
    result = deepcopy(batch)
    original = [r for r in result["drafts"] if r["source_proposal_key"] == proposal_key]
    target = [r for r in result["drafts"] if r["draft_id"] == target_draft_id]
    if len(original) != 1 or len(target) != 1:
        raise ValueError("existing unique opportunity and target slot required")
    source, destination = original[0], target[0]
    for row in (source, destination):
        if (row["family_binding_status"] != "unverified_hypothesis"
            or row["primary_layer"] != "A"
            or any(v.get("semantic_contract") or v.get("instruction_schedule") for v in row["arms"].values())
            or any(k in row for k in ("authored_pcu_path", "agent_pilot_evidence",
                                      "agent_branch_pilot_run", "exploratory_evidence"))):
            raise ValueError("only untouched unbound A allocations can be exchanged")
    if source is destination:
        return result, source
    old = [{k: r[k] for k in ("draft_id", "source_proposal_key", "task_family")} for r in (source, destination)]
    source["task_family"], destination["task_family"] = destination["task_family"], source["task_family"]
    for row in (source, destination):
        row["arms"] = {a: {"instruction_template_not_executable": t, "instruction_schedule": None, "semantic_contract": None}
                       for a, t in zip(("R0", "R1"), TEMPLATES[row["task_family"].split("_")[0]])}
    result.setdefault("allocation_history", []).append({"before": old, "reason": "bind observed branch meaning; displaced slot remains unverified",
                                                       "qualified": False, "human_reviewed": False})
    validate_draft_batch(result)
    return result, source


def replace_unbound_branch_opportunity(root, batch, candidate_path, target_draft_id, reason,
                                       roles, turn_context, task_family="A2_straight_turn",
                                       supplementary_acquisition=None):
    """Keep quotas and provenance while replacing an untouched branch hypothesis."""
    from .speed_authoring import _replace_unbound_opportunity
    paired_branch_utterances(roles, turn_context, task_family)
    return _replace_unbound_opportunity(root, batch, candidate_path, target_draft_id, reason,
                                       supplementary_acquisition, task_family=task_family)


def assemble_branch_pcu(root, native_path, batch_path, target_draft_id, roles, turn_context="entrance",
                        task_family="A2_straight_turn", *, replacement_reason=None,
                        supplementary_acquisition=None):
    root, native_path, batch_path = Path(root).resolve(), Path(native_path).resolve(), Path(batch_path).resolve()
    if (not native_path.is_relative_to(root / ".runtime/curation_review/spatial")
        or not batch_path.is_relative_to(root / "data/private/agent_drafts")):
        raise ValueError("isolated project authoring inputs required")
    utterances = paired_branch_utterances(roles, turn_context, task_family)
    native = load_json(native_path)
    run = Path(native["run_dir"])
    candidate = validate_agent_runs(root, [run])
    semantics = inspect_branch_semantics(run)
    if not all(semantics[k] for k in ("actual_branch_prefilter_passed", "sampled_runtime_safety_passed", "preintervention_non_language_equal")):
        raise ValueError("current complete actual branch pilot must pass")
    if (native["scope"] != "actual_branch_gate_full_verifier_pilot_not_qualification"
        or native["complete"] is not True or native["all_own_pass_opposing_fail"] is not True
        or native["qualified"] is not False or native["human_reviewed"] is not False
        or native["proposal_key"] != candidate["proposal_key"]):
        raise ValueError("complete matching native gate evidence required")
    sidecar = load_json(run / "branch-pair-sidecar.private.json")
    ledger = {(s, p["arm_id"], p["seed"]) for s, p in sidecar["sessions"].items()}
    if len(native["arms"]) != 2 or {(r["session_uuid"], r["arm_id"], r["seed"]) for r in native["arms"]} != ledger:
        raise ValueError("exact actual two-arm ledger required")
    for row in native["arms"]:
        actual = semantics["actual_sources"][row["arm_id"]]
        if (any(str(row[k]) != str(actual[k]) for k in ("session_uuid", "source_asl", "source_size_bytes", "source_mtime_ns"))
            or row["own"]["passed"] is not True or row["opposing"]["passed"] is not False or row["invalid_frame_times_s"]):
            raise ValueError("unchanged passing actual ASL evidence required")
    original_batch = load_json(batch_path)
    if replacement_reason is not None:
        source = load_json(run / "launch.private.json")["agent_candidate_source"]
        original_batch = replace_unbound_branch_opportunity(root, original_batch, source, target_draft_id,
            replacement_reason, roles, turn_context, task_family, supplementary_acquisition)
    elif supplementary_acquisition is not None:
        raise ValueError("supplementary replacement requires an explicit reason")
    batch, slot = allocate_unbound_opportunity(original_batch, candidate["proposal_key"], target_draft_id)
    if slot["task_family"] != task_family:
        raise ValueError("allocation must match the explicitly selected branch family")
    ids = {k: k + "-" + uuid.uuid4().hex for k in ("pcu", "group", "trigger", "support", "recording", "gate0", "gate1")}
    gate_ids = dict(zip(("R0", "R1"), (ids["gate0"], ids["gate1"])))
    annotations = deepcopy(native["annotations_draft"])
    annotations["gates"] = {gate_ids[a]: annotations["gates"]["draft-gate-" + a] for a in ("R0", "R1")}
    search = load_json(load_json(run / "branch-pair-plan.private.json")["diagnostic_search_reference"]["path"])
    geometry = load_json(search["source_cache"])
    # Use one unchanged upstream section containing the intervention pose.
    # Unioning touching original sections can create degenerate seam rings.
    from shapely.geometry import Point, shape
    selected_lane = next(a["lane_id"] for a in candidate["selected_arm_pair"]["arms"] if a["arm"] == "R0")
    sections = geometry["arms"]["R0"]["lanes"]
    upstream = sections[:next(i for i, r in enumerate(sections) if r["lane_id"] == selected_lane)]
    scene = candidate["scene_execution_proposal_not_frozen"]
    plans = {p["arm_id"]: deepcopy(p["oracle_plan"]) for p in sidecar["sessions"].values()}
    prefix = [w for w in plans["R0"]["waypoints"] if w["time_s"] <= 1.7]
    issue_pose = next(w for w in prefix if w["time_s"] == 1.7)
    triggers = [r["polygon"] for r in upstream if shape(r["polygon"]).covers(Point(issue_pose["x"], issue_pose["y"]))]
    if len(triggers) != 1:
        raise ValueError("unique original upstream section at intervention required")
    annotations["triggers"] = {ids["trigger"]: deepcopy(triggers[0])}
    inspection = load_json(run / "branch-pair-inspection.private.json")
    prefix_evidence = inspection["arms"]["R0"]["prefix"]
    duration = semantics["rollout_duration_s"]
    contracts = {a: {"id": "contract-" + uuid.uuid4().hex, "kind": "gate", "signature_set": ["signature-" + uuid.uuid4().hex],
                     "target_gate": gate_ids[a], "opposing_gate": gate_ids["R1" if a == "R0" else "R0"], "deadline_s": duration}
                 for a in ("R0", "R1")}
    pcu = {"schema_version": SCHEMA_VERSION, "pcu_id": ids["pcu"], "meta_group_id": ids["group"],
        "backend": {"type": "alpasim", "alpasim_commit": ALPASIM_COMMIT, "renderer_type": "nurec", "dataset_version": "26.01"},
        "primary_layer": "A", "task_family": task_family,
        "scene": {"artifact_id": candidate["artifact_uuid"], "scene_id_private": candidate["scene_id_private"],
            "recording_id_private": ids["recording"], **{k: scene[k] for k in ("recording_start_timestamp_us", "start_timestamp_us", "start_time_offset_us")},
            "initial_pose": {k: prefix[0][k] for k in ("x", "y", "z", "yaw_rad")},
            "camera_configuration": {"logical_ids": prefix_evidence["preintervention_camera_logical_ids"], "height": 320, "width": 576},
            "reconstruction_support_id": ids["support"]},
        "shared_non_language_input": {"route_envelope": [[w["x"], w["y"]] for w in prefix], "route_envelope_kind": "COMMON_SUPPORT",
            "route_semantic_target": None, "route_neutrality_reviewed": False, "command_policy": "UNKNOWN_NEUTRAL",
            "coupled_exogenous_seeds": list(COUPLED_SEEDS), "preintervention_history_id": prefix_evidence["preintervention_non_language_sha256"],
            "simulator_configuration": {"renderer": {"kind": "sensorsim", "implementation": "NuRec"}, "route_use_waypoint_commands": False}},
        "decision_window": {"trigger": ids["trigger"], "start_s": 1.7, "deadline_s": duration, "commitment_gates": list(gate_ids.values())},
        "hard_constraints": {"collision": False, "offroad": False, "legality": True, "minimum_progress_m": 10., "deadline_s": duration},
        "arms": {a: {"instruction_schedule": [{"issue_at_us": scene["start_timestamp_us"] + 1700000, "utterances": utterances[a]}],
                     "semantic_contract": c, "opposing_contract": deepcopy(contracts["R1" if a == "R0" else "R0"])} for a, c in contracts.items()},
        "private_annotations": annotations,
        "oracle_plans_by_seed": {str(s): deepcopy(plans) for s in COUPLED_SEEDS}}
    validate_pcu(pcu, allow_pending_reviews=True)
    audit = audit_branch_draft(pcu, native, load_json(root / "config/threshold_profile.json"))
    if not audit["all_own_pass_opposing_fail"]:
        raise ValueError("new source-shaped contract failed on actual pilot rows")
    slot.update(family_binding_status=f"source_shaped_{task_family.split('_')[0]}_draft_pending_final_review", topology_pair_hypothesis=deepcopy(candidate["selected_arm_pair"]),
                authored_start_offset_us_not_frozen=scene["start_time_offset_us"], agent_branch_pilot_run=str(run))
    slot["arms"] = deepcopy(pcu["arms"])
    validate_draft_batch(batch)
    draft = {"scope": "source_shaped_agent_pcu_draft_pending_review_not_qualified", "pcu": pcu, "source_draft_id": slot["draft_id"],
             "proposal_key": candidate["proposal_key"], "qualified": False, "final_human_review": "pending", "formal_witnesses_created": 0,
             "structural_validation_passed_with_pending_reviews": True, "evidence_paths": [str(native_path)],
             "actual_pilot_seeds": sorted({r["seed"] for r in native["arms"]}),
             "turn_context": turn_context,
             "assembled_actual_contract_audit": audit,
             "plan_seed_status": {str(s): "deterministic_plan_hypothesis_not_formal_seed_validation" for s in COUPLED_SEEDS},
             "remaining": ["final language/route/legality/balance/render review", "source freeze", "three fresh formal seed pairs", "qualification and model evaluation"]}
    return batch, draft
