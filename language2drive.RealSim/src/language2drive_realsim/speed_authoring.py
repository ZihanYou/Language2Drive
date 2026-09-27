"""Author B1 drafts from existing actual pilots, without inventing legality."""
from copy import deepcopy
from pathlib import Path
import uuid
import math

from .agent_drafts import validate_draft_batch
from .constants import ALPASIM_COMMIT, COUPLED_SEEDS, SCHEMA_VERSION
from .feasibility_summary import summarize_speed_seeds, summarize_speed_pilot
from .io import load_json
from .pcu_authoring import relabel_actual_spatial_rows
from .schema import SchemaError, validate_pcu
from .verifier import verify_trace
from .speed_timing import validate_speed_runtime_timing


def original_speed_trigger_anchor(candidate, native):
    """Anchor a single-lane speed task at its actual bound lane, not a speed-arm label."""
    if any(key in native for key in ('source_map_binding', 'sign_scope_reference', 'crosswalk_scope_reference')):
        return candidate['selected_arm_pair']['arms'][0]['lane_id']
    from .speed_spatial_draft import original_candidate_scope_geometry
    lane_id, _ = original_candidate_scope_geometry(candidate, native['source_lane_id'])
    return lane_id


def original_intervention_trigger(vector_map, lane_id, prefix, *, predecessor_lookback_depth=0,
                                  history_audit=None):
    """Select an original issue section; opt-in lookback uses only the 0–1.7s prefix."""
    import numpy as np
    from shapely.geometry import Point, Polygon
    from .lane_path_search import lane_corridor
    history = np.asarray([[w[k] for k in ("time_s", "x", "y", "z", "yaw_rad")] for w in prefix])
    _, ids, _ = lane_corridor(vector_map, lane_id, 70., observed_prefix=history,
        predecessor_lookback_depth=predecessor_lookback_depth, history_audit=history_audit)
    point = Point(prefix[-1]["x"], prefix[-1]["y"])
    found = []
    for key in ids:
        lane = vector_map.get_road_lane(key)
        vertices = np.concatenate((lane.left_edge.xyz[:, :2], lane.right_edge.xyz[::-1, :2])).tolist()
        polygon = Polygon(vertices)
        if polygon.is_valid and polygon.covers(point):
            found.append((key, vertices))
    if len(found) != 1:
        raise ValueError("unique unchanged original section at intervention required")
    return found[0]


def move_unbound_opportunity(batch, proposal_key, target_draft_id):
    """Move real opportunity payloads between unbound A/B slots, not evidence.

    Slot identities, families, instructions and requirements stay in place.
    Neither displaced opportunity gains evidence or family verification.
    """
    validate_draft_batch(batch)
    result = deepcopy(batch)
    source = [r for r in result["drafts"] if r["source_proposal_key"] == proposal_key]
    target = [r for r in result["drafts"] if r["draft_id"] == target_draft_id]
    if len(source) != 1 or len(target) != 1:
        raise ValueError("unique existing source opportunity and target slot required")
    source, target = source[0], target[0]
    for row in (source, target):
        if (row["family_binding_status"] != "unverified_hypothesis" or row["primary_layer"] not in {"A", "B"}
            or any(a.get("semantic_contract") or a.get("instruction_schedule") for a in row["arms"].values())
            or any(k in row for k in ("authored_pcu_path", "agent_pilot_evidence", "agent_branch_pilot_run"))):
            raise ValueError("only untouched unbound A/B opportunity slots can move")
    if source is target:
        return result, target
    fixed = {"draft_id", "task_family", "primary_layer", "arms", "required_bindings", "grounding_bindings", "family_binding_status"}
    before = [{k: r[k] for k in ("draft_id", "source_proposal_key", "task_family")} for r in (source, target)]
    source_payload = {k: deepcopy(v) for k, v in source.items() if k not in fixed}
    target_payload = {k: deepcopy(v) for k, v in target.items() if k not in fixed}
    for row, payload in ((source, target_payload), (target, source_payload)):
        for key in list(row):
            if key not in fixed:
                del row[key]
        row.update(payload)
    result.setdefault("allocation_history", []).append({"before": before,
        "reason": "move existing successful hypothesis into matching family slot; displaced opportunity remains unverified",
        "qualified": False, "human_reviewed": False})
    validate_draft_batch(result)
    return result, target


def speed_utterances(lower_mps, upper_mps):
    """Expose the exact quantitative band in every equivalent model utterance."""
    lower, upper = lower_mps * 3.6, upper_mps * 3.6
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper:
        raise ValueError("ordered nonnegative speed band required")
    return [f"Through the upcoming road segment, keep your speed between {lower:g} and {upper:g} km/h.",
            f"Maintain {lower:g}–{upper:g} km/h while driving through the next stretch of this road.",
            f"Travel along the upcoming stretch at a speed from {lower:g} to {upper:g} km/h."]


def replace_unbound_speed_opportunity(root, batch, candidate_path, target_draft_id, reason,
                                     supplementary_acquisition=None, *, task_family="B1_segment_speed_band",
                                     actual_preview_run=None):
    """Replace an unused speed lead, not its family or unverified grounding.

    B3 must be explicitly selected. This allocates a hypothesis only; actual
    crosswalk evidence and review remain mandatory in B3 source assembly.
    """
    if task_family not in {"B1_segment_speed_band", "B3_triggered_speed_constraint"}:
        raise ValueError("explicit B1 or B3 speed replacement family required")
    return _replace_unbound_opportunity(root, batch, candidate_path, target_draft_id, reason,
                                       supplementary_acquisition, task_family=task_family,
                                       actual_preview_run=actual_preview_run)


def _replace_unbound_opportunity(root, batch, candidate_path, target_draft_id, reason,
                                 supplementary_acquisition=None, *, task_family, actual_preview_run=None):
    """Shared evidence checks for explicit speed/branch replacement, never qualification."""
    from .agent_curation import load_agent_candidate
    from .curation_review import arm_pair_candidates
    if task_family not in {"B1_segment_speed_band", "B2_scoped_lane_preference", "B3_triggered_speed_constraint", "A1_intersection_left_right", "A2_straight_turn"}:
        raise ValueError("explicit supported replacement family required")
    pair_kinds = ({"keep_vs_left", "keep_vs_right"} if task_family in {"B1_segment_speed_band", "B2_scoped_lane_preference", "B3_triggered_speed_constraint"}
                  else {"recorded_vs_counterfactual_branch"})
    root = Path(root).resolve()
    validate_draft_batch(batch)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("explicit replacement reason required")
    candidate = load_agent_candidate(root, candidate_path)
    key = candidate["proposal_key"]
    if any(r["source_proposal_key"] == key for r in batch["drafts"]):
        raise ValueError("replacement opportunity is already allocated")
    targets = [r for r in batch["drafts"] if r["draft_id"] == target_draft_id]
    if len(targets) != 1:
        raise ValueError("one existing matching family slot required")
    old = targets[0]
    if (old["task_family"] != task_family or old["primary_layer"] != task_family[0]
        or old["family_binding_status"] != "unverified_hypothesis"
        or any(a.get("semantic_contract") or a.get("instruction_schedule") for a in old["arms"].values())
        or any(k in old for k in ("authored_pcu_path", "agent_pilot_evidence", "agent_branch_pilot_run", "exploratory_evidence"))):
        raise ValueError("only untouched unbound matching family slots may be replaced")
    inventory_path = root / "generated/curation_review_inventory.private.json"
    acquisition_path = root / "generated/acquisition_report.json"
    reviews_path = root / "data/private/curation_decisions.private.json"
    bindings = {Path(b["path"]).resolve() for b in load_json(candidate_path)["source_bindings"]}
    supplementary = supplementary_acquisition is not None
    if supplementary:
        acquisition_path = Path(supplementary_acquisition).resolve()
        base = root / ".runtime/supplementary-acquisition"
        if not acquisition_path.is_relative_to(base) or load_json(acquisition_path).get("complete") is not True:
            raise ValueError("completed project-local supplementary acquisition required")
        original = load_json(root / "config/mining_shortlist_v0.1.json")
        original_scenes = {s for group in original["groups"].values() for s in group}
        if candidate["scene_id_private"].removeprefix("clipgt-") in original_scenes:
            raise ValueError("supplementary inputs cannot bypass original scene review bindings")
        inventories = [p for p in bindings if p.is_relative_to(base) and p.suffix == ".json"
            and any(s.get("artifact_uuid") == candidate["artifact_uuid"]
                    and any(r.get("proposal_key") == key for r in s.get("proposals", []))
                    for s in load_json(p).get("scenes", []))]
        if len(inventories) != 1:
            raise ValueError("one source-bound supplementary proposal inventory required")
        inventory_path = inventories[0]
        from .curation_review_server import additional_review_inputs
        for entry in additional_review_inputs(root).values():
            decisions = entry["decisions_path"]
            if decisions.exists() and load_json(decisions)["decisions"].get(key, {}).get("decision") == "reject":
                raise ValueError("human-rejected supplementary opportunity cannot be reallocated")
    elif not {inventory_path, acquisition_path, reviews_path}.issubset(bindings):
        raise ValueError("bound original inventory, acquisition and review records required")
    if load_json(reviews_path)["decisions"].get(key, {}).get("decision") == "reject":
        raise ValueError("human-rejected opportunity cannot be reallocated")
    proposals = [(s, p) for s in load_json(inventory_path)["scenes"] for p in s["proposals"] if p["proposal_key"] == key]
    assets = [a["inspection"] for a in load_json(acquisition_path)["artifacts"] if a["artifact_uuid"] == candidate["artifact_uuid"]]
    if len(proposals) != 1 or len(assets) != 1:
        raise ValueError("unique original proposal and artifact required")
    scene, proposal = proposals[0]
    asset = assets[0]
    if supplementary and not all(asset.get(k) is True for k in
                                ("readable_usdz", "identity_matches_catalog", "required_members_present")):
        raise ValueError("verified complete supplementary asset required")
    execution = candidate["scene_execution_proposal_not_frozen"]
    path = Path(execution["artifact_path"]).resolve()
    pair = candidate["selected_arm_pair"]
    metadata = asset["metadata"]
    if (scene["artifact_uuid"] != candidate["artifact_uuid"] or scene["scene_id_private"] != candidate["scene_id_private"]
        or metadata["scene_id"] != candidate["scene_id_private"] or asset["errors"] or not asset["identity_matches_catalog"]
        or path not in bindings or not path.is_relative_to(root) or path != Path(asset["path"]).resolve()
        or path.stat().st_size != asset["size_bytes"] or pair not in arm_pair_candidates(proposal)
        or pair["kind"] not in pair_kinds
        or candidate["lane_geometry_private_not_qualified"] != proposal["topology_private"]["lane_geometry_private_not_qualified"]
        or execution["recording_start_timestamp_us"] != metadata["start_timestamp_us"]
        or execution["recording_end_timestamp_us"] != metadata["end_timestamp_us"]
        or execution["start_timestamp_us"] != metadata["start_timestamp_us"] + execution["start_time_offset_us"]
        or execution["start_time_offset_us"] < 0 or execution["start_timestamp_us"] + 13000000 >= metadata["end_timestamp_us"]):
        raise ValueError("replacement geometry, asset or real recording window mismatch")
    preview_replacement = None
    for field in ("evidence_sheet", "map_evidence_svg"):
        evidence = (root / proposal[field]).resolve()
        if not evidence.is_relative_to(root) or not evidence.is_file():
            if (field != "evidence_sheet" or actual_preview_run is None or not supplementary
                or task_family != "B1_segment_speed_band" or proposal.get("evidence_rendered") is not False
                or proposal.get("evidence_error") != "preview_video_not_found"):
                raise ValueError("original local review evidence required")
            from .speed_preview_binding import bind_actual_speed_preview
            preview_replacement = bind_actual_speed_preview(root, candidate, actual_preview_run)
    if actual_preview_run is not None and preview_replacement is None:
        raise ValueError("actual preview alternative is only for explicitly missing recorded previews")
    fixed = {"draft_id", "task_family", "primary_layer", "arms", "required_bindings", "grounding_bindings",
             "family_binding_status", "qualified", "execution_ready", "final_human_review", "formal_witnesses", "coupled_seeds"}
    row = {k: deepcopy(old[k]) for k in fixed}
    row.update(source_proposal_key=key, scene_id_private=candidate["scene_id_private"], artifact_uuid=candidate["artifact_uuid"],
        artifact_path=str(path), artifact_size_bytes=path.stat().st_size,
        suggested_start_offset_us_not_frozen=execution["start_time_offset_us"],
        recording_start_timestamp_us=metadata["start_timestamp_us"], recording_duration_s=metadata["duration_s"],
        original_timing_prefilter_passed=proposal["timing_review"]["automatic_timing_prefilter_pass"],
        planning_horizon_13s_available_at_suggested_window=True, topology_pair_hypothesis=deepcopy(pair),
        evidence_sheet=proposal["evidence_sheet"], map_evidence_svg=proposal["map_evidence_svg"],
        initial_human_decision_reference=None, authoring_authority="user_authorized_agent_not_human_signoff")
    if preview_replacement is not None:
        row["missing_recorded_preview_evidence"] = proposal["evidence_sheet"]
        row["actual_preview_alternative_not_original_evidence"] = preview_replacement
        row["evidence_sheet"] = None
    result = deepcopy(batch)
    result["drafts"] = [row if r["draft_id"] == target_draft_id else r for r in result["drafts"]]
    result.setdefault("allocation_history", []).append({"replaced_row": deepcopy(old), "new_proposal_key": key,
        "source_candidate": str(Path(candidate_path).resolve()), "reason": reason,
        "supplementary_acquisition": supplementary,
        "source_evidence_bindings": [{"path": str(p), "size_bytes": p.stat().st_size,
                                      "mtime_ns": str(p.stat().st_mtime_ns)}
                                     for p in (inventory_path, acquisition_path, reviews_path)],
        "qualified": False, "human_reviewed": False})
    validate_draft_batch(result)
    return result


def audit_speed_draft(pcu, native, profile):
    from .compiler import compile_source
    validate_pcu(pcu, allow_pending_reviews=True)
    try:
        compile_source({"schema_version": SCHEMA_VERSION, "pcus": [pcu]})
    except SchemaError as error:
        blocker = str(error)
    else:
        raise ValueError("pending-review draft must not enter the formal compiler")
    results = []
    for actual in native["arms"]:
        arm = actual["arm_id"]
        rows = relabel_actual_spatial_rows(actual["normalized_actual_samples"], pcu["private_annotations"])
        own = verify_trace(pcu, arm, rows, pcu["private_annotations"], profile)
        other = verify_trace(pcu, "R1" if arm == "R0" else "R0", rows, pcu["private_annotations"], profile)
        results.append({"seed": actual["seed"], "arm_id": arm, "session_uuid": actual["session_uuid"],
                        "own": own.to_dict(), "opposing": other.to_dict()})
    ledger = {(r["seed"], r["arm_id"]) for r in results}
    seeds = {r["seed"] for r in results}
    passed = len(seeds) in (1, 3) and seeds.issubset(COUPLED_SEEDS) and len(results) == 2 * len(seeds) and len({r["session_uuid"] for r in results}) == len(results) and ledger == {(s, a) for s in seeds for a in ("R0", "R1")} and all(
        r["own"]["passed"] and not r["opposing"]["passed"] for r in results)
    family_prefix = "B3" if pcu["task_family"] == "B3_triggered_speed_constraint" else "B1"
    return {"scope": "assembled_" + family_prefix + "_actual_contract_audit_not_qualification", "qualified": False,
            "all_own_pass_opposing_fail": passed, "results": results, "formal_compile_blocker": blocker,
            "actual_pilot_seeds": sorted(seeds), "three_seed_pilot_complete": len(seeds) == 3,
            "formal_witnesses_created": 0}


def assemble_speed_pcu(root, native_path, run_dirs, batch_path, target_draft_id, *, crosswalk_approach_path=None,
                       sign_approach_path=None, trigger_predecessor_lookback_depth=0):
    if (isinstance(trigger_predecessor_lookback_depth, bool)
        or not isinstance(trigger_predecessor_lookback_depth, int)
        or not 0 <= trigger_predecessor_lookback_depth <= 8):
        raise ValueError("trigger predecessor lookback depth must be an integer from 0 to 8")
    if crosswalk_approach_path is not None and sign_approach_path is not None:
        raise ValueError("crosswalk and sign authoring are mutually exclusive")
    root, native_path, batch_path = Path(root).resolve(), Path(native_path).resolve(), Path(batch_path).resolve()
    if (not native_path.is_relative_to(root / ".runtime/curation_review/spatial")
        or not batch_path.is_relative_to(root / "data/private/agent_drafts")
        or any(not Path(r).resolve().is_relative_to(root / ".runtime") for r in run_dirs)):
        raise ValueError("isolated project-local evidence required")
    policy = load_json(root / "config/curation_workflow.private.json")
    if (policy.get("mode") != "agent_authoring_then_final_human_review"
        or policy.get("agent_may_author_drafts") is not True
        or policy.get("require_final_human_review_before_release") is not True
        or policy.get("missing_evidence_may_be_marked_passed") is not False):
        raise ValueError("explicit delegated drafting authority required")
    if len(run_dirs) not in (1, 3):
        raise ValueError("one exploratory pair or three actual seed pairs required")
    summary = summarize_speed_pilot(run_dirs[0]) if len(run_dirs) == 1 else summarize_speed_seeds(run_dirs)
    native = load_json(native_path)
    if (native["scope"] != "actual_B1_map_segment_full_verifier_pilot_not_qualification"
        or native["candidate_binding"] != summary["candidate_binding"] or native["complete"] is not True
        or native["all_own_pass_opposing_fail"] is not True or native["qualified"] is not False
        or native["human_reviewed"] is not False or native["formal_witnesses_created"] != 0):
        raise ValueError("complete current unqualified native spatial evidence required")
    expected = {(p["seed"], a, r["session_uuid"]) for p in summary["pairs"] for a, r in p["arms"].items()}
    if (len(native["arms"]) != len(expected) or {(r["seed"], r["arm_id"], r["session_uuid"]) for r in native["arms"]} != expected
        or any(r["invalid_frame_times_s"] or r["own"]["passed"] is not True or r["opposing"]["passed"] is not False for r in native["arms"])):
        raise ValueError("exact actual passing session ledger required")
    # The road-level report reuses a stat-bound single-lane normalization cache.
    # Confirm that relabeling never changed actual poses, times or safety data.
    if "actual_cache_binding" in native:
        binding = native["actual_cache_binding"]
        path = Path(binding["path"]).resolve()
        if not path.is_relative_to(native_path.parent):
            raise ValueError("original actual cache must remain in the isolated spatial directory")
        stat = path.stat()
        if stat.st_size != binding["size_bytes"] or str(stat.st_mtime_ns) != binding["mtime_ns"]:
            raise ValueError("actual normalization cache changed")
        previous = load_json(path)
        if previous["candidate_binding"] != native["candidate_binding"]:
            raise ValueError("actual normalization cache belongs to another candidate")
        for row in native["arms"]:
            old = next(r for r in previous["arms"] if r["session_uuid"] == row["session_uuid"])
            expected_rows = relabel_actual_spatial_rows(old["normalized_actual_samples"], native["annotations_draft"])
            if row["normalized_actual_samples"] != expected_rows:
                raise ValueError("actual cache fields changed beyond geometry relabeling")
    first = Path(summary["pairs"][0]["run_dir"])
    timing = validate_speed_runtime_timing(first, load_json(first / "speed-pair-sidecar.private.json"))
    deadline_s = timing["rollout_duration_s"]
    if any(c.get("deadline_s") != deadline_s for c in native["contracts_draft"].values()):
        raise ValueError("actual speed scope deadline differs from prepared timing")
    candidate = load_json(first / "launch.private.json")["candidate"]
    if candidate.get("curation_authority"):
        from .agent_curation import validate_agent_runs
        if validate_agent_runs(root, run_dirs) != candidate:
            raise ValueError("delegated pilot candidate changed")
    batch, slot = move_unbound_opportunity(load_json(batch_path), candidate["proposal_key"], target_draft_id)
    approach_path = sign_approach_path if sign_approach_path is not None else crosswalk_approach_path
    family = "B3_triggered_speed_constraint" if approach_path is not None else "B1_segment_speed_band"
    if slot["task_family"] != family:
        raise ValueError("matching " + family + " slot required")
    approach, frame_refs, frame_bindings = None, None, None
    if "sign_scope_reference" in native and sign_approach_path is None:
        raise ValueError("preselected sign native evidence requires explicit B3 sign authoring")
    if sign_approach_path is not None:
        from .sign_authoring import validate_sign_authoring_inputs
        approach, frame_refs, frame_bindings = validate_sign_authoring_inputs(
            root, candidate, native, sign_approach_path, summary)
    if crosswalk_approach_path is not None:
        from .crosswalk_authoring import validate_crosswalk_authoring_inputs
        approach, frame_refs, frame_bindings = validate_crosswalk_authoring_inputs(
            root, candidate, native, crosswalk_approach_path, summary)
    ids = {k: k + "-" + uuid.uuid4().hex for k in ("pcu", "group", "trigger", "scope", "entry", "exit", "recording", "support")}
    annotations = deepcopy(native["annotations_draft"])
    annotations["scopes"] = {ids["scope"]: annotations["scopes"]["draft-speed-segment"]}
    annotations["gates"] = {ids[k]: annotations["gates"]["draft-" + k] for k in ("entry", "exit")}
    artifact = None
    if "source_map_binding" in native:
        from alpasim_utils.artifact import Artifact
        from .speed_spatial_draft import downstream_road_geometry
        binding = native["source_map_binding"]
        path = Path(binding["source_artifact"]).resolve()
        if (path != Path(candidate["scene_execution_proposal_not_frozen"]["artifact_path"]).resolve()
            or path.stat().st_size != binding["source_size_bytes"] or str(path.stat().st_mtime_ns) != binding["source_mtime_ns"]):
            raise ValueError("downstream original map source changed")
        artifact = Artifact(str(path))
        if artifact.metadata.uuid != candidate["artifact_uuid"] or artifact.metadata.scene_id != candidate["scene_id_private"]:
            raise ValueError("downstream original map identity differs")
        actual = downstream_road_geometry(artifact.map, candidate["selected_arm_pair"]["arms"][0]["lane_id"],
                                         native["source_lane_id"], binding["adjacent_lane_id"], candidate["selected_arm_pair"]["kind"])
        if any(binding[k] != v for k, v in actual.items()):
            raise ValueError("downstream geometry differs from the actual original map")
        geometry = actual["source_lane_geometry"]
    else:
        geometry = candidate["lane_geometry_private_not_qualified"][native["source_lane_id"]]
    annotations["triggers"] = {ids["trigger"]: deepcopy(geometry["lane_region_polygon_xy_not_qualified"])}
    for key in ("legality_reviewed", "balance_reviewed", "render_reviewed"):
        annotations[key] = False
    contracts = deepcopy(native["contracts_draft"])
    for c in contracts.values():
        c.update(id="contract-" + uuid.uuid4().hex, signature_set=["signature-" + uuid.uuid4().hex],
                 scope=ids["scope"], entry_gate=ids["entry"], exit_gate=ids["exit"])
    plans, scene = {}, candidate["scene_execution_proposal_not_frozen"]
    for pair in summary["pairs"]:
        sidecar = load_json(Path(pair["run_dir"]) / "speed-pair-sidecar.private.json")
        plans[str(pair["seed"])] = {r["arm_id"]: deepcopy(r["oracle_plan"]) for r in sidecar["sessions"].values()}
        for private in sidecar["sessions"].values():
            c = contracts[private["arm_id"]]
            if (private["diagnostic_speed_band_mps"] != [c["speed_min_mps"], c["speed_max_mps"]]
                or private["start_timestamp_us"] != scene["start_timestamp_us"]
                or private["instruction_issued_at_us"] != scene["start_timestamp_us"] + 1700000):
                raise ValueError("actual plan timing or speed band differs from contract")
    actual_seeds = sorted(int(s) for s in plans)
    first_plans = plans[str(actual_seeds[0])]
    for seed in COUPLED_SEEDS:
        if str(seed) not in plans:
            plans[str(seed)] = deepcopy(first_plans)  # Hypothesis only, never an unrun actual witness.
    prefix = [w for w in first_plans["R0"]["waypoints"] if w["time_s"] <= 1.7]
    from shapely.geometry import Point, Polygon
    issue_point = Point(prefix[-1]["x"], prefix[-1]["y"])
    trigger_history_audit = []
    if not Polygon(annotations["triggers"][ids["trigger"]]).covers(issue_point):
        from alpasim_utils.artifact import Artifact
        artifact = artifact or Artifact(scene["artifact_path"])
        if artifact.metadata.uuid != candidate["artifact_uuid"] or artifact.metadata.scene_id != candidate["scene_id_private"]:
            raise ValueError("trigger map artifact identity differs")
        # The scope can be several sections downstream of the observed prefix.
        # Its immediate predecessors need not have been observed yet, so history
        # cannot directly resolve a merge there. Anchor at the selected approach;
        # optional bounded lookback must prove a unique original incoming path
        # to the exact shared prefix. It never uses post-intervention trajectories.
        approach_lane = original_speed_trigger_anchor(candidate, native)
        trigger_lane, vertices = original_intervention_trigger(artifact.map, approach_lane, prefix,
            predecessor_lookback_depth=trigger_predecessor_lookback_depth,
            history_audit=trigger_history_audit)
        annotations["triggers"] = {ids["trigger"]: vertices}
    else:
        if trigger_predecessor_lookback_depth:
            raise ValueError("trigger predecessor lookback is unused: original scope already contains issue pose")
        trigger_lane = native["source_lane_id"]
    annotations["trigger_map_basis"] = {"original_lane_id": trigger_lane, "contains_instruction_rig_pose": True,
                                        "geometry_modified": False}
    if trigger_predecessor_lookback_depth:
        annotations["trigger_map_basis"].update(predecessor_lookback_depth=trigger_predecessor_lookback_depth,
            observed_history_interval_s=[0., 1.7], future_poses_used=False,
            original_incoming_path_audit=trigger_history_audit)
    utterances = {arm: speed_utterances(c["speed_min_mps"], c["speed_max_mps"]) for arm, c in contracts.items()}
    if approach is not None:
        from .crosswalk_authoring import crosswalk_speed_utterances
        from .sign_authoring import sign_speed_utterances
        is_sign = sign_approach_path is not None
        source_key = "source_sign" if is_sign else "source_crosswalk"
        map_id_key = "original_sign_map_id" if is_sign else "original_crosswalk_map_id"
        description = "Upcoming speed-limit sign; original ClipGT traffic sign " if is_sign else "Upcoming marked pedestrian crosswalk; original ClipGT crosswalk "
        make_utterances = sign_speed_utterances if is_sign else crosswalk_speed_utterances
        activation_id = "trigger-" + uuid.uuid4().hex
        annotations["triggers"][activation_id] = deepcopy(annotations["scopes"][ids["scope"]])
        annotations["trigger_referent"] = {
            "referent_id": "referent-" + uuid.uuid4().hex, "referent_kind": "landmark", "trigger_id": activation_id,
            "private_description": description + approach[source_key]["key"]["map_id"],
            "model_input_resolution": [320, 576], "frame_refs": frame_refs,
            "identity_reviewed": False, "visibility_reviewed": False, "trigger_geometry_reviewed": False,
            map_id_key: approach[source_key]["key"]["map_id"],
            "approach_distance_definition": approach["segment"]["distance_convention"],
            "far_distance_m": approach["segment"]["far_distance_m"], "near_distance_m": approach["segment"]["near_distance_m"],
        }
        for arm, c in contracts.items():
            c.update(kind="triggered_speed_band", trigger_id=activation_id)
            utterances[arm] = make_utterances(c["speed_min_mps"], c["speed_max_mps"],
                approach["segment"]["far_distance_m"], approach["segment"]["near_distance_m"])
    inspection = load_json(first / "speed-pair-inspection.private.json")
    p = inspection["arms"]["R0"]["prefix"]
    pcu = {"schema_version": SCHEMA_VERSION, "pcu_id": ids["pcu"], "meta_group_id": ids["group"],
        "backend": {"type": "alpasim", "alpasim_commit": ALPASIM_COMMIT, "renderer_type": "nurec", "dataset_version": "26.01"},
        "primary_layer": "B", "task_family": family,
        "scene": {"artifact_id": candidate["artifact_uuid"], "scene_id_private": candidate["scene_id_private"],
            "recording_id_private": ids["recording"], **{k: scene[k] for k in ("recording_start_timestamp_us", "start_timestamp_us", "start_time_offset_us")},
            "initial_pose": {k: prefix[0][k] for k in ("x", "y", "z", "yaw_rad")},
            "camera_configuration": {"logical_ids": p["preintervention_camera_logical_ids"], "height": 320, "width": 576},
            "reconstruction_support_id": ids["support"]},
        "shared_non_language_input": {"route_envelope": [[w["x"], w["y"]] for w in prefix], "route_envelope_kind": "COMMON_SUPPORT",
            "route_semantic_target": None, "route_neutrality_reviewed": False, "command_policy": "UNKNOWN_NEUTRAL",
            "coupled_exogenous_seeds": list(COUPLED_SEEDS), "preintervention_history_id": p["preintervention_non_language_sha256"],
            "simulator_configuration": {"renderer": {"kind": "sensorsim", "implementation": "NuRec"}, "route_use_waypoint_commands": False}},
        "decision_window": {"trigger": ids["trigger"], "start_s": 1.7, "deadline_s": deadline_s, "commitment_gates": []},
        "hard_constraints": {"collision": False, "offroad": False, "legality": True, "minimum_progress_m": 10., "deadline_s": deadline_s},
        "arms": {a: {"instruction_schedule": [{"issue_at_us": scene["start_timestamp_us"] + 1700000,
                      "utterances": utterances[a]}],
                     "semantic_contract": c, "opposing_contract": deepcopy(contracts["R1" if a == "R0" else "R0"])} for a, c in contracts.items()},
        "private_annotations": annotations, "oracle_plans_by_seed": plans}
    audit = audit_speed_draft(pcu, native, load_json(root / "config/threshold_profile.json"))
    if not audit["all_own_pass_opposing_fail"]:
        raise ValueError("assembled speed contracts fail on actual evidence")
    slot.update(family_binding_status="source_shaped_" + family[:2] + "_draft_pending_final_review",
                authored_start_offset_us_not_frozen=scene["start_time_offset_us"], arms=deepcopy(pcu["arms"]),
                speed_hypothesis_note="same road and entry/exit gates; topology lane pair is provenance, not a lane-change instruction")
    validate_draft_batch(batch)
    draft = {"scope": "source_shaped_agent_pcu_draft_pending_review_not_qualified", "pcu": pcu,
        "source_draft_id": slot["draft_id"], "proposal_key": candidate["proposal_key"], "qualified": False,
        "final_human_review": "pending", "formal_witnesses_created": 0, "actual_pilot_seeds": actual_seeds,
        "plan_seed_status": {str(s): "actual_exploratory_plan_not_formal_witness" if s in actual_seeds else
                             "deterministic_plan_hypothesis_not_seed_validation" for s in COUPLED_SEEDS},
        "structural_validation_passed_with_pending_reviews": True, "assembled_actual_contract_audit": audit,
        "evidence_paths": [str(native_path)], "actual_seed_summary": summary, "speed_limit_basis": native.get("speed_limit_basis"),
        "remaining": ["visible shared road-segment language/scope review", "posted/applicable speed and rail-approach legality evidence",
                      "final route/balance/render review", "source freeze", "fresh formal oracle qualification", "model evaluation"]}
    if approach is not None:
        draft["sign_approach_evidence" if sign_approach_path is not None else "crosswalk_approach_evidence"] = approach
        draft["referent_frame_bindings"] = frame_bindings
        draft["evidence_paths"].append(str(Path(approach_path).resolve()))
        label = "speed-limit sign" if sign_approach_path is not None else "crosswalk"
        draft["remaining"] += ["fixed " + label + " identity and visual grounding review", "object-relative approach distance language review"]
    return batch, draft
