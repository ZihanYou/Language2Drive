"""Prepare evaluator-private feasibility work from explicit human arm reviews.

No source PCUs, eligibility flags, safety claims, or oracle witnesses are
created here. The queue preserves selected arms and keeps missing evidence
explicit so topology-family suggestions cannot inflate release coverage.
"""

from collections import Counter
import math
from pathlib import Path

from .constants import COUPLED_SEEDS, FAMILY_TARGETS, NUREC_VERSION_STRING
from .curation_review_server import ReviewStore, additional_review_inputs
from .io import load_json


def _geometry_errors(geometry, arms):
    errors = []
    for arm in arms:
        lane = arm["lane_id"]
        row = geometry.get(lane, {})
        center = row.get("centerline_xyz", [])
        if len(center) < 2 or any(
            not isinstance(point, list) or len(point) < 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in point[:3])
            for point in center
        ):
            errors.append("invalid_lane_centerline:" + lane)
        if not row.get("lane_region_polygon_xy_not_qualified"):
            errors.append("missing_lane_region:" + lane)
    return errors


def _compatible_families(kind):
    # Compatibility is an upper bound, not qualified or frozen family coverage.
    if kind in {"keep_vs_left", "keep_vs_right"}:
        return ["A5_lane_change_keep", "B2_scoped_lane_preference"]
    if kind == "left_vs_right":
        return ["A4_lane_change_left_right", "B2_scoped_lane_preference"]
    if kind == "recorded_vs_counterfactual_branch":
        return ["A1_intersection_left_right", "A2_straight_turn", "A3_fork_left_right"]
    return []


def prepare_feasibility(root, inventory_path=None, decisions_path=None, acquisition_path=None):
    root = Path(root).resolve()
    inventory_path = inventory_path or root / "generated/curation_review_inventory.private.json"
    decisions_path = decisions_path or root / "data/private/curation_decisions.private.json"
    inventory = load_json(inventory_path)
    # An absent log means no submissions, not permission to invent approvals.
    log = (load_json(decisions_path) if Path(decisions_path).exists() else
           {"schema_version": "l2d.realsim.curation_decisions.v0.1", "decisions": {}})
    acquisition = load_json(acquisition_path or root / "generated/acquisition_report.json")
    artifacts = {}
    for artifact in acquisition["artifacts"]:
        uuid = artifact["artifact_uuid"]
        if uuid in artifacts:
            raise ValueError("duplicate acquisition artifact: " + uuid)
        artifacts[uuid] = artifact
    store = ReviewStore(inventory_path, decisions_path)
    if log.get("schema_version") != "l2d.realsim.curation_decisions.v0.1":
        raise ValueError("unsupported review log schema")
    decisions = log["decisions"]
    proposals = {}
    for scene in inventory["scenes"]:
        for proposal in scene["proposals"]:
            key = proposal["proposal_key"]
            if key in proposals:
                raise ValueError("duplicate proposal key: " + key)
            proposals[key] = (scene, proposal)
    queue, invalid, deferred = [], [], []
    counts = Counter()
    for key, review in sorted(decisions.items()):
        try:
            normalized = store.validate_review(key, review)
            scene, proposal = proposals[key]
            for field, expected in (
                ("proposal_key", key), ("artifact_uuid", scene["artifact_uuid"]),
                ("scene_id_private", scene["scene_id_private"]),
                ("reviewed_arm_pair", normalized["reviewed_arm_pair"]),
                ("evidence_sheet", proposal["evidence_sheet"]),
                ("map_evidence_svg", proposal["map_evidence_svg"]),
            ):
                if review.get(field) != expected:
                    raise ValueError("review_inventory_mismatch:" + field)
            revision = review.get("revision")
            if type(revision) is not int or revision < 1 or not review.get("reviewed_at"):
                raise ValueError("missing_review_revision_or_timestamp")
        except (ValueError, TypeError, KeyError) as error:
            invalid.append({"proposal_key": key, "error": str(error)})
            continue
        counts[review["decision"]] += 1
        if review["decision"] != "keep":
            deferred.append({"proposal_key": key, "decision": review["decision"],
                             "notes": review["notes"], "rejection_reasons": review["rejection_reasons"]})
            continue
        pair = normalized["reviewed_arm_pair"]
        geometry = proposal["topology_private"].get("lane_geometry_private_not_qualified", {})
        errors = _geometry_errors(geometry, pair["arms"])
        artifact = artifacts.get(scene["artifact_uuid"], {})
        inspection = artifact.get("inspection", {})
        metadata = inspection.get("metadata", {})
        artifact_path = Path(inspection.get("path", ""))
        if (artifact.get("scene_id_private") != scene["scene_id_private"]
            or any(inspection.get(field) is not True for field in
                   ("readable_usdz", "required_members_present", "identity_matches_catalog"))
            or inspection.get("errors")
            or metadata.get("artifact_uuid") != scene["artifact_uuid"]
            or metadata.get("scene_id") != scene["scene_id_private"]
            or metadata.get("nre_version") != NUREC_VERSION_STRING
            or not artifact_path.is_file()
            or artifact_path.stat().st_size != inspection.get("size_bytes")):
            errors.append("artifact_missing_changed_or_unverified")
        offset = proposal["suggested_start_time_offset_us_not_qualified"]
        recording_start = metadata.get("start_timestamp_us")
        recording_end = metadata.get("end_timestamp_us")
        if (type(recording_start) is not int or type(recording_end) is not int
            or type(offset) is not int or offset < 0
            or recording_start + offset >= recording_end):
            errors.append("invalid_scene_time_binding")
        for field in ("evidence_sheet", "map_evidence_svg"):
            path = (root / proposal[field]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                errors.append("evidence_missing_or_outside_project:" + field)
        requirements = ["dual_arm_nre_render_support", "safe_oracle_paths_by_seed",
                        "shared_initial_state_and_prefix", "language_and_contract_freeze",
                        "six_real_oracle_rollouts"]
        if proposal["proposal_type"] == "route_branch":
            requirements += ["branch_direction_classification", "commitment_gates_and_deadline"]
        else:
            requirements += ["shared_scope_and_stable_lane_dwell"]
        queue.append({
            "proposal_key": key, "artifact_uuid": scene["artifact_uuid"],
            "scene_id_private": scene["scene_id_private"],
            "scene_execution_proposal_not_frozen": {
                "artifact_path": inspection.get("path"),
                "nre_version": metadata.get("nre_version"),
                "recording_start_timestamp_us": recording_start,
                "start_time_offset_us": offset,
                "start_timestamp_us": recording_start + offset if type(recording_start) is int else None,
                "recording_end_timestamp_us": recording_end,
            },
            "review_revision": revision, "reviewer": review["reviewer"],
            "reviewed_at": review["reviewed_at"], "selected_arm_pair": pair,
            "state": "INPUT_REPAIR_REQUIRED" if errors else "NRE_FEASIBILITY_REQUIRED",
            "input_errors": errors,
            "compatible_families_not_qualified": _compatible_families(pair["kind"]),
            "requirements_remaining": requirements,
            "decision_offset_us": proposal["decision_offset_us"],
            "suggested_start_time_offset_us_not_qualified": proposal["suggested_start_time_offset_us_not_qualified"],
            "timing_review": proposal["timing_review"],
            "evidence_sheet": proposal["evidence_sheet"], "map_evidence_svg": proposal["map_evidence_svg"],
            "lane_geometry_private_not_qualified": {
                lane: geometry[lane] for lane in dict.fromkeys(
                    [proposal["lane_id_private"]] + [arm["lane_id"] for arm in pair["arms"]]
                ) if lane in geometry
            },
            "required_oracle_runs": [{"arm": arm, "seed": seed}
                                     for seed in COUPLED_SEEDS for arm in ("R0", "R1")],
        })
    queue.sort(key=lambda row: (bool(row["input_errors"]),
                               -min(row["timing_review"]["pre_decision_context_s"],
                                    row["timing_review"]["post_decision_context_s"]),
                               row["proposal_key"]))
    family_counts = Counter(f for row in queue if not row["input_errors"]
                            for f in row["compatible_families_not_qualified"])
    survivors = {key for key, (_, proposal) in proposals.items()
                 if proposal["timing_review"]["automatic_timing_prefilter_pass"]}
    summary = {
        "timing_survivor_count": len(survivors), "submitted_count": len(decisions),
        "unreviewed_count": len(survivors - decisions.keys()),
        "valid_review_counts": dict(counts), "invalid_review_count": len(invalid),
        "feasibility_candidate_count": len(queue),
        "input_ready_count": sum(not row["input_errors"] for row in queue),
        "selected_pair_kinds": dict(Counter(row["selected_arm_pair"]["kind"] for row in queue)),
        "qualified_pcu_count": 0,
    }
    return {
        "schema_version": "l2d.realsim.feasibility_queue.private.v0.1",
        "scope": "reviewed_pair_feasibility_work_queue_not_pcu_source",
        "summary": summary, "invalid_reviews": invalid, "deferred_reviews": deferred,
        "family_capacity_upper_bounds_not_qualified": {
            family: {"required": target, "compatible_candidates": family_counts[family],
                     "minimum_additional_candidates": max(0, target - family_counts[family])}
            for family, target in FAMILY_TARGETS.items()
        },
        "candidates": queue,
    }


def prepare_additional_feasibility(root, batch):
    inputs = additional_review_inputs(root)
    if batch not in inputs or "acquisition_path" not in inputs[batch]:
        raise ValueError("registered additional review batch and explicit acquisition report required")
    paths = inputs[batch]
    report = prepare_feasibility(root, **paths)
    report["review_batch"] = batch
    report["review_input_paths"] = {k: str(v) for k, v in paths.items()}
    return report
