"""PCU report checks, not a full benchmark or raw-evidence release certificate."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict

from .constants import ALPASIM_COMMIT, FAMILY_TARGETS, LAYER_TARGETS
from .cohort_integrity import audit_cohort_integrity
from .io import stable_hash
from .family_allocation import family_targets
from .release_profile import release_layer_targets


def provenance_integrity_errors(provenance: Dict[str, Any]) -> list[str]:
    """Check the small manifest's integrity, not execution-time authenticity.

    Even a correctly self-bound manifest can describe the wrong/current tree.
    No weights, ASLs or source trees are read by this check.
    """
    errors = []
    if provenance.get("schema_version") != "l2d.realsim.provenance.v0.1":
        errors.append("unsupported provenance schema")
    for field in ("language2drive_git_sha", "alpasim_git_sha"):
        if not isinstance(provenance.get(field), str) or not re.fullmatch(r"[0-9a-f]{40}", provenance[field]):
            errors.append("malformed provenance field: " + field)
    for field in (
        "binding_hash", "alpasim_config_hash", "pcu_source_hash",
        "private_sidecar_hash", "verifier_source_hash", "threshold_profile_hash",
        "oracle_source_hash", "scene_candidate_inventory_hash",
        "curation_review_inventory_hash", "qualification_report_hash", "code_tree_hash",
    ):
        if not isinstance(provenance.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", provenance[field]):
            errors.append("malformed provenance field: " + field)
    if provenance.get("alpasim_git_sha") != ALPASIM_COMMIT:
        errors.append("AlpaSim revision differs from pin")
    try:
        expected = stable_hash({key: value for key, value in provenance.items() if key != "binding_hash"})
    except (TypeError, ValueError):
        expected = None
    if expected is None or provenance.get("binding_hash") != expected:
        errors.append("provenance self-binding mismatch")
    return errors


def final_report(
    source: Dict[str, Any],
    qualification: Dict[str, Any],
    leakage: Dict[str, Any],
    witness_index: Dict[str, Any],
    verifier_report: Dict[str, Any],
    provenance: Dict[str, Any],
    inspect_report: Dict[str, Any],
    candidate_inventory: Dict[str, Any] = None,
    curation_review: Dict[str, Any] = None,
) -> Dict[str, Any]:
    pcus = source.get("pcus", [])
    eligible_rows = [row for row in qualification.get("pcus", []) if row.get("eligible") is True]
    layer_counts = Counter(row.get("primary_layer") for row in eligible_rows)
    family_counts = Counter(row.get("task_family") for row in eligible_rows)
    scene_count = len({pcu.get("scene", {}).get("artifact_id") for pcu in pcus if pcu.get("scene")})
    unresolved_invalid = sum(
        witness.get("own_verdict", {}).get("failure") == "INVALID_SIM"
        for witness in witness_index.get("witnesses", [])
        if isinstance(witness.get("own_verdict"), dict)
    )
    source_binding = stable_hash(source)
    allocation_error = None
    try:
        expected_families = family_targets(source)
        expected_layers = release_layer_targets(source)
    except ValueError as error:
        expected_families = dict(FAMILY_TARGETS)
        expected_layers = dict(LAYER_TARGETS)
        allocation_error = str(error)
    expected_pcus = sum(expected_layers.values())
    cohort = audit_cohort_integrity(source, qualification, witness_index)
    provenance_errors = provenance_integrity_errors(provenance)
    gates = {
        f"pcus_{expected_pcus}_of_{expected_pcus}_eligible": len(eligible_rows) == expected_pcus and len(pcus) == expected_pcus,
        **{f"layer_{layer}_{count}_of_{count}": layer_counts.get(layer, 0) == count for layer, count in expected_layers.items()},
        "task_family_allocation_valid": allocation_error is None,
        "task_family_distribution": Counter(family_counts) == Counter(expected_families),
        "two_contrast_arms_each": all(set(pcu.get("arms", {})) == {"R0", "R1"} for pcu in pcus) and len(pcus) == expected_pcus,
        "coupled_seeds_at_least_3": all(len(pcu.get("shared_non_language_input", {}).get("coupled_exogenous_seeds", [])) >= 3 for pcu in pcus) and len(pcus) == expected_pcus,
        "cohort_report_integrity": cohort["structural_pass"],
        "qualification_global_gates": qualification.get("pass") is True
        and qualification.get("all_global_gates_pass") is True
        and qualification.get("global_errors") == [],
        "oracle_witness_coverage_100_percent": cohort["structural_pass"],
        "route_leakage_audit": leakage.get("pass") is True,
        "paired_input_audit": leakage.get("pass") is True,
        "verifier_tests": verifier_report.get("pass") is True,
        "reconstruction_render_validity": all(
            witness.get("render_diagnostics", {}).get("valid") is True
            for witness in witness_index.get("witnesses", [])
        ) and cohort["structural_pass"],
        "unresolved_invalid_sim_zero": unresolved_invalid == 0,
        "missing_semantic_contracts_zero": all(
            all(arm.get("semantic_contract") for arm in pcu.get("arms", {}).values()) for pcu in pcus
        ) and len(pcus) == expected_pcus,
        "missing_opposing_contracts_zero": all(
            all(arm.get("opposing_contract") for arm in pcu.get("arms", {}).values()) for pcu in pcus
        ) and len(pcus) == expected_pcus,
        "evidence_source_bindings_match": all(
            report.get("source_binding") == source_binding
            for report in (qualification, leakage, witness_index)
        ),
        "missing_provenance_binding_zero": not provenance_errors,
        "real_alpasim_execution_environment": (
            inspect_report.get("schema_version") == "l2d.realsim.inspect.v0.1"
            and inspect_report.get("alpasim_git_sha") == ALPASIM_COMMIT
            and inspect_report.get("oracle_execution_ready") is True
            and inspect_report.get("release_execution_ready") is True
        ),
    }
    pcu_report_checks_passed = all(gates.values())
    blockers = [name for name, passed in gates.items() if not passed]
    blockers.extend(cohort["errors"])
    blockers.extend(provenance_errors)
    blockers.extend(str(error) for error in (qualification.get("global_errors") or []))
    if allocation_error:
        blockers.append(allocation_error)
    candidate_inventory = candidate_inventory or {}
    rejected_scene_candidates = [
        row
        for row in candidate_inventory.get("candidates", [])
        if row.get("automatic_prefilter_pass") is not True
    ]
    scene_rejection_reasons = Counter(
        reason
        for row in rejected_scene_candidates
        for reason in row.get("rejection_reasons", [])
    )
    curation_review = curation_review or {}
    opportunity_rows = [
        proposal
        for scene in curation_review.get("scenes", [])
        for proposal in scene.get("proposals", [])
    ]
    timing_rejected = [
        proposal
        for proposal in opportunity_rows
        if proposal.get("timing_review", {}).get(
            "automatic_timing_prefilter_pass"
        )
        is not True
    ]
    opportunity_rejection_reasons = Counter(
        reason
        for proposal in timing_rejected
        for reason in proposal.get("timing_review", {}).get(
            "automatic_rejection_reasons", []
        )
    )
    if opportunity_rows:
        discovered_candidates = len(opportunity_rows)
        rejected_candidates = len(timing_rejected)
        pending_candidates = len(opportunity_rows) - len(timing_rejected)
        rejection_reasons = dict(opportunity_rejection_reasons)
    else:
        # Compatibility for pre-curation reports. These are scene-level rows,
        # not claims about executable PCU opportunities.
        discovered_candidates = candidate_inventory.get("candidate_count", 0)
        rejected_candidates = len(rejected_scene_candidates)
        pending_candidates = 0
        rejection_reasons = dict(scene_rejection_reasons)
    return {
        "schema_version": "l2d.realsim.final_report.v0.2",
        "status": "PCU-REPORT-CHECKS-PASSED" if pcu_report_checks_passed else "NOT RELEASE-READY",
        "scope": "pcu_cohort_and_oracle_report_checks_only",
        "pcu_report_checks_passed": pcu_report_checks_passed,
        "release_ready": False,
        "full_benchmark_release": False,
        "model_evaluation_assessed": False,
        "raw_trace_authenticity_verified": False,
        "execution_provenance_verified": False,
        "provenance_manifest_binding": provenance.get("binding_hash"),
        "cohort_integrity": cohort,
        "eligible_pcus": len(eligible_rows),
        "source_pcus": len(pcus),
        "layer_counts": dict(layer_counts),
        "task_family_counts": dict(family_counts),
        "task_family_targets": expected_families,
        "target_pcu_count": expected_pcus,
        "layer_targets": expected_layers,
        "release_profile": source.get("release_profile"),
        "c_family_reallocation": source.get("c_family_reallocation"),
        "b4_family_reallocation": source.get("b4_family_reallocation"),
        "independent_reconstructed_scenes": scene_count,
        "discovered_candidates": discovered_candidates,
        "candidate_scope": (
            "route_aligned_pcu_opportunity"
            if opportunity_rows
            else candidate_inventory.get("candidate_scope", "legacy_scene_candidate")
        ),
        "rejected_candidates": rejected_candidates,
        "pending_curator_candidates": pending_candidates,
        "candidate_rejection_reasons": rejection_reasons,
        "discovered_scene_candidates": candidate_inventory.get(
            "candidate_count", 0
        ),
        "rejected_scene_candidates": len(rejected_scene_candidates),
        "scene_candidate_rejection_reasons": dict(scene_rejection_reasons),
        "acceptance_gates": gates,
        "blockers": sorted(set(blockers)),
        "unassessed_release_requirements": [
            "complete_registered_model_campaign_and_terminal_evidence",
            "raw_oracle_and_model_trace_authenticity",
            "execution_time_code_and_checkpoint_provenance",
            "complete_safe_legal_evidence",
            "publication_licenses_and_private_data_exclusion",
        ],
        "limitations": {
            "traffic_interaction_support": "Actor-relative verification is supported, but qualification requires stable private actor IDs on every seed.",
            "semantic_asset_availability": "Only curator-verified visible assets may enter Layer C.",
            "ocr_visibility": "Text must be reviewed at actual model input resolution; privileged metadata is insufficient.",
            "reconstructed_counterfactual_range": "Both arms must stay within frozen reconstruction support on every seed.",
            "crossed_grounding_coverage": "A/C use first irreversible registered-gate commitment; richer referent relations require new annotations.",
            "layer_D": "RealSim v0.1 reserves an extension point but excludes D because multi-window program state and revision/memory qualification are not yet frozen.",
        },
    }
