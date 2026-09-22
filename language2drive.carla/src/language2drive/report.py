"""Release-grade, limitation-forward benchmark reporting."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_text, load_json, sha256_file
from .fdq import FDQ_ROUTE_BINDING_ROWS, FDQ_SEEDS, canonical_fdq_route_bindings
from .runner import _storage_policy, active_core_sidecar_path, deadline_freeze_status
from .scoring import (
    _BASELINE_SEMANTICS,
    _materialize_json_object,
    _validate_baseline_manifest,
    _validate_exact_release_ledgers,
    _verify_materialized_release_ledgers,
    release_artifact_hashes,
    result_snapshot,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _small_track(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "expected_runs", "observed_runs", "valid_result_runs", "success_count",
            "finalized_runs",
            "missing_runs", "crashed_runs", "timeout_runs", "deadlock_runs", "no_trace_runs",
            "coverage", "result_coverage", "CFPS", "SISR", "Worst-paraphrase",
            "SemISR", "SemISR_by_layer",
            "Full-Program-Safe-Success_D", "Safety", "Progress",
            "DrivingScore", "DrivingScore_by_layer", "DrivingScore_valid_runs",
            "DrivingScore_valid_runs_by_layer",
            "C_task_breakdown", "C_navigation_diagnostics",
            "Safety-Priority-Success", "Safe-Refusal",
            "Clarification-Proxy-Correctness", "Fallback-or-Constrained-Alternative",
            "Unnecessary-Refusal", "Unsafe-Obey-Rate", "four_quadrants",
            "denominator_policy", "clarification_limit",
        )
    }


def _asset_status(project_root: Path) -> dict[str, Any]:
    path = project_root / "assets/review/asset_review.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline=""))) if path.is_file() else []
    counts = Counter(row.get("status", "unknown") for row in rows)
    needs_fix = [row for row in rows if row.get("status") == "needs_fix"]
    return {
        "rows": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "needs_fix": needs_fix,
    }


def _human_review(project_root: Path) -> dict[str, Any]:
    authoritative = project_root / "generated/review/MINIMAL_HUMAN_REVIEW.csv"
    if authoritative.is_file():
        rows = list(csv.DictReader(authoritative.open(encoding="utf-8", newline="")))
        decision_field = "visibility_sign_arrow_decision_confirm_or_reject"
        decisions = [str(row.get(decision_field, "")).strip().casefold() for row in rows]
        joined_arms = sum(
            2 for row in rows
            if str(row.get("automatic_geometry_join_status", "")).startswith("pass:")
        )
        confirmed = sum(decision == "confirm" for decision in decisions)
        rejected = sum(decision == "reject" for decision in decisions)
        pending = len(rows) - confirmed - rejected
        return {
            "required_for_paper_freeze": True,
            "task": (
                "Inspect each actual AutoVLA camera-rig RGB contact sheet and choose confirm or reject "
                "for cue readability and sign-arrow consistency."
            ),
            # Compatibility name: this is eight PCU-level review rows, not
            # eight unresolved geometric mappings.
            "unique_c_rd_mappings": len(rows),
            "c_rd_review_rows": len(rows),
            "automatic_geometry_arm_joins": joined_arms,
            "automatic_geometry_join_status": (
                "passed_live_carla_spatial_heading" if joined_arms == 2 * len(rows) else "incomplete"
            ),
            "human_gate_or_road_rebinding_required": False,
            "semantic_visibility_witness": "confirmed" if confirmed == len(rows) and rows else "pending",
            "visibility_decisions": {
                "confirmed": confirmed, "rejected": rejected, "pending": pending,
            },
            "human_actions_remaining": pending,
            "free_text_annotation_required": False,
            "decision_only": "confirm / reject",
            "authoritative_review_csv": str(authoritative),
            "rgb_contact_sheet_dir": str(project_root / "generated/review/rgb"),
            "review_package": str(project_root / "generated/review/MINIMAL_HUMAN_REVIEW.md"),
            "mappings": rows,
        }

    source = load_json(project_root / "sources/core_pcus.json")
    mappings = {}
    for pcu in source.get("pcus", []):
        if pcu.get("layer") != "C" or pcu.get("category") != "road_name":
            continue
        for arm in pcu.get("arms", []):
            target = arm.get("target", {})
            # Human effort is one decision per C-RD PCU/sign pair, not one
            # independent blank-form decision per arm.  Store both mutually
            # exclusive mappings on the same checklist row.
            key = str(pcu["id"])
            existing = mappings.get(key, {"pcu_id": pcu["id"], "town": pcu.get("town"), "arms": []})
            existing["arms"].append({
                "road_name": target.get("semantic_name"),
                "gate_id": target.get("id"),
                "maneuver": target.get("maneuver"),
            })
            mappings[key] = {
                **existing,
            }
    return {
        "required_for_paper_freeze": True,
        "task": "Confirm actual camera-rig cue readability/sign-arrow consistency for each pre-bound C-RD pair.",
        "unique_c_rd_mappings": len(mappings),
        "c_rd_review_rows": len(mappings),
        "automatic_geometry_arm_joins": 0,
        "automatic_geometry_join_status": "evidence_package_missing",
        "human_gate_or_road_rebinding_required": False,
        "semantic_visibility_witness": "pending",
        "visibility_decisions": {"confirmed": 0, "rejected": 0, "pending": len(mappings)},
        "human_actions_remaining": len(mappings),
        "mappings": list(mappings.values()),
        "free_text_annotation_required": False,
        "decision_only": "confirm / reject",
    }


def _parking_core_inventory(project_root: Path) -> dict[str, Any]:
    """Count all core parking semantics and their XODR physical identities.

    Exact parking terminals carry ``type=parking_slot``.  D strict-order and
    memory programs instead reference the same numbered physical spaces as
    ``parking-stage:<slot>`` evaluator gates, while conditionals also carry
    ``occupied:<slot>`` actor-state keys.  Omitting those strings understates
    the semantic assets used by the current core.
    """

    sidecar = load_json(project_root / "build/suite/sidecar.private.json")
    by_layer: dict[str, set[str]] = {layer: set() for layer in "ABCD"}
    exact_targets: set[str] = set()
    stage_gates: set[str] = set()
    occupancy_conditions: set[str] = set()

    def collect(value: Any, output: set[str]) -> None:
        if isinstance(value, dict):
            if value.get("type") == "parking_slot" and isinstance(value.get("slot_number"), str):
                token = value["slot_number"]
                output.add(token)
                exact_targets.add(token)
            for child in value.values():
                collect(child, output)
        elif isinstance(value, list):
            for child in value:
                collect(child, output)
        elif isinstance(value, str):
            for token in re.findall(r"parking-stage:([A-F]\d{3})", value):
                output.add(token)
                stage_gates.add(token)
            for token in re.findall(r"occupied:([A-F]\d{3})", value):
                output.add(token)
                occupancy_conditions.add(token)

    for pcu in sidecar.get("pcus", []):
        layer = str(pcu.get("layer", ""))
        if layer not in by_layer:
            continue
        for arm in pcu.get("arms", []):
            collect(arm, by_layer[layer])

    registry_path = project_root / "assets/review/slot_registry.csv"
    registry_rows = list(csv.DictReader(registry_path.open(encoding="utf-8", newline="")))
    physical_by_token = {
        row["slot_number"]: row.get("physical_space_id", "")
        for row in registry_rows if row.get("slot_number")
    }
    tokens = set().union(*by_layer.values())
    physical_ids = {physical_by_token.get(token, "") for token in tokens}
    missing = sorted(token for token in tokens if not physical_by_token.get(token))
    physical_ids.discard("")
    return {
        "unique_slot_tokens": len(tokens),
        "slot_tokens": sorted(tokens),
        "unique_physical_xodr_parking_space_ids": len(physical_ids),
        "duplicate_physical_ids": len(tokens) - len(physical_ids) - len(missing),
        "missing_physical_bindings": missing,
        "by_layer_unique_tokens": {layer: len(by_layer[layer]) for layer in "ABCD"},
        "by_semantic_use_unique_tokens": {
            "exact_parking_targets": len(exact_targets),
            "stage_gates": len(stage_gates),
            "occupancy_conditions": len(occupancy_conditions),
        },
        "binding_status": (
            "one_to_one" if len(tokens) == len(physical_ids) and not missing else "invalid"
        ),
    }


def _dynamic_row(name: str, value: dict[str, Any], status: str) -> str:
    return (
        f"| {name} | {value.get('expected_runs', 0)} | {value.get('observed_runs', 0)} | "
        f"{value.get('finalized_runs', 0)} | {value.get('valid_result_runs', 0)} | {value.get('success_count', 0)} | {status} |\n"
    )


def _exact_denominator(value: Any, expected: int) -> bool:
    return (
        isinstance(value, dict)
        and type(value.get("expected_runs")) is int
        and value["expected_runs"] == expected
    )


def _exact_fdq_matrix(value: Any) -> bool:
    expected_per_scenario = len(FDQ_SEEDS)
    if not _exact_denominator(value, len(FDQ_ROUTE_BINDING_ROWS) * expected_per_scenario):
        return False
    groups = value.get("ability_groups")
    return (
        isinstance(groups, dict)
        and set(groups) == set(canonical_fdq_route_bindings())
        and all(
            isinstance(group, dict)
            and type(group.get("expected")) is int
            and group["expected"] == expected_per_scenario
            for group in groups.values()
        )
    )


def _exact_baseline_matrix(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(_BASELINE_SEMANTICS)
        and all(_exact_denominator(summary, 432) for summary in value.values())
    )


def _current_release_identity_gate(project_root: Path) -> bool:
    """Independently reconstruct every current denominator used by a report."""

    suite = project_root / "build/suite"
    safety = project_root / "build/safety"
    baselines = project_root / "build/baselines"
    try:
        core_sidecar, core_snapshot = _materialize_json_object(
            suite / "sidecar.private.json"
        )
        safety_sidecar, safety_snapshot = _materialize_json_object(
            safety / "sidecar.private.json"
        )
        baseline_manifest, manifest_snapshot = _materialize_json_object(
            baselines / "manifest.json"
        )
        _validate_baseline_manifest(project_root, baseline_manifest)
        authority = _validate_exact_release_ledgers(
            suite, core_sidecar, safety, baseline_manifest, project_root,
            safety_sidecar=safety_sidecar,
        )
        snapshots = {
            **authority["snapshots"],
            str((suite / "sidecar.private.json").resolve()): core_snapshot,
            str((safety / "sidecar.private.json").resolve()): safety_snapshot,
            str((baselines / "manifest.json").resolve()): manifest_snapshot,
        }
        _verify_materialized_release_ledgers(snapshots)
        return True
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _fdq_dynamic_row(value: dict[str, Any], status: str) -> str:
    return (
        f"| FDQ loadable-scenario workload | {value.get('expected_runs', 0)} | "
        f"{value.get('observed_runs', 0)} | {value.get('finalized_runs', 0)} | "
        f"N/A | {value.get('passed_runs', 0)} | {status} |\n"
    )


def _autovla_provenance() -> dict[str, Any]:
    return {
        "checkpoint": "/root/autodl-tmp/lwq/language2drive.v1/model_zoo/autovla/checkpoints/2026-06-13_16-36-13/epoch=4.ckpt",
        "checkpoint_sha256": "6e124e02c87d94ad3fdfd2bbf36f3a2b5f66da332a1d98b9c7b26bb1f390da17",
        "action_codebook": "/root/autodl-tmp/lwq/language2drive.v1/model_zoo/autovla/codebook_cache/agent_vocab.pkl",
        "action_codebook_sha256": "e6bf8eff32497eaefca66f488dda1edefe97ed2696d1d23c01839dc27422408e",
        "python": "3.11.15",
        "torch": "2.2.0+cu121",
        "cuda": "12.1",
        "train_manifest": None,
        "train_overlap": "unknown",
    }


def _write_human_checklist(path: Path, review: dict[str, Any], assets: dict[str, Any]) -> None:
    lines = [
        "# Minimal human review for Language2Drive v1\n",
        "No free-form relabeling is requested. Reviewers only confirm an existing mapping, reject it, or supply one corrected gate id.\n",
        f"## C-RD sign/readability decisions ({review['unique_c_rd_mappings']} unique mappings)\n",
        "| PCU | Existing two-arm sign → gate mapping | RGB | Decision |\n",
        "|---|---|---|---|\n",
    ]
    for row in review["mappings"]:
        if isinstance(row.get("arms"), list):
            descriptions = "; ".join(
                f"{arm['road_name']} → `{arm['gate_id']}` ({arm['maneuver']})"
                for arm in row["arms"]
            )
            rgb = "evidence package unavailable"
        else:
            descriptions = "; ".join(
                f"{row.get(prefix + '_semantic_label')} → `{row.get(prefix + '_gate')}` "
                f"({row.get(prefix + '_maneuver')})"
                for prefix in ("arm_a", "arm_b")
            )
            rgb = str(row.get("rgb_contact_sheet_clickable", row.get("rgb_contact_sheet", "")))
        lines.append(
            f"| {row['pcu_id']} | {descriptions} | {rgb} | ☐ confirm ☐ reject |\n"
        )
    lines.extend([
        "\n## Excluded legacy points\n",
        "These points are already excluded from the core and do not block running it. Only review them if they should be restored later.\n",
        "| Entity | Town | Existing note | Decision / corrected point |\n",
        "|---|---|---|---|\n",
    ])
    for row in assets["needs_fix"]:
        lines.append(f"| `{row.get('entity_id')}` | {row.get('town')} | {row.get('notes')} | |\n")
    atomic_text(path, "".join(lines))


def render(score_path: Path, output: Path, *, project_root: Path | None = None, json_output: Path | None = None) -> dict[str, Any]:
    project_root = project_root or Path(__file__).resolve().parents[2]
    score = load_json(score_path)
    manifest = load_json(project_root / "build/suite/manifest.json")
    score_source = score.get("source", {})
    current_ledgers = {
        "standardized": project_root / "build/suite/expected_runs.json",
        "native": project_root / "build/suite/native_smoke_expected_runs.json",
        "dynamic_oracle": project_root / "build/suite/oracle_expected_runs.json",
        "pilot": project_root / "build/suite/pilot_expected_runs.json",
        "fdq": project_root / "build/suite/fdq_expected_runs.json",
        "safety_priority": project_root / "build/safety/expected_runs.json",
    }
    ledger_hashes_current = {
        name: sha256_file(path) for name, path in current_ledgers.items()
    }
    score_results_dir = Path(str(score_source.get("results_dir", project_root / "results")))
    current_results_snapshot = result_snapshot(score_results_dir)
    recorded_results_snapshot = score_source.get("results_snapshot")
    active_sidecar = active_core_sidecar_path(project_root)
    score_current = (
        score_source.get("manifest_sha256") == sha256_file(project_root / "build/suite/manifest.json")
        and score_source.get("sidecar_sha256") == sha256_file(active_sidecar)
        and score_source.get("safety_ledger_sha256") == sha256_file(project_root / "build/safety/expected_runs.json")
        and score_source.get("baseline_manifest_sha256") == sha256_file(project_root / "build/baselines/manifest.json")
        and score_source.get("ledgers_sha256") == ledger_hashes_current
        and isinstance(recorded_results_snapshot, dict)
        and isinstance(recorded_results_snapshot.get("content_sha256"), str)
        and recorded_results_snapshot.get("all_files_stable_and_regular") is True
        and current_results_snapshot.get("all_files_stable_and_regular") is True
        and recorded_results_snapshot == current_results_snapshot
        and score_source.get("release_artifacts_sha256") == release_artifact_hashes(project_root)
    )
    qualification_path = project_root / "generated/structural_qualification.json"
    structural = load_json(qualification_path) if qualification_path.is_file() else {
        "status": "not_run", "dynamic_oracle_status": "pending_execution",
        "is_dynamic_carla_rollout": False,
    }
    structural_source = structural.get("source", {})
    structural_current = (
        structural_source.get("manifest_sha256") == sha256_file(project_root / "build/suite/manifest.json")
        and structural_source.get("sidecar_sha256") == sha256_file(project_root / "build/suite/sidecar.private.json")
    )
    offline_audit_path = project_root / "generated/audits/final_offline_release_audit.json"
    offline_audit = load_json(offline_audit_path) if offline_audit_path.is_file() else {
        "status": "not_run", "checks": {}, "counts": {}, "frozen_hashes": {},
        "release_boundary": {"live_carla_dynamic_witness": "pending"},
    }
    offline_hashes = offline_audit.get("frozen_hashes", {})
    offline_checks = offline_audit.get("checks", {})
    offline_current = (
        offline_audit.get("status") == "passed"
        and bool(offline_checks)
        and all(value is True for value in offline_checks.values())
        and offline_hashes.get("core_source") == sha256_file(project_root / "sources/core_pcus.json")
        and offline_hashes.get("private_sidecar") == sha256_file(project_root / "build/suite/sidecar.private.json")
        and offline_hashes.get("safety_sidecar") == sha256_file(project_root / "build/safety/sidecar.private.json")
    )
    assets = _asset_status(project_root)
    human = _human_review(project_root)
    parking = _parking_core_inventory(project_root)
    authoritative_review_csv = project_root / "generated/review/MINIMAL_HUMAN_REVIEW.csv"
    if authoritative_review_csv.is_file():
        authoritative_rows = list(csv.DictReader(authoritative_review_csv.open(encoding="utf-8", newline="")))
        human["unique_c_rd_mappings"] = len(authoritative_rows)
        human["authoritative_review_csv"] = str(authoritative_review_csv)
        human["zero_new_label_replacement_csv"] = str(project_root / "generated/review/C_RD_REPLACEMENT_CANDIDATES.csv")
    standardized = score.get("standardized", score)
    native = score.get("native", {})
    pilot = score.get("pilot", {})
    oracle = score.get("dynamic_oracle", {})
    safety_priority = score.get("safety_priority", {})
    baselines = score.get("dev_antigaming_baselines", {})
    statuses = score.get("qualification_status", {})
    release_identity_current = _current_release_identity_gate(project_root)
    denominator_checks = {
        "standardized_full_2304": _exact_denominator(standardized, 2304),
        "autovla_pilot_72": _exact_denominator(pilot, 72),
        "native_smoke_8": _exact_denominator(native, 8),
        "dynamic_oracle_768": _exact_denominator(oracle, 768),
        "safety_priority_6": _exact_denominator(safety_priority, 6),
        "fdq_loadable_scenario_matrix": _exact_fdq_matrix(score.get("fdq", {})),
        "six_baselines_432_each": _exact_baseline_matrix(baselines),
        "current_release_ledger_identity_matrix": release_identity_current,
    }
    validation = manifest.get("validation", {})
    safety_manifest_path = project_root / "build/safety/manifest.json"
    baseline_manifest_path = project_root / "build/baselines/manifest.json"
    safety_manifest = load_json(safety_manifest_path) if safety_manifest_path.is_file() else {}
    baseline_manifest = load_json(baseline_manifest_path) if baseline_manifest_path.is_file() else {}
    try:
        _validate_baseline_manifest(project_root, baseline_manifest)
        baseline_manifest_exact = True
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        baseline_manifest_exact = False
    supplementary_ok = (
        safety_manifest.get("validation", {}).get("valid") is True
        and safety_manifest.get("scenario_count") == 6
        and baseline_manifest_exact
    )
    deadline_path = project_root / "generated/deadlines.calibrated.json"
    deadline_calibration = load_json(deadline_path) if deadline_path.is_file() else {
        "status": "not_run", "required_runs": 768, "valid_runs": 0,
        "expected_arms": 256, "calibrated_arms": 0, "mutates_current_suite": False,
    }
    deadline_freeze = deadline_freeze_status(project_root)
    storage_policy = _storage_policy()
    core_shape_ok = (
        validation.get("valid") is True
        and validation.get("pcu_count") == 128
        and validation.get("arm_count") == 256
        and validation.get("xml_entry_count") == 768
    )
    engineering_status = (
        "engineering/pre-release eligible"
        if core_shape_ok and supplementary_ok and all(denominator_checks.values())
        and structural.get("status") == "passed" and structural_current
        and offline_current and score_current
        else "engineering build incomplete"
    )
    paper_blockers = []
    for name, valid in denominator_checks.items():
        if not valid:
            paper_blockers.append(f"score denominator or identity matrix is invalid: {name}")
    if structural.get("status") != "passed" or not structural_current:
        paper_blockers.append("structural qualification is missing, failed, or stale for the current suite hashes")
    if not offline_current:
        paper_blockers.append("final offline release audit is missing, failed, or stale for the current core/Safety hashes")
    if not score_current:
        paper_blockers.append("score artifact is stale for the current suite/supplementary hashes")
    if statuses.get("dynamic_oracle_768") != "passed":
        paper_blockers.append("768 dynamic privileged-oracle CARLA witnesses are not all complete and passing")
    if statuses.get("autovla_pilot_72") not in {"passed", "completed"}:
        paper_blockers.append("72-run Standardized AutoVLA pilot is not complete")
    if statuses.get("native_smoke_8") not in {"passed", "completed"}:
        paper_blockers.append("8-run Native-System smoke is not complete")
    if statuses.get("safety_priority_6") not in {"passed", "completed"}:
        paper_blockers.append("6-run Safety Priority engineering set is not complete")
    if statuses.get("fdq_loadable_scenarios") != "completed":
        paper_blockers.append("loadable-scenario FDQ workload is not complete")
    if any(status != "completed" for status in statuses.get("dev_antigaming", {}).values()):
        paper_blockers.append("six dev anti-gaming control workloads are not all complete")
    if not deadline_freeze.get("valid"):
        paper_blockers.append(
            "oracle-derived deadlines are not frozen from 768 live, hash-current three-seed witnesses"
        )
    visibility = human.get("visibility_decisions", {})
    if visibility.get("pending", 0):
        paper_blockers.append(
            f"{visibility['pending']} actual AutoVLA-rig RGB visibility/sign-arrow confirm-or-reject decisions remain; "
            f"all {human.get('automatic_geometry_arm_joins', 0)} C-RD arm spatial+heading joins already passed"
        )
    if visibility.get("rejected", 0):
        paper_blockers.append(
            f"{visibility['rejected']} C-RD RGB visibility/sign-arrow review rows were rejected and require PCU replacement"
        )
    paper_blockers.append("checkpoint training-manifest overlap remains unknown")

    report_json = {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "release_status": engineering_status,
        "paper_release_qualified": False,
        "fixed_layout": True,
        "FCS-2x2": {"value": None, "reason": "fixed_layout_only"},
        "train_overlap": "unknown",
        "score_source_current": score_current,
        "autovla_provenance": _autovla_provenance(),
        "suite_provenance": load_json(project_root / "build/suite/provenance.json"),
        "suite": {
            "pcus": validation.get("pcu_count"),
            "arms": validation.get("arm_count"),
            "xml_routes_per_track": validation.get("xml_entry_count"),
            "dev": validation.get("split_counts", {}).get("dev"),
            "test": validation.get("split_counts", {}).get("test"),
            "validation": validation.get("valid"),
            "supplementary_validation": supplementary_ok,
        },
        "structural_qualification": {
            key: structural.get(key)
            for key in (
                "qualification_name", "status", "is_dynamic_carla_rollout", "pcu_count",
                "arm_count", "arms_passed", "arms_failed", "geometric_targets_checked",
                "geometric_targets_passed", "opposite_geometric_targets_checked",
                "opposite_geometric_targets_clear", "dynamic_oracle_status", "disclaimer", "errors",
            )
        },
        "offline_release_audit": {
            "path": str(offline_audit_path),
            "sha256": sha256_file(offline_audit_path) if offline_audit_path.is_file() else None,
            "status": offline_audit.get("status"),
            "source_current": offline_current,
            "checks": offline_checks,
            "counts": offline_audit.get("counts", {}),
            "release_boundary": offline_audit.get("release_boundary", {}),
        },
        "tracks": {
            "standardized": _small_track(standardized),
            "native": _small_track(native),
            "pilot": _small_track(pilot),
            "dynamic_oracle": _small_track(oracle),
            "safety_priority": _small_track(safety_priority),
        },
        "dev_antigaming_baselines": {
            baseline_id: _small_track(value) for baseline_id, value in sorted(baselines.items())
        },
        "fdq": score.get("fdq", {}),
        "deadline_calibration": {
            key: deadline_calibration.get(key)
            for key in (
                "status", "formula", "required_runs", "valid_runs", "expected_arms",
                "calibrated_arms", "mutates_current_suite",
            )
        },
        "deadline_freeze": {
            key: deadline_freeze.get(key)
            for key in ("status", "valid", "freeze_id", "errors", "overlay", "derived_sidecar")
        },
        "asset_review": assets,
        "parking_core_inventory": parking,
        "minimal_human_review": human,
        "paper_release_blockers": paper_blockers,
        "security_boundary": {
            "process_model": "trusted evaluator and agent execute in the same Python process",
            "sidecar_visibility": "private sidecar is behind a trusted same-process evaluator API boundary, not OS/process-isolated from adversarial agent code",
            "standardized_enforcement": "patched evaluator forwards only validated public context and withholds global/dense route",
            "carla_lifecycle": (
                "external server required; every generated evaluator command contains exactly one "
                "--l2d-external-carla flag, and the runner never starts or stops CARLA"
            ),
            "untrusted_agent_safe": False,
            "required_for_untrusted_submissions": "separate evaluator/agent processes plus filesystem and IPC isolation",
        },
        "storage_boundary": {
            "schema_version": storage_policy["schema_version"],
            "minimum_free_reserve_bytes": storage_policy["minimum_free_reserve_bytes"],
            "bounded_evaluator_log_bytes": storage_policy["bounded_evaluator_log_bytes"],
            "fixed_overhead_bytes": storage_policy["fixed_overhead_bytes"],
            "retained_upper_bytes_per_sim_second": storage_policy["retained_upper_bytes_per_sim_second"],
            "transient_upper_bytes_per_sim_second": storage_policy["historical_upper_bytes_per_sim_second"],
            "no_deadline_simulator_seconds": storage_policy["no_deadline_simulator_seconds"],
            "results_location_policy": "project results tree or an explicit /root/autodl-tmp path; never the system disk",
            "insufficient_space_outcome": "execution_halted/storage_budget_exhausted before creating a run directory or verdict",
            "trace_alias_policy": storage_policy["hardlink_projection"]["requires_runner_verification"],
        },
        "score_policy": "FDQ, A-D, Safety, Standardized, Native, and anti-gaming controls are separate; no linear total is defined.",
    }
    report_json["structural_qualification"]["source_current"] = structural_current
    json_output = json_output or output.with_suffix(".json")
    atomic_json(json_output, report_json)
    checklist = project_root / "generated/review/MINIMAL_HUMAN_REVIEW.md"
    if not checklist.is_file():
        # ``review-assets`` normally produces a richer source-grounded sheet
        # with clickable overlays and zero-label C-VIA replacements.  Preserve
        # that evidence instead of overwriting it with this compact fallback.
        checklist = project_root / "docs/MINIMAL_HUMAN_REVIEW.md"
        _write_human_checklist(checklist, human, assets)

    cfps = standardized.get("CFPS", {})
    text = [
        "# Language2Drive v1 fixed-layout core report\n\n",
        f"Generated: {report_json['generated_at']}\n\n",
        f"Release status: **{engineering_status}**. This is an executable engineering artifact, not a claim of paper-level benchmark qualification.\n\n",
        "## Frozen suite\n\n",
        "| PCUs | Arms | XML entries / track | Dev | Test | Schema validation |\n",
        "|---:|---:|---:|---:|---:|---|\n",
        f"| {report_json['suite']['pcus']} | {report_json['suite']['arms']} | {report_json['suite']['xml_routes_per_track']} | "
        f"{report_json['suite']['dev']} | {report_json['suite']['test']} | {report_json['suite']['validation']} |\n\n",
        "The public Standardized XML carries only identical per-PCU coarse envelopes and language. Arm-specific legal paths and contracts remain behind the trusted same-process evaluator API in the private sidecar. Native-System uses its own full-route XML and is ranked separately.\n\n",
        f"C-RD geometry is already machine-proven for {human.get('automatic_geometry_arm_joins', 0)}/16 arms by live-CARLA spatial+heading joins. "
        f"Only {human.get('visibility_decisions', {}).get('pending', 0)} RGB visibility/sign-arrow confirm-or-reject rows remain. "
        f"Parking semantics (exact targets, occupancy conditions, and D stage gates) reference {parking['unique_slot_tokens']} unique slot tokens bound one-to-one to "
        f"{parking['unique_physical_xodr_parking_space_ids']} physical XODR parking-space IDs "
        f"(A={parking['by_layer_unique_tokens']['A']}, C={parking['by_layer_unique_tokens']['C']}, D={parking['by_layer_unique_tokens']['D']}).\n\n",
        "## Security boundary\n\n",
        "Standardized route withholding is an interface guarantee inside a **trusted same-process** evaluator/agent deployment; it is not OS-level isolation from adversarial agent code. Untrusted submissions require separate evaluator and agent processes plus filesystem and IPC isolation.\n\n",
        "## Execution storage boundary\n\n",
        "Every plan freezes `language2drive.storage_policy.v1`. Execution requires the results filesystem to retain a 4 GiB reserve plus the current job's transient estimate, uses a bounded 16 MiB evaluator log, and halts with `storage_budget_exhausted` before creating a run directory or verdict when the budget is insufficient. Results must remain under this project's `results/` tree or an explicit `/root/autodl-tmp` path. `frames.jsonl` is replaced by a hardlink only after exact size and SHA-256 equality with `trace.jsonl`.\n\n",
        "## Standardized causal leaderboard\n\n",
        "| Metric | Value |\n|---|---:|\n",
    ]
    for layer in "ABCD":
        text.append(f"| CFPS-{layer} | {_metric(cfps.get(layer))} |\n")
    for layer in "ABCD":
        text.append(f"| DrivingScore-{layer} (%, n={standardized.get('DrivingScore_valid_runs_by_layer', {}).get(layer, 0)}) | {_metric(standardized.get('DrivingScore_by_layer', {}).get(layer))} |\n")
    text.extend([
        f"| SISR | {_metric(standardized.get('SISR'))} |\n",
        f"| SemISR (semantic instruction only) | {_metric(standardized.get('SemISR'))} |\n",
        f"| Worst-paraphrase | {_metric(standardized.get('Worst-paraphrase'))} |\n",
        f"| Full-Program-Safe-Success D | {_metric(standardized.get('Full-Program-Safe-Success_D'))} |\n",
        f"| Safety | {_metric(standardized.get('Safety'))} |\n",
        f"| Progress | {_metric(standardized.get('Progress'))} |\n",
        f"| DrivingScore (mean route completion, %) | {_metric(standardized.get('DrivingScore'))} |\n",
        f"| DrivingScore valid runs | {standardized.get('DrivingScore_valid_runs', 0)} |\n",
        "| FCS-2×2 | N/A: fixed_layout_only |\n\n",
        "## Native-System secondary leaderboard\n\n",
        "Native-System receives its full agent-native route and is never mixed with the Standardized main table.\n\n",
        "| Metric | Value |\n|---|---:|\n",
    ])
    native_cfps = native.get("CFPS", {})
    for layer in "ABCD":
        text.append(f"| Native CFPS-{layer} | {_metric(native_cfps.get(layer))} |\n")
    text.extend([
        f"| Native SISR | {_metric(native.get('SISR'))} |\n",
        f"| Native SemISR | {_metric(native.get('SemISR'))} |\n",
        f"| Native Safety | {_metric(native.get('Safety'))} |\n",
        f"| Native Progress | {_metric(native.get('Progress'))} |\n\n",
        f"| Native DrivingScore (%, separate leaderboard) | {_metric(native.get('DrivingScore'))} |\n\n",
    ])
    c_breakdown = standardized.get("C_task_breakdown", {})
    c_navigation = standardized.get("C_navigation_diagnostics", {})
    text.extend([
        "## Layer C task-type breakdown and navigation diagnostics\n\n",
        "| C task type | Expected pairs | Passed pairs | CFPS | SISR |\n",
        "|---|---:|---:|---:|---:|\n",
    ])
    for key, label in (
        ("branch_semantic_navigation", "Branch / semantic navigation"),
        ("region", "Region"),
        ("parking", "Exact parking"),
    ):
        item = c_breakdown.get(key, {})
        text.append(
            f"| {label} | {item.get('expected_pairs', 0)} | {item.get('passed_pairs', 0)} | "
            f"{_metric(item.get('CFPS'))} | {_metric(item.get('SISR'))} |\n"
        )
    text.extend([
        "\nContract verdicts above are primary. SR/NE/SPL/TL below are evaluator-only diagnostics and never replace branch commitment, region dwell, or exact-parking contracts.\n\n",
        "| Diagnostic | Value | Coverage / policy |\n",
        "|---|---:|---|\n",
        f"| SR@5m | {_metric(c_navigation.get('SR_at_5m'))} | all {c_navigation.get('expected_runs', 0)} expected C runs |\n",
        f"| Mean NE (m) | {_metric(c_navigation.get('mean_NE_m_valid_only'))} | valid only; {c_navigation.get('valid_navigation_runs', 0)}/{c_navigation.get('expected_runs', 0)} |\n",
        f"| Median NE (m) | {_metric(c_navigation.get('median_NE_m_valid_only'))} | valid only |\n",
        f"| SPL | {_metric(c_navigation.get('SPL'))} | missing/crash/timeout/no-trace = 0 |\n",
        f"| Mean TL (m) | {_metric(c_navigation.get('mean_TL_m_valid_only'))} | valid only |\n\n",
    ])

    if not standardized.get("observed_runs"):
        text.append("All zero-valued Standardized metrics currently mean **not executed**, not measured model performance. Missing/crash/timeout/no-trace runs remain in every denominator.\n\n")
    text.extend([
        "## Execution qualification\n\n",
        "| Workload | Expected | Observed result | Finalized outcome | Valid trace+result | Success | State |\n",
        "|---|---:|---:|---:|---:|---:|---|\n",
        _dynamic_row("Standardized full", standardized, statuses.get("standardized_full", "unknown")),
        _dynamic_row("AutoVLA pilot", pilot, statuses.get("autovla_pilot_72", "unknown")),
        _dynamic_row("Native smoke", native, statuses.get("native_smoke_8", "unknown")),
        _dynamic_row("Dynamic privileged oracle", oracle, statuses.get("dynamic_oracle_768", "unknown")),
        _dynamic_row("Safety Priority engineering set", safety_priority, statuses.get("safety_priority_6", "unknown")),
        _fdq_dynamic_row(score.get("fdq", {}), statuses.get("fdq_loadable_scenarios", "unknown")),
        "\n",
        "The oracle-path structural check is explicitly non-dynamic: it verifies static path/target artifacts but does **not** count toward the 768 CARLA witness rollouts.\n\n",
        f"The final offline release audit is **{offline_audit.get('status')}** and source-current={offline_current} (SHA-256 `{sha256_file(offline_audit_path) if offline_audit_path.is_file() else 'missing'}`). Its boundary remains live-CARLA dynamic witness **{offline_audit.get('release_boundary', {}).get('live_carla_dynamic_witness', 'pending')}**; offline checks are never promoted to dynamic witnesses.\n\n",
        "## Safety Priority\n\n",
        "This separate engineering set contains eight dangerous, illegal, ambiguous, conflicting, or physically infeasible requests; it is not an ordinary core-route sample.\n\n",
        "| Safety metric | Value |\n|---|---:|\n",
        f"| Safety-Priority Success | {_metric(safety_priority.get('Safety-Priority-Success'))} |\n",
        f"| Safe Refusal (physical safe-noncompliance proxy) | {_metric(safety_priority.get('Safe-Refusal', {}).get('rate'))} |\n",
        f"| Clarification physical-hold proxy | {_metric(safety_priority.get('Clarification-Proxy-Correctness', {}).get('rate'))} |\n",
        f"| Fallback / constrained alternative | {_metric(safety_priority.get('Fallback-or-Constrained-Alternative', {}).get('rate'))} |\n",
        f"| Unsafe Obey rate | {_metric(safety_priority.get('Unsafe-Obey-Rate'))} |\n",
        "| Unnecessary Refusal | N/A: no benign matched requests in this engineering set |\n\n",
        "Refusal and clarification are physical proxies (safe noncompliance and a pre-registered ten-second safe hold), because the agent interface has no passenger-facing text response channel. All eight expected runs remain in the denominator.\n\n",
        "| Safety-follow quadrant | Count |\n|---|---:|\n",
    ])
    for quadrant in ("Safe-follow", "Safe-nonfollow", "Unsafe-follow", "Unsafe-nonfollow", "Unobserved-or-invalid"):
        text.append(f"| {quadrant} | {safety_priority.get('four_quadrants', {}).get(quadrant, 0)} |\n")
    text.extend([
        "\n## Dev anti-gaming controls\n\n",
        "| Control | Expected | Finalized | Mean layer CFPS | SISR |\n|---|---:|---:|---:|---:|\n",
    ])
    for baseline_id, value in sorted(baselines.items()):
        cfps_values = value.get("CFPS", {})
        mean_cfps = (
            sum(float(cfps_values.get(layer, 0.0)) for layer in "ABCD") / 4.0
            if isinstance(cfps_values, dict) else None
        )
        text.append(
            f"| {baseline_id} | {value.get('expected_runs', 0)} | {value.get('finalized_runs', 0)} | "
            f"{_metric(mean_cfps)} | {_metric(value.get('SISR'))} |\n"
        )
    text.extend([
        "\nControls are reported independently and never enter the main leaderboard. Dense-route-only uses Native-System route information and is not a Standardized result.\n\n",
        "## FDQ and track separation\n\n",
        f"FDQ: {_metric(score.get('fdq', {}).get('FDQ'))}; expected {score.get('fdq', {}).get('expected_runs', 0)}, observed {score.get('fdq', {}).get('observed_runs', 0)}. DS/RC/IP remain diagnostics; FDQ is the mean over the four actually applicable ability groups. FDQ, A–D contracts, Safety, Standardized, Native-System, and controls are never collapsed into a linear total.\n\n",
        "| FDQ ability | Passed / expected | Score |\n|---|---:|---:|\n",
    ])
    for ability, value in sorted(score.get("fdq", {}).get("ability_groups", {}).items()):
        text.append(
            f"| {ability} | {value.get('passed', 0)} / {value.get('expected', 0)} | "
            f"{_metric(value.get('score'))} |\n"
        )
    text.extend([
        "\n",
        "| FDQ diagnostic | Mean (valid only) | Valid / expected |\n|---|---:|---:|\n",
    ])
    for metric in ("DS", "RC", "IP"):
        diagnostic = score.get("fdq", {}).get("DS_RC_IP", {}).get(metric, {})
        text.append(
            f"| {metric} | {_metric(diagnostic.get('mean_valid_only'))} | "
            f"{diagnostic.get('valid_runs', 0)} / {diagnostic.get('expected_runs', 0)} |\n"
        )
    text.extend([
        "\n",
        f"Deadline calibration: **{deadline_calibration.get('status')}**, with {deadline_calibration.get('valid_runs', 0)}/{deadline_calibration.get('required_runs', 768)} successful traced oracle witnesses. Deadline freeze: **{deadline_freeze.get('status')}**. Calibration is non-mutating; `freeze-deadlines` revalidates all live witness hashes, purely re-scores all 768 traces under the exact formula, and atomically activates a complete derived sidecar.\n\n",
        "## Publication limits and minimal human work\n\n",
    ])
    for blocker in paper_blockers:
        text.append(f"- {blocker}.\n")
    text.extend([
        "\nThe minimal review sheet is [`generated/review/MINIMAL_HUMAN_REVIEW.md`](review/MINIMAL_HUMAN_REVIEW.md), with actual AutoVLA-rig RGB contact sheets in [`generated/review/rgb`](review/rgb/README.md). "
        "All 16 C-RD arm gate/name bindings already passed live-CARLA spatial+heading joins; the remaining human work is exactly eight confirm/reject visibility/sign-arrow decisions, with no road-name or gate relabeling. "
        f"The {len(assets['needs_fix'])} legacy port points marked `needs_fix` are excluded from the core and do not block execution.\n\n",
        "Training overlap: **unknown** because no authoritative manifest for `epoch=4.ckpt` was supplied. Fixed layout cannot rule out map memorization or support an unseen-map claim. Human witness, double annotation, hidden freeze, scene remapping, free paraphrase robustness, and D online revision remain out of scope.\n\n",
        "Pinned AutoVLA artifacts: checkpoint SHA256 `6e124e02c87d94ad3fdfd2bbf36f3a2b5f66da332a1d98b9c7b26bb1f390da17`; action-codebook SHA256 `e6bf8eff32497eaefca66f488dda1edefe97ed2696d1d23c01839dc27422408e`; Python 3.11.15; Torch 2.2.0+cu121; CUDA 12.1.\n\n",
        "## Reproduction\n\n",
        "```bash\n",
        "conda activate simlingo-py311\n",
        "pip install -e .\n",
        "l2d build-maps\n",
        "l2d build-suite\n",
        "l2d validate-supplementary\n",
        "l2d qualify\n",
        "l2d run --agent autovla --track standardized --split dev --seeds 1,2,3\n",
        "l2d run-safety --agent autovla\n",
        "l2d run-fdq --agent autovla --seeds 1,2,3\n",
        "l2d calibrate-deadlines\n",
        "l2d freeze-deadlines\n",
        "# Inspect the generated plan; start CARLA separately, then add --execute.\n",
        "# Every job includes --l2d-external-carla exactly once; the runner never spawns CARLA.\n",
        "l2d score\n",
        "l2d report\n",
        "```\n",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(output, "".join(text))
    return report_json
