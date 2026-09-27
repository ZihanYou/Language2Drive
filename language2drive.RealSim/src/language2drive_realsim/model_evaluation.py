"""Formal model evidence authority, separate from diagnostic trace scoring."""

from pathlib import Path

from .constants import ALPASIM_COMMIT
from .evaluation import evaluate_rollouts
from .io import file_hash, load_json, stable_hash
from .metrics import aggregate_results
from .model_identity import checkpoint_identity
from .runtime_client import build_run_plan
from .schema import validate_source


MODEL_PREFIXES = {
    "Language2Drive/SimLingo-Posttrained-epoch004": "l2d_simlingo-driver-",
    "nvidia/Alpamayo-1.5-10B": "l2d_alpamayo1_5-driver-",
    "nvidia/Alpamayo2-Super": "l2d_alpamayo2_super-driver-",
    "RenzKa/simlingo": "l2d_simlingo-driver-",
    "Qwen/Qwen-Drive-1.0-4B": "l2d_qwen_drive-driver-",
    "Zewei-Zhou/AutoVLA": "l2d_autovla-driver-",
    "OpenDriveVLA/OpenDriveVLA-0.5B": "l2d_opendrivevla-driver-",
    "OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0": "l2d_lmdrive_null_lidar-driver-",
    # Explicitly names the disclosed non-official variant.  This identity is
    # intentionally distinct from official Senna and cannot satisfy the
    # formal single-seed manifest adapter gate.
    "rb93dett/Senna@48aa4218b586ce89c1778a5dedecbbe0e2887fcf:l2d-meta-action-v1":
        "l2d_senna_meta_action_variant-driver-",
}

# These two adapters were audited against the pinned AlpaSim Driver 0.91.0
# service.  Unlike historical models, a future package version needs a fresh
# protocol review rather than silently inheriting prefix authority.
MODEL_DRIVER_VERSION_IDS = {
    "Qwen/Qwen-Drive-1.0-4B": "l2d_qwen_drive-driver-0.91.0",
    "Zewei-Zhou/AutoVLA": "l2d_autovla-driver-0.91.0",
    "OpenDriveVLA/OpenDriveVLA-0.5B": "l2d_opendrivevla-driver-0.91.0",
    "rb93dett/Senna@48aa4218b586ce89c1778a5dedecbbe0e2887fcf:l2d-meta-action-v1":
        "l2d_senna_meta_action_variant-driver-0.91.0",
}


def model_manifest_errors(source, public, sidecar, manifest, qualification):
    """Bind full-cohort execution, prior qualification and local model identity."""
    errors = []
    binding = stable_hash(source)
    expected = len(source.get("pcus", [])) * 6
    try:
        validate_source(source, require_exact_release=True)
        plan = build_run_plan(public, sidecar)
        if manifest.get("run_plan_hash") != stable_hash(plan):
            errors.append("manifest_run_plan_mismatch")
    except (KeyError, TypeError, ValueError) as error:
        errors.append("invalid_frozen_cohort:%s" % error)
    if any(value.get("source_binding") != binding for value in (public, sidecar, manifest, qualification)):
        errors.append("source_binding_mismatch")
    if manifest.get("schema_version") != "l2d.realsim.run.v0.1":
        errors.append("invalid_manifest_schema")
    if manifest.get("manifest_binding") != stable_hash({k: v for k, v in manifest.items() if k != "manifest_binding"}):
        errors.append("manifest_binding_mismatch")
    if manifest.get("track") != "benchmark_model" or manifest.get("dry_run") is not False:
        errors.append("manifest_not_actual_benchmark_execution")
    if manifest.get("expected_run_count") != expected or expected == 0:
        errors.append("manifest_denominator_mismatch")
    pcus = source.get("pcus", [])
    qualified_rows = qualification.get("pcus", [])
    if (qualification.get("pass") is not True
        or qualification.get("global_errors") != []
        or qualification.get("eligible_count") != len(pcus)
        or qualification.get("candidate_count") != len(pcus)
        or len(qualified_rows) != len(pcus)
        or {r.get("pcu_id") for r in qualified_rows} != {p["pcu_id"] for p in pcus}
        or any(r.get("eligible") is not True for r in qualified_rows)):
        errors.append("full_cohort_oracle_qualification_missing")
    identity = manifest.get("model_identity")
    if not isinstance(identity, dict) or identity.get("model_id") not in MODEL_PREFIXES:
        errors.append("unsupported_or_missing_model_identity")
    else:
        try:
            actual = checkpoint_identity(Path(identity["path"]), identity["model_id"])
            if (identity.get("complete") is not True or not identity.get("snapshot_revision")
                or actual != identity):
                errors.append("checkpoint_identity_incomplete_or_changed")
        except (KeyError, TypeError, ValueError, OSError):
            errors.append("checkpoint_identity_unreadable")
    return errors


def model_receipt_errors(trace_path, receipt, session, model_id):
    """Check production bridge receipts; never hash USDZs or checkpoint shards."""
    errors = []
    if receipt.get("schema_version") != "l2d.realsim.asl_receipt.v0.1":
        errors.append("invalid_asl_receipt_schema")
    if receipt.get("synthetic") is not False:
        errors.append("synthetic_or_unmarked_evidence")
    expected = dict(session_uuid=trace_path.stem, artifact_id=session["artifact_id"],
        scene_id_private=session["scene_id_private"], start_timestamp_us=session["start_timestamp_us"],
        random_seed=session["seed"], alpasim_commit=ALPASIM_COMMIT)
    errors.extend("receipt_%s_mismatch" % k for k, v in expected.items() if receipt.get(k) != v)
    prefix = MODEL_PREFIXES.get(model_id)
    version = receipt.get("egodriver_version_id")
    exact_version = MODEL_DRIVER_VERSION_IDS.get(model_id)
    if (exact_version is not None and version != exact_version) or (
        exact_version is None
        and (not prefix or not isinstance(version, str) or not version.startswith(prefix))
    ):
        errors.append("receipt_wrong_benchmark_driver")
    if receipt.get("oracle_controller") is not None:
        errors.append("oracle_authority_in_model_evidence")
    fingerprint = receipt.get("preintervention_non_language_sha256")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)):
        errors.append("actual_preintervention_evidence_missing")
    asl = receipt.get("source_asl")
    if not isinstance(asl, str) or not Path(asl).is_file():
        errors.append("source_asl_missing")
    elif file_hash(Path(asl)) != receipt.get("source_asl_sha256"):
        errors.append("source_asl_hash_mismatch")
    if file_hash(trace_path) != receipt.get("normalized_trace_sha256"):
        errors.append("normalized_trace_hash_mismatch")
    return errors


def evaluate_model_rollouts(source, public, sidecar, evidence_dir, thresholds, manifest, qualification):
    """Preserve every slot, including model failures and unauthenticated evidence."""
    report = evaluate_rollouts(source, sidecar, evidence_dir, thresholds)
    global_errors = model_manifest_errors(source, public, sidecar, manifest, qualification)
    identity = manifest.get("model_identity") or {}
    model_id = identity.get("model_id") if isinstance(identity, dict) else None
    receipts, errors = {}, {}
    for row in report["results"]:
        sid = row["session_uuid"]
        issues = errors.setdefault((row["pcu_id"], row["seed"], row["arm_id"]), [])
        trace = Path(evidence_dir) / (str(sid) + ".jsonl")
        try:
            receipt = load_json(trace.with_suffix(".receipt.json"))
            if not isinstance(receipt, dict):
                raise ValueError("receipt must be an object")
            issues.extend(model_receipt_errors(trace, receipt, sidecar["sessions"][sid], model_id))
            receipts[sid] = receipt
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append("model_receipt_unreadable:%s" % error)
    used_asls = {}
    for sid, receipt in receipts.items():
        digest = receipt.get("source_asl_sha256")
        if isinstance(digest, str):
            used_asls.setdefault(digest, []).append(sid)
    duplicated_asls = {sid for sids in used_asls.values() if len(sids) > 1 for sid in sids}
    pairs = {}
    for row in report["results"]:
        if row["session_uuid"] in duplicated_asls:
            errors[(row["pcu_id"], row["seed"], row["arm_id"])].append("source_asl_reused_across_sessions")
        pairs.setdefault((row["pcu_id"], row["seed"]), []).append(row)
    for rows in pairs.values():
        fingerprints = [receipts.get(r["session_uuid"], {}).get("preintervention_non_language_sha256") for r in rows]
        if len(rows) != 2 or not fingerprints[0] or fingerprints[0] != fingerprints[1]:
            for row in rows:
                errors[(row["pcu_id"], row["seed"], row["arm_id"])].append("actual_paired_prefix_missing_or_mismatched")
    audit_rows = []
    for row in report["results"]:
        issues = errors[(row["pcu_id"], row["seed"], row["arm_id"])]
        audit_rows.append(dict(session_uuid=row["session_uuid"], pcu_id=row["pcu_id"],
            seed=row["seed"], arm_id=row["arm_id"], errors=issues))
        if global_errors or issues:
            # Diagnostics preserve raw scoring, but unauthenticated successes
            # cannot inflate formal metrics. Failed slots are never removed.
            original = {k: row[k] for k in ("passed", "failure", "diagnostics") if k in row}
            row.update(passed=False, failure="INVALID_SIM", diagnostics=dict(
                reason="formal_model_evidence_authority_failed", authority_errors=issues,
                global_authority_errors=global_errors, untrusted_semantic_result=original))
    passed = not global_errors and all(not r["errors"] for r in audit_rows)
    invalid_sim_count = sum(r.get("failure") == "INVALID_SIM" for r in report["results"])
    report.update(source_binding=stable_hash(source), model_identity=identity,
        run_manifest_binding=manifest.get("manifest_binding"),
        qualification_report_binding=stable_hash(qualification),
        evaluation_scope="formal_model_evidence_checked",
        formal_model_evaluation_complete=passed and report["denominator_complete"] and invalid_sim_count == 0,
        invalid_sim_count=invalid_sim_count,
        model_authority_audit=dict(passed=passed, global_errors=global_errors, runs=audit_rows),
        metrics=aggregate_results(report["results"]))
    return report
