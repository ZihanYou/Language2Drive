"""Audit existing single-seed model evidence before skipping any new runs.

All completed selected-seed evidence is considered, including task failures.
Legacy artifacts are read-only. ASL/trace integrity is verified once per audit;
checkpoint shards and USDZs are never hashed. This is evidence compatibility,
not full legality attestation or a completed campaign report.
"""

from collections import Counter
from pathlib import Path

from .component_scoring import score_trace_components
from .io import load_json, load_jsonl, stable_hash
from .keep_change_pair_smoke import reference
from .model_evaluation import model_manifest_errors, model_receipt_errors
from .single_seed_execution import build_single_seed_execution_plan, single_seed_manifest_errors


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_legacy_single_seed_reuse(source, public, sidecar, campaign, qualification, *,
                                  target_manifest, inference_configuration, old_attempt,
                                  thresholds):
    """Read the original stopped attempt; no caller-supplied favourable subset."""
    plan = build_single_seed_execution_plan(source, public, sidecar, campaign, qualification)
    errors = single_seed_manifest_errors(source, public, sidecar, campaign, qualification,
        target_manifest, inference_configuration, require_actual=False)
    root = Path(old_attempt).resolve()
    manifest = load_json(root / "run_manifest.json")
    preparation = load_json(root / "model-preparation.private.json")
    launch = load_json(root / "launch-context.private.json")
    state = load_json(root / "execution-status.private.json")
    errors.extend(model_manifest_errors(source, public, sidecar, manifest, qualification))
    if manifest.get("model_identity") != target_manifest.get("model_identity"):
        errors.append("legacy_model_identity_differs")
    if (state.get("state") != "stopped_on_error"
            or state.get("owned_cleanup_complete") is not True
            or state.get("cleanup_errors") != []
            or state.get("failures") != ["Supervisor received signal 15"]):
        errors.append("legacy_attempt_not_clean_administrative_stop")
    if state.get("source_binding") != plan["source_binding"]:
        errors.append("legacy_execution_source_mismatch")
    references = launch.get("execution_file_references", [])
    if (not references or stable_hash(references) != launch.get("execution_file_snapshot_binding")
            or manifest.get("execution_file_snapshot_binding") != launch.get("execution_file_snapshot_binding")
            or launch.get("qualification_binding") != stable_hash(qualification)
            or launch.get("source_binding") != plan["source_binding"]):
        errors.append("legacy_launch_binding_mismatch")
    try:
        if launch.get("preparation") != reference(root / "model-preparation.private.json"):
            errors.append("legacy_preparation_changed")
        if any(reference(item["path"]) != item for item in references + preparation["references"]):
            errors.append("legacy_bound_files_changed")
        config_path = root / "driver-config.yaml"
        threshold_path = Path(__file__).resolve().parents[2] / "config/threshold_profile.json"
        bound_paths = {Path(item["path"]).resolve() for item in references + preparation["references"]}
        if config_path not in bound_paths or threshold_path not in bound_paths:
            errors.append("legacy_config_or_threshold_not_bound")
        if load_json(config_path) != inference_configuration:
            errors.append("legacy_inference_configuration_differs")
        if load_json(threshold_path) != thresholds:
            errors.append("legacy_threshold_profile_differs")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append("legacy_file_reference_error:" + str(exc))
    # Do not hash large evidence until all cheap provenance gates succeed.
    if errors:
        return dict(schema_version="l2d.realsim.single_seed_reuse.v0.1",
            scope="legacy_evidence_compatibility_not_complete_model_evaluation",
            reuse_compatible=False, formal_model_evaluation_complete=False,
            global_errors=errors, records=[], asl_integrity_checks=0, trace_integrity_checks=0)
    expected = {r["session_uuid"]: r for r in plan["runs"]}
    pcus = {p["pcu_id"]: p for p in source["pcus"]}
    evidence = root / "model_evidence"
    selected = sorted(sid for sid in expected if (evidence / (sid + ".jsonl")).exists()
                      or (evidence / (sid + ".receipt.json")).exists())
    records, receipts = [], {}
    asl_checks = trace_checks = 0
    for sid in selected:
        slot = expected[sid]
        trace = evidence / (sid + ".jsonl")
        receipt_path = trace.with_suffix(".receipt.json")
        record = dict(session_uuid=sid, pcu_id=slot["pcu_id"], arm_id=slot["arm_id"],
                      seed=slot["random_seed"], origin_trace=str(trace),
                      origin_receipt=str(receipt_path), errors=[])
        try:
            receipt = load_json(receipt_path)
            _require(isinstance(receipt, dict), "receipt must be an object")
            _require(trace.is_file(), "normalized trace missing")
            # The unchanged formal bridge audit checks real bytes, driver,
            # source scene/start/seed and actual non-language prefix evidence.
            record["errors"].extend(model_receipt_errors(trace, receipt,
                sidecar["sessions"][sid], manifest["model_identity"]["model_id"]))
            trace_checks += 1
            asl_checks += int(isinstance(receipt.get("source_asl"), str)
                              and Path(receipt["source_asl"]).is_file())
            receipts[sid] = receipt
            record["source_asl"] = receipt.get("source_asl")
            record["source_asl_sha256"] = receipt.get("source_asl_sha256")
            record["normalized_trace_sha256"] = receipt.get("normalized_trace_sha256")
            record["actual_prefix"] = receipt.get("preintervention_non_language_sha256")
            record["file_references"] = [reference(trace), reference(receipt_path)]
            if isinstance(receipt.get("source_asl"), str) and Path(receipt["source_asl"]).is_file():
                record["file_references"].append(reference(receipt["source_asl"]))
            score = score_trace_components(pcus[slot["pcu_id"]], slot["arm_id"], load_jsonl(trace),
                annotations=sidecar["sessions"][sid]["annotations"], threshold_profile=thresholds)
            record["component_scoring"] = score
            if not score["trace_usable"]:
                record["errors"].append("infrastructure_invalid_trace")
        except (OSError, KeyError, ValueError, TypeError) as exc:
            record["errors"].append("legacy_evidence_unreadable:" + str(exc))
        records.append(record)
    pairs = {}
    digests = Counter(r.get("source_asl_sha256") for r in receipts.values())
    for record in records:
        pairs.setdefault(record["pcu_id"], []).append(record)
        digest = receipts.get(record["session_uuid"], {}).get("source_asl_sha256")
        if digest and digests[digest] != 1:
            record["errors"].append("source_asl_reused_across_sessions")
    for pair in pairs.values():
        prefixes = [r.get("actual_prefix") for r in pair]
        if (len(pair) != 2 or {r["arm_id"] for r in pair} != {"R0", "R1"}
                or not prefixes[0] or len(set(prefixes)) != 1):
            for record in pair:
                record["errors"].append("actual_paired_prefix_missing_or_mismatched")
    compatible = bool(records) and all(not r["errors"] for r in records)
    reusable = selected if compatible else []  # No partial success cherry-picking.
    result = dict(schema_version="l2d.realsim.single_seed_reuse.v0.1",
        scope="legacy_evidence_compatibility_not_complete_model_evaluation",
        reuse_compatible=compatible, formal_model_evaluation_complete=False,
        protocol_id=campaign["protocol_id"], seed=campaign["model_seeds"][0],
        model_name=target_manifest["model_name"], source_binding=plan["source_binding"],
        configuration_id=target_manifest["configuration_id"],
        target_manifest_binding=target_manifest["manifest_binding"],
        origin_manifest_binding=manifest["manifest_binding"], old_attempt=str(root),
        candidate_runs=len(records), compatible_runs=len(reusable),
        compatible_pairs=len(reusable) // 2, expected_runs=len(expected),
        remaining_sessions=sorted(set(expected) - set(reusable)),
        global_errors=[], records=records, asl_integrity_checks=asl_checks,
        trace_integrity_checks=trace_checks, checkpoint_shard_hashes_computed=0,
        asset_hashes_computed=0, missing_legality_not_assumed_passed=True,
        reuse_cache_policy="stat references are edit guards, not a substitute for release integrity audit")
    result["audit_binding"] = stable_hash(result)
    return result


def reusable_sessions_from_cache(audit, plan, target_manifest):
    """Validate recorded audit and fast edit guards before scheduling skips.

    This avoids repeating multi-GB ASL reads for preparation/status checks. It
    is not the release-time integrity check and never produces model scores.
    The target is the original preparation manifest, not a relabelled manifest
    for a later execution attempt. All failures raise; no subset is selected.
    """
    _require(audit.get("schema_version") == "l2d.realsim.single_seed_reuse.v0.1"
             and audit.get("reuse_compatible") is True and audit.get("global_errors") == []
             and audit.get("formal_model_evaluation_complete") is False,
             "reuse audit is not compatible")
    _require(audit.get("audit_binding") == stable_hash({k:v for k,v in audit.items() if k != "audit_binding"}),
             "reuse audit binding changed")
    _require(target_manifest.get("manifest_binding") == stable_hash({k:v for k,v in
             target_manifest.items() if k != "manifest_binding"})
             and target_manifest.get("run_plan_hash") == stable_hash(plan),
             "reuse target manifest or plan changed")
    _require(audit.get("target_manifest_binding") == target_manifest.get("manifest_binding")
             and audit.get("source_binding") == plan.get("source_binding")
             and audit.get("protocol_id") == plan.get("protocol_id")
             and plan.get("model_seeds") == [audit.get("seed")]
             and audit.get("configuration_id") == target_manifest.get("configuration_id")
             and audit.get("model_name") == target_manifest.get("model_name"),
             "reuse audit target mismatch")
    expected = {r["session_uuid"]:r for r in plan["runs"]}
    _require(len(expected) == plan["run_count"] == audit.get("expected_runs"),
             "reuse full denominator mismatch")
    records = audit.get("records", [])
    _require(bool(records) and len(records) == audit.get("candidate_runs") == audit.get("compatible_runs")
             and all(not r.get("errors") for r in records), "reuse run audit incomplete")
    sessions, pairs = set(), {}
    for record in records:
        sid = record["session_uuid"]
        _require(sid in expected and sid not in sessions, "duplicate or unexpected reuse session")
        slot = expected[sid]
        _require((record["pcu_id"],record["arm_id"],record["seed"]) ==
                 (slot["pcu_id"],slot["arm_id"],slot["random_seed"]), "reuse identity mismatch")
        refs = record.get("file_references", [])
        required_paths = {str(Path(record[k]).resolve()) for k in ("origin_trace", "origin_receipt", "source_asl")}
        _require(len(refs) == 3 and {r["path"] for r in refs} == required_paths
                 and all(reference(r["path"]) == r for r in refs), "reuse evidence files changed")
        sessions.add(sid)
        pairs.setdefault(record["pcu_id"], []).append(record["arm_id"])
    _require(all(sorted(arms) == ["R0", "R1"] for arms in pairs.values())
             and len(pairs) == audit.get("compatible_pairs"), "reuse must contain complete pairs")
    _require(audit.get("remaining_sessions") == sorted(set(expected) - sessions), "reuse remainder mismatch")
    root = Path(audit["old_attempt"])
    manifest = load_json(root / "run_manifest.json")
    launch = load_json(root / "launch-context.private.json")
    preparation = load_json(root / "model-preparation.private.json")
    _require(manifest.get("manifest_binding") == audit.get("origin_manifest_binding")
             and manifest["manifest_binding"] == stable_hash({k:v for k,v in manifest.items() if k != "manifest_binding"}),
             "origin manifest changed")
    refs = launch.get("execution_file_references", [])
    _require(bool(refs) and stable_hash(refs) == manifest.get("execution_file_snapshot_binding")
             and launch.get("preparation") == reference(root / "model-preparation.private.json")
             and all(reference(r["path"]) == r for r in refs + preparation["references"]),
             "origin bound files changed")
    return sessions
