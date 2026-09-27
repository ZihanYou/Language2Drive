"""Dual-arm, all-seed PCU qualification and oracle witness indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import ARMS, COUPLED_SEEDS, ORACLE_CONTROLLER_ID
from .io import file_hash, load_json, load_jsonl, stable_hash
from .schema import SchemaError, validate_pcu, validate_source
from .verifier import verify_trace
from .render_validity import NATIVE_MODE


def _receipt_for(trace_path: Path) -> Path:
    return trace_path.with_suffix(".receipt.json")


def _authority_check(
    trace_path: Path, receipt: Dict[str, Any], session: Dict[str, Any], alpasim_commit: str
) -> List[str]:
    errors = []
    if receipt.get("schema_version") != "l2d.realsim.asl_receipt.v0.1":
        errors.append("invalid_asl_receipt_schema")
    if receipt.get("synthetic") is not False:
        errors.append("synthetic_or_unmarked_evidence")
    if receipt.get("session_uuid") != trace_path.stem:
        errors.append("receipt_session_mismatch")
    if receipt.get("artifact_id") != session["artifact_id"]:
        errors.append("receipt_artifact_mismatch")
    if receipt.get("scene_id_private") != session["scene_id_private"]:
        errors.append("receipt_scene_mismatch")
    if receipt.get("start_timestamp_us") != session["start_timestamp_us"]:
        errors.append("receipt_start_timestamp_mismatch")
    if receipt.get("random_seed") != session["seed"]:
        errors.append("receipt_random_seed_mismatch")
    if receipt.get("alpasim_commit") != alpasim_commit:
        errors.append("receipt_alpasim_commit_mismatch")
    driver_version = receipt.get("egodriver_version_id")
    if not isinstance(driver_version, str) or not driver_version.startswith(
        "l2d_privileged_oracle-driver-"
    ):
        errors.append("evidence_not_generated_by_privileged_oracle_driver")
    if receipt.get("oracle_controller") != ORACLE_CONTROLLER_ID:
        errors.append("oracle_controller_authority_missing")
    source_asl = receipt.get("source_asl")
    if not isinstance(source_asl, str) or not Path(source_asl).is_file():
        errors.append("source_asl_missing")
    elif file_hash(Path(source_asl)) != receipt.get("source_asl_sha256"):
        errors.append("source_asl_hash_mismatch")
    if file_hash(trace_path) != receipt.get("normalized_trace_sha256"):
        errors.append("normalized_trace_hash_mismatch")
    return errors


def qualify_source(
    source: Dict[str, Any],
    sidecar: Dict[str, Any],
    leakage_audit: Dict[str, Any],
    evidence_dir: Path,
    threshold_profile: Dict[str, Any],
    verifier_test_report: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from .release_profile import release_layer_targets
    errors = []
    expected_pcus = 50
    try:
        validate_source(source, require_exact_release=True)
    except SchemaError as error:
        errors.append(str(error))
    try:
        expected_pcus = sum(release_layer_targets(source).values())
    except ValueError as error:
        if str(error) not in errors:
            errors.append(str(error))
    source_binding = stable_hash(source)
    if sidecar.get("source_binding") != source_binding:
        errors.append("sidecar_source_binding_mismatch")
    if leakage_audit.get("source_binding") != source_binding or leakage_audit.get("pass") is not True:
        errors.append("leakage_audit_failed")
    if verifier_test_report.get("pass") is not True:
        errors.append("verifier_tests_failed")

    sessions_by_identity = {}
    for session_uuid, session in sidecar.get("sessions", {}).items():
        key = (session.get("pcu_id"), session.get("arm_id"), session.get("seed"))
        if key in sessions_by_identity:
            errors.append("duplicate_sidecar_identity:%r" % (key,))
        sessions_by_identity[key] = (session_uuid, session)

    expected_session_identities = {
        (pcu.get("pcu_id"), arm_id, seed)
        for pcu in source.get("pcus", [])
        for arm_id in ARMS
        for seed in COUPLED_SEEDS
    }
    observed_session_identities = set(sessions_by_identity)
    if (
        observed_session_identities != expected_session_identities
        or len(sidecar.get("sessions", {})) != len(expected_session_identities)
    ):
        missing = expected_session_identities - observed_session_identities
        extra = observed_session_identities - expected_session_identities
        errors.append(
            "sidecar_identity_ledger_mismatch:missing=%d:extra=%d"
            % (len(missing), len(extra))
        )

    expected_leakage_pairs = {
        (pcu.get("pcu_id"), seed)
        for pcu in source.get("pcus", [])
        for seed in COUPLED_SEEDS
    }
    observed_leakage_pairs = []
    leakage_pair_invalid = False
    for row in leakage_audit.get("pairs", []):
        identity = (row.get("pcu_id"), row.get("seed"))
        observed_leakage_pairs.append(identity)
        fingerprint = row.get("paired_input_hash")
        if (
            row.get("pass") is not True
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
        ):
            leakage_pair_invalid = True
    if (
        len(observed_leakage_pairs) != len(set(observed_leakage_pairs))
        or set(observed_leakage_pairs) != expected_leakage_pairs
        or leakage_pair_invalid
    ):
        errors.append("leakage_pair_ledger_incomplete_or_invalid")

    pcu_rows = []
    witness_rows = []
    for pcu in source.get("pcus", []):
        pcu_errors = []
        seed_rows = []
        try:
            validate_pcu(pcu)
        except SchemaError as error:
            pcu_errors.append(str(error))
        for seed in COUPLED_SEEDS:
            arm_rows = {}
            for arm_id in ARMS:
                identity = (pcu.get("pcu_id"), arm_id, seed)
                entry = sessions_by_identity.get(identity)
                run_errors = []
                own_result = None
                opposite_result = None
                trace_hash = None
                render_diagnostics = {}
                if entry is None:
                    run_errors.append("missing_compiled_session")
                    session_uuid = None
                    session = None
                else:
                    session_uuid, session = entry
                    trace_path = Path(evidence_dir) / (session_uuid + ".jsonl")
                    receipt_path = _receipt_for(trace_path)
                    if not trace_path.is_file():
                        run_errors.append("normalized_trace_missing")
                    elif not receipt_path.is_file():
                        run_errors.append("asl_authority_receipt_missing")
                    else:
                        try:
                            receipt = load_json(receipt_path)
                            run_errors.extend(
                                _authority_check(
                                    trace_path,
                                    receipt,
                                    session,
                                    pcu.get("backend", {}).get(
                                        "alpasim_commit", ""
                                    ),
                                )
                            )
                            trace = load_jsonl(trace_path)
                            own_result = verify_trace(
                                pcu,
                                arm_id,
                                trace,
                                pcu.get("private_annotations", {}),
                                threshold_profile,
                            )
                            other_arm = "R1" if arm_id == "R0" else "R0"
                            opposite_result = verify_trace(
                                pcu,
                                other_arm,
                                trace,
                                pcu.get("private_annotations", {}),
                                threshold_profile,
                            )
                            trace_hash = file_hash(trace_path)
                            invalid_frames = [row["time_s"] for row in trace if not row["render_valid"]]
                            deviations = [
                                float(row["reconstruction_deviation_m"])
                                for row in trace
                                if isinstance(
                                    row.get("reconstruction_deviation_m"), (int, float)
                                )
                                and not isinstance(
                                    row.get("reconstruction_deviation_m"), bool
                                )
                            ]
                            max_deviation = max(deviations) if deviations else None
                            render_diagnostics = {
                                "valid": not invalid_frames,
                                "invalid_frame_times_s": invalid_frames,
                                "maximum_deviation_m": max_deviation,
                                "deviation_reference": "reviewed_support_polygon_boundary",
                                "validity_modes": sorted({r.get("render_validity_mode", "legacy_support_polygon") for r in trace}),
                                "maximum_recorded_path_deviation_m": max(
                                    (r["recorded_path_deviation_m"] for r in trace
                                     if r.get("recorded_path_deviation_m") is not None), default=None),
                                "recorded_path_deviation_is_instruction_correctness": False,
                            }
                            if render_diagnostics["validity_modes"] == [NATIVE_MODE]:
                                render_diagnostics["maximum_deviation_m"] = render_diagnostics["maximum_recorded_path_deviation_m"]
                                render_diagnostics["deviation_reference"] = "same_timestamp_recorded_ego_AABB_xy_diagnostic_only"
                            if own_result.passed is not True:
                                run_errors.append("oracle_does_not_satisfy_own_contract")
                            if opposite_result.passed is True:
                                run_errors.append("oracle_also_satisfies_opposing_contract")
                        except (OSError, TypeError, ValueError, KeyError) as error:
                            run_errors.append("trace_or_verifier_error:%s" % error)
                row = {
                    "session_uuid": session_uuid,
                    "arm_id": arm_id,
                    "seed": seed,
                    "qualified": not run_errors,
                    "errors": run_errors,
                    "own_verdict": own_result.to_dict() if own_result else None,
                    "opposing_verdict": opposite_result.to_dict() if opposite_result else None,
                    "normalized_trace_sha256": trace_hash,
                    "render_diagnostics": render_diagnostics,
                }
                arm_rows[arm_id] = row
                witness_rows.append(dict(row, pcu_id=pcu.get("pcu_id")))
                pcu_errors.extend("%s/%s:%s" % (seed, arm_id, item) for item in run_errors)
            seed_rows.append({"seed": seed, "arms": arm_rows, "qualified": all(row["qualified"] for row in arm_rows.values())})
        pcu_rows.append(
            {
                "pcu_id": pcu.get("pcu_id"),
                "primary_layer": pcu.get("primary_layer"),
                "task_family": pcu.get("task_family"),
                "eligible": not pcu_errors,
                "errors": pcu_errors,
                "seeds": seed_rows,
            }
        )
    eligible = [row for row in pcu_rows if row["eligible"]]
    report = {
        "schema_version": "l2d.realsim.qualification.v0.1",
        "source_binding": source_binding,
        "eligible_count": len(eligible),
        "candidate_count": len(source.get("pcus", [])),
        "all_global_gates_pass": not errors,
        "global_errors": errors,
        "pcus": pcu_rows,
    }
    report["target_pcu_count"] = expected_pcus
    report["pass"] = not errors and len(eligible) == expected_pcus and all(row["eligible"] for row in pcu_rows)
    witness_index = {
        "schema_version": "l2d.realsim.oracle_witness_index.v0.1",
        "source_binding": source_binding,
        "coverage_numerator": sum(row["qualified"] for row in witness_rows),
        "coverage_denominator": expected_pcus * len(ARMS) * len(COUPLED_SEEDS),
        "coverage_complete": not errors and len(witness_rows) == expected_pcus * len(ARMS) * len(COUPLED_SEEDS) and all(row["qualified"] for row in witness_rows),
        "witnesses": witness_rows,
    }
    return report, witness_index
