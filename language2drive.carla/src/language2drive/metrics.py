from __future__ import annotations

import json
import math
import os
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_json, stable_hash
from .core_driving_score import valid_driving_score
from .schema import ArmIdentity, SchemaError


_CRASH_WORDS = ("crash", "agent error", "simulation error")
_FINAL_TERMINAL_STATUSES = {"crash", "timeout", "deadlock"}
_STRICT_CORE_STATUSES = {
    "complete", "crash", "timeout", "deadlock", "no_trace",
    "invalid_trace", "capture_failure", "runtime_authority_failure",
}
_VERDICT_FIELDS = (
    "own_contract", "opposite_contract", "safe_legal", "progress",
    "deadline", "stability",
)
_RESULT_IDENTITY_FIELDS = (
    "run_id", "pcu_id", "arm_id", "paraphrase", "seed", "track",
    "route_entry_id",
)
_FULL_EXECUTION_BUNDLE_FIELDS = {
    "schema_version", "route_xml_sha256", "agent_sha256",
    "agent_config_sha256", "evaluator_sha256", "workload",
    "benchmark_track", "evaluator_track", "dependency_profile",
    "dependency_hashes", "dependency_manifest_sha256",
    "execution_environment_sha256", "storage_policy_sha256",
    "wall_timeout_policy_sha256", "carla_runtime_root",
    "carla_runtime_asset_manifest_sha256",
    "carla_runtime_asset_inventory_sha256",
    "carla_runtime_asset_files_sha256", "carla_runtime_asset_file_count",
    "carla_runtime_asset_total_size_bytes",
    "carla_runtime_shipping_binary_sha256",
    "carla_runtime_asset_provenance_sha256",
    "carla_runtime_available_maps_sha256",
    "carla_runtime_available_map_count", "carla_runtime_client_version",
    "carla_runtime_server_version", "carla_runtime_boot_id",
    "carla_runtime_freeze_monotonic_ticks",
    "carla_runtime_clock_ticks_per_second",
    "carla_runtime_server_readonly_permissions_sha256",
    "carla_runtime_server_readonly_permission_entry_count",
    "carla_runtime_server_uid", "carla_runtime_server_gids",
    "carla_runtime_asset_validator_sha256", "carla_server_process_receipt",
    "carla_server_process_receipt_sha256",
}


def _exact_bool(value: Any) -> bool:
    return value is True


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _strict_json_object(path: Path) -> dict[str, Any]:
    """Load one standards-compliant JSON object, rejecting duplicate keys."""

    source = Path(path)
    metadata = source.lstat()
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("JSON evidence is not a real regular file")
    value = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value is not an object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _execution_bundle_is_full(bundle: Any, *, workload: str | None = None) -> bool:
    """Recognize only the current, complete execution-authority bundle."""

    if not isinstance(bundle, dict) or bundle.get("schema_version") != "language2drive.execution_bundle.v2":
        return False
    expected_fields = set(_FULL_EXECUTION_BUNDLE_FIELDS)
    if workload != "fdq":
        expected_fields.add("private_sidecar_sha256")
    if set(bundle) != expected_fields:
        return False
    if workload != "fdq" and "private_sidecar_sha256" not in bundle:
        return False
    if workload == "fdq" and "private_sidecar_sha256" in bundle:
        return False
    hash_fields = {
        key for key in _FULL_EXECUTION_BUNDLE_FIELDS
        if key.endswith("_sha256")
    } | ({"private_sidecar_sha256"} if workload != "fdq" else set())
    if any(not _is_sha256(bundle.get(key)) for key in hash_fields):
        return False
    if not isinstance(bundle.get("dependency_hashes"), dict) or not bundle["dependency_hashes"]:
        return False
    if any(
        not isinstance(key, str) or not _is_sha256(value)
        for key, value in bundle["dependency_hashes"].items()
    ):
        return False
    if not isinstance(bundle.get("carla_server_process_receipt"), dict):
        return False
    if bundle.get("dependency_manifest_sha256") != stable_hash(bundle["dependency_hashes"]):
        return False
    if bundle.get("carla_server_process_receipt_sha256") != stable_hash(
        bundle["carla_server_process_receipt"]
    ):
        return False
    for key in ("workload", "benchmark_track", "evaluator_track", "dependency_profile", "carla_runtime_root"):
        if not isinstance(bundle.get(key), str) or not bundle[key]:
            return False
    for key in (
        "carla_runtime_asset_file_count", "carla_runtime_asset_total_size_bytes",
        "carla_runtime_available_map_count", "carla_runtime_freeze_monotonic_ticks",
        "carla_runtime_clock_ticks_per_second",
        "carla_runtime_server_readonly_permission_entry_count",
        "carla_runtime_server_uid",
    ):
        if type(bundle.get(key)) is not int or bundle[key] < 0:
            return False
    if (
        not isinstance(bundle.get("carla_runtime_server_gids"), list)
        or not bundle["carla_runtime_server_gids"]
        or any(type(value) is not int or value < 0 for value in bundle["carla_runtime_server_gids"])
        or bundle["carla_runtime_server_gids"] != sorted(set(bundle["carla_runtime_server_gids"]))
    ):
        return False
    return workload is None or bundle.get("workload") == workload


def _layer(run: dict[str, Any]) -> str:
    explicit = run.get("layer")
    if explicit in {"A", "B", "C", "D"}:
        return explicit
    prefix = str(run.get("pcu_id", "")).split("-", 1)[0]
    return prefix if prefix in {"A", "B", "C", "D"} else "unknown"


def _status_class(value: Any) -> str:
    return value if type(value) is str and value in _STRICT_CORE_STATUSES else "invalid_status"


def _trace_state(path: Path) -> tuple[bool, str | None]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False, "trace_missing"
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        return False, "trace_not_real_regular_file"
    try:
        row_count = 0
        previous_time: float | None = None
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(
                    line, object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(value, dict):
                    return False, f"trace_row_{line_number}_not_object"
                sim_time = value.get("sim_time")
                if (not isinstance(sim_time, (int, float)) or isinstance(sim_time, bool)
                        or not math.isfinite(float(sim_time))):
                    return False, f"trace_row_{line_number}_invalid_sim_time"
                current_time = float(sim_time)
                if previous_time is not None and current_time < previous_time:
                    return False, f"trace_row_{line_number}_nonmonotonic_sim_time"
                previous_time = current_time
                row_count += 1
        if row_count:
            return True, None
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        return False, f"trace_invalid:{type(error).__name__}"
    return False, "trace_empty"


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read expected-run ledger {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("expected-run ledger must be an object containing a runs list")

    runs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(payload["runs"]):
        if not isinstance(raw, dict):
            raise ValueError(f"ledger run {index} must be an object")
        try:
            identity = ArmIdentity(raw["pcu_id"], raw["arm_id"], raw["paraphrase"], raw["seed"], raw["track"])
        except (KeyError, SchemaError) as error:
            raise ValueError(f"invalid ledger run {index}: {error}") from error
        expected_run_id = identity.run_id
        if raw.get("run_id") != expected_run_id:
            raise ValueError(f"ledger run {index} has run_id inconsistent with its identity")
        identity_key = (identity.track, identity.pcu_id, identity.arm_id, identity.paraphrase, identity.seed)
        if expected_run_id in seen_ids or identity_key in seen_identities:
            raise ValueError(f"duplicate expected run {expected_run_id}")
        seen_ids.add(expected_run_id)
        seen_identities.add(identity_key)
        runs.append(dict(raw))
    return runs


def _read_result(
    run: dict[str, Any],
    results_dir: Path,
    *,
    expected_input_hashes: dict[str, Any] | None = None,
    result_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = results_dir / run["run_id"]
    try:
        run_dir_metadata = run_dir.lstat()
        run_dir_exists = True
        run_dir_safe = not run_dir.is_symlink() and stat.S_ISDIR(run_dir_metadata.st_mode)
    except FileNotFoundError:
        run_dir_exists = False
        run_dir_safe = False
    result_path = run_dir / "contract_result.json"
    trace_path = run_dir / "trace.jsonl"
    observed = (run_dir_safe and result_path.is_file()) or result_override is not None
    base = {**run, "layer": _layer(run), "observed": observed}

    def rejected(
        status: str, error: str, *, trace_present: bool = False,
        trace_error: str | None = None,
    ) -> dict[str, Any]:
        return {
            **base, "status": status, "success": False,
            "semantic_success": False,
            "declared_success": False, "trace_present": trace_present,
            "trace_error": trace_error, "result_valid": False,
            "finalized": False, "result_error": error,
            "navigation_diagnostics": None, "verdict_complete": False,
            **{field: False for field in _VERDICT_FIELDS},
        }

    if run_dir_exists and not run_dir_safe:
        return rejected("stale_provenance", "run_directory_not_real")

    if not observed:
        return {**base, "status": "missing", "success": False, "trace_present": False,
                "semantic_success": False,
                "trace_error": "trace_not_checked_without_result", "result_valid": False,
                "finalized": False, "declared_success": False,
                "navigation_diagnostics": None, "verdict_complete": False,
                "own_contract": False, "opposite_contract": False, "safe_legal": False,
                "progress": False, "deadline": False, "stability": False}

    # A score is current only when the caller supplies the freshly resolved,
    # complete bundle and the per-run receipt binds that exact bundle and run.
    workload = (
        expected_input_hashes.get("workload")
        if isinstance(expected_input_hashes, dict) else None
    )
    if not _execution_bundle_is_full(expected_input_hashes, workload=workload):
        return rejected("stale_provenance", "expected_execution_bundle_missing_or_incomplete")
    if expected_input_hashes.get("benchmark_track") != run.get("track"):
        return rejected("stale_provenance", "execution_bundle_benchmark_track_mismatch")
    try:
        runner_command = _strict_json_object(run_dir / "runner_command.json")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        return rejected("stale_provenance", f"runner_command_invalid:{type(error).__name__}")
    if (
        not isinstance(runner_command.get("run"), dict)
        or stable_hash(runner_command["run"]) != stable_hash(run)
    ):
        return rejected("stale_provenance", "runner_command_run_identity_mismatch")
    observed_bundle = runner_command.get("input_hashes")
    if (
        not _execution_bundle_is_full(observed_bundle, workload=workload)
        or observed_bundle != expected_input_hashes
    ):
        return rejected("stale_provenance", "execution_bundle_mismatch_or_incomplete")

    if result_override is not None:
        if not isinstance(result_override, dict):
            return rejected("invalid_result", "in_memory_result_not_object")
        result = dict(result_override)
    else:
        try:
            result = _strict_json_object(result_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            return rejected("invalid_result", f"result_json_invalid:{type(error).__name__}")

    status = _status_class(result.get("status"))
    trace_present, trace_error = _trace_state(trace_path)
    if status == "invalid_status":
        return rejected(status, "missing_or_noncanonical_status", trace_present=trace_present, trace_error=trace_error)
    if result.get("schema_version") != "language2drive.contract_result.v1":
        return rejected("invalid_result", "missing_or_noncanonical_result_schema", trace_present=trace_present, trace_error=trace_error)
    if result.get("scorer") != "language2drive.v1.offline_contract_engine":
        return rejected("invalid_result", "missing_or_noncanonical_scorer", trace_present=trace_present, trace_error=trace_error)
    if any(
        type(result.get(field)) is not type(run.get(field))
        or result.get(field) != run.get(field)
        for field in _RESULT_IDENTITY_FIELDS
    ):
        return rejected("invalid_result", "result_identity_mismatch", trace_present=trace_present, trace_error=trace_error)
    if type(result.get("success")) is not bool:
        return rejected("invalid_result", "success_must_be_boolean", trace_present=trace_present, trace_error=trace_error)
    if type(result.get("capture_complete")) is not bool:
        return rejected("invalid_result", "capture_complete_must_be_boolean", trace_present=trace_present, trace_error=trace_error)
    if "driving_score" in result and result["driving_score"] is not None and not valid_driving_score(result["driving_score"]):
        return rejected("invalid_result", "driving_score_must_be_finite_percent", trace_present=trace_present, trace_error=trace_error)
    if any(type(result.get(field)) is not bool for field in _VERDICT_FIELDS):
        return rejected("invalid_result", "complete_six_field_verdict_required", trace_present=trace_present, trace_error=trace_error)

    components = {field: result[field] for field in _VERDICT_FIELDS}
    declared_semantic_success = result.get(
        "semantic_success", components["own_contract"]
    )
    if type(declared_semantic_success) is not bool:
        return rejected(
            "invalid_result",
            "semantic_success_must_be_boolean",
            trace_present=trace_present,
            trace_error=trace_error,
        )
    component_success = (
        components["own_contract"] and not components["opposite_contract"] and components["safe_legal"]
        and components["progress"] and components["deadline"] and components["stability"]
    )
    declared_success = result["success"]
    if declared_success != component_success:
        return rejected("inconsistent_result", "success_differs_from_six_field_conjunction", trace_present=trace_present, trace_error=trace_error)
    if status != "complete" and declared_success:
        return rejected("inconsistent_result", "noncomplete_status_declares_success", trace_present=trace_present, trace_error=trace_error)

    capture_complete = result["capture_complete"]
    if status == "complete" and capture_complete is not True:
        return rejected("capture_failure", "complete_result_without_complete_capture", trace_present=trace_present, trace_error=trace_error)
    if status != "complete" and capture_complete is not False:
        return rejected("inconsistent_result", "noncomplete_result_claims_complete_capture", trace_present=trace_present, trace_error=trace_error)

    result_valid = status == "complete" and capture_complete and trace_present
    success = result_valid and declared_success
    if status == "complete" and not trace_present:
        status = "no_trace"
    finalized = result_valid or status in _FINAL_TERMINAL_STATUSES
    return {
        **base,
        "status": status,
        "success": success,
        "semantic_success": result_valid and declared_semantic_success,
        "declared_success": declared_success,
        "trace_present": trace_present,
        "trace_error": trace_error,
        "result_valid": result_valid,
        "finalized": finalized,
        "navigation_diagnostics": result.get("navigation_diagnostics"),
        "verdict_complete": True,
        "capture_complete": capture_complete,
        "driving_score": result.get("driving_score") if result_valid else None,
        **components,
    }


def _ratio(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _pair_results(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], bool]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["track"], row["pcu_id"], row["paraphrase"], row["seed"])].append(row)
    values: dict[tuple[str, str, int, int], bool] = {}
    for key, pair in grouped.items():
        arm_ids = {row["arm_id"] for row in pair}
        values[key] = len(pair) == 2 and len(arm_ids) == 2 and all(row["success"] for row in pair)
    return values


def _leaderboard_summary(
    rows: list[dict[str, Any]], pair_values: dict[tuple[str, str, int, int], bool]
) -> dict[str, Any]:
    pair_layers = {key: _layer({"pcu_id": key[1]}) for key in pair_values}
    cfps = {
        layer: _ratio(value for key, value in pair_values.items() if pair_layers[key] == layer)
        for layer in "ABCD"
    }
    cfps_counts = {
        layer: {
            "passed": sum(value for key, value in pair_values.items() if pair_layers[key] == layer),
            "expected": sum(1 for key in pair_values if pair_layers[key] == layer),
        }
        for layer in "ABCD"
    }
    paraphrase_values: dict[int, list[bool]] = defaultdict(list)
    semantic_seed_groups: dict[tuple[str, str, int], dict[int, bool]] = defaultdict(dict)
    for (track, pcu_id, paraphrase, seed), value in pair_values.items():
        paraphrase_values[paraphrase].append(value)
        semantic_seed_groups[(track, pcu_id, seed)][paraphrase] = value
    cfps_by_paraphrase = {str(index): _ratio(paraphrase_values[index]) for index in (0, 1, 2)}
    worst_units = [set(values) == {0, 1, 2} and all(values.values()) for values in semantic_seed_groups.values()]
    worst_by_layer = {
        layer: _ratio(
            set(values) == {0, 1, 2} and all(values.values())
            for (_, pcu_id, _), values in semantic_seed_groups.items()
            if _layer({"pcu_id": pcu_id}) == layer
        )
        for layer in "ABCD"
    }
    sisr_by_layer = {layer: _ratio(row["success"] for row in rows if row["layer"] == layer) for layer in "ABCD"}
    semantic_by_layer = {
        layer: _ratio(
            row["semantic_success"] for row in rows if row["layer"] == layer
        )
        for layer in "ABCD"
    }
    safety_by_layer = {layer: _ratio(row["safe_legal"] for row in rows if row["layer"] == layer) for layer in "ABCD"}
    progress_by_layer = {layer: _ratio(row["progress"] for row in rows if row["layer"] == layer) for layer in "ABCD"}
    score_values = [float(row["driving_score"]) for row in rows if row.get("result_valid") and valid_driving_score(row.get("driving_score"))]
    driving_score_by_layer = {}
    driving_score_valid_runs_by_layer = {}
    for layer in "ABCD":
        values = [float(row["driving_score"]) for row in rows if row["layer"] == layer and row.get("result_valid") and valid_driving_score(row.get("driving_score"))]
        driving_score_by_layer[layer] = sum(values) / len(values) if values else None
        driving_score_valid_runs_by_layer[layer] = len(values)
    return {
        "expected_runs": len(rows),
        "CFPS": cfps,
        "CFPS_counts": cfps_counts,
        "CFPS_by_paraphrase": cfps_by_paraphrase,
        # Unit-level worst case: all three wording variants must achieve strict
        # paired success for a PCU/seed.
        "Worst-paraphrase": _ratio(worst_units),
        "Worst-paraphrase_by_layer": worst_by_layer,
        "SISR": _ratio(row["success"] for row in rows),
        "SISR_by_layer": sisr_by_layer,
        "SemISR": _ratio(row["semantic_success"] for row in rows),
        "SemISR_by_layer": semantic_by_layer,
        "Full-Program-Safe-Success_D": sisr_by_layer["D"],
        "Safety": _ratio(row["safe_legal"] for row in rows),
        "Safety_by_layer": safety_by_layer,
        "Progress": _ratio(row["progress"] for row in rows),
        "Progress_by_layer": progress_by_layer,
        "DrivingScore": sum(score_values) / len(score_values) if score_values else None,
        "DrivingScore_by_layer": driving_score_by_layer,
        "DrivingScore_valid_runs": len(score_values),
        "DrivingScore_valid_runs_by_layer": driving_score_valid_runs_by_layer,
    }


def aggregate(
    ledger_path: Path,
    results_dir: Path,
    *,
    expected_input_hashes: dict[str, Any] | None = None,
    result_overrides: dict[str, dict[str, Any]] | None = None,
    materialized_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate exactly the expected runs; absent/invalid/crashed runs stay in denominators."""

    expected = (
        [dict(run) for run in materialized_runs]
        if materialized_runs is not None else _load_ledger(Path(ledger_path))
    )
    def bundle_for(run: dict[str, Any]) -> dict[str, Any] | None:
        if isinstance(expected_input_hashes, dict) and expected_input_hashes.get("schema_version") == "language2drive.execution_bundle.v2":
            return expected_input_hashes
        if isinstance(expected_input_hashes, dict):
            candidate = expected_input_hashes.get(str(run.get("track")))
            return candidate if isinstance(candidate, dict) else None
        return None

    rows = [
        _read_result(
            run, Path(results_dir), expected_input_hashes=bundle_for(run),
            result_override=(result_overrides or {}).get(run["run_id"]),
        )
        for run in expected
    ]
    pair_values = _pair_results(rows)
    tracks = sorted({row["track"] for row in rows})
    leaderboard_track = "standardized" if "standardized" in tracks else (tracks[0] if tracks else "standardized")
    leaderboard_rows = [row for row in rows if row["track"] == leaderboard_track]
    leaderboard_pairs = {key: value for key, value in pair_values.items() if key[0] == leaderboard_track}
    leaderboard = _leaderboard_summary(leaderboard_rows, leaderboard_pairs)
    metrics_by_track = {
        track: _leaderboard_summary(
            [row for row in rows if row["track"] == track],
            {key: value for key, value in pair_values.items() if key[0] == track},
        )
        for track in tracks
    }
    status_counts = Counter(row["status"] for row in rows)

    expected_count = len(rows)
    observed_count = sum(row["observed"] for row in rows)
    logged_count = sum(row["trace_present"] for row in rows)
    valid_count = sum(row["result_valid"] for row in rows)
    finalized_count = sum(row.get("finalized", False) for row in rows)
    oracle_rows = [row for row in rows if row["track"] == "oracle"]
    output = {
        "expected_runs": expected_count,
        "observed_runs": observed_count,
        "recorded_runs": sum(row["observed"] and row["status"] != "invalid_result" for row in rows),
        "logged_runs": logged_count,
        "valid_result_runs": valid_count,
        "finalized_runs": finalized_count,
        "missing_runs": status_counts["missing"],
        "crashed_runs": status_counts["crash"],
        "timeout_runs": status_counts["timeout"],
        "deadlock_runs": status_counts["deadlock"],
        "no_trace_runs": status_counts["no_trace"],
        "invalid_trace_runs": status_counts["invalid_trace"],
        "capture_failure_runs": status_counts["capture_failure"],
        "stale_provenance_runs": status_counts["stale_provenance"],
        "invalid_runs": sum(count for status, count in status_counts.items() if status in {
            "invalid_result", "invalid_status", "inconsistent_result"
        }),
        "status_counts": dict(sorted(status_counts.items())),
        "coverage": logged_count / expected_count if expected_count else 0.0,
        "result_coverage": valid_count / expected_count if expected_count else 0.0,
        "success_count": sum(row["success"] for row in rows),
        "leaderboard_track": leaderboard_track,
        "metrics_by_track": metrics_by_track,
        "oracle_witness": {
            "passed": sum(row["success"] for row in oracle_rows),
            "expected": len(oracle_rows),
            "pass_rate": _ratio(row["success"] for row in oracle_rows),
        },
        **{key: value for key, value in leaderboard.items() if key != "expected_runs"},
        "FCS-2x2": {"value": None, "reason": "fixed_layout_only"},
        "train_overlap": "unknown",
        "rows": rows,
    }
    return output


def score_to_file(ledger_path: Path, results_dir: Path, output: Path) -> dict[str, Any]:
    value = aggregate(ledger_path, results_dir)
    atomic_json(output, value)
    return value
