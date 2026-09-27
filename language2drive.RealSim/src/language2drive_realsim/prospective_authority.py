"""Prospective request and paired-prefix authority for live model campaigns."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from .io import atomic_json, load_json, stable_hash
from .runtime_client import simulation_request_payload


REQUEST_SCHEMA = "l2d.realsim.prospective_request.v1"
PREFIX_SCHEMA = "l2d.realsim.prospective_paired_prefix.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_request_authority(
    pair_directory: Path,
    plan: Dict[str, Any],
    *,
    runtime_address: str,
    driver_address: str,
    concurrency: int,
    use_registered_driver: bool,
    manifest_binding: str,
    preparation_binding: str,
    formal_candidate: bool,
) -> Dict[str, Dict[str, Any]]:
    """Exclusively publish per-slot records before the corresponding RPC.

    The exact request payload comes from the helper consumed by
    :func:`submit_run_plan`.  Existing records are never overwritten.
    """

    pair_directory = Path(pair_directory)
    payload = simulation_request_payload(
        plan, driver_address, concurrency, use_registered_driver
    )
    payload_binding = stable_hash(payload)
    rows = plan["runs"]
    pair_ids = {row.get("pcu_id") for row in rows}
    seeds = {row.get("random_seed") for row in rows}
    arms = {row.get("arm_id") for row in rows}
    if len(rows) != 2 or len(pair_ids) != 1 or len(seeds) != 1 or arms != {"R0", "R1"}:
        raise ValueError("prospective request authority requires one complete R0/R1 pair")
    recorded_at = _utc_now()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        session_uuid = row["session_uuid"]
        path = pair_directory / "requests" / f"{session_uuid}.json"
        record: Dict[str, Any] = {
            "schema_version": REQUEST_SCHEMA,
            "recorded_at_utc": recorded_at,
            "recorded_before_submission": True,
            "formal_candidate": bool(formal_candidate),
            "session_uuid": session_uuid,
            "pcu_id": row["pcu_id"],
            "arm_id": row["arm_id"],
            "seed": row["random_seed"],
            "request_path": str(path.resolve()),
            "runtime_address": runtime_address,
            "driver_address": driver_address,
            "manifest_binding": manifest_binding,
            "preparation_binding": preparation_binding,
            "simulation_request_payload": payload,
            "simulation_request_binding": payload_binding,
        }
        record["request_binding"] = stable_hash(record)
        atomic_json(path, record, overwrite=False)
        result[session_uuid] = record
    return result


def validate_request_authority(
    record: Dict[str, Any],
    *,
    session_uuid: str,
    manifest_binding: str,
    preparation_binding: str,
) -> list[str]:
    errors = []
    if record.get("schema_version") != REQUEST_SCHEMA:
        errors.append("request schema mismatch")
    if record.get("recorded_before_submission") is not True:
        errors.append("request was not prospectively recorded")
    if record.get("session_uuid") != session_uuid:
        errors.append("request session mismatch")
    if record.get("manifest_binding") != manifest_binding:
        errors.append("request manifest mismatch")
    if record.get("preparation_binding") != preparation_binding:
        errors.append("request preparation mismatch")
    payload = record.get("simulation_request_payload")
    if not isinstance(payload, dict) or stable_hash(payload) != record.get(
        "simulation_request_binding"
    ):
        errors.append("simulation request payload binding mismatch")
    candidate = dict(record)
    binding = candidate.pop("request_binding", None)
    if stable_hash(candidate) != binding:
        errors.append("request record binding mismatch")
    return errors


def write_paired_prefix_authority(
    pair_directory: Path,
    plan: Dict[str, Any],
    receipts: Iterable[Dict[str, Any]],
    request_records: Dict[str, Dict[str, Any]],
    *,
    manifest_binding: str,
    preparation_binding: str,
    formal_candidate: bool,
) -> Dict[str, Any]:
    """Publish one same-prefix attestation only after both receipts exist."""

    rows = plan["runs"]
    receipt_by_session = {row.get("session_uuid"): row for row in receipts}
    sessions = {row["session_uuid"] for row in rows}
    if len(rows) != 2 or set(receipt_by_session) != sessions:
        raise ValueError("paired-prefix authority requires two exact session receipts")
    prefix_values = {
        receipt_by_session[session].get("preintervention_non_language_sha256")
        for session in sessions
    }
    if len(prefix_values) != 1 or not next(iter(prefix_values), None):
        raise ValueError("paired non-language prefixes are absent or different")
    for session in sessions:
        request = request_records.get(session)
        if not isinstance(request, dict) or validate_request_authority(
            request,
            session_uuid=session,
            manifest_binding=manifest_binding,
            preparation_binding=preparation_binding,
        ):
            raise ValueError("paired-prefix authority lacks valid prospective requests")
    prefix = next(iter(prefix_values))
    record: Dict[str, Any] = {
        "schema_version": PREFIX_SCHEMA,
        "recorded_at_utc": _utc_now(),
        "formal_candidate": bool(formal_candidate),
        "pcu_id": rows[0]["pcu_id"],
        "seed": rows[0]["random_seed"],
        "manifest_binding": manifest_binding,
        "preparation_binding": preparation_binding,
        "non_language_prefix_binding": prefix,
        "slots": [
            {
                "session_uuid": row["session_uuid"],
                "arm_id": row["arm_id"],
                "request_path": request_records[row["session_uuid"]]["request_path"],
                "request_binding": request_records[row["session_uuid"]]["request_binding"],
                "source_asl_sha256": receipt_by_session[row["session_uuid"]].get(
                    "source_asl_sha256"
                ),
            }
            for row in rows
        ],
    }
    record["paired_prefix_binding"] = stable_hash(record)
    atomic_json(
        Path(pair_directory) / "paired-prefix-authority.json",
        record,
        overwrite=False,
    )
    return record


def validate_paired_prefix_authority(
    record: Dict[str, Any],
    *,
    manifest_binding: str,
    preparation_binding: str,
) -> list[str]:
    errors = []
    if record.get("schema_version") != PREFIX_SCHEMA:
        errors.append("paired-prefix schema mismatch")
    if record.get("manifest_binding") != manifest_binding:
        errors.append("paired-prefix manifest mismatch")
    if record.get("preparation_binding") != preparation_binding:
        errors.append("paired-prefix preparation mismatch")
    slots = record.get("slots")
    if (
        not isinstance(slots, list)
        or len(slots) != 2
        or {row.get("arm_id") for row in slots if isinstance(row, dict)} != {"R0", "R1"}
        or len({row.get("session_uuid") for row in slots if isinstance(row, dict)}) != 2
    ):
        errors.append("paired-prefix slots are not one complete pair")
    if not isinstance(record.get("non_language_prefix_binding"), str) or not record[
        "non_language_prefix_binding"
    ]:
        errors.append("paired-prefix binding is absent")
    candidate = dict(record)
    binding = candidate.pop("paired_prefix_binding", None)
    if stable_hash(candidate) != binding:
        errors.append("paired-prefix record binding mismatch")
    return errors


def load_completed_pair_authority(
    pair_directory: Path,
    *,
    manifest_binding: str,
    preparation_binding: str,
) -> Dict[str, Any]:
    """Read and strictly validate one terminal pair for resume accounting."""

    pair_directory = Path(pair_directory)
    prefix = load_json(pair_directory / "paired-prefix-authority.json")
    errors = validate_paired_prefix_authority(
        prefix,
        manifest_binding=manifest_binding,
        preparation_binding=preparation_binding,
    )
    if prefix.get("formal_candidate") is not True:
        errors.append("paired-prefix record is not formal")
    for slot in prefix.get("slots", []):
        path = Path(slot.get("request_path", ""))
        try:
            path.resolve().relative_to(pair_directory.resolve())
        except (ValueError, OSError):
            errors.append("request path escapes pair directory")
            continue
        if not path.is_file():
            errors.append("request record is missing")
            continue
        request = load_json(path)
        if request.get("formal_candidate") is not True:
            errors.append("request record is not formal")
        errors.extend(
            validate_request_authority(
                request,
                session_uuid=slot.get("session_uuid"),
                manifest_binding=manifest_binding,
                preparation_binding=preparation_binding,
            )
        )
        if request.get("request_binding") != slot.get("request_binding"):
            errors.append("prefix record request binding mismatch")
    if errors:
        raise ValueError("invalid completed pair authority: " + repr(errors))
    return prefix
