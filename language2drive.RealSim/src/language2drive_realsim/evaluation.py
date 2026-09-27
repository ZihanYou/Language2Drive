"""Evaluate frozen model rollouts while preserving the complete denominator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .constants import ARMS, COUPLED_SEEDS
from .io import load_jsonl
from .metrics import aggregate_results
from .verifier import verify_trace


def evaluate_rollouts(
    source: Dict[str, Any],
    sidecar: Dict[str, Any],
    evidence_dir: Path,
    threshold_profile: Dict[str, Any],
) -> Dict[str, Any]:
    pcu_by_id = {pcu["pcu_id"]: pcu for pcu in source.get("pcus", [])}
    sessions_by_identity = {}
    duplicate_identities = []
    for session_uuid, session in sidecar.get("sessions", {}).items():
        identity = (session.get("pcu_id"), session.get("arm_id"), session.get("seed"))
        if identity in sessions_by_identity:
            duplicate_identities.append(identity)
        sessions_by_identity[identity] = (session_uuid, session)

    rows: List[Dict[str, Any]] = []
    expected_identities = {
        (pcu_id, arm_id, seed)
        for pcu_id in pcu_by_id
        for arm_id in ARMS
        for seed in COUPLED_SEEDS
    }
    for pcu_id in sorted(pcu_by_id):
        pcu = pcu_by_id[pcu_id]
        for seed in COUPLED_SEEDS:
            for arm_id in ARMS:
                entry = sessions_by_identity.get((pcu_id, arm_id, seed))
                if entry is None:
                    session_uuid = None
                    result = {
                        "passed": False,
                        "failure": "INVALID_SIM",
                        "diagnostics": {"reason": "missing_compiled_session"},
                    }
                else:
                    session_uuid, session = entry
                    trace_path = Path(evidence_dir) / (session_uuid + ".jsonl")
                    if not trace_path.is_file():
                        result = {
                            "passed": False,
                            "failure": "INVALID_SIM",
                            "diagnostics": {"reason": "missing_trace"},
                        }
                    else:
                        try:
                            result = verify_trace(
                                pcu,
                                arm_id,
                                load_jsonl(trace_path),
                                pcu.get("private_annotations", {}),
                                threshold_profile,
                            ).to_dict()
                        except (OSError, TypeError, ValueError, KeyError) as error:
                            result = {
                                "passed": False,
                                "failure": "INVALID_SIM",
                                "diagnostics": {"reason": str(error)},
                            }
                rows.append(
                    dict(
                        result,
                        session_uuid=session_uuid,
                        pcu_id=pcu_id,
                        arm_id=arm_id,
                        seed=seed,
                    )
                )
    expected = len(source.get("pcus", [])) * len(ARMS) * len(COUPLED_SEEDS)
    sidecar_identity_complete = (
        not duplicate_identities
        and set(sessions_by_identity) == expected_identities
        and len(sidecar.get("sessions", {})) == expected
    )
    denominator_complete = (
        expected > 0 and len(rows) == expected and sidecar_identity_complete
    )
    metrics = aggregate_results(rows)
    return {
        "schema_version": "l2d.realsim.evaluation.v0.1",
        "expected_rollouts": expected,
        "observed_rollouts": len(rows),
        "denominator_complete": denominator_complete,
        "sidecar_identity_complete": sidecar_identity_complete,
        "duplicate_sidecar_identities": duplicate_identities,
        "results": rows,
        "metrics": metrics,
    }
