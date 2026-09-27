"""Fail-closed structural checks for PCU qualification report bundles.

This module checks JSON structure and cross-document consistency only.  It does
not read ASL files or model weights, and it does not establish raw trajectory
authenticity, execution provenance, model-evaluation completion, or release
authority.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Mapping, Tuple

from .constants import FAILURES
from .io import stable_hash
from .schema import validate_source


QUALIFICATION_SCHEMA = "l2d.realsim.qualification.v0.1"
WITNESS_SCHEMA = "l2d.realsim.oracle_witness_index.v0.1"
INTEGRITY_SCHEMA = "l2d.realsim.cohort_integrity.v0.1"

_GATES = (
    "source_schema_and_release_profile",
    "qualification_schema_and_source_binding",
    "qualification_global_summary",
    "qualification_rows_match_source",
    "qualification_tuple_coverage",
    "witness_schema_and_source_binding",
    "witness_summary_matches_expected",
    "witness_tuple_coverage",
    "qualification_witness_records_match",
)


class _IntegrityError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _IntegrityError(message)


def _source_contract(source: Any) -> Dict[str, Any]:
    validate_source(source, require_exact_release=True)
    pcus = source["pcus"]
    ids = [pcu["pcu_id"] for pcu in pcus]
    _require(len(ids) == len(set(ids)), "source PCU IDs are not unique")
    by_id = {pcu["pcu_id"]: pcu for pcu in pcus}
    expected = set()
    for pcu in pcus:
        arms = tuple(pcu["arms"])
        seeds = tuple(
            pcu["shared_non_language_input"]["coupled_exogenous_seeds"]
        )
        _require(all(type(seed) is int for seed in seeds), "source seeds must be integers")
        _require(len(arms) == len(set(arms)), "source arms are not unique")
        _require(len(seeds) == len(set(seeds)), "source seeds are not unique")
        expected.update((pcu["pcu_id"], arm, seed) for arm in arms for seed in seeds)
    return {
        "pcus": pcus,
        "ids": ids,
        "by_id": by_id,
        "expected": expected,
        "source_binding": stable_hash(source),
    }


def _qualification_header(
    qualification: Any, source_binding: str
) -> None:
    _require(isinstance(qualification, dict), "qualification must be an object")
    _require(
        qualification.get("schema_version") == QUALIFICATION_SCHEMA,
        "unsupported qualification schema",
    )
    _require(
        qualification.get("source_binding") == source_binding,
        "qualification source binding mismatch",
    )


def _qualification_summary(qualification: Mapping[str, Any], count: int) -> None:
    _require(qualification.get("pass") is True, "qualification pass is not true")
    _require(
        qualification.get("all_global_gates_pass") is True,
        "qualification global gates did not pass",
    )
    _require(
        qualification.get("global_errors") == [],
        "qualification global_errors is not empty",
    )
    for field in ("eligible_count", "target_pcu_count", "candidate_count"):
        value = qualification.get(field)
        _require(
            type(value) is int and value == count,
            "qualification %s does not equal %d" % (field, count),
        )


def _qualification_records(
    qualification: Mapping[str, Any], contract: Mapping[str, Any]
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    rows = qualification.get("pcus")
    ids = contract["ids"]
    _require(isinstance(rows, list), "qualification pcus must be a list")
    _require(len(rows) == len(ids), "qualification row count mismatch")
    _require(
        Counter(row.get("pcu_id") for row in rows if isinstance(row, dict))
        == Counter(ids),
        "qualification PCU IDs do not exactly match source",
    )
    nested: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "qualification row must be an object")
        pcu = contract["by_id"][row["pcu_id"]]
        _require(
            row.get("eligible") is True and row.get("errors") == [],
            "qualification PCU is not cleanly eligible: %s" % row["pcu_id"],
        )
        for field in ("primary_layer", "task_family"):
            _require(
                row.get(field) == pcu[field],
                "qualification %s disagrees with source for %s"
                % (field, row["pcu_id"]),
            )
        expected_seeds = tuple(
            pcu["shared_non_language_input"]["coupled_exogenous_seeds"]
        )
        expected_arms = set(pcu["arms"])
        seed_rows = row.get("seeds")
        _require(isinstance(seed_rows, list), "qualification seeds must be a list")
        _require(
            len(seed_rows) == len(expected_seeds),
            "qualification seed count mismatch for %s" % row["pcu_id"],
        )
        seen_seeds = set()
        for seed_row in seed_rows:
            _require(isinstance(seed_row, dict), "qualification seed row must be an object")
            seed = seed_row.get("seed")
            _require(
                type(seed) is int
                and seed in expected_seeds
                and seed not in seen_seeds,
                "unexpected or duplicate qualification seed for %s" % row["pcu_id"],
            )
            seen_seeds.add(seed)
            _require(
                seed_row.get("qualified") is True,
                "qualification seed is not qualified for %s" % row["pcu_id"],
            )
            arms = seed_row.get("arms")
            _require(
                isinstance(arms, dict) and set(arms) == expected_arms,
                "qualification arms mismatch for %s/%s" % (row["pcu_id"], seed),
            )
            for arm, record in arms.items():
                _require(isinstance(record, dict), "qualification arm record must be an object")
                key = (row["pcu_id"], arm, seed)
                _require(key not in nested, "duplicate qualification tuple")
                _require(
                    record.get("arm_id") == arm
                    and type(record.get("seed")) is int
                    and record["seed"] == seed,
                    "nested qualification identity mismatch",
                )
                _require(
                    record.get("qualified") is True and record.get("errors") == [],
                    "nested qualification record is not cleanly qualified",
                )
                nested[key] = record
        _require(
            seen_seeds == set(expected_seeds),
            "qualification seed coverage mismatch for %s" % row["pcu_id"],
        )
    _require(
        set(nested) == contract["expected"],
        "qualification tuple coverage mismatch",
    )
    return nested


def _witness_header(witnesses: Any, source_binding: str) -> None:
    _require(isinstance(witnesses, dict), "witness index must be an object")
    _require(
        witnesses.get("schema_version") == WITNESS_SCHEMA,
        "unsupported witness schema",
    )
    _require(
        witnesses.get("source_binding") == source_binding,
        "witness source binding mismatch",
    )


def _witness_summary(witnesses: Mapping[str, Any], expected_count: int) -> None:
    _require(
        witnesses.get("coverage_complete") is True,
        "witness coverage is not declared complete",
    )
    for field in ("coverage_numerator", "coverage_denominator"):
        value = witnesses.get(field)
        _require(
            type(value) is int and value == expected_count,
            "witness %s does not equal %d" % (field, expected_count),
        )


def _witness_records(
    witnesses: Mapping[str, Any], expected: set
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    records = witnesses.get("witnesses")
    _require(isinstance(records, list), "witnesses must be a list")
    _require(len(records) == len(expected), "witness row count mismatch")
    seen: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    sessions = set()
    for record in records:
        _require(isinstance(record, dict), "witness record must be an object")
        key = (record.get("pcu_id"), record.get("arm_id"), record.get("seed"))
        _require(
            type(record.get("seed")) is int and key in expected and key not in seen,
            "unexpected or duplicate witness tuple",
        )
        session = record.get("session_uuid")
        _require(
            isinstance(session, str)
            and bool(session.strip())
            and session not in sessions,
            "missing or reused witness session",
        )
        sessions.add(session)
        _require(
            record.get("qualified") is True and record.get("errors") == [],
            "witness is not cleanly qualified",
        )
        own = record.get("own_verdict")
        opposing = record.get("opposing_verdict")
        _require(
            isinstance(own, dict)
            and own.get("passed") is True
            and own.get("failure") is None,
            "own witness verdict is invalid",
        )
        _require(
            isinstance(opposing, dict)
            and opposing.get("passed") is False
            and opposing.get("failure") in FAILURES - {"INVALID_SIM"},
            "opposing witness verdict is invalid",
        )
        _require(
            record.get("render_diagnostics", {}).get("valid") is True,
            "witness render validity is missing",
        )
        trace_hash = record.get("normalized_trace_sha256")
        _require(
            isinstance(trace_hash, str)
            and len(trace_hash) == 64
            and all(character in "0123456789abcdef" for character in trace_hash),
            "witness normalized trace binding is malformed",
        )
        seen[key] = record
    _require(set(seen) == expected, "witness tuple coverage mismatch")
    return seen


def _records_match(
    qualification_records: Mapping[Tuple[str, str, int], Dict[str, Any]],
    witness_records: Mapping[Tuple[str, str, int], Dict[str, Any]],
) -> None:
    _require(
        set(qualification_records) == set(witness_records),
        "qualification and witness tuple sets disagree",
    )
    for key, witness in witness_records.items():
        without_pcu = {name: value for name, value in witness.items() if name != "pcu_id"}
        _require(
            without_pcu == qualification_records[key],
            "qualification and witness records disagree for %s/%s/%s" % key,
        )


def audit_cohort_integrity(
    source: Any, qualification: Any, witnesses: Any
) -> Dict[str, Any]:
    """Return structural gates and errors for a qualification report bundle.

    A passing result means only that the supplied source, qualification report,
    and witness index are internally consistent at the JSON-report layer.
    """

    gates = {name: False for name in _GATES}
    errors = []

    def run(gate: str, check: Callable[[], Any]) -> Any:
        try:
            result = check()
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            errors.append("%s: %s" % (gate, error))
            return None
        gates[gate] = True
        return result

    contract = run(
        "source_schema_and_release_profile", lambda: _source_contract(source)
    )
    if contract is None:
        for gate in _GATES[1:]:
            errors.append("%s: source contract unavailable" % gate)
    else:
        run(
            "qualification_schema_and_source_binding",
            lambda: _qualification_header(qualification, contract["source_binding"]),
        )
        run(
            "qualification_global_summary",
            lambda: _qualification_summary(qualification, len(contract["pcus"])),
        )
        qualification_records = run(
            "qualification_rows_match_source",
            lambda: _qualification_records(qualification, contract),
        )
        # Row validation includes complete arm/seed coverage. Keep this as a
        # separate consumable gate even though both are established together.
        if qualification_records is not None:
            gates["qualification_tuple_coverage"] = True
        else:
            errors.append(
                "qualification_tuple_coverage: qualification rows unavailable"
            )
        run(
            "witness_schema_and_source_binding",
            lambda: _witness_header(witnesses, contract["source_binding"]),
        )
        run(
            "witness_summary_matches_expected",
            lambda: _witness_summary(witnesses, len(contract["expected"])),
        )
        witness_records = run(
            "witness_tuple_coverage",
            lambda: _witness_records(witnesses, contract["expected"]),
        )
        if qualification_records is None or witness_records is None:
            errors.append(
                "qualification_witness_records_match: validated records unavailable"
            )
        else:
            run(
                "qualification_witness_records_match",
                lambda: _records_match(qualification_records, witness_records),
            )

    expected_pcus = len(contract["pcus"]) if contract is not None else None
    expected_tuples = len(contract["expected"]) if contract is not None else None
    return {
        "schema_version": INTEGRITY_SCHEMA,
        "scope": "pcu_cohort_and_oracle_report_structure_only",
        "structural_pass": all(gates.values()),
        "gates": gates,
        "errors": errors,
        "expected_pcus": expected_pcus,
        "expected_oracle_tuples": expected_tuples,
        "source_binding": contract["source_binding"] if contract is not None else None,
        "raw_trace_authenticity_verified": False,
        "execution_provenance_verified": False,
        "model_evaluation_complete": False,
        "full_benchmark_release": False,
    }
