"""Paper 1.0 RealSim metrics: scheduled cohorts, PCU macro means, and L2D.

Pure aggregation accepts multiple expressions per PCU. The document interface
enforces the paper's RealSim protocol: 30 PCUs, one expression, seed 1701.
Legacy evidence-interval reports remain available in their original modules.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .paired_semantic_metrics_all30 import score_candidate
from .physical_only_metrics import STRICT, route_completion_percent, validated_trace_rows

PROFILE = "realsim-paper-1.0"
ARM_METRICS = ("SISR", "SemISR", "SCS", "Safety", "Progress", "DrivingScore")


def _bounded(value, maximum=1.0):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f"metric must be finite and between 0 and {maximum}")
    return float(value)


def l2d_summary(metrics: dict) -> dict:
    """Equations 4–5 / 18–20; input and output scores are on the 0–100 scale."""
    scores = {name: _bounded(metrics[name], 100.0) for name in
              ("SISR", "CFPS", "SemISR", "Sem-CFPS", "SCS", "PCS", "Safety", "DrivingScore")}
    families = [(scores[a] + scores[b]) / 2 for a, b in
                (("SISR", "CFPS"), ("SemISR", "Sem-CFPS"), ("SCS", "PCS"))]
    balanced = (sum(math.sqrt(value) for value in families) / 3) ** 2
    driving = (scores["Safety"] + scores["DrivingScore"]) / 2
    return dict(a1=families[0], a2=families[1], a3=families[2],
                Lbal=balanced, driving_component=driving,
                L2D=driving * (1 + 0.99 * balanced) / 100)


def aggregate_pairs(pairs: list[dict]) -> dict:
    """Average expressions inside each PCU, then PCUs; never pool unequal counts.

    Each row has pcu_id, rewrite_id and arms {R0: metrics, R1: metrics}.
    Every per-arm metric is a fraction, including DrivingScore. SISR and
    SemISR are binary 0/1. Invalid scheduled executions must be supplied as
    zero-credit arms, not omitted. The caller fixes the scheduled inventory.
    """
    groups, seen = {}, set()
    for pair in pairs:
        key = (pair["pcu_id"], pair["rewrite_id"])
        if not all(isinstance(value, str) and value for value in key) or key in seen:
            raise ValueError("invalid or duplicate paired instance")
        seen.add(key)
        if set(pair["arms"]) != {"R0", "R1"}:
            raise ValueError("paired instance must contain both scheduled arms")
        arms = [{name: _bounded(pair["arms"][arm][name]) for name in ARM_METRICS}
                for arm in ("R0", "R1")]
        if any(arm[name] not in (0.0, 1.0) for arm in arms
               for name in ("SISR", "SemISR", "Safety", "Progress")):
            raise ValueError("binary verdicts must be zero or one")
        first, second = arms
        values = {name: (first[name] + second[name]) / 2 for name in ARM_METRICS}
        values.update(CFPS=first["SISR"] * second["SISR"],
                      PCS=min(first["SCS"], second["SCS"]),
                      **{"Sem-CFPS": first["SemISR"] * second["SemISR"],
                         "one_sided_success": abs(first["SISR"] - second["SISR"]),
                         "graded_arm_gap": abs(first["SCS"] - second["SCS"])})
        groups.setdefault(key[0], []).append(values)
    if not groups:
        raise ValueError("scheduled pair set cannot be empty")
    names = next(iter(groups.values()))[0].keys()
    macro = {name: 100 * sum(sum(row[name] for row in rows) / len(rows)
                            for rows in groups.values()) / len(groups) for name in names}
    one_sided = macro.pop("one_sided_success")
    graded_gap = macro.pop("graded_arm_gap")
    strict_residual = macro["SISR"] - macro["CFPS"] - one_sided / 2
    graded_residual = macro["SCS"] - macro["PCS"] - graded_gap / 2
    if abs(strict_residual) > 1e-10 or abs(graded_residual) > 1e-10:
        raise AssertionError("individual/paired identity mismatch")
    summary = l2d_summary(macro)
    macro["L2D"] = summary["L2D"]
    return dict(metrics=macro, summary_components=summary,
                aggregation="equal_expressions_within_PCU_then_equal_PCUs",
                pcu_count=len(groups), paired_instances=len(seen),
                identities=dict(one_sided_success_percent=one_sided,
                                graded_arm_gap_percent=graded_gap,
                                strict_residual=strict_residual, graded_residual=graded_residual))


def task_deadline_s(pcu: dict, arm: str) -> float:
    """An arm cannot earn route credit after its contract or decision deadline."""
    times = [pcu["decision_window"]["deadline_s"], pcu["arms"][arm]["semantic_contract"]["deadline_s"]]
    if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("invalid task deadline")
    return float(min(times))


def score_document(document: dict, pcus: list[dict], source_label: str = "") -> dict:
    """Score all scheduled arms. Unavailable executions get zero, not imputed success.

    Missing scheduled records are input errors. Explicit failed/unavailable
    records get zero credit and remain in the fixed denominator. A record
    claiming valid evidence must have a readable, valid trace and component
    verdicts; malformed evidence is rejected rather than silently repaired.
    """
    by_id = {pcu["pcu_id"]: pcu for pcu in pcus}
    if len(pcus) != 30 or len(by_id) != 30:
        raise ValueError("expected 30 unique frozen PCUs")
    if document.get("seed") != 1701 or document.get("expected_runs") != 60:
        raise ValueError("paper protocol requires seed 1701 and 60 scheduled slots")
    records = document.get("results")
    if not isinstance(records, list) or len(records) != 60:
        raise ValueError("expected exactly 60 scheduled records")
    scored, statuses, seen = {}, [], set()
    for record in records:
        key = (record.get("pcu_id"), record.get("arm_id"))
        if key[0] not in by_id or key[1] not in ("R0", "R1") or key in seen:
            raise ValueError("invalid, unknown or duplicate slot")
        if record.get("seed") != 1701:
            raise ValueError("slot seed mismatch")
        seen.add(key)
        usable = record.get("execution_status") == "completed" and record.get("evidence_valid") is True
        values = dict.fromkeys(ARM_METRICS, 0.0)
        if usable:
            components = record.get("components", {})
            if not isinstance(components, dict) or any(type(components.get(k)) is not bool for k in STRICT):
                raise ValueError("valid execution lacks binary strict component verdicts")
            own, relaxed = components["own_contract"], components.get("semantic_success")
            if not own and type(relaxed) is not bool:
                raise ValueError("valid execution lacks relaxed semantic verdict")
            trace = record.get("origin", {}).get("trace")
            if not isinstance(trace, str) or not trace:
                raise ValueError("valid execution lacks normalized trace")
            rows = validated_trace_rows(Path(trace))
            pcu, arm = by_id[key[0]], key[1]
            safe = not any(row["collision"] or row["offroad"] for row in rows)
            strict = (own and not components["opposing_contract"] and safe
                      and all(components[k] for k in ("progress", "deadline", "stability")))
            values.update(SISR=float(strict), SemISR=float(own or relaxed is True),
                          Safety=float(safe), Progress=float(components["progress"]),
                          SCS=score_candidate(pcu, arm, rows)["compliance"],
                          DrivingScore=route_completion_percent(
                              rows, pcu, arm, 1701, horizon_s=task_deadline_s(pcu, arm)) / 100)
        scored.setdefault(key[0], {})[key[1]] = values
        statuses.append(dict(pcu_id=key[0], arm_id=key[1], evidence_valid=usable,
                             credited=usable, metrics_fraction=values))
    pairs = [dict(pcu_id=pcu_id, rewrite_id="expression-0", arms=arms)
             for pcu_id, arms in sorted(scored.items())]
    result = aggregate_pairs(pairs)
    result.update(profile=PROFILE, scale="0-100", model=document.get("model_name"),
                  configuration_id=document.get("configuration_id"), source=source_label, seed=1701,
                  coverage=dict(scheduled_slots=60, valid_slots=sum(s["evidence_valid"] for s in statuses),
                                zero_credit_slots=sum(not s["evidence_valid"] for s in statuses),
                                scheduled_pairs=30), slots=statuses)
    return result


def load_experiment(path: Path) -> dict:
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    for record in document.get("results", []):
        trace = record.get("origin", {}).get("trace")
        if isinstance(trace, str) and trace and not Path(trace).is_absolute():
            record["origin"]["trace"] = str(path.parent / trace)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcus", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pcus = json.loads(args.pcus.read_text(encoding="utf-8"))["pcus"]
    result = score_document(load_experiment(args.experiment), pcus, args.experiment.name)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
