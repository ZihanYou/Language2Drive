"""Small, prospectively selected controls and overlap-safe failure taxonomy.

This is supplemental observational analysis, not a replacement leaderboard.
Unknown evidence is never converted into an output/control failure.
"""
from collections import Counter
from copy import deepcopy
from itertools import combinations
import uuid

from .paired_semantic_metrics_all30 import score_candidate, score_pair, task_group

PROFILE = "realsim-supplemental-diagnostics-v1-20260926"
CHOICE_MARGIN = 0.10
CONDITIONS = ("normal", "repeat_1", "repeat_2", "opposite", "empty", "neutral", "shuffled")
NEUTRAL = "Continue driving safely."
LABELS = ("wrong_behavior", "correct_incomplete", "safety_violation", "output_control_failure")
PRIMARY = ("evidence_unknown", "output_control_failure", "safety_violation",
           "wrong_behavior", "correct_incomplete", "completed", "ambiguous_incomplete")


def select_pcus(pcus):
    """First frozen-source PCU of each family; no outcome-based selection."""
    return [next(p for p in pcus if task_group(p) == group)
            for group in ("speed", "lane", "route")]


def make_runs(plan, sidecar, selected, namespace):
    indexed = {(r["pcu_id"], r["arm_id"]): r for r in plan["runs"]}
    output_sidecar = {**deepcopy(sidecar), "sessions": {}}
    runs = []
    for index, pcu in enumerate(selected):
        donor = selected[(index + 1) % len(selected)]
        base = indexed[(pcu["pcu_id"], "R0")]
        donor_session = sidecar["sessions"][indexed[(donor["pcu_id"], "R0")]["session_uuid"]]
        for condition in CONDITIONS:
            arm = "R1" if condition == "opposite" else "R0"
            original = indexed[(pcu["pcu_id"], arm)]
            row = deepcopy(original)
            row["session_uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}/{pcu['pcu_id']}/{condition}"))
            row["diagnostic_condition"] = condition
            session = deepcopy(sidecar["sessions"][original["session_uuid"]])
            normal_schedule = sidecar["sessions"][base["session_uuid"]]["instruction_schedule"]
            if len(normal_schedule) != 1 or len(session["instruction_schedule"]) != 1:
                raise ValueError("v1 controls require a single frozen instruction issuance")
            if condition in ("normal", "repeat_1", "repeat_2"):
                text = normal_schedule[0]["text"]
            elif condition == "empty":
                text = ""
            elif condition == "neutral":
                text = NEUTRAL
            elif condition == "shuffled":
                text = donor_session["instruction_schedule"][0]["text"]
            else:
                text = session["instruction_schedule"][0]["text"]
            session["instruction_schedule"][0]["text"] = text
            session["diagnostic"] = {"profile": PROFILE, "condition": condition,
                "score_target_arm": arm, "original_session_uuid": original["session_uuid"],
                "donor_pcu_id": donor["pcu_id"] if condition == "shuffled" else None,
                "simulation_and_inference_seed": 1701, "model_input_instruction": text}
            output_sidecar["sessions"][row["session_uuid"]] = session
            runs.append(row)
    return {**deepcopy(plan), "runs": runs, "run_count": len(runs),
            "runtime_paired_input_assertions": {}, "diagnostic_profile": PROFILE}, output_sidecar


def classify(record, pcu, rows=None, *, technical_failure=None):
    """Independent labels + priority-ordered, mutually exclusive primary class.

    Choice is an observable target preference, NOT an inferred model intention.
    The threshold is declared before examining diagnostic results.
    """
    result = {"pcu_id": pcu["pcu_id"], "arm_id": record.get("arm_id"),
              "labels": [], "choice": "unknown", "task_completed": None,
              "physical_safe": None, "technical_failure": technical_failure}
    if technical_failure is not None:
        result["labels"] = ["output_control_failure"]
        result["primary"] = "output_control_failure"
        return result
    if rows is None:
        result["primary"] = "evidence_unknown"
        return result
    arm = record["arm_id"]
    other = "R1" if arm == "R0" else "R0"
    own = score_candidate(pcu, arm, rows)
    opposing = score_candidate(pcu, other, rows)
    pair = score_pair(pcu, rows, rows)
    preference = pair["preference_R0_trajectory"]
    oriented = preference if arm == "R0" else -preference
    # No advance and no semantic target hit cannot identify a chosen behavior.
    if own["terminal_forward_distance_m"] <= 1e-6:
        choice = "ambiguous"
    elif oriented > CHOICE_MARGIN:
        choice = "correct"
    elif oriented < -CHOICE_MARGIN:
        choice = "wrong"
    else:
        choice = "ambiguous"
    components = record.get("components", {})
    # Use the frozen verifier when available, not a new C=1 threshold.
    semantic = components.get("own_contract")
    if semantic is True:
        choice = "correct"
    required = [semantic, components.get("progress"), components.get("deadline"), components.get("stability")]
    complete = False if any(v is False for v in required) else (
        True if all(v is True for v in required) else None)
    safe = not any(row["collision"] or row["offroad"] for row in rows)
    labels = []
    if choice == "wrong": labels.append("wrong_behavior")
    if choice == "correct" and complete is False: labels.append("correct_incomplete")
    if not safe: labels.append("safety_violation")
    primary = next((label for label in PRIMARY if label in labels), None)
    if primary is None:
        primary = "completed" if complete is True else (
            "ambiguous_incomplete" if complete is False else "evidence_unknown")
    result.update(primary=primary, labels=labels, choice=choice, task_completed=complete,
                  physical_safe=safe, oriented_preference=oriented,
                  preference_samples=pair["common_effective_samples"],
                  own_compliance=own["compliance"], opposing_compliance=opposing["compliance"],
                  own_active_samples=own.get("active_samples"),
                  own_satisfaction_ratio=own["satisfaction_ratio"],
                  forward_m=own["terminal_forward_distance_m"], components=components)
    return result


def summarize(rows):
    primary = {name: sum(r["primary"] == name for r in rows) for name in PRIMARY}
    marginal = {name: sum(name in r["labels"] for r in rows) for name in LABELS}
    patterns = Counter("+".join(sorted(r["labels"])) or "none" for r in rows)
    overlap = {f"{a}&{b}": sum(a in r["labels"] and b in r["labels"] for r in rows)
               for a, b in combinations(LABELS, 2)}
    if sum(primary.values()) != len(rows):
        raise AssertionError("primary categories must partition slots")
    known_choice = sum(r["choice"] in ("correct", "wrong") for r in rows)
    known_correct_completion = sum(r["choice"] == "correct" and r.get("task_completed") is not None for r in rows)
    known_safety = sum(r["physical_safe"] is not None for r in rows)
    def percentage(numerator, denominator):
        return 100 * numerator / denominator if denominator else None
    return {"denominator": len(rows), "primary_counts": primary,
            "multilabel_counts_do_not_sum": marginal, "pairwise_overlaps": overlap,
            "exact_label_patterns": dict(sorted(patterns.items())),
            "unique_any_labeled_failure": sum(bool(r["labels"]) for r in rows),
            "known_choice_denominator": known_choice,
            "correct_choice_denominator": sum(r["choice"] == "correct" for r in rows),
            "known_correct_choice_completion_denominator": known_correct_completion,
            "known_safety_denominator": known_safety,
            "conditional_rates_percent": {
                "wrong_given_known_choice": percentage(marginal["wrong_behavior"], known_choice),
                "incomplete_given_correct_and_known_completion": percentage(marginal["correct_incomplete"], known_correct_completion),
                "safety_violation_given_known_safety": percentage(marginal["safety_violation"], known_safety)},
            "observed_unsuccessful_primary_count": len(rows)-primary["completed"]-primary["evidence_unknown"]}
