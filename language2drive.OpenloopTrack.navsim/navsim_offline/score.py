"""Reproduce the manuscript's original NAVSIM scoring environment.

This historical scorer intentionally has no Shapely preflight check. In an
environment without Shapely, geometric contracts raise ImportError, which the
original verifier absorbs during scoring. Use the isolated legacy
runner for exact manuscript reproduction; see README.md.
"""
import argparse
import json
import random
from pathlib import Path

from verify import (normalize_prediction, score_pcus, task_completion,
                    validate_trajectory)


def metric_summary(result):
    return {key: result[key] for key in (
        "n_pcu", "ol_sem_sisr", "ol_sem_cfps", "trajectory_valid_rate",
        "kinematic_feasibility_rate", "contract_margin", "failure_counts")}


def grouped_scores(sidecars, predictions, key, horizon=4.0):
    groups = {}
    for sidecar in sidecars:
        value = sidecar.get(key, "unknown")
        groups.setdefault(value, []).append(sidecar)
    return {str(value): metric_summary(score_pcus(items, predictions, horizon))
            for value, items in sorted(groups.items(), key=lambda item: str(item[0]))}


def cluster_bootstrap(sidecars, pcu_results, samples=1000, seed=0):
    groups = {}
    for sidecar in sidecars:
        key = sidecar.get("recording_id", sidecar["pcu_id"])
        groups.setdefault(key, []).append(pcu_results[sidecar["pcu_id"]])
    if not groups:
        return None
    rng = random.Random(seed)
    values = list(groups.values())
    sisr, cfps = [], []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        rows = [row for group in draw for row in group]
        sisr.append(sum(sum(row["arms"]) for row in rows) / (2 * len(rows)))
        cfps.append(sum(row["paired"] for row in rows) / len(rows))
    sisr.sort()
    cfps.sort()
    lo = int(0.025 * samples)
    hi = min(samples - 1, int(0.975 * samples))
    return {"unit": "original_recording", "n_clusters": len(values), "samples": samples,
            "ol_sem_sisr_95ci": [sisr[lo], sisr[hi]], "ol_sem_cfps_95ci": [cfps[lo], cfps[hi]]}


def score_rewrites(sidecars, rewrite_predictions, horizon=4.0):
    """Equal-weight rewrites within each PCU, then equal-weight PCUs.

    rewrite_predictions is an ordered mapping from rewrite label to a normal
    prediction dictionary. C[j][k] denotes completion of candidate contract j
    by the trajectory generated for instruction k.
    """
    if not isinstance(rewrite_predictions, dict) or not rewrite_predictions:
        raise ValueError("rewrite predictions must be a non-empty object")
    labels = list(rewrite_predictions)
    per_rewrite_binary = {label: score_pcus(sidecars, rewrite_predictions[label], horizon)
                          for label in labels}
    pcu_rows = {}
    for sidecar in sidecars:
        pcu_id = sidecar["pcu_id"]
        rewrite_rows = {}
        for label in labels:
            arms = rewrite_predictions[label].get(pcu_id, {})
            matrix = [[0.0, 0.0], [0.0, 0.0]]
            valid = [False, False]
            for trajectory_arm in (0, 1):
                try:
                    poses = normalize_prediction(arms[str(trajectory_arm)])
                    validate_trajectory(poses)
                    valid[trajectory_arm] = True
                    for candidate in (0, 1):
                        matrix[candidate][trajectory_arm] = task_completion(
                            poses, sidecar["contracts"][candidate], horizon)
                except (KeyError, TypeError, ValueError, IndexError, ImportError):
                    pass
            own = [matrix[0][0], matrix[1][1]]
            binary = per_rewrite_binary[label]["pcus"][pcu_id]
            rewrite_rows[label] = {
                "valid": valid,
                "completion_matrix": matrix,
                "own_completion": own,
                "scs": 100.0 * sum(own) / 2.0,
                "pcs": 100.0 * min(own),
                "sem_sisr": 100.0 * sum(binary["arms"]) / 2.0,
                "sem_cfps": 100.0 * binary["paired"],
            }
        pcu_rows[pcu_id] = {
            "rewrites": rewrite_rows,
            "scs": sum(row["scs"] for row in rewrite_rows.values()) / len(labels),
            "pcs": sum(row["pcs"] for row in rewrite_rows.values()) / len(labels),
            "sem_sisr": sum(row["sem_sisr"] for row in rewrite_rows.values()) / len(labels),
            "sem_cfps": sum(row["sem_cfps"] for row in rewrite_rows.values()) / len(labels),
        }
    n = len(pcu_rows)
    def mean(key):
        return sum(row[key] for row in pcu_rows.values()) / n if n else None
    scs = mean("scs")
    pcs = mean("pcs")
    imbalance = (50.0 * sum(
        abs(row["own_completion"][0] - row["own_completion"][1])
        for pcu in pcu_rows.values() for row in pcu["rewrites"].values()) /
        (n * len(labels))) if n else None
    return {
        "n_pcu": n,
        "n_rewrites_per_arm": len(labels),
        "rewrite_labels": labels,
        "aggregation": "equal rewrites within PCU, then equal PCUs",
        "sem_sisr": mean("sem_sisr"),
        "sem_cfps": mean("sem_cfps"),
        "scs": scs,
        "pcs": pcs,
        "paired_imbalance": imbalance,
        "identity_residual": ((scs - pcs) - imbalance) if n else None,
        "pcus": pcu_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--masked-predictions", type=Path, default=None)
    parser.add_argument("--shuffled-predictions", type=Path, default=None)
    parser.add_argument("--generic-predictions", type=Path, default=None)
    parser.add_argument("--paraphrase-predictions", type=Path, default=None,
                        help="JSON array of three prediction dictionaries")
    parser.add_argument("--rewrite-predictions", type=Path, default=None,
                        help="JSON object mapping rewrite labels to prediction dictionaries")
    args = parser.parse_args()
    sidecars = [json.loads(p.read_text()) for p in sorted(args.sidecars.glob("*.json"))]
    predictions = json.loads(args.predictions.read_text())
    result = score_pcus(sidecars, predictions)
    result["cluster_bootstrap"] = cluster_bootstrap(sidecars, result["pcus"], args.bootstrap_samples)
    result["by_tier"] = grouped_scores(sidecars, predictions, "annotation_tier")
    result["by_family"] = grouped_scores(sidecars, predictions, "family")
    common_sidecars = [item for item in sidecars if item.get("common_horizon_eligible", True)]
    common = score_pcus(common_sidecars, predictions, horizon=3.0)
    result["common_3s"] = {key: value for key, value in common.items() if key != "pcus"}
    result["common_3s"]["cluster_bootstrap"] = cluster_bootstrap(
        common_sidecars, common["pcus"], args.bootstrap_samples)
    result["common_3s"]["by_tier"] = grouped_scores(
        common_sidecars, predictions, "annotation_tier", horizon=3.0)
    result["common_3s"]["by_family"] = grouped_scores(
        common_sidecars, predictions, "family", horizon=3.0)
    for label, path in (("masked", args.masked_predictions),
                        ("generic", args.generic_predictions),
                        ("shuffled", args.shuffled_predictions)):
        if path:
            control = score_pcus(sidecars, json.loads(path.read_text()))
            result[label + "_language"] = {"ol_sem_sisr": control["ol_sem_sisr"],
                                            "ol_sem_cfps": control["ol_sem_cfps"],
                                            "cfps_gap": result["ol_sem_cfps"] - control["ol_sem_cfps"]}
    if args.paraphrase_predictions:
        variants = json.loads(args.paraphrase_predictions.read_text())
        if not isinstance(variants, list) or len(variants) != 3:
            parser.error("--paraphrase-predictions must contain exactly three prediction dictionaries")
        scores = [score_pcus(sidecars, variant) for variant in variants]
        n = len(sidecars)
        result["paraphrase"] = {
            "sisr_by_variant": [score["ol_sem_sisr"] for score in scores],
            "cfps_by_variant": [score["ol_sem_cfps"] for score in scores],
            "worst_pcu_success": (sum(min(score["pcus"][sidecar["pcu_id"]]["paired"]
                                          for score in scores) for sidecar in sidecars) / n) if n else None,
        }
    if args.rewrite_predictions:
        result["equal_weight_rewrites"] = score_rewrites(
            sidecars, json.loads(args.rewrite_predictions.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    concise = {key: value for key, value in result.items()
               if key not in ("pcus", "equal_weight_rewrites")}
    if "equal_weight_rewrites" in result:
        concise["equal_weight_rewrites"] = {
            key: value for key, value in result["equal_weight_rewrites"].items()
            if key != "pcus"
        }
    print(json.dumps(concise))


if __name__ == "__main__":
    main()
