#!/usr/bin/env python3
"""
Compute evaluation metrics for a single-route leaderboard result JSON.

Usage:
    python tools/eval_single_route.py --result result_0.json
    python tools/eval_single_route.py --result result_0.json --results-root results/
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# Reuse official smoothness computation from Bench2Drive
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "language2drive", "tools"))
from efficiency_smoothness_benchmark import seg_compute_comfort_metric  # noqa: E402


def load_result(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_record(data):
    records = data["_checkpoint"]["records"]
    if not records:
        raise ValueError("No route records found in result JSON.")
    if len(records) > 1:
        print(f"Warning: found {len(records)} records, using the first one.", file=sys.stderr)
    return records[0]


def is_success(record):
  """Same rule as Bench2Drive merge_route_json.py."""
  if record["status"] not in ("Completed", "Perfect"):
    return False
  for key, value in record["infractions"].items():
    if key == "min_speed_infractions":
      continue
    if len(value) > 0:
      return False
  return True


def compute_efficiency(record):
  """Driving efficiency from min_speed_infractions messages."""
  messages = record["infractions"].get("min_speed_infractions", [])
  if not messages:
    return None

  values = []
  for msg in messages:
    match = re.search(r"\b\d+\.?\d*%", msg)
    if match is None:
      continue
    pct = float(match.group().rstrip("%"))
    if pct > 1000:
      continue
    values.append(pct)

  if not values:
    return None
  return sum(values) / len(values)


def find_metric_info(save_name, results_root):
  """Try official path first, then SimLingo debug_viz layout."""
  candidates = [
    os.path.join(results_root, save_name, "metric_info.json"),
  ]
  candidates.extend(
    glob.glob(
      os.path.join(results_root, save_name, "debug_viz", "**", "metric", "metric_info.json"),
      recursive=True,
    )
  )
  candidates.extend(
    glob.glob(
      os.path.join(results_root, "**", "metric", "metric_info.json"),
      recursive=True,
    )
  )

  for path in candidates:
    if os.path.isfile(path):
      return path
  return None


def load_metric_arrays(metric_info_path):
  with open(metric_info_path, encoding="utf-8") as f:
    data = json.load(f)

  arrays = {
    "acceleration": [],
    "angular_velocity": [],
    "forward_vector": [],
    "right_vector": [],
    "location": [],
    "rotation": [],
  }
  for step in sorted(data.keys(), key=int):
    frame = data[step]
    for key in arrays:
      arrays[key].append(frame[key])

  for key in arrays:
    arrays[key] = np.array(arrays[key])
  return arrays


def compute_smoothness(metric_info_path):
  arrays = load_metric_arrays(metric_info_path)
  return seg_compute_comfort_metric(**arrays)


def summarize_infractions(record):
  summary = {}
  for key, events in record["infractions"].items():
    summary[key] = len(events)
  return summary


def build_report(result_path, record, metric_info_path=None):
  scores = record["scores"]
  driving_score = scores["score_composed"]
  route_completion = scores["score_route"]
  infraction_penalty = scores["score_penalty"]

  report = {
    "result_file": os.path.abspath(result_path),
    "route_id": record["route_id"],
    "save_name": record.get("save_name"),
    "status": record["status"],
    "town_name": record.get("town_name"),
    "driving_score": driving_score,
    "route_completion": route_completion,
    "infraction_penalty": infraction_penalty,
    "success": is_success(record),
    "success_rate": 1.0 if is_success(record) else 0.0,
    "efficiency": compute_efficiency(record),
    "smoothness": None,
    "metric_info_path": metric_info_path,
    "route_length_m": record["meta"]["route_length"],
    "duration_game_s": record["meta"]["duration_game"],
    "infraction_counts": summarize_infractions(record),
    "infraction_messages": {
      k: v for k, v in record["infractions"].items() if len(v) > 0
    },
  }

  if metric_info_path:
    report["smoothness"] = compute_smoothness(metric_info_path)

  return report


def print_report(report):
  print("=" * 60)
  print("Single Route Evaluation Summary")
  print("=" * 60)
  print(f"Result file      : {report['result_file']}")
  print(f"Route ID         : {report['route_id']}")
  print(f"Status           : {report['status']}")
  print(f"Town             : {report.get('town_name', 'N/A')}")
  print("-" * 60)
  print(f"Driving Score    : {report['driving_score']:.6f}")
  print(f"Route Completion : {report['route_completion']:.6f}%")
  print(f"Infraction Penalty: {report['infraction_penalty']:.6f}")
  print(f"Success          : {report['success']}")
  print(f"Success Rate     : {report['success_rate']:.4f}")

  if report["efficiency"] is not None:
    print(f"Efficiency       : {report['efficiency']:.4f}%")
  else:
    print("Efficiency       : N/A (no min_speed_infractions)")

  if report["smoothness"] is not None:
    print(f"Smoothness       : {report['smoothness']:.4f}")
  else:
    print("Smoothness       : N/A (metric_info.json not found)")

  print("-" * 60)
  print(f"Route length     : {report['route_length_m']:.3f} m")
  print(f"Game duration    : {report['duration_game_s']:.3f} s")

  if report["infraction_messages"]:
    print("-" * 60)
    print("Infractions:")
    for key, messages in report["infraction_messages"].items():
      print(f"  [{key}]")
      for msg in messages:
        print(f"    - {msg}")
  print("=" * 60)


def main():
  parser = argparse.ArgumentParser(description="Evaluate metrics for a single route result JSON.")
  parser.add_argument(
    "--result",
    type=str,
    default=os.path.join(_REPO_ROOT, "result_0.json"),
    help="Path to a single-route leaderboard result JSON.",
  )
  parser.add_argument(
    "--results-root",
    type=str,
    default=os.path.join(_REPO_ROOT, "results"),
    help="Root folder used to auto-find metric_info.json from save_name.",
  )
  parser.add_argument(
    "--metric-info",
    type=str,
    default=None,
    help="Optional explicit path to metric_info.json.",
  )
  parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Optional path to save the report as JSON.",
  )
  args = parser.parse_args()

  data = load_result(args.result)
  record = get_record(data)

  metric_info_path = args.metric_info
  if metric_info_path is None and record.get("save_name"):
    metric_info_path = find_metric_info(record["save_name"], args.results_root)

  report = build_report(args.result, record, metric_info_path)
  print_report(report)

  if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
  main()
  
"""
# 在 simlingo 环境中运行
conda activate simlingo-py311
python tools/eval_single_route.py --result result_0.json

# 可选：保存 JSON 报告
python tools/eval_single_route.py --result result_0.json --output result_0_metrics.json

# 可选：手动指定 metric_info.json
python tools/eval_single_route.py --result result_0.json \
  --metric-info results/.../metric/metric_info.json
"""
