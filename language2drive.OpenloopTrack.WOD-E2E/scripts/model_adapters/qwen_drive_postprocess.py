#!/usr/bin/env python3
"""Convert one-sample Qwen-Drive 10 Hz predictions to public 4 Hz scoring input.

No private targets are read. Invalid, duplicate or missing model outputs remain missing.
Reasoning text stays in the raw private prediction log and is never copied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


VERSION = "qwen_drive_10hz_to_wod_4hz_linear_v1"


def read_lines(path: Path) -> list[tuple[bytes, dict | None, str | None]]:
    result = []
    for raw in path.read_bytes().splitlines(keepends=True):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError("line is not an object")
            result.append((raw, item, None))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            result.append((raw, None, f"invalid_json:{exc}"))
    return result


def to_4hz(record: dict) -> list[list[float]]:
    samples = record.get("trajectories")
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError("expected_exactly_one_trajectory_sample")
    trajectory = samples[0]
    if not isinstance(trajectory, list) or len(trajectory) != 50:
        raise ValueError("expected_50_points_at_10hz")
    points = []
    for i, point in enumerate(trajectory):
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"point_{i}_needs_xyz")
        try:
            xyz = [float(value) for value in point]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"point_{i}_nonnumeric") from exc
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError(f"point_{i}_nonfinite")
        points.append(xyz)
    # Source point j is at (j+1)/10 seconds. Target k is at (k+1)/4 seconds.
    # Integer tenths are read directly; quarter-second positions use the two
    # surrounding 10 Hz points with linear interpolation. No extrapolation.
    result = []
    for k in range(20):
        t = (k + 1) / 4.0
        coordinate = t * 10 - 1
        lo = int(math.floor(coordinate))
        hi = int(math.ceil(coordinate))
        if lo < 0 or hi >= 50:
            raise ValueError("target_time_outside_10hz_window")
        weight = coordinate - lo
        result.append([
            round((1 - weight) * points[lo][axis] + weight * points[hi][axis], 5)
            for axis in (0, 1)
        ])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="official run_planning output")
    parser.add_argument("--workload", type=Path, required=True, help="public workload JSONL")
    parser.add_argument("--output", type=Path, required=True, help="4 Hz scoring predictions JSONL")
    parser.add_argument("--audit", type=Path, required=True, help="per-request conversion audit JSONL")
    parser.add_argument("--report", type=Path, required=True, help="summary with input hash")
    args = parser.parse_args()
    expected = []
    for _, row, error in read_lines(args.workload):
        if error or row is None or not row.get("request_id"):
            raise ValueError(f"invalid public workload: {error or row}")
        expected.append(str(row["request_id"]))
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate request_id in public workload")
    expected_set = set(expected)
    records: dict[str, list[tuple[bytes, dict]]] = {}
    malformed = []
    for line_number, (raw, row, error) in enumerate(read_lines(args.raw), 1):
        if error or row is None:
            malformed.append({"line": line_number, "reason": error, "sha256": hashlib.sha256(raw).hexdigest()})
            continue
        identifier = row.get("token", row.get("request_id"))
        if not isinstance(identifier, str) or not identifier:
            malformed.append({"line": line_number, "reason": "missing_token", "sha256": hashlib.sha256(raw).hexdigest()})
            continue
        records.setdefault(identifier, []).append((raw, row))
    predictions, audit = [], []
    for identifier in expected:
        entries = records.get(identifier, [])
        status = {"request_id": identifier, "conversion_version": VERSION}
        if not entries:
            status.update(status="missing", reason="missing_prediction")
        elif len(entries) > 1:
            status.update(status="invalid", reason="duplicate_prediction")
            status["source_line_sha256"] = [hashlib.sha256(raw).hexdigest() for raw, _ in entries]
        else:
            raw, row = entries[0]
            status["source_line_sha256"] = hashlib.sha256(raw).hexdigest()
            try:
                points = to_4hz(row)
            except ValueError as exc:
                status.update(status="invalid", reason=str(exc))
            else:
                status.update(status="valid", reason=None)
                predictions.append({"request_id": identifier, "trajectory": points})
        audit.append(status)
    unknown = sorted(set(records) - expected_set)
    for path in (args.output, args.audit, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in predictions), encoding="utf-8")
    args.audit.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in audit), encoding="utf-8")
    report = {
        "conversion_version": VERSION,
        "source_prediction_file": str(args.raw),
        "source_prediction_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "public_workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "expected_requests": len(expected),
        "valid_predictions": len(predictions),
        "missing_predictions": sum(row["status"] == "missing" for row in audit),
        "invalid_predictions": sum(row["status"] == "invalid" for row in audit),
        "malformed_output_lines": malformed,
        "unknown_output_tokens": unknown,
        "sample_policy": "exactly_one; no reference-based selection",
        "time_mapping": "10 Hz t=0.1..5.0 s to 4 Hz t=0.25..5.0 s by linear interpolation",
        "label_access": "none",
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(predictions)}/{len(expected)} valid 4 Hz predictions, "
          f"{report['missing_predictions']} missing, {report['invalid_predictions']} invalid; {args.output}")


if __name__ == "__main__":
    main()
