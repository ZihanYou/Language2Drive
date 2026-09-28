#!/usr/bin/env python3
"""Prepare and expand deterministic Qwen current-only requests without label access.

Only byte-equivalent model inputs after removing the output token are deduplicated.
Official reasoning is greedy and the planner noise is seeded anew for each request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def input_digest(scene: dict) -> str:
    payload = json.loads(json.dumps(scene))
    payload["meta_info"].pop("token", None)
    return sha(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8"))


def prepare(args) -> None:
    public = {r["request_id"]: r for r in rows(args.workload)}
    seen_keys, unique, mapping = {}, [], []
    requests = set()
    for scene in rows(args.scenes):
        meta = scene["meta_info"]
        identifier = meta["token"]
        if identifier not in public or identifier in requests:
            raise ValueError(f"unexpected/duplicate scene request_id: {identifier}")
        requests.add(identifier)
        if meta["temporal_sampling"] != "current_only_3view":
            raise ValueError(f"scene {identifier} is not current_only_3view")
        request = public[identifier]
        key = (request["input_id"], request["instruction"])
        digest = input_digest(scene)
        if key in seen_keys:
            representative, previous_digest = seen_keys[key]
            if digest != previous_digest:
                raise ValueError(f"same public input/instruction has different model input: {identifier}")
        else:
            representative = identifier
            seen_keys[key] = representative, digest
            unique.append(scene)
        mapping.append({"request_id": identifier, "source_request_id": representative,
                        "input_id": request["input_id"], "input_sha256": digest,
                        "protocol": "current_only_3view"})
    for path in (args.unique_scenes, args.mapping, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.unique_scenes.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                           for r in unique), encoding="utf-8")
    args.mapping.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in mapping), encoding="utf-8")
    report = {"mode": "prepare", "protocol": "current_only_3view",
              "requests": len(mapping), "unique_model_inputs": len(unique),
              "reused_requests": len(mapping) - len(unique),
              "equivalence": "identical canonical scene excluding output token",
              "official_reasoning_decoding": "greedy; do_sample=False; num_beams=1",
              "official_planner_noise": "per-call torch.Generator manual_seed(42)",
              "scene_file_sha256": sha(args.scenes.read_bytes()),
              "unique_scene_file_sha256": sha(args.unique_scenes.read_bytes()),
              "mapping_file_sha256": sha(args.mapping.read_bytes()), "label_access": "none"}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(unique)} unique scenes for {len(mapping)} requests")


def fanout(args) -> None:
    mapping = list(rows(args.mapping))
    public_ids = [r["request_id"] for r in rows(args.workload)]
    if len(public_ids) != len(set(public_ids)):
        raise ValueError("duplicate public workload request_id")
    by_id = {r["request_id"]: r for r in mapping}
    if len(by_id) != len(mapping) or not set(by_id).issubset(public_ids):
        raise ValueError("invalid mapping request IDs")
    source_ids = {r["source_request_id"] for r in mapping}
    predicted = {}
    for row in rows(args.unique_predictions):
        identifier = row.get("request_id")
        if identifier not in source_ids or identifier in predicted:
            raise ValueError(f"unexpected/duplicate unique prediction: {identifier}")
        if not isinstance(row.get("trajectory"), list) or len(row["trajectory"]) != 20:
            raise ValueError(f"invalid 20-point trajectory: {identifier}")
        predicted[identifier] = row["trajectory"]
    expanded, audit = [], []
    for identifier in public_ids:
        if identifier not in by_id:
            continue
        record = by_id[identifier]
        source = record["source_request_id"]
        audit_row = {**record, "status": "valid" if source in predicted else "missing",
                     "prediction_reused": source != identifier}
        if source in predicted:
            expanded.append({"request_id": identifier, "trajectory": predicted[source]})
        audit.append(audit_row)
    for path in (args.output, args.audit, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, allow_nan=False) + "\n"
                                   for r in expanded), encoding="utf-8")
    args.audit.write_text("".join(json.dumps(r) + "\n" for r in audit), encoding="utf-8")
    report = {"mode": "fanout", "protocol": "current_only_3view",
              "mapped_requests": len(mapping), "unique_predictions": len(predicted),
              "valid_expanded_predictions": len(expanded),
              "missing_expanded_predictions": len(mapping) - len(expanded),
              "reused_valid_predictions": sum(r["status"] == "valid" and r["prediction_reused"] for r in audit),
              "mapping_file_sha256": sha(args.mapping.read_bytes()),
              "unique_prediction_file_sha256": sha(args.unique_predictions.read_bytes()),
              "expanded_prediction_file_sha256": sha(args.output.read_bytes()),
              "label_access": "none"}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Expanded {len(predicted)} unique predictions to {len(expanded)}/{len(mapping)} requests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--unique-scenes", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    f = sub.add_parser("fanout")
    f.add_argument("--mapping", type=Path, required=True)
    f.add_argument("--workload", type=Path, required=True)
    f.add_argument("--unique-predictions", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--audit", type=Path, required=True)
    f.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else fanout)(args)


if __name__ == "__main__":
    main()
