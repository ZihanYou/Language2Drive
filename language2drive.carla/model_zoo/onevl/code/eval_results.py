#!/usr/bin/env python3
"""Unified evaluation for OneVL opensource inference outputs: **roadwork**, **impromptu**, **ar1**.

Usage:
  python eval_results.py roadwork [--json_path ...] [--tail_points 15] [--no_denorm]
  python eval_results.py impromptu [--results_json ... | --results_jsonl ...] [--test_jsonl ...]
  python eval_results.py ar1 [--results_json ...] [--test_jsonl ...]

Roadwork: 0–1000 coordinates → pixels via first user image; ADE/FDE on last ``tail_points``
GT waypoints vs predictions (mean L2 / final L2); reports mean ``latency`` (seconds) on evaluated samples.

Impromptu: BEV meters; same metrics as ``eval_onevl.py`` (L2 @ 1–4s, ADE, FDE).

  - ``--results_json``: OneVL ``impromptu_results.json`` (GT often empty → needs ``--test_jsonl``).
  - ``--results_jsonl``: ms-swift infer JSONL (per line: ``response``, ``labels``); GT from
    ``labels`` unless ``--test_jsonl`` is given for fallback.

AR1: ego-frame trajectory in **meters**; GT from ``ar1_test.jsonl`` assistant ``<answer>``;
``output_text`` may omit a leading ``[[``; match by first image path suffix ``ar1_labels.v2/...``.
ADE / FDE align ``eval_ade_fde_onevl_jsonl.py``; reports mean **latency**.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Roadwork: parsing + denorm (inlined from ms-swift eval_ade_fde_merged)
# ---------------------------------------------------------------------------

_PAIR_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def first_user_image_path_from_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    for turn in messages:
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image":
                continue
            img = part.get("image") or part.get("image_url")
            if isinstance(img, str) and img.strip():
                return img.strip()
    return None


def image_wh(path: str, cache: Dict[str, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if path in cache:
        return cache[path]
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with Image.open(p) as im:
            w, h = im.size
    except OSError:
        return None
    cache[path] = (int(w), int(h))
    return cache[path]


def denorm_xy_norm1000(pts: np.ndarray, w: float, h: float) -> np.ndarray:
    out = np.asarray(pts, dtype=np.float64)
    if out.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if out.ndim == 1 and out.shape[0] == 2:
        out = out.reshape(1, 2)
    if out.ndim != 2 or out.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    out = out.copy()
    out[:, 0] = out[:, 0] / 1000.0 * w
    out[:, 1] = out[:, 1] / 1000.0 * h
    return out


def parse_gt_waypoints(gt_str: str) -> List[List[float]]:
    if not gt_str or not isinstance(gt_str, str):
        return []
    s = gt_str.strip()
    if not s:
        return []
    try:
        data = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        try:
            data = ast.literal_eval("[" + s + "]")
        except (SyntaxError, ValueError):
            return []
    if not data:
        return []
    if isinstance(data[0], (int, float)):
        return [list(map(float, data))]
    return [list(map(float, p)) for p in data]


def parse_output_waypoints(text: str) -> List[List[float]]:
    if not text or not isinstance(text, str):
        return []
    cut = text.split("</answer>", 1)[0]
    cut = cut.split("</redacted_thinking>", 1)[0]
    pts: List[List[float]] = []
    for m in _PAIR_RE.finditer(cut):
        pts.append([float(m.group(1)), float(m.group(2))])
    return pts


def ade_fde_gt_tail(
    gt: np.ndarray,
    pred: np.ndarray,
    tail_points: int = 15,
) -> Tuple[float, float, int]:
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape[1] < 2 or pred.shape[1] < 2:
        return float("nan"), float("nan"), 0
    k = min(tail_points, gt.shape[0])
    if k <= 0:
        return float("nan"), float("nan"), 0
    gt_tail = gt[-k:]
    n = int(min(gt_tail.shape[0], pred.shape[0]))
    if n <= 0:
        return float("nan"), float("nan"), 0
    err = np.linalg.norm(pred[:n, :2] - gt_tail[:n, :2], axis=1)
    return float(err.mean()), float(err[-1]), n


def cmd_roadwork(argv: Optional[Sequence[str]] = None) -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Roadwork ADE/FDE (image pixels after denorm)")
    ap.add_argument(
        "--json_path",
        type=Path,
        default=script_dir / "output/roadwork/roadwork_results.json",
    )
    ap.add_argument("--tail_points", type=int, default=15)
    ap.add_argument(
        "--no_denorm",
        action="store_true",
        help="Coordinates already in pixels",
    )
    args = ap.parse_args(argv)

    with open(args.json_path, encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    ades: List[float] = []
    fdes: List[float] = []
    latencies: List[float] = []
    skipped: List[Tuple[int, str]] = []
    hw_cache: Dict[str, Tuple[int, int]] = {}

    for i, item in enumerate(data):
        gt_list = parse_gt_waypoints(item.get("GT", ""))
        pred_list = parse_output_waypoints(item.get("output_text", ""))

        if not args.no_denorm:
            img_path = first_user_image_path_from_messages(item.get("messages"))
            if not img_path:
                skipped.append((i, "no_image_in_messages"))
                continue
            wh = image_wh(img_path, hw_cache)
            if wh is None:
                skipped.append((i, f"image_unreadable:{img_path}"))
                continue
            w, h = wh
            gt = denorm_xy_norm1000(np.asarray(gt_list, dtype=np.float64), w, h)
            pred = denorm_xy_norm1000(np.asarray(pred_list, dtype=np.float64), w, h)
        else:
            gt = np.asarray(gt_list, dtype=np.float64)
            pred = np.asarray(pred_list, dtype=np.float64)
            if gt.ndim == 1 and gt.size == 2:
                gt = gt.reshape(1, 2)
            if pred.ndim == 1 and pred.size == 2:
                pred = pred.reshape(1, 2)

        if gt.shape[0] == 0:
            skipped.append((i, "empty_gt"))
            continue
        if pred.shape[0] == 0:
            skipped.append((i, "no_pred_points"))
            continue

        ade, fde, n = ade_fde_gt_tail(gt, pred, tail_points=args.tail_points)
        if not math.isfinite(ade) or n == 0:
            skipped.append((i, "n_eval=0"))
            continue
        ades.append(ade)
        fdes.append(fde)
        latencies.append(float(item.get("latency", 0.0)))

    print(f"json_path: {args.json_path}")
    if args.no_denorm:
        print("denorm: off")
    else:
        print("denorm: x_px=x/1000*w, y_px=y/1000*h")
    print(
        f"ADE on last {args.tail_points} GT points (mean L2 px); "
        f"samples: {len(ades)} / {len(data)}, skipped: {len(skipped)}"
    )
    if not ades:
        for idx, r in skipped[:15]:
            print(f"  skip[{idx}]: {r}")
        return 1
    print(f"ADE (mean over samples): {float(np.mean(ades)):.6f}")
    print(f"FDE (mean over samples): {float(np.mean(fdes)):.6f}")
    print(f"avg latency (evaluated samples): {float(np.mean(latencies)):.4f} s")
    return 0


# ---------------------------------------------------------------------------
# Impromptu
# ---------------------------------------------------------------------------

IMPROMPTU_TRAJ_LEN = 10
PATTERN_GT_IMP = re.compile(r"<answer>\s*(\[\[.*?\]\])\s*\.?\s*\n", re.DOTALL)
PATTERN_PRED_IMP = re.compile(
    r"(\[\[.*?\]\])\s*\.?\s*\n?\s*</answer>", re.DOTALL
)


def _impromptu_image_id(messages: Any) -> str:
    try:
        for p in messages[0]["content"]:
            if isinstance(p, dict) and p.get("type") == "image":
                path = p.get("image") or ""
                return os.path.basename(str(path)).split(".CAM_")[0]
    except (KeyError, IndexError, TypeError):
        pass
    return "unknown"


def extract_gt_trajectory_imp(text: str) -> List[List[float]]:
    try:
        m = PATTERN_GT_IMP.search(text)
        if not m:
            return []
        traj_str = m.group(1).strip().replace("\n", "").replace(" ", "")
        traj = ast.literal_eval(traj_str)
        if isinstance(traj, list) and len(traj) == IMPROMPTU_TRAJ_LEN:
            return [list(map(float, p)) for p in traj]
    except (SyntaxError, ValueError, TypeError):
        pass
    return []


def extract_pred_trajectory_imp(output_text: str) -> List[List[float]]:
    if not output_text or not isinstance(output_text, str):
        return []
    try:
        m = PATTERN_PRED_IMP.search(output_text)
        if m:
            traj_str = m.group(1).strip().replace("\n", "").replace(" ", "")
            traj = ast.literal_eval(traj_str)
        else:
            cut = output_text.split("</answer>", 1)[0].strip().rstrip(".").strip()
            if not cut.startswith("[["):
                cut = "[[" + cut
            traj = ast.literal_eval(cut)
        if isinstance(traj, list) and len(traj) == IMPROMPTU_TRAJ_LEN:
            return [list(map(float, p)) for p in traj]
    except (SyntaxError, ValueError, TypeError):
        pass
    return []


def _l2(p: List[float], q: List[float]) -> float:
    return float(np.linalg.norm(np.array(p) - np.array(q)))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _impromptu_latency_seconds(row: Dict[str, Any]) -> float:
    for key in ("latency", "infer_time", "generation_time", "time_seconds"):
        v = row.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def cmd_impromptu(argv: Optional[Sequence[str]] = None) -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Impromptu metrics (BEV meters)")
    ap.add_argument(
        "--results_json",
        type=Path,
        default=script_dir / "output/impromptu/impromptu_results.json",
        help="OneVL-style array JSON (e.g. impromptu_results.json)",
    )
    ap.add_argument(
        "--results_jsonl",
        type=Path,
        default=None,
        help="ms-swift infer JSONL (fields: response, labels, messages, …)",
    )
    ap.add_argument(
        "--test_jsonl",
        type=Path,
        default=None,
        help="GT fallback: assistant <answer> per line. Default when using --results_json only.",
    )
    ap.add_argument("--output_eval_json", type=Path, default=None)
    ap.add_argument("--output_excel", type=Path, default=None)
    args = ap.parse_args(argv)

    results_path_display: str
    test_jsonl_resolved: Optional[Path] = None
    if args.results_jsonl is not None:
        p = Path(args.results_jsonl)
        if not p.is_file():
            print(f"ERROR: results_jsonl not found: {p}", file=sys.stderr)
            return 1
        results = load_jsonl(p)
        results_path_display = str(p)
        if args.test_jsonl is None:
            test_rows: Optional[List[Dict[str, Any]]] = None
        else:
            tp = Path(args.test_jsonl)
            if not tp.is_file():
                print(f"ERROR: test_jsonl not found: {tp}", file=sys.stderr)
                return 1
            test_rows = load_jsonl(tp)
            test_jsonl_resolved = tp
    else:
        pj = Path(args.results_json)
        if not pj.is_file():
            print(f"ERROR: results_json not found: {pj}", file=sys.stderr)
            return 1
        with open(pj, encoding="utf-8") as f:
            results = json.load(f)
        results_path_display = str(pj)
        tpath = (
            Path(args.test_jsonl)
            if args.test_jsonl is not None
            else script_dir / "test_data/impromptu_test.jsonl"
        )
        if not tpath.is_file():
            print(f"ERROR: test_jsonl not found: {tpath}", file=sys.stderr)
            return 1
        test_rows = load_jsonl(tpath)
        test_jsonl_resolved = tpath

    if test_rows is not None and len(results) != len(test_rows):
        print(
            f"ERROR: length mismatch results={len(results)} test_jsonl={len(test_rows)}",
            file=sys.stderr,
        )
        return 1

    keys = ["l2_1s", "l2_2s", "l2_3s", "l2_4s", "ade", "fde", "latency"]
    total_metrics: Dict[str, List[float]] = {k: [] for k in keys}
    per_sample: List[Dict[str, Any]] = []
    skipped = 0

    for idx, sample_res in enumerate(results):
        try:
            latency = _impromptu_latency_seconds(sample_res)
            image_id = _impromptu_image_id(sample_res["messages"])

            gt_src = (sample_res.get("GT") or "").strip()
            if not gt_src:
                lab = sample_res.get("labels")
                if isinstance(lab, str):
                    gt_src = lab.strip()

            if gt_src:
                gt_traj = extract_gt_trajectory_imp(gt_src)
            elif test_rows is not None:
                sample_test = test_rows[idx]
                assistant = ""
                if len(sample_test.get("messages", [])) > 1:
                    assistant = sample_test["messages"][1].get("content", "") or ""
                gt_traj = extract_gt_trajectory_imp(assistant)
            else:
                gt_traj = []

            pred_raw = (
                sample_res.get("output_text")
                or sample_res.get("response")
                or ""
            )
            pred_traj = extract_pred_trajectory_imp(
                pred_raw if isinstance(pred_raw, str) else ""
            )
            if not gt_traj or not pred_traj:
                skipped += 1
                continue

            l2_1s = _l2(pred_traj[1], gt_traj[1])
            l2_2s = _l2(pred_traj[3], gt_traj[3])
            l2_3s = _l2(pred_traj[5], gt_traj[5])
            l2_4s = _l2(pred_traj[7], gt_traj[7])
            all_l2 = [_l2(pred_traj[i], gt_traj[i]) for i in range(IMPROMPTU_TRAJ_LEN)]
            ade = float(np.mean(all_l2))
            fde = _l2(pred_traj[9], gt_traj[9])

            per_sample.append(
                {
                    "index": idx,
                    "image_id": image_id,
                    "latency": round(latency, 4),
                    "l2_1s": round(l2_1s, 4),
                    "l2_2s": round(l2_2s, 4),
                    "l2_3s": round(l2_3s, 4),
                    "l2_4s": round(l2_4s, 4),
                    "ADE": round(ade, 4),
                    "FDE": round(fde, 4),
                }
            )
            for k, v in zip(keys, [l2_1s, l2_2s, l2_3s, l2_4s, ade, fde, latency]):
                total_metrics[k].append(v)
        except Exception:
            skipped += 1
            continue

    n = len(per_sample)
    if n == 0:
        print("No valid impromptu samples.", file=sys.stderr)
        return 1

    if test_rows is None:
        gt_note = "per-row labels/GT"
    else:
        gt_note = f"row labels/GT if present, else {test_jsonl_resolved}"
    final_eval = {
        "task": "impromptu",
        "results_path": results_path_display,
        "gt_source": gt_note,
        "total_samples": n,
        "skipped": skipped,
        "avg_latency": round(float(np.mean(total_metrics["latency"])), 4),
        "avg_l2_1s": round(float(np.mean(total_metrics["l2_1s"])), 4),
        "avg_l2_2s": round(float(np.mean(total_metrics["l2_2s"])), 4),
        "avg_l2_3s": round(float(np.mean(total_metrics["l2_3s"])), 4),
        "avg_l2_4s": round(float(np.mean(total_metrics["l2_4s"])), 4),
        "avg_ADE": round(float(np.mean(total_metrics["ade"])), 4),
        "avg_FDE": round(float(np.mean(total_metrics["fde"])), 4),
    }

    print("=" * 50)
    print("Impromptu (eval_onevl.py style, meters)")
    print(f"results: {results_path_display}")
    if test_rows is None:
        print("GT from: row labels/GT only (no test_jsonl)")
    else:
        print(f"GT from: row labels/GT if present, else {test_jsonl_resolved}")
    print(f"有效样本：{n} / {len(results)}，skip：{skipped}")
    print(f"平均延迟：{final_eval['avg_latency']} s")
    print(
        f"1s/2s/3s/4s L2: {final_eval['avg_l2_1s']} / {final_eval['avg_l2_2s']} / "
        f"{final_eval['avg_l2_3s']} / {final_eval['avg_l2_4s']}"
    )
    print(f"ADE: {final_eval['avg_ADE']} | FDE: {final_eval['avg_FDE']}")
    print("=" * 50)

    if args.output_eval_json:
        args.output_eval_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_eval_json, "w", encoding="utf-8") as f:
            json.dump(final_eval, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote {args.output_eval_json}")

    if args.output_excel:
        try:
            import pandas as pd
        except ImportError as e:
            print(f"Skip Excel: {e}", file=sys.stderr)
            return 0
        args.output_excel.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_sample).to_excel(
            args.output_excel, index=False, engine="openpyxl"
        )
        print(f"Wrote {args.output_excel}")

    return 0


# ---------------------------------------------------------------------------
# AR1 (ego-frame trajectory, meters)
# ---------------------------------------------------------------------------

_AR1_ANSWER_RE = re.compile(
    r"<answer>\s*(\[\[.*?\]\])\s*</answer>", re.DOTALL | re.IGNORECASE
)


def _canonical_ar1_image_key(path: str) -> str:
    p = path.replace("\\", "/")
    i = p.find("ar1_labels.v2/")
    if i >= 0:
        return p[i:]
    return p


def _ar1_first_user_image(entry: Dict[str, Any]) -> Optional[str]:
    for m in entry.get("messages") or []:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image":
                    p = part.get("image") or part.get("image_url")
                    if isinstance(p, str) and p.strip():
                        return p.strip()
        break
    return None


def _ar1_parse_gt_from_messages(obj: Dict[str, Any]) -> Optional[List[List[float]]]:
    for m in obj.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        ma = _AR1_ANSWER_RE.search(content)
        if not ma:
            continue
        try:
            raw = json.loads(ma.group(1))
            return [[float(p[0]), float(p[1])] for p in raw]
        except (json.JSONDecodeError, TypeError, ValueError, IndexError):
            continue
    return None


def _load_ar1_gt_map(gt_path: Path) -> Dict[str, List[List[float]]]:
    by_image: Dict[str, List[List[float]]] = {}
    with gt_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            traj = _ar1_parse_gt_from_messages(obj)
            imgs = obj.get("images") or []
            if traj is None or not imgs:
                continue
            key0 = imgs[0] if isinstance(imgs[0], str) else ""
            if not key0:
                continue
            k = _canonical_ar1_image_key(key0)
            by_image[k] = traj
    return by_image


def _ar1_parse_pred_from_output_text(text: str) -> Optional[List[List[float]]]:
    if not text:
        return None
    text = text.strip()
    pairs = re.findall(
        r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", text
    )
    m0 = re.match(
        r"^([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*,",
        text,
    )
    if m0:
        pairs = [(m0.group(1), m0.group(2))] + list(pairs)
    else:
        m0b = re.match(r"^([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", text)
        if m0b:
            pairs = [(m0b.group(1), m0b.group(2))] + list(pairs)
    if not pairs:
        return None
    return [[float(a), float(b)] for a, b in pairs]


def _ar1_ade_fde_pair(
    pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]]
) -> Optional[Tuple[float, float, int]]:
    if not pred or not gt:
        return None
    t = min(len(pred), len(gt))
    if t <= 0:
        return None
    errs: List[float] = []
    for i in range(t):
        dx = float(pred[i][0]) - float(gt[i][0])
        dy = float(pred[i][1]) - float(gt[i][1])
        errs.append(math.hypot(dx, dy))
    return sum(errs) / t, errs[-1], t


def cmd_ar1(argv: Optional[Sequence[str]] = None) -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="AR1 ADE/FDE (meters) + latency; GT from test jsonl"
    )
    ap.add_argument(
        "--results_json",
        type=Path,
        default=script_dir / "output/ar1/ar1_results.json",
    )
    ap.add_argument(
        "--test_jsonl",
        type=Path,
        default=script_dir / "test_data/ar1_test.jsonl",
    )
    ap.add_argument("--output_eval_json", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.results_json.is_file():
        print(f"ERROR: results_json not found: {args.results_json}", file=sys.stderr)
        return 1
    if not args.test_jsonl.is_file():
        print(f"ERROR: test_jsonl not found: {args.test_jsonl}", file=sys.stderr)
        return 1

    gt_map = _load_ar1_gt_map(args.test_jsonl)
    if not gt_map:
        print(f"No GT trajectories loaded from {args.test_jsonl}", file=sys.stderr)
        return 1

    with open(args.results_json, encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    if not isinstance(data, list):
        print("results_json must be a JSON array", file=sys.stderr)
        return 1

    ades: List[float] = []
    fdes: List[float] = []
    latencies: List[float] = []
    skip_no_pred = skip_no_gt = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue
        pred = _ar1_parse_pred_from_output_text(entry.get("output_text") or "")
        img_raw = _ar1_first_user_image(entry)
        if pred is None:
            skip_no_pred += 1
            continue
        if not img_raw:
            skip_no_gt += 1
            continue
        img_k = _canonical_ar1_image_key(img_raw)
        if img_k not in gt_map:
            skip_no_gt += 1
            continue
        gt = gt_map[img_k]
        out = _ar1_ade_fde_pair(pred, gt)
        if out is None:
            skip_no_pred += 1
            continue
        ade, fde, _ = out
        ades.append(ade)
        fdes.append(fde)
        latencies.append(_impromptu_latency_seconds(entry))

    n = len(ades)
    print("=" * 50)
    print("AR1 (ego frame, meters)")
    print(f"results: {args.results_json}")
    print(f"GT: {args.test_jsonl} (by first image key)")
    print(f"n_infer_rows: {len(data)}, evaluated: {n}")
    print(f"skip_no_pred: {skip_no_pred}, skip_no_gt: {skip_no_gt}")
    if n == 0:
        print("No matched samples.", file=sys.stderr)
        return 1
    avg_lat = float(np.mean(latencies))
    print(f"ADE (mean over samples): {sum(ades) / n:.6f}")
    print(f"FDE (mean over samples): {sum(fdes) / n:.6f}")
    print(f"avg latency (evaluated samples): {avg_lat:.4f} s")
    print("=" * 50)

    if args.output_eval_json:
        summary = {
            "task": "ar1",
            "results_path": str(args.results_json),
            "test_jsonl": str(args.test_jsonl),
            "n": n,
            "n_infer_rows": len(data),
            "skip_no_pred": skip_no_pred,
            "skip_no_gt": skip_no_gt,
            "avg_ADE": round(sum(ades) / n, 6),
            "avg_FDE": round(sum(fdes) / n, 6),
            "avg_latency": round(avg_lat, 4),
        }
        args.output_eval_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_eval_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote {args.output_eval_json}")

    return 0


def main() -> None:
    prog = Path(sys.argv[0]).name
    if len(sys.argv) < 2:
        print(
            f"Usage: {prog} roadwork|impromptu|ar1 [task options...]\n"
            f"  {prog} roadwork --help\n"
            f"  {prog} impromptu --help\n"
            f"  {prog} ar1 --help",
            file=sys.stderr,
        )
        raise SystemExit(2)
    task, rest = sys.argv[1], sys.argv[2:]
    if task == "roadwork":
        raise SystemExit(cmd_roadwork(rest))
    if task == "impromptu":
        raise SystemExit(cmd_impromptu(rest))
    if task == "ar1":
        raise SystemExit(cmd_ar1(rest))
    print(f"Unknown task {task!r}. Use: roadwork | impromptu | ar1", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
