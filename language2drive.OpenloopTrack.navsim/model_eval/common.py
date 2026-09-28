#!/usr/bin/env python3
"""Shared, public-input-only utilities for l2d-offline-navsim model runs."""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data/public"
SENSORS = Path(os.environ.get(
    "L2D_NAVSIM_SENSOR_ROOT",
    str(ROOT / "assets/openscene-v1.1/sensor_blobs/test"),
))
TIMESTAMPS = np.arange(0.5, 4.01, 0.5, dtype=np.float64)
REWRITE_NAMES = ("canonical", "paraphrase_1", "paraphrase_2", "paraphrase_3")


@dataclass(frozen=True)
class Query:
    pcu_id: str
    family: str
    rewrite: str
    rewrite_index: int
    arm: int
    instruction: str
    public_input: dict

    @property
    def key(self) -> str:
        return f"{self.pcu_id}|{self.rewrite}|{self.arm}"


def load_inputs() -> list[dict]:
    manifest = json.loads((PUBLIC / "manifest.json").read_text())
    rows = []
    for pcu_id in manifest["pcu_ids"]:
        row = json.loads((PUBLIC / "public_inputs" / f"{pcu_id}.json").read_text())
        if row["pcu_id"] != pcu_id:
            raise ValueError(f"PCU mismatch for {pcu_id}")
        rows.append(row)
    return rows


def iter_queries() -> Iterable[Query]:
    for row in load_inputs():
        for rewrite_index, rewrite in enumerate(REWRITE_NAMES):
            for arm in (0, 1):
                instruction = (row["instructions"][arm] if rewrite_index == 0
                               else row["paraphrases"][arm][rewrite_index - 1])
                yield Query(row["pcu_id"], row["family"], rewrite,
                            rewrite_index, arm, instruction, row)


def image_path(row: dict, frame: int, camera: str) -> Path:
    path = SENSORS / row["camera_paths_history"][frame][camera]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def ego_history(row: dict) -> np.ndarray:
    """Return the four observed poses as [x,y,heading], oldest to anchor."""
    xy = np.asarray([[p["x"], p["y"]] for p in row["ego_history"]], dtype=np.float64)
    delta = np.diff(xy, axis=0)
    seg_heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    heading = np.r_[seg_heading, 0.0]
    heading -= heading[-1]
    return np.c_[xy, heading]


def velocity_history(row: dict) -> np.ndarray:
    return np.asarray([p["velocity_xy"] for p in row["ego_history"]], dtype=np.float64)


def acceleration_history(row: dict) -> np.ndarray:
    velocity = velocity_history(row)
    acceleration = np.empty_like(velocity)
    acceleration[1:] = np.diff(velocity, axis=0) / 0.5
    acceleration[0] = acceleration[1]
    return acceleration


def interpolate_history(values: np.ndarray, count: int = 16) -> np.ndarray:
    """Interpolate the observed -1.5..0 s window without extrapolation."""
    values = np.asarray(values, dtype=np.float64)
    old_t = np.linspace(-1.5, 0.0, len(values))
    new_t = np.linspace(-1.5, 0.0, count)
    columns = [np.interp(new_t, old_t, values[:, i]) for i in range(values.shape[1])]
    return np.stack(columns, axis=1)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def headings_from_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    points = np.vstack([[0.0, 0.0], xy])
    delta = np.diff(points, axis=0)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    if len(headings) > 1:
        stationary = np.linalg.norm(delta, axis=1) < 1e-4
        for i in range(1, len(headings)):
            if stationary[i]:
                headings[i] = headings[i - 1]
    return np.unwrap(headings)


def resample_trajectory(points: Sequence[Sequence[float]], source_times: Sequence[float],
                        target_times: Sequence[float] = TIMESTAMPS) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) != len(source_times):
        raise ValueError(f"bad trajectory shape {points.shape}")
    if not np.all(np.isfinite(points[:, :2])):
        raise ValueError("trajectory contains non-finite coordinates")
    x = np.interp(target_times, source_times, points[:, 0])
    y = np.interp(target_times, source_times, points[:, 1])
    return np.c_[x, y]


def submission_trajectory(points_xy: Sequence[Sequence[float]],
                          headings: Sequence[float] | None = None) -> dict:
    xy = np.asarray(points_xy, dtype=np.float64)
    if xy.shape != (8, 2) or not np.all(np.isfinite(xy)):
        raise ValueError(f"expected finite (8,2), got {xy.shape}")
    if headings is None:
        headings = headings_from_xy(xy)
    headings = np.asarray(headings, dtype=np.float64)
    if headings.shape != (8,) or not np.all(np.isfinite(headings)):
        raise ValueError(f"expected finite (8,) headings, got {headings.shape}")
    return {
        "coordinate_frame": "ego_at_anchor",
        "reference_point": "rear_axle",
        "timestamps": TIMESTAMPS.tolist(),
        "positions_xy": xy.tolist(),
        "headings": headings.tolist(),
    }


def parse_numeric_points(text: str, min_points: int = 5) -> list[list[float]]:
    """Extract a model's last plausible list of 2D/3D points."""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    candidates: list[list[list[float]]] = []
    # Prefer Python/JSON-like nested lists.
    starts = [m.start() for m in re.finditer(r"\[", cleaned)]
    for start in starts:
        for end in range(len(cleaned), start + 1, -1):
            if cleaned[end - 1] != "]":
                continue
            try:
                obj = ast.literal_eval(cleaned[start:end])
            except Exception:
                continue
            if isinstance(obj, (list, tuple)):
                pts = []
                for p in obj:
                    if (isinstance(p, (list, tuple)) and len(p) >= 2
                            and all(isinstance(x, (int, float)) for x in p[:3])):
                        pts.append([float(x) for x in p[:3]])
                if len(pts) >= min_points:
                    candidates.append(pts)
            break
    # Also accept tuples or individually bracketed points without an outer list.
    point_re = re.compile(
        r"[\[(]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        r"(?:\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))?\s*[\])]"
    )
    pts = [[float(v) for v in m.groups() if v is not None]
           for m in point_re.finditer(cleaned)]
    if len(pts) >= min_points:
        candidates.append(pts)
    if not candidates:
        raise ValueError(f"could not parse {min_points} waypoints from {text[:500]!r}")
    return candidates[-1]


def load_progress(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row
    return rows


def append_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def consolidate(progress_path: Path, output_dir: Path) -> tuple[Path, Path]:
    rows = load_progress(progress_path)
    expected = list(iter_queries())
    missing = [q.key for q in expected if q.key not in rows or "trajectory" not in rows[q.key]]
    if missing:
        raise RuntimeError(f"missing {len(missing)} predictions; first: {missing[:5]}")
    rewrites = {name: {} for name in REWRITE_NAMES}
    for q in expected:
        rewrites[q.rewrite].setdefault(q.pcu_id, {})[str(q.arm)] = rows[q.key]["trajectory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "rewrite_predictions.json"
    canonical_path = output_dir / "canonical_predictions.json"
    all_path.write_text(json.dumps(rewrites, indent=2))
    canonical_path.write_text(json.dumps(rewrites["canonical"], indent=2))
    return canonical_path, all_path
