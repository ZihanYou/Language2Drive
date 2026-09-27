"""Deterministic review artifacts for downloaded NuRec preview videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from .io import atomic_json, load_json


def _video_path(preview_root: Path, clip_id: str) -> Path:
    return (
        Path(preview_root)
        / "sample_set"
        / "26.01_release"
        / clip_id
        / "camera_front_wide_120fov.mp4"
    )


def _probe(video: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration:format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_s": float(duration),
    }


def _review_one(scene: Dict[str, Any], preview_root: Path, output_dir: Path) -> Dict[str, Any]:
    clip_id = scene["clip_id"]
    video = _video_path(preview_root, clip_id)
    sheet = output_dir / (clip_id + ".jpg")
    row: Dict[str, Any] = {
        "clip_id": clip_id,
        "stratum": scene["stratum"],
        "labels": scene["labels"],
        "video_path": str(video.resolve()),
        "video_present": video.is_file(),
        "video_valid": False,
        "contact_sheet": str(sheet.resolve()),
        # Schema convention is [height, width]; dimensions below remove any
        # ambiguity when displaying the usual width x height notation.
        "model_input_resolution": [320, 576],
        "model_input_dimensions": {"width": 576, "height": 320},
        "sample_times_s": [0, 4, 8, 12, 16],
        "sampling_method": "independent_input_seek_not_fps_bin_selection",
        "sample_times_are_seek_requests": True,
        "exact_decoded_timestamps_attested": False,
        "ocr_text": "",
        "ocr_available": shutil.which("tesseract") is not None,
        "ocr_error": None,
        "errors": [],
    }
    if not video.is_file():
        row["errors"].append("preview_not_downloaded")
        return row
    try:
        row.update(_probe(video))
        # fps=1/4 selects a representative near the middle of each output
        # interval, not necessarily the advertised 0/4/8/12/16-second frames.
        # Reuse the independent-seek renderer without rewriting old evidence.
        from .curation_review import _render_sheet
        _render_sheet(video, row["sample_times_s"], sheet)
        row["video_valid"] = True
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        row["errors"].append("preview_review_failed:%s" % type(error).__name__)
        return row

    # OCR is a separate review aid.  A missing OCR executable must not turn a
    # successfully decoded real video into an invalid-video result.  Layer C
    # still requires curator-confirmed readability at model resolution.
    if not row["ocr_available"]:
        row["ocr_error"] = "tesseract_not_available"
        return row
    try:
        ocr = subprocess.run(
            ["tesseract", str(sheet), "stdout", "--psm", "11"],
            check=True,
            capture_output=True,
            text=True,
        )
        row["ocr_text"] = " ".join(ocr.stdout.split())
    except (OSError, subprocess.CalledProcessError) as error:
        row["ocr_error"] = "preview_ocr_failed:%s" % type(error).__name__
    return row


def build_preview_review(
    selection_path: Path,
    preview_root: Path,
    output_dir: Path,
    report_path: Path,
    workers: int = 8,
) -> Dict[str, Any]:
    selection = load_json(selection_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def review(scene: Dict[str, Any]) -> Dict[str, Any]:
        return _review_one(scene, Path(preview_root), output_dir)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(review, selection["scenes"]))
    report = {
        "schema_version": "l2d.realsim.preview_review.v0.1",
        "candidate_count": len(rows),
        "downloaded_preview_count": sum(row["video_present"] for row in rows),
        "valid_preview_count": sum(row["video_valid"] for row in rows),
        "ocr_available_count": sum(row["ocr_available"] for row in rows),
        "ocr_error_count": sum(row["ocr_error"] is not None for row in rows),
        "ocr_nonempty_count": sum(bool(row["ocr_text"]) for row in rows),
        "scenes": rows,
    }
    atomic_json(report_path, report)
    return report
