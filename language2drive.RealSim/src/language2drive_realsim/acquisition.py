"""Auditable acquisition of full NuRec artifacts selected from real previews."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import NUREC_DATASET_REVISION, NUREC_VERSION_STRING
from .discovery import _suite_uuids, inspect_downloaded_artifact
from .io import load_json


def _hf_executable() -> Optional[str]:
    """Prefer the CLI installed beside the active RealSim interpreter."""

    sibling = Path(sys.executable).with_name("hf")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("hf")


def _scene_ids(shortlist: Dict[str, Any]) -> List[str]:
    values = [scene for scenes in shortlist["groups"].values() for scene in scenes]
    if len(values) != len(set(values)):
        raise ValueError("shortlist contains duplicate reconstructed scenes")
    return values


def _catalog_rows(
    scenes_csv: Path, revision: str = NUREC_DATASET_REVISION,
) -> Dict[str, Dict[str, str]]:
    """Resolve the pinned revision before indexing clips shared by releases."""
    rows = {}
    with Path(scenes_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["hf_revision"] != revision:
                continue
            scene = row["scene_id"].removeprefix("clipgt-")
            if scene in rows and rows[scene] != row:
                raise ValueError("ambiguous_catalog_scene_in_revision:" + scene)
            rows[scene] = row
    return rows


def _download_one(
    executable: str,
    repository: str,
    revision: str,
    row: Dict[str, str],
    scene_cache: Path,
) -> Dict[str, Any]:
    local_path = scene_cache / row["path"]
    before = inspect_downloaded_artifact(local_path, row)
    if before["readable_usdz"] and before["identity_matches_catalog"] and before["required_members_present"]:
        return {"path": row["path"], "download_attempted": False, "download_error": None}

    environment = os.environ.copy()
    environment.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    environment.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    error = None
    for _attempt in range(3):
        completed = subprocess.run(
            [
                executable,
                "download",
                repository,
                row["path"],
                "--repo-type",
                "dataset",
                "--revision",
                revision,
                "--local-dir",
                str(scene_cache),
                "--quiet",
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode == 0:
            error = None
            break
        error = "hf_download_exit_%d:%s" % (
            completed.returncode,
            completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no_stderr",
        )
    return {"path": row["path"], "download_attempted": True, "download_error": error}


def _batches(rows: List[Dict[str, str]], size: int) -> Any:
    iterator = iter(rows)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def acquire_shortlist(
    shortlist_path: Path,
    scenes_csv: Path,
    suites_csv: Path,
    scene_cache: Path,
    workers: int = 4,
    inventory_only: bool = False,
    minimum_free_gib: float = 20.0,
) -> Dict[str, Any]:
    shortlist = load_json(shortlist_path)
    scene_ids = _scene_ids(shortlist)
    rows_by_scene = _catalog_rows(scenes_csv, shortlist["revision"])
    suite_uuids = _suite_uuids(suites_csv, shortlist["suite"])
    errors: List[str] = []
    if shortlist.get("revision") != NUREC_DATASET_REVISION:
        errors.append("shortlist_revision_must_be:" + NUREC_DATASET_REVISION)
    if shortlist.get("nre_version_string") != NUREC_VERSION_STRING:
        errors.append("shortlist_nre_version_must_be:" + NUREC_VERSION_STRING)
    rows: List[Dict[str, str]] = []
    for scene_id in scene_ids:
        row = rows_by_scene.get(scene_id)
        if row is None:
            errors.append("scene_missing_from_catalog:" + scene_id)
            continue
        if row["uuid"] not in suite_uuids:
            errors.append("scene_missing_from_suite:" + scene_id)
        if row["hf_revision"] != shortlist["revision"]:
            errors.append("revision_mismatch:" + scene_id)
        if row["nre_version_string"] != shortlist["nre_version_string"]:
            errors.append("nre_version_mismatch:" + scene_id)
        rows.append(row)

    worker_count = max(1, int(workers))
    minimum_free_bytes = max(0, int(float(minimum_free_gib) * 1024**3))
    scene_cache = Path(scene_cache)
    scene_cache.mkdir(parents=True, exist_ok=True)
    free_before_bytes = shutil.disk_usage(scene_cache).free
    disk_guard_stopped = False
    disk_guard_skipped_count = 0

    executable = _hf_executable()
    download_rows: List[Dict[str, Any]] = []
    if not inventory_only and not errors:
        if executable is None:
            errors.append("hf_cli_not_available")
        else:
            def download(row: Dict[str, str]) -> Dict[str, Any]:
                return _download_one(
                    executable,
                    shortlist["dataset"],
                    shortlist["revision"],
                    row,
                    Path(scene_cache),
                )

            # Submit only one worker-sized wave at a time. This lets the shared
            # workspace disk guard stop scheduling new multi-GiB artifacts;
            # already running files are allowed to finish cleanly so resumable
            # Hugging Face state is not discarded.
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                for row_index, batch in enumerate(_batches(rows, worker_count)):
                    free_now = shutil.disk_usage(scene_cache).free
                    if free_now < minimum_free_bytes:
                        disk_guard_stopped = True
                        remaining = rows[row_index * worker_count :]
                        disk_guard_skipped_count = len(remaining)
                        download_rows.extend(
                            {
                                "path": row["path"],
                                "download_attempted": False,
                                "download_error": "disk_reserve_reached",
                            }
                            for row in remaining
                        )
                        break
                    download_rows.extend(pool.map(download, batch))

    download_by_path = {item["path"]: item for item in download_rows}
    artifacts = []
    for row in rows:
        inspected = inspect_downloaded_artifact(scene_cache / row["path"], row)
        artifacts.append(
            {
                "scene_id_private": row["scene_id"],
                "artifact_uuid": row["uuid"],
                "artifact_path": row["path"],
                "download": download_by_path.get(row["path"]),
                "inspection": inspected,
            }
        )
    verified = sum(
        item["inspection"]["readable_usdz"]
        and item["inspection"]["identity_matches_catalog"]
        and item["inspection"]["required_members_present"]
        for item in artifacts
    )
    errors.extend(
        item["download"]["download_error"]
        for item in artifacts
        if item["download"] and item["download"]["download_error"]
    )
    return {
        "schema_version": "l2d.realsim.acquisition.v0.1",
        "download_transport": {
            "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            "proxy_environment_present": any(
                bool(os.environ.get(name))
                for name in (
                    "http_proxy",
                    "https_proxy",
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "all_proxy",
                )
            ),
        },
        "shortlist_stage": shortlist["stage"],
        "dataset": shortlist["dataset"],
        "revision": shortlist["revision"],
        "nre_version_string": shortlist["nre_version_string"],
        "suite": shortlist["suite"],
        "selected_scene_count": len(scene_ids),
        "catalog_resolved_count": len(rows),
        "downloaded_artifact_count": sum(item["inspection"]["present"] for item in artifacts),
        "verified_artifact_count": verified,
        "inventory_only": inventory_only,
        "disk_guard": {
            "minimum_free_gib": float(minimum_free_gib),
            "minimum_free_bytes": minimum_free_bytes,
            "free_before_bytes": free_before_bytes,
            "free_after_bytes": shutil.disk_usage(scene_cache).free,
            "stopped_scheduling": disk_guard_stopped,
            "skipped_artifact_count": disk_guard_skipped_count,
        },
        "complete": not errors and verified == len(scene_ids),
        "errors": errors,
        "artifacts": artifacts,
    }
