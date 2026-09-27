"""Provenance binding for sources, configs, code, and evidence."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .io import file_hash, stable_hash


_StatFields = Tuple[int, int, int, int, int, int]
_PathState = Optional[Tuple[_StatFields, Optional[_StatFields]]]


def _stat_fields(value: Any) -> _StatFields:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _path_state(path: Path) -> _PathState:
    """Record both a directory entry and its target without following it silently."""

    try:
        link_stat = path.lstat()
    except FileNotFoundError:
        return None
    try:
        target_stat = path.stat()
    except FileNotFoundError:
        target_stat = None
    return (_stat_fields(link_stat), _stat_fields(target_stat) if target_stat else None)


def _snapshot(paths: Iterable[Path]) -> Dict[str, _PathState]:
    # absolute() keeps the lexical path, unlike resolve(), so symlink replacement
    # remains observable in addition to changes to its target.
    unique = {str(Path(path).absolute()): Path(path) for path in paths}
    return {name: _path_state(path) for name, path in sorted(unique.items())}


def _python_sources(project_root: Path) -> Tuple[Path, ...]:
    source_dir = project_root / "src" / "language2drive_realsim"
    return tuple(sorted(source_dir.glob("*.py"), key=lambda path: str(path.absolute())))


def _git_sha(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(path), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tree_hash(paths: Iterable[Path], root: Optional[Path] = None) -> str:
    """Hash file contents with relocation-stable logical paths."""

    resolved_root = Path(root).resolve() if root is not None else None
    rows = []
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        resolved = path.resolve()
        if resolved_root is not None:
            try:
                logical_path = resolved.relative_to(resolved_root).as_posix()
            except ValueError:
                logical_path = path.name
        else:
            logical_path = path.name if path.is_absolute() else path.as_posix()
        rows.append({"path": logical_path, "sha256": file_hash(path)})
    return stable_hash(rows)


def build_provenance(
    project_root: Path,
    source_path: Path,
    sidecar_path: Optional[Path] = None,
    qualification_path: Optional[Path] = None,
) -> Dict[str, Any]:
    project_root = Path(project_root)
    code_files = _python_sources(project_root)
    verifier_files = [project_root / "src" / "language2drive_realsim" / "events.py", project_root / "src" / "language2drive_realsim" / "verifier.py"]
    oracle_files = [
        project_root / "src" / "language2drive_realsim" / "oracle_model.py",
        project_root
        / "src"
        / "language2drive_realsim"
        / "alpasim_oracle_driver_service.py",
    ]
    threshold_path = project_root / "config" / "threshold_profile.json"
    candidate_path = project_root / "generated" / "candidate_inventory.json"
    curation_path = (
        project_root / "generated" / "curation_review_inventory.private.json"
    )
    alpasim_config_path = project_root / "config" / "alpasim_realsim.json"
    language2drive_git_root = project_root.parent / "language2drive.v1"
    alpasim_git_root = project_root / "vendor" / "alpasim"
    watched_paths = list(code_files) + verifier_files + oracle_files + [
        alpasim_config_path,
        Path(source_path),
        threshold_path,
        candidate_path,
        curation_path,
    ]
    if sidecar_path is not None:
        watched_paths.append(Path(sidecar_path))
    if qualification_path is not None:
        watched_paths.append(Path(qualification_path))

    before_files = _snapshot(watched_paths)
    before_git = {
        "language2drive_git_sha": _git_sha(language2drive_git_root),
        "alpasim_git_sha": _git_sha(alpasim_git_root),
    }
    try:
        result = {
            "schema_version": "l2d.realsim.provenance.v0.1",
            **before_git,
            "alpasim_config_hash": file_hash(alpasim_config_path),
            "pcu_source_hash": file_hash(source_path),
            "private_sidecar_hash": file_hash(sidecar_path) if sidecar_path and sidecar_path.exists() else None,
            "verifier_source_hash": tree_hash(verifier_files, project_root),
            "threshold_profile_hash": file_hash(threshold_path),
            "oracle_source_hash": tree_hash(oracle_files, project_root),
            "scene_candidate_inventory_hash": file_hash(candidate_path) if candidate_path.exists() else None,
            "curation_review_inventory_hash": (
                file_hash(curation_path) if curation_path.exists() else None
            ),
            "qualification_report_hash": file_hash(qualification_path) if qualification_path and qualification_path.exists() else None,
            "code_tree_hash": tree_hash(code_files, project_root),
        }
    except OSError as error:
        raise ValueError("provenance input changed or became unavailable during snapshot") from error

    after_git = {
        "language2drive_git_sha": _git_sha(language2drive_git_root),
        "alpasim_git_sha": _git_sha(alpasim_git_root),
    }
    if after_git != before_git:
        raise ValueError("Git revision changed during provenance snapshot")
    if _python_sources(project_root) != code_files:
        raise ValueError("Python source file set changed during provenance snapshot")
    if _snapshot(watched_paths) != before_files:
        raise ValueError("provenance input changed during snapshot")

    result["binding_hash"] = stable_hash(result)
    return result
