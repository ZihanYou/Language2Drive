"""Recoverable Language2Drive evaluator job planning and execution.

The runner deliberately does not start or stop CARLA.  A server is shared by
the existing Bench2Drive workflow, so process lifecycle remains an explicit
operator responsibility.  Every evaluator invocation handles exactly one
route entry and one seed; this makes retries idempotent and prevents language
context from leaking between routes.
"""

from __future__ import annotations

import errno
import fcntl
import copy
import hashlib
import json
import math
import os
import pwd
import re
import secrets
import select
import signal
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib.metadata import PackageNotFoundError, distribution
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from . import carla_runtime_assets as carla_runtime_assets_module
from .fdq import FDQ_ROUTE_BINDING_ROWS, canonical_fdq_route_bindings
from .io import _fsync_directory, atomic_json, load_json, sha256_file, stable_hash
from .carla_runtime_assets import (
    freeze_manifest as freeze_carla_runtime_manifest,
    validate_manifest as validate_carla_runtime_manifest,
    validate_provenance as validate_carla_runtime_provenance,
    validate_server_readonly_permissions,
)


L2D_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_ROOT = L2D_ROOT
BENCHMARK_ROOT = L2D_ROOT / "language2drive"
MODEL_ZOO_ROOT = L2D_ROOT / "model_zoo"
CONDA_PREFIX = Path(sys.prefix)
DEFAULT_CHECKPOINT = (
    MODEL_ZOO_ROOT
    / "autovla/checkpoints/2026-06-13_16-36-13/epoch=4.ckpt"
)
DEFAULT_EVALUATOR = (
    BENCHMARK_ROOT
    / "leaderboard/leaderboard/leaderboard_evaluator.py"
)
EVALUATOR_RUNTIME_DEPENDENCIES = (
    L2D_ROOT / "src/language2drive/io.py",
    L2D_ROOT / "src/language2drive/fdq_metrics.py",
    L2D_ROOT / "team_code/__init__.py",
    BENCHMARK_ROOT / "leaderboard/leaderboard/utils/closed_loop_logger.py",
    CONDA_PREFIX / "lib/python3.11/site-packages/carla/__init__.py",
    CONDA_PREFIX
    / "lib/python3.11/site-packages/carla/libcarla.cpython-311-x86_64-linux-gnu.so",
    L2D_ROOT / "team_code/language2drive_context.py",
    L2D_ROOT / "team_code/language2drive_trace.py",
    L2D_ROOT / "team_code/language2drive_runtime.py",
    L2D_ROOT / "team_code/language2drive_signal_free_proof.py",
    L2D_ROOT / "team_code/language2drive_parking_manoeuvre.py",
    L2D_ROOT / "team_code/language2drive_external_carla.py",
)
EVALUATOR_RUNTIME_DATA_TREES = (
    BENCHMARK_ROOT / "leaderboard/leaderboard",
    BENCHMARK_ROOT / "scenario_runner/srunner",
    L2D_ROOT / "carla_runtime/PythonAPI/carla",
)
BASE_RUNTIME_DISTRIBUTIONS = (
    "carla", "py-trees", "numpy", "shapely", "networkx", "ephem",
    "dictor", "transforms3d", "pydot", "pyparsing", "requests", "urllib3",
    "certifi", "idna", "charset-normalizer", "PyYAML", "Pillow", "six",
    "tabulate", "setuptools", "mkl-service", "pexpect", "ptyprocess",
    "simple-watchdog-timer", "wcwidth",
)
AUTOVLA_RUNTIME_DISTRIBUTIONS = (
    "torch", "transformers", "accelerate", "tokenizers", "safetensors",
)
DEFAULT_AGENT = L2D_ROOT / "team_code/agent_autovla.py"
DEFAULT_ORACLE_AGENT = L2D_ROOT / "team_code/l2d_oracle_agent.py"
DEFAULT_FDQ_ROUTES = BENCHMARK_ROOT / "leaderboard/data/bench2drive220.xml"
_FDQ_ROUTE_BINDING_ROWS = FDQ_ROUTE_BINDING_ROWS
_FDQ_RECORD_STATUSES = frozenset({
    "Completed", "Perfect", "Failed", "Failed - Agent timed out",
    "Failed - Agent deviated from the route", "Failed - Agent got blocked",
})


def _canonical_fdq_route_bindings() -> dict[str, dict[str, str]]:
    """Reconstruct FDQ authority from immutable scalar rows.

    Each caller receives fresh nested dictionaries so a plan/job mutation can
    never mutate the module authority through an alias.
    """

    return canonical_fdq_route_bindings()


# Compatibility/readability export.  Security-sensitive comparisons always
# call _canonical_fdq_route_bindings() rather than trusting this mutable view.
FDQ_ROUTE_BINDINGS = _canonical_fdq_route_bindings()
DEADLINE_FORMULA = "ceil(1.5 * max(seed_1, seed_2, seed_3 completion_time) + 5s)"
DEADLINE_OVERLAY_NAME = "deadlines.frozen.private.json"
DEADLINE_SIDECAR_NAME = "sidecar.deadlines-frozen.private.json"
STORAGE_RESERVE_BYTES = 4 * 1024 ** 3
STORAGE_UPPER_BYTES_PER_SIM_SECOND = 150_000
STORAGE_HARDLINK_PROJECTION_BYTES_PER_SIM_SECOND = 75_000
STORAGE_NO_DEADLINE_SIM_SECONDS = 900.0
EVALUATOR_LOG_MAX_BYTES = 16 * 1024 ** 2
EVALUATOR_TERMINATE_GRACE_SECONDS = 10.0
STORAGE_NONTRACE_FIXED_OVERHEAD_BYTES = 1024 ** 2
AUTOVLA_BASE_MODEL_DIR = (
    MODEL_ZOO_ROOT
    / "autovla/checkpoints/pretrained/Qwen2.5-VL-3B-Instruct"
)
AUTOVLA_BASE_MODEL_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
)
AUTOVLA_RUNTIME_DEPENDENCIES = (
    MODEL_ZOO_ROOT / "autovla/config.yaml",
    MODEL_ZOO_ROOT / "autovla/codebook_cache/agent_vocab.pkl",
    MODEL_ZOO_ROOT / "autovla/code/models/action_tokenizer.py",
    MODEL_ZOO_ROOT / "autovla/code/models/autovla.py",
    MODEL_ZOO_ROOT / "autovla/code/models/utils/score.py",
    L2D_ROOT / "team_code/config_autovla.py",
    L2D_ROOT / "team_code/language2drive_context.py",
)

CARLA_RUNTIME_ROOT = L2D_ROOT / "carla_runtime"
CARLA_RUNTIME_MANIFEST = L2D_ROOT / ".runtime/carla_runtime_assets.current.json"
CARLA_RUNTIME_PROVENANCE = (
    L2D_ROOT / ".runtime/carla_runtime_assets.current.provenance.json"
)
CARLA_RUNTIME_MAP_WITNESS = (
    L2D_ROOT / ".runtime/carla_runtime_available_maps.current.json"
)
CARLA_SERVER_RUNTIME_ROOT = L2D_ROOT / ".runtime/carla-server-release-v1"
CARLA_SERVER_LAUNCH_RECEIPT = CARLA_SERVER_RUNTIME_ROOT / "launch-receipt.json"
CARLA_SERVER_ACCOUNT = "sim"
_EVALUATOR_ENVIRONMENT_TEMPLATE_ROOT = (
    L2D_ROOT / ".runtime/job-environments/__attempt_runtime_v1__"
)
_EVALUATOR_ENVIRONMENT_PARENT = L2D_ROOT / ".runtime/job-environments"
_LOWER_HEX = frozenset("0123456789abcdef")


def carla_server_runtime_root(port: int) -> Path:
    """Return the isolated release-runtime root for one CARLA endpoint."""

    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("CARLA server port must be in 1..65535")
    if port == 2000:
        return CARLA_SERVER_RUNTIME_ROOT
    return CARLA_SERVER_RUNTIME_ROOT.with_name(
        f"{CARLA_SERVER_RUNTIME_ROOT.name}-port-{port}"
    )


def _strict_json_loads(raw: str, label: str) -> Any:
    """Parse strict JSON, rejecting duplicate keys and non-finite constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label}: non-finite JSON constant {value}")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label}: invalid JSON: {error}") from error


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _current_file_sha256(path: Path) -> str:
    """Freshly hash an opened regular inode with pre/post identity checks."""

    resolved = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"execution dependency is not a regular file: {resolved}")
        key = (
            str(resolved), int(before.st_dev), int(before.st_ino), int(before.st_size),
            int(before.st_mtime_ns), int(before.st_ctime_ns),
        )
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
        digest = hasher.hexdigest()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        str(resolved), int(after.st_dev), int(after.st_ino), int(after.st_size),
        int(after.st_mtime_ns), int(after.st_ctime_ns),
    )
    current = resolved.lstat()
    current_identity = (
        str(resolved), int(current.st_dev), int(current.st_ino), int(current.st_size),
        int(current.st_mtime_ns), int(current.st_ctime_ns),
    )
    if identity != key or current_identity != key:
        raise ValueError(f"execution dependency changed while hashing: {resolved}")
    return digest


def _dependency_key(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return "project/" + resolved.relative_to(L2D_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _python_site_startup_hash(site_packages: Path) -> str:
    """Hash the exact set of files Python may execute from ``site`` startup."""

    root = Path(site_packages)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("target Python site-packages root is missing")
    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError as error:
        raise ValueError(f"cannot enumerate Python startup surface: {error}") from error
    rows = []
    for entry in entries:
        name = entry.name
        relevant = (
            name.endswith(".pth")
            or name == "sitecustomize"
            or name == "usercustomize"
            or name.startswith("sitecustomize.")
            or name.startswith("usercustomize.")
        )
        if not relevant:
            continue
        try:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError(
                    f"Python startup dependency is non-regular: {entry.path}"
                )
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"cannot inspect Python startup dependency {entry.path}: {error}"
            ) from error
        rows.append({
            "name": name,
            "size_bytes": metadata.st_size,
            "sha256": _current_file_sha256(Path(entry.path)),
        })
    return stable_hash(rows)


def _distribution_runtime_hash(package: Any) -> str:
    """Hash declared files plus actual import roots for one distribution.

    Conda occasionally installs legacy ``egg-info/SOURCES.txt`` inventories
    containing source-tree files that are intentionally absent, and those
    inventories may omit installed extension modules.  Record both the exact
    present/absent declaration state and the actual top-level import trees so
    neither case creates a fail-open.
    """

    prefix = CONDA_PREFIX.resolve(strict=True)
    site_packages = (CONDA_PREFIX / "lib/python3.11/site-packages").resolve(strict=True)
    declared_rows = []
    for record_path in sorted(package.files or [], key=str):
        path = Path(package.locate_file(record_path))
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            declared_rows.append({"record_path": str(record_path), "present": False})
            continue
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"runtime distribution file is non-regular: {path}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(prefix)
        except ValueError as error:
            raise ValueError(
                f"runtime distribution file escapes the target prefix: {resolved}"
            ) from error
        declared_rows.append({
            "record_path": str(record_path), "present": True,
            "resolved_path": str(resolved), "size_bytes": metadata.st_size,
            "sha256": _current_file_sha256(resolved),
        })

    actual_roots: set[Path] = set()
    raw_top_level = package.read_text("top_level.txt") or ""
    for line in raw_top_level.splitlines():
        name = line.strip()
        if not name:
            continue
        if not name.isidentifier():
            raise ValueError(f"runtime distribution top-level name is invalid: {name!r}")
        directory = site_packages / name
        module = site_packages / f"{name}.py"
        if directory.exists():
            actual_roots.add(directory)
        elif module.exists():
            actual_roots.add(module)
        else:
            raise ValueError(f"runtime distribution import root is missing: {name}")
    metadata_root = Path(package._path).resolve(strict=True)
    try:
        metadata_root.relative_to(site_packages)
    except ValueError as error:
        raise ValueError("runtime distribution metadata escapes site-packages") from error
    actual_roots.add(metadata_root)

    actual_rows = []
    for root in sorted(actual_roots, key=str):
        if root.is_symlink():
            raise ValueError(f"runtime distribution import root is a symlink: {root}")
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = []
            for current, names, filenames in os.walk(
                root, followlinks=False,
                onerror=lambda error: (_ for _ in ()).throw(
                    ValueError(f"cannot enumerate runtime distribution: {error}")
                ),
            ):
                current_path = Path(current)
                if "__pycache__" in current_path.relative_to(root).parts:
                    names[:] = []
                    continue
                names.sort()
                filenames.sort()
                for name in names:
                    child = current_path / name
                    if child.is_symlink() or not child.is_dir():
                        raise ValueError(
                            f"runtime distribution directory is non-real: {child}"
                        )
                candidates.extend(current_path / name for name in filenames)
        else:
            raise ValueError(f"runtime distribution import root is non-regular: {root}")
        for path in candidates:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"runtime distribution file is non-regular: {path}")
            resolved = path.resolve(strict=True)
            actual_rows.append({
                "path": resolved.relative_to(site_packages).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": _current_file_sha256(resolved),
            })
    actual_rows.sort(key=lambda row: row["path"])
    return stable_hash({
        "name": package.metadata["Name"], "version": package.version,
        "declared_files": declared_rows, "actual_import_files": actual_rows,
    })


def _python_prefix_runtime_hash() -> str:
    """Hash the complete target Python prefix, excluding isolated bytecode.

    The evaluator can load Python modules, extension modules and native
    libraries indirectly.  A whole-prefix authority is deliberately broader
    than a hand-maintained import list and therefore closes future transitive
    imports too.  ``PYTHONPYCACHEPREFIX`` points at an empty read-only tree, so
    source-adjacent ``__pycache__`` directories are the only excluded bytes.
    """

    root = CONDA_PREFIX.resolve(strict=True)

    def inventory(
        *, hash_files: bool,
    ) -> tuple[dict[str, tuple[Any, ...]], list[dict[str, Any]]]:
        identities: dict[str, tuple[Any, ...]] = {}
        rows: list[dict[str, Any]] = []

        def onerror(error: OSError) -> None:
            raise ValueError(f"cannot enumerate target Python prefix: {error}") from error

        for current_text, names, filenames in os.walk(
            root, followlinks=False, onerror=onerror,
        ):
            current = Path(current_text)
            relative_current = current.relative_to(root)
            if "__pycache__" in relative_current.parts:
                names[:] = []
                continue
            kept_names = []
            for name in sorted(names):
                child = current / name
                relative = child.relative_to(root).as_posix()
                metadata = child.lstat()
                if child.is_symlink():
                    target = os.readlink(child)
                    resolved = child.resolve(strict=True)
                    try:
                        resolved.relative_to(root)
                    except ValueError as error:
                        raise ValueError(
                            f"Python prefix directory symlink escapes prefix: {child}"
                        ) from error
                    resolved_metadata = resolved.stat()
                    if not stat.S_ISDIR(resolved_metadata.st_mode):
                        raise ValueError(
                            f"Python prefix directory symlink target is not a directory: {child}"
                        )
                    identities[relative] = (
                        "symlink-directory", metadata.st_dev, metadata.st_ino,
                        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
                        target, str(resolved), resolved_metadata.st_dev,
                        resolved_metadata.st_ino, resolved_metadata.st_mtime_ns,
                        resolved_metadata.st_ctime_ns,
                    )
                    if hash_files:
                        rows.append({
                            "path": relative, "type": "symlink-directory",
                            "target": target, "resolved_path": str(resolved),
                        })
                    continue
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(f"Python prefix directory is non-real: {child}")
                identities[relative] = (
                    "directory", metadata.st_dev, metadata.st_ino,
                    metadata.st_mtime_ns, metadata.st_ctime_ns,
                )
                kept_names.append(name)
            names[:] = kept_names
            for name in sorted(filenames):
                path = current / name
                if "__pycache__" in path.relative_to(root).parts:
                    continue
                relative = path.relative_to(root).as_posix()
                metadata = path.lstat()
                if path.is_symlink():
                    target = os.readlink(path)
                    try:
                        resolved = path.resolve(strict=True)
                    except FileNotFoundError:
                        identities[relative] = (
                            "symlink-broken", metadata.st_dev, metadata.st_ino,
                            metadata.st_size, metadata.st_mtime_ns,
                            metadata.st_ctime_ns, target,
                        )
                        if hash_files:
                            rows.append({
                                "path": relative, "type": "symlink-broken",
                                "target": target,
                            })
                        continue
                    resolved_metadata = resolved.stat()
                    if not stat.S_ISREG(resolved_metadata.st_mode):
                        raise ValueError(
                            f"Python prefix file symlink target is non-regular: {path}"
                        )
                    identities[relative] = (
                        "symlink-file", metadata.st_dev, metadata.st_ino,
                        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
                        target, str(resolved), resolved_metadata.st_dev,
                        resolved_metadata.st_ino, resolved_metadata.st_size,
                        resolved_metadata.st_mtime_ns, resolved_metadata.st_ctime_ns,
                    )
                    if hash_files:
                        rows.append({
                            "path": relative, "type": "symlink-file", "target": target,
                            "resolved_path": str(resolved),
                            "size_bytes": int(resolved_metadata.st_size),
                            "sha256": _current_file_sha256(resolved),
                        })
                elif stat.S_ISREG(metadata.st_mode):
                    identities[relative] = (
                        "file", metadata.st_dev, metadata.st_ino, metadata.st_size,
                        metadata.st_mtime_ns, metadata.st_ctime_ns,
                    )
                    if hash_files:
                        rows.append({
                            "path": relative, "type": "file",
                            "size_bytes": int(metadata.st_size),
                            "sha256": _current_file_sha256(path),
                        })
                else:
                    raise ValueError(f"Python prefix entry is non-regular: {path}")
        rows.sort(key=lambda row: row["path"])
        return identities, rows

    before, rows = inventory(hash_files=True)
    after, _ = inventory(hash_files=False)
    if before != after:
        raise ValueError("target Python prefix changed while being hashed")
    return stable_hash({
        "schema_version": "language2drive.python_prefix_runtime.v1",
        "root": str(root), "entry_count": len(before), "entries": rows,
    })


def _mapped_evaluator_native_runtime_hash() -> str:
    """Import the oracle workload cleanly and bind every mapped regular file."""

    script = r'''
import json
import os
import leaderboard.leaderboard_evaluator
import team_code.l2d_oracle_agent
paths = set()
with open("/proc/self/maps", "r", encoding="ascii") as handle:
    for line in handle:
        fields = line.rstrip().split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = os.path.realpath(fields[5])
            if os.path.isfile(path):
                paths.add(path)
print(json.dumps(sorted(paths), separators=(",", ":"), allow_nan=False))
'''
    try:
        with _temporary_evaluator_environment("mapped-native") as environment:
            completed = subprocess.run(
                [str(CONDA_PREFIX / "bin/python"), "-c", script],
                cwd=str(EVALUATOR_ROOT), env=environment, text=True,
                capture_output=True, timeout=60, check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot probe mapped evaluator runtime: {error}") from error
    if completed.returncode != 0 or completed.stderr:
        diagnostic = (completed.stderr or completed.stdout).strip()[-1000:]
        raise ValueError(
            f"mapped evaluator runtime probe failed ({completed.returncode}): {diagnostic}"
        )
    value = _strict_json_loads(completed.stdout, "mapped evaluator runtime probe")
    if (
        not isinstance(value, list) or value != sorted(set(value))
        or not all(isinstance(item, str) and Path(item).is_absolute() for item in value)
    ):
        raise ValueError("mapped evaluator runtime probe shape differs")
    rows = []
    for raw_path in value:
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"mapped evaluator runtime file is non-regular: {path}")
        rows.append({
            "path": str(path.resolve(strict=True)), "size_bytes": path.stat().st_size,
            "sha256": _current_file_sha256(path),
        })
    return stable_hash(rows)


def _execution_dependency_hashes(profile: str | None) -> dict[str, str]:
    """Hash deterministic runtime dependencies without copying model assets."""

    paths = [
        L2D_ROOT / "environment.lock.yml",
        L2D_ROOT / "tools/start_external_carla.sh",
        Path(__file__),
        Path(carla_runtime_assets_module.__file__),
        *EVALUATOR_RUNTIME_DEPENDENCIES,
    ]
    if profile == "session-runtime.v1":
        # High-throughput experiment profile: bind the evaluator entrypoint,
        # runner, explicit bridge files, Python executable/version, and CARLA
        # extension, while deliberately avoiding recursive scans of the whole
        # conda prefix and source trees.  The agent/config themselves are
        # hashed separately in ``expected_input_hashes``.  This profile is
        # auditable but is not the byte-complete release-publication profile.
        paths.append(DEFAULT_EVALUATOR)
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise ValueError(
                "session dependency does not exist: "
                + ", ".join(str(path) for path in missing)
            )
        hashes = {
            _dependency_key(path.resolve()): _current_file_sha256(path)
            for path in sorted(set(paths), key=lambda item: _dependency_key(item))
        }
        executable = Path(sys.executable).resolve(strict=True)
        expected_executable = (CONDA_PREFIX / "bin/python").resolve(strict=True)
        if executable != expected_executable:
            raise ValueError(
                "session dependency closure must run under the target SimLingo Python"
            )
        hashes["python-runtime/executable"] = _current_file_sha256(executable)
        hashes["python-runtime/version"] = stable_hash({
            "version": sys.version,
            "executable": str(executable),
        })
        hashes["python-runtime/site-startup-files"] = _python_site_startup_hash(
            CONDA_PREFIX / "lib/python3.11/site-packages"
        )
        return dict(sorted(hashes.items()))
    for root in EVALUATOR_RUNTIME_DATA_TREES:
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"evaluator runtime dependency tree is missing: {root}")
        def onerror(error: OSError) -> None:
            raise ValueError(f"cannot enumerate evaluator dependency tree: {error}") from error
        for current, names, filenames in os.walk(root, followlinks=False, onerror=onerror):
            current_path = Path(current)
            relative_parts = current_path.relative_to(root).parts
            if "__pycache__" in relative_parts:
                names[:] = []
                continue
            names.sort()
            filenames.sort()
            for name in names:
                child = current_path / name
                if child.is_symlink() or not child.is_dir():
                    raise ValueError(f"evaluator dependency directory is not real: {child}")
            for name in filenames:
                path = current_path / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"evaluator runtime dependency is non-regular: {path}")
                paths.append(path)
    stdlib_root = CONDA_PREFIX / "lib/python3.11"
    if not stdlib_root.is_dir() or stdlib_root.is_symlink():
        raise ValueError("target Python standard-library root is missing")
    for current, names, filenames in os.walk(
        stdlib_root, followlinks=False,
        onerror=lambda error: (_ for _ in ()).throw(
            ValueError(f"cannot enumerate Python stdlib: {error}")
        ),
    ):
        current_path = Path(current)
        names[:] = sorted(
            name for name in names
            if name not in {"site-packages", "__pycache__"}
        )
        filenames.sort()
        for name in names:
            child = current_path / name
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"Python stdlib directory is not real: {child}")
        for name in filenames:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Python stdlib dependency is non-regular: {path}")
            paths.append(path)
    if profile == "autovla-runtime.v1":
        config_path = MODEL_ZOO_ROOT / "autovla/config.yaml"
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            model = config["model"]
            configured_base = Path(model["pretrained_model_path"]).resolve()
            configured_codebook = Path(model["codebook_cache_path"]).resolve()
        except (OSError, UnicodeError, TypeError, KeyError, yaml.YAMLError) as error:
            raise ValueError(f"invalid AutoVLA dependency config: {error}") from error
        if configured_base != AUTOVLA_BASE_MODEL_DIR.resolve():
            raise ValueError(
                "AutoVLA pretrained_model_path changed outside the frozen dependency profile"
            )
        expected_codebook = (
            MODEL_ZOO_ROOT / "autovla/codebook_cache/agent_vocab.pkl"
        ).resolve()
        if configured_codebook != expected_codebook:
            raise ValueError(
                "AutoVLA codebook_cache_path changed outside the frozen dependency profile"
            )
        paths.extend(AUTOVLA_RUNTIME_DEPENDENCIES)
        paths.extend(AUTOVLA_BASE_MODEL_DIR / name for name in AUTOVLA_BASE_MODEL_FILES)
    elif profile not in {None, "base-runtime.v1"}:
        raise ValueError(f"unsupported execution dependency profile: {profile}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            "execution dependency does not exist: "
            + ", ".join(str(path) for path in missing)
        )
    paths = sorted({path.resolve() for path in paths}, key=lambda path: _dependency_key(path))
    hashes = {
        _dependency_key(path): _current_file_sha256(path)
        for path in sorted(paths, key=lambda item: _dependency_key(item))
    }
    executable = Path(sys.executable).resolve(strict=True)
    expected_executable = (CONDA_PREFIX / "bin/python").resolve(strict=True)
    if executable != expected_executable:
        raise ValueError(
            "execution dependency closure must run under the target SimLingo Python"
        )
    hashes["python-runtime/executable"] = _current_file_sha256(executable)
    hashes["python-runtime/version"] = stable_hash({
        "version": sys.version,
        "executable": str(executable),
    })
    # Python processes the top-level ``*.pth`` files during interpreter
    # startup and may import ``sitecustomize``/``usercustomize`` before the
    # evaluator entrypoint.  Bind the exact startup surface, including the
    # empty-set case, so adding or removing one of these files invalidates a
    # saved plan even when no installed distribution RECORD mentions it.
    hashes["python-runtime/site-startup-files"] = _python_site_startup_hash(
        CONDA_PREFIX / "lib/python3.11/site-packages"
    )
    hashes["python-runtime/prefix-manifest"] = _python_prefix_runtime_hash()
    if profile in {None, "base-runtime.v1"}:
        hashes["native-runtime/mapped-oracle-workload"] = (
            _mapped_evaluator_native_runtime_hash()
        )
    names = list(BASE_RUNTIME_DISTRIBUTIONS)
    if profile == "autovla-runtime.v1":
        names.extend(AUTOVLA_RUNTIME_DISTRIBUTIONS)
    for name in sorted(names):
        try:
            package = distribution(name)
        except PackageNotFoundError as error:
            raise ValueError(f"required runtime distribution is not installed: {name}") from error
        if not list(package.files or []):
            raise ValueError(f"runtime distribution has no RECORD inventory: {name}")
        hashes[f"python-distribution/{name}"] = _distribution_runtime_hash(package)
    loader = Path("/lib64/ld-linux-x86-64.so.2").resolve(strict=True)
    linked_binaries = {
        Path(CONDA_PREFIX / "bin/python").resolve(strict=True),
        Path(
            CONDA_PREFIX
            / "lib/python3.11/site-packages/carla/"
            "libcarla.cpython-311-x86_64-linux-gnu.so"
        ).resolve(strict=True),
    }
    linked_libraries = {loader}
    link_environment = {
        "LD_LIBRARY_PATH": os.pathsep.join((
            str(CONDA_PREFIX / "lib/python3.11/site-packages/pillow.libs"),
            str(CONDA_PREFIX / "lib"),
        )),
    }
    for binary in sorted(linked_binaries, key=str):
        completed = subprocess.run(
            [str(loader), "--list", str(binary)],
            cwd="/", env=link_environment, text=True, capture_output=True,
            timeout=20, check=False,
        )
        if completed.returncode != 0 or completed.stderr or "not found" in completed.stdout:
            diagnostic = (completed.stderr or completed.stdout).strip()[-1000:]
            raise ValueError(f"cannot resolve native runtime closure: {diagnostic}")
        for line in completed.stdout.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("linux-vdso.so"):
                continue
            candidate = stripped.split(" => ", 1)[-1].split(" (", 1)[0]
            if not candidate.startswith("/"):
                raise ValueError(f"native runtime dependency is unresolved: {stripped}")
            linked_libraries.add(Path(candidate).resolve(strict=True))
    for library in sorted(linked_libraries, key=str):
        if library.is_symlink() or not library.is_file():
            raise ValueError(f"native runtime dependency is non-regular: {library}")
        hashes[f"native-runtime/{library}"] = _current_file_sha256(library)
    return dict(sorted(hashes.items()))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_runs(path: Path) -> list[dict[str, Any]]:
    value = load_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
        raise ValueError(f"{path}: expected an object containing a runs list")
    if value.get("expected_count") != len(value["runs"]):
        raise ValueError(f"{path}: expected_count does not match runs")
    seen: set[str] = set()
    output = []
    for index, raw in enumerate(value["runs"]):
        if not isinstance(raw, dict) or not isinstance(raw.get("run_id"), str):
            raise ValueError(f"{path}: invalid run at index {index}")
        if raw["run_id"] in seen:
            raise ValueError(f"{path}: duplicate run_id {raw['run_id']}")
        seen.add(raw["run_id"])
        output.append(dict(raw))
    return output


def default_ledger(build_dir: Path, track: str, *, pilot: bool = False, tier: str = "large") -> Path:
    if track == "standardized":
        if tier not in {"small", "base", "large"}:
            raise ValueError("benchmark tier must be small, base, or large")
        if pilot and tier != "large":
            raise ValueError("--pilot only supports the complete large tier")
        if pilot:
            return build_dir / "pilot_expected_runs.json"
        return build_dir / f"{tier}_expected_runs.json"
    if track == "native":
        return build_dir / "native_smoke_expected_runs.json"
    if track == "oracle":
        return build_dir / "oracle_expected_runs.json"
    if track == "fdq":
        return build_dir / "fdq_expected_runs.json"
    raise ValueError("runner track must be standardized, native, oracle, or fdq")


def _strict_json_object_file(path: Path, label: str) -> dict[str, Any]:
    """Read one unique regular JSON object without permissive JSON aliases."""

    source = Path(path)
    metadata = source.lstat()
    if (
        source.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"{label} is not unique regular evidence")
    value = _strict_json_loads(source.read_text(encoding="utf-8"), label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


_CORE_RESULT_COMPONENT_KEYS = (
    "own_contract", "opposite_contract", "safe_legal", "progress",
    "deadline", "stability",
)
_TERMINAL_FAILURE_STATUSES = frozenset({"crash", "timeout", "deadlock", "stuck"})


def _normalized_result_status(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return value.casefold()


def _core_result_is_final(
    value: dict[str, Any], path: Path, expected_run: dict[str, Any] | None,
) -> bool:
    status = _normalized_result_status(value.get("status"))
    success = value.get("success")
    if type(success) is not bool or status is None:
        return False
    semantic_success = value.get("semantic_success", value.get("own_contract"))
    if type(semantic_success) is not bool:
        return False
    if "driving_score" in value and value["driving_score"] is not None:
        from .core_driving_score import valid_driving_score
        if not valid_driving_score(value["driving_score"]):
            return False
    if expected_run is not None and (
        value.get("run_id") != expected_run.get("run_id")
        or value.get("pcu_id") != expected_run.get("pcu_id")
        or value.get("arm_id") != expected_run.get("arm_id")
        or value.get("paraphrase") != expected_run.get("paraphrase")
        or value.get("seed") != expected_run.get("seed")
        or value.get("track") != expected_run.get("track")
        or value.get("route_entry_id") != expected_run.get("route_entry_id")
    ):
        # Evaluator-produced complete results intentionally expose only the
        # route identity; their remaining identity is bound by route_meta and
        # runner_command below.  Runner-produced terminal results always carry
        # the entire ledger row identity and must never be reusable cross-run.
        if status in _TERMINAL_FAILURE_STATUSES:
            return False
    if status in _TERMINAL_FAILURE_STATUSES:
        return (
            success is False
            and value.get("schema_version") == "language2drive.contract_result.v1"
            and value.get("capture_complete") is False
        )
    if status not in {"complete", "completed", "success", "passed", "failed", "failure"}:
        return False
    if (
        value.get("capture_complete") is not True
        or bool(value.get("unsupported_required_fields"))
        or bool(value.get("postprocess_error"))
        or not path.with_name("trace.jsonl").is_file()
        or any(type(value.get(key)) is not bool for key in _CORE_RESULT_COMPONENT_KEYS)
    ):
        return False
    expected_success = bool(
        value["own_contract"]
        and not value["opposite_contract"]
        and value["safe_legal"]
        and value["progress"]
        and value["deadline"]
        and value["stability"]
    )
    if success != expected_success:
        return False
    if expected_run is not None and (
        value.get("pcu_id") != expected_run.get("pcu_id")
        or value.get("arm_id") != expected_run.get("arm_id")
        or value.get("route_entry_id") != expected_run.get("route_entry_id")
    ):
        return False
    return True


def _fdq_result_is_final(
    value: dict[str, Any], path: Path, expected_run: dict[str, Any] | None,
) -> bool:
    if value.get("schema_version") != "language2drive.fdq_result.v1":
        return False
    status = _normalized_result_status(value.get("status"))
    if type(value.get("success")) is not bool or status is None:
        return False
    if expected_run is not None:
        ability = expected_run.get("ability")
        binding = _canonical_fdq_route_bindings().get(str(ability))
        if binding is None or any(
            (
                value.get("run_id") != expected_run.get("run_id"),
                value.get("ability") != ability,
                value.get("seed") != expected_run.get("seed"),
                value.get("route_id") != binding["route_id"],
                value.get("town") != binding["town"],
                value.get("scenario") != binding["scenario"],
            )
        ):
            return False
    if status in _TERMINAL_FAILURE_STATUSES:
        return (
            value["success"] is False
            and value.get("capture_complete") is False
        )
    if status != "complete" or value.get("capture_complete") is not True:
        return False
    if set(value) != {
        "schema_version", "status", "success", "capture_complete", "run_id",
        "ability", "seed", "route_id", "town", "scenario", "record_status",
        "DS", "RC", "IP", "infraction_free", "source", "source_sha256",
        "checkpoint_record_sha256",
    }:
        return False
    if type(value.get("infraction_free")) is not bool:
        return False
    for key in ("DS", "RC", "IP"):
        metric = value.get(key)
        if (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(float(metric))
        ):
            return False
    expected_bounds = {"DS": (0.0, 100.0), "RC": (0.0, 100.0), "IP": (0.0, 1.0)}
    if any(
        not minimum <= float(value[key]) <= maximum
        for key, (minimum, maximum) in expected_bounds.items()
    ):
        return False
    source_raw = value.get("source")
    if not isinstance(source_raw, str) or not source_raw:
        return False
    source = Path(source_raw)
    if (
        source != path.with_name("bench2drive_result.json").resolve()
        or not source.is_file()
        or value.get("source_sha256") != _current_file_sha256(source)
    ):
        return False
    try:
        checkpoint = _strict_json_object_file(source, "Bench2Drive checkpoint")
        records = checkpoint["_checkpoint"]["records"]
        record = records[0]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, IndexError):
        return False
    if (
        not isinstance(records, list)
        or len(records) != 1
        or not isinstance(record, dict)
        or value.get("checkpoint_record_sha256") != stable_hash(record)
    ):
        return False
    binding = _canonical_fdq_route_bindings().get(str(value.get("ability")))
    if binding is None:
        return False
    scores = record.get("scores")
    infractions = record.get("infractions")
    canonical_infraction_keys = {
        "collisions_layout", "collisions_pedestrian", "collisions_vehicle",
        "red_light", "stop_infraction", "outside_route_lanes",
        "min_speed_infractions", "yield_emergency_vehicle_infractions",
        "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout",
    }
    score_bounds = {
        "score_composed": (0.0, 100.0),
        "score_route": (0.0, 100.0),
        "score_penalty": (0.0, 1.0),
    }
    if (
        record.get("route_id") != f"RouteScenario_{binding['route_id']}_rep0"
        or record.get("town_name") != binding["town"]
        or record.get("scenario_name") != f"{binding['scenario']}_1"
        or record.get("status") != value.get("record_status")
        or record.get("status") not in _FDQ_RECORD_STATUSES
        or not isinstance(scores, dict)
        or any(
            not isinstance(scores.get(key), (int, float))
            or isinstance(scores.get(key), bool)
            or not math.isfinite(float(scores[key]))
            or not minimum <= float(scores[key]) <= maximum
            for key, (minimum, maximum) in score_bounds.items()
        )
        or not isinstance(infractions, dict)
        or set(infractions) != canonical_infraction_keys
        or any(not isinstance(item, list) for item in infractions.values())
        or type(record.get("num_infractions")) is not int
        or record["num_infractions"] < 0
        or record["num_infractions"]
        != sum(len(item) for item in infractions.values())
        or value["DS"] != scores.get("score_composed")
        or value["RC"] != scores.get("score_route")
        or value["IP"] != scores.get("score_penalty")
        or value["infraction_free"] != all(not item for item in infractions.values())
    ):
        return False
    expected_success = bool(
        value.get("record_status") in {"Completed", "Perfect"}
        and value["infraction_free"]
    )
    return bool(
        value["success"] == expected_success
        and (
            not expected_success
            or (value["DS"], value["RC"], value["IP"])
            == (100.0, 100.0, 1.0)
        )
    )


def _result_is_final(
    path: Path, expected_run: dict[str, Any] | None = None,
) -> bool:
    try:
        value = _strict_json_object_file(path, "result")
    except (OSError, UnicodeError, ValueError):
        return False
    if Path(path).name == "fdq_result.json":
        return _fdq_result_is_final(value, Path(path), expected_run)
    return _core_result_is_final(value, Path(path), expected_run)


def _public_context_matches_evaluator_track(
    public: Any, evaluator_track: Any, seed: Any,
) -> bool:
    """Validate public context without leaking Standardized private identity."""

    if not isinstance(public, dict) or public.get("track") != evaluator_track:
        return False
    if evaluator_track == "standardized":
        return "seed" not in public and "route_name" not in public
    return public.get("seed") == seed


def _result_is_current_for_sidecar(
    path: Path,
    sidecar_path: Path | None,
    expected_input_hashes: dict[str, Any] | None = None,
    *,
    expected_run: dict[str, Any] | None = None,
    workload: str = "core",
) -> bool:
    """Require finalized core results to match the exact private suite.

    Stable run IDs are useful for recovery, but they deliberately do not
    encode a multi-megabyte sidecar hash.  Without this check a successful or
    crashed rollout from an older build could be skipped after regeneration.
    Runner provenance is available even when the evaluator crashes before it
    can emit ``run_meta.json``; evaluator provenance remains a compatibility
    fallback for earlier runs.
    """

    if expected_run is None or not _result_is_final(path, expected_run):
        return False
    command_path = path.with_name("runner_command.json")
    try:
        command = _strict_json_object_file(command_path, "runner command")
        if command.get("run") != expected_run:
            return False
        if path.parent.name != expected_run.get("run_id"):
            return False
        observed_hashes = command.get("input_hashes", {})
        if not isinstance(observed_hashes, dict):
            return False
        observed = observed_hashes.get("private_sidecar_sha256")
    except (OSError, UnicodeError, ValueError, TypeError):
        observed_hashes = {}
        observed = None
    if expected_input_hashes is not None:
        # Bundle equality is deliberate: silently accepting a subset would
        # let results from an old agent, XML, config, evaluator or information
        # track suppress a required retry after a release input changes.
        if observed_hashes != expected_input_hashes:
            return False
        if workload == "fdq":
            return Path(path).name == "fdq_result.json"
        try:
            result_value = _strict_json_object_file(path, "result")
        except (OSError, UnicodeError, ValueError):
            return False
        if _normalized_result_status(result_value.get("status")) in _TERMINAL_FAILURE_STATUSES:
            # A launch crash may precede evaluator metadata, but its exact
            # runner command/run identity and full execution bundle still
            # make it a legitimate finalized benchmark outcome.
            return True
        try:
            metadata = _strict_json_object_file(
                path.with_name("route_meta.json"), "route metadata",
            )
            run_metadata = _strict_json_object_file(
                path.with_name("run_meta.json"), "run metadata",
            )
        except (OSError, UnicodeError, ValueError):
            return False
        evaluator_track = expected_input_hashes.get("evaluator_track")
        expected_meta_run_id = (
            f"{evaluator_track}__{expected_run.get('pcu_id')}__"
            f"{expected_run.get('arm_id')}__p{expected_run.get('paraphrase')}__"
            f"s{expected_run.get('seed')}"
        )
        provenance = metadata.get("private_capture_provenance")
        return bool(
            metadata == run_metadata
            and metadata.get("run_id") == expected_meta_run_id
            and metadata.get("route_entry_id") == expected_run.get("route_entry_id")
            and metadata.get("route_id")
            == f"RouteScenario_{expected_run.get('route_entry_id')}"
            and _public_context_matches_evaluator_track(
                metadata.get("language2drive_public_context"),
                evaluator_track,
                expected_run.get("seed"),
            )
            and provenance == {
                "route_entry_id": expected_run.get("route_entry_id"),
                "private_sidecar_sha256": observed,
            }
        )
    if sidecar_path is None:
        return True
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        return False
    expected = sha256_file(sidecar_path)
    if observed is None:
        for name in ("run_meta.json", "route_meta.json"):
            try:
                meta = load_json(path.with_name(name))
                observed = meta.get("private_capture_provenance", {}).get(
                    "private_sidecar_sha256"
                )
            except (OSError, UnicodeError, ValueError, TypeError):
                continue
            if observed is not None:
                break
    return isinstance(observed, str) and observed == expected


def select_runs(
    ledger_path: Path,
    results_dir: Path,
    *,
    track: str,
    split: str,
    seeds: Iterable[int],
    route_entry_ids: Iterable[int] | None = None,
    sidecar_path: Path | None = None,
    expected_input_hashes: dict[str, Any] | None = None,
    retry_finalized: bool = False,
    max_runs: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select pending jobs while retaining legitimate failed verdicts.

    A finalized failed/crash/timeout result is an observed benchmark outcome,
    not an automatically retryable infrastructure gap.  ``retry_finalized``
    is therefore explicit.
    """

    seed_set = {int(value) for value in seeds}
    if not seed_set or any(value < 0 for value in seed_set):
        raise ValueError("seeds must contain non-negative integers")
    route_entry_set = None if route_entry_ids is None else {int(value) for value in route_entry_ids}
    if route_entry_set is not None and (not route_entry_set or any(value < 0 for value in route_entry_set)):
        raise ValueError("route_entry_ids must contain non-negative integers")
    selected, skipped_final, filtered = [], 0, 0
    for run in _read_runs(ledger_path):
        if (
            run.get("track") != track
            or run.get("split") != split
            or run.get("seed") not in seed_set
            or (route_entry_set is not None and int(run.get("route_entry_id", -1)) not in route_entry_set)
        ):
            filtered += 1
            continue
        finalized = _result_is_current_for_sidecar(
            results_dir / run["run_id"] / "contract_result.json",
            sidecar_path,
            expected_input_hashes,
            expected_run=run,
            workload="core",
        )
        if finalized and not retry_finalized:
            skipped_final += 1
            continue
        selected.append(run)
        if max_runs is not None and len(selected) >= max_runs:
            break
    return selected, {
        "selected": len(selected),
        "skipped_finalized": skipped_final,
        "filtered": filtered,
    }


def select_fdq_runs(
    ledger_path: Path,
    results_dir: Path,
    *,
    seeds: Iterable[int],
    expected_input_hashes: dict[str, Any] | None = None,
    retry_finalized: bool = False,
    max_runs: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select recoverable FDQ jobs from its independent expected ledger."""

    seed_set = {int(value) for value in seeds}
    if not seed_set or any(value < 0 for value in seed_set):
        raise ValueError("seeds must contain non-negative integers")
    selected, skipped_final, filtered = [], 0, 0
    for run in _read_runs(ledger_path):
        if run.get("seed") not in seed_set:
            filtered += 1
            continue
        ability = run.get("ability")
        if ability not in _canonical_fdq_route_bindings():
            raise ValueError(f"FDQ run {run.get('run_id')} has unsupported ability {ability!r}")
        finalized = _result_is_current_for_sidecar(
            results_dir / run["run_id"] / "fdq_result.json",
            None,
            expected_input_hashes,
            expected_run=run,
            workload="fdq",
        )
        if finalized and not retry_finalized:
            skipped_final += 1
            continue
        selected.append(run)
        if max_runs is not None and len(selected) >= max_runs:
            break
    return selected, {"selected": len(selected), "skipped_finalized": skipped_final, "filtered": filtered}


RELEASE_AGENTS = frozenset(['alpamayo1.5', 'impromptu-vla', 'lmdrive', 'onevl', 'qwen-drive-1.0-4b', 'senna', 'simlingo'])


def _agent_paths(agent: str, agent_path: Path | None, agent_config: Path | None) -> tuple[Path, Path]:
    if agent_path is not None:
        selected_agent = Path(agent_path)
    elif agent in RELEASE_AGENTS:
        selected_agent = MODEL_ZOO_ROOT / agent / "agent.py"
    elif agent == "autovla":
        selected_agent = DEFAULT_AGENT
    elif agent == "oracle":
        selected_agent = DEFAULT_ORACLE_AGENT
    else:
        raise ValueError(f"unknown built-in agent {agent!r}; provide --agent-path")
    if agent_config is not None:
        selected_config = Path(agent_config)
    elif agent in RELEASE_AGENTS:
        selected_config = MODEL_ZOO_ROOT / agent / ("adapter_config.py" if agent == "lmdrive" else "adapter_config.json")
    elif agent == "autovla":
        selected_config = DEFAULT_CHECKPOINT
    elif agent == "oracle":
        # The privileged Language2Drive oracle reads its private program from
        # the evaluator capability channel.  The evaluator API still expects
        # a concrete configuration argument, even though the oracle does not
        # use an ML checkpoint.
        selected_config = Path(__file__).resolve().parents[2] / "config/defaults.json"
    elif agent_path is not None:
        raise ValueError("custom agents require --agent-config")
    else:
        raise ValueError("custom agents require --agent-config")
    for label, path in (("agent", selected_agent), ("agent config", selected_config)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    return selected_agent.resolve(), selected_config.resolve()


def _deadline_artifact_paths(project_root: Path) -> tuple[Path, Path, Path]:
    build_dir = Path(project_root) / "build/suite"
    return (
        build_dir / "sidecar.private.json",
        build_dir / DEADLINE_SIDECAR_NAME,
        build_dir / DEADLINE_OVERLAY_NAME,
    )


def _derive_deadline_sidecar(
    base_sidecar: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Create an exact base-sidecar copy with deadline fields replaced."""

    by_arm: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("deadline freeze entries must be objects")
        key = (str(entry.get("pcu_id", "")), str(entry.get("arm_id", "")))
        deadline = entry.get("deadline_seconds")
        previous = entry.get("previous_deadline_seconds")
        if not all(key) or key in by_arm:
            raise ValueError(f"duplicate or invalid deadline arm identity: {key!r}")
        if (
            not isinstance(deadline, (int, float))
            or isinstance(deadline, bool)
            or not math.isfinite(float(deadline))
            or float(deadline) <= 0
            or not isinstance(previous, (int, float))
            or isinstance(previous, bool)
            or not math.isfinite(float(previous))
            or float(previous) <= 0
        ):
            raise ValueError(f"invalid deadline values for {key[0]}/{key[1]}")
        by_arm[key] = entry
    if len(by_arm) != 256:
        raise ValueError(f"deadline freeze must contain exactly 256 arms, got {len(by_arm)}")

    derived = copy.deepcopy(base_sidecar)
    observed_arms: set[tuple[str, str]] = set()
    for pcu in derived.get("pcus", []):
        pcu_id = str(pcu.get("id", ""))
        for arm in pcu.get("arms", []):
            key = (pcu_id, str(arm.get("id", "")))
            if key not in by_arm:
                raise ValueError(f"deadline freeze has no entry for {key[0]}/{key[1]}")
            expected_previous = float(by_arm[key]["previous_deadline_seconds"])
            if not math.isclose(float(arm.get("deadline_seconds")), expected_previous, abs_tol=1e-9):
                raise ValueError(f"base arm deadline changed for {key[0]}/{key[1]}")
            arm["deadline_seconds"] = int(by_arm[key]["deadline_seconds"])
            observed_arms.add(key)
    if observed_arms != set(by_arm):
        raise ValueError("deadline freeze arm identities do not exactly match the base sidecar")

    routes = derived.get("routes", [])
    if not isinstance(routes, list) or len(routes) != 768:
        raise ValueError(f"base sidecar must contain exactly 768 routes, got {len(routes) if isinstance(routes, list) else 0}")
    for route in routes:
        key = (str(route.get("pcu_id", "")), str(route.get("arm_id", "")))
        if key not in by_arm:
            raise ValueError(f"route has no deadline freeze entry: {key!r}")
        expected_previous = float(by_arm[key]["previous_deadline_seconds"])
        if not math.isclose(float(route.get("deadline_seconds")), expected_previous, abs_tol=1e-9):
            raise ValueError(f"base route deadline changed for {key[0]}/{key[1]}")
        route["deadline_seconds"] = int(by_arm[key]["deadline_seconds"])
    return derived


def deadline_freeze_status(
    project_root: Path, *, verify_witnesses: bool = True
) -> dict[str, Any]:
    """Validate the committed deadline overlay and full derived sidecar."""

    root = Path(project_root)
    base_path, derived_path, overlay_path = _deadline_artifact_paths(root)
    present = {"derived_sidecar": derived_path.is_file(), "overlay": overlay_path.is_file()}
    if not any(present.values()):
        return {"status": "not_frozen", "valid": False, "errors": [], "artifacts": present}
    errors: list[str] = []
    if not all(present.values()):
        errors.append("deadline freeze commit is incomplete")
        return {"status": "invalid", "valid": False, "errors": errors, "artifacts": present}
    try:
        base = load_json(base_path)
        derived = load_json(derived_path)
        overlay = load_json(overlay_path)
        if overlay.get("schema_version") != "language2drive.deadline_freeze.v1":
            errors.append("unsupported deadline freeze schema")
        if overlay.get("status") != "frozen" or overlay.get("formula") != DEADLINE_FORMULA:
            errors.append("deadline freeze status/formula is invalid")
        source = overlay.get("source", {})
        expected_sources = {
            "base_sidecar_sha256": sha256_file(base_path),
            "oracle_ledger_sha256": sha256_file(root / "build/suite/oracle_expected_runs.json"),
            "manifest_sha256": sha256_file(root / "build/suite/manifest.json"),
            "provenance_sha256": sha256_file(root / "build/suite/provenance.json"),
        }
        for key, expected in expected_sources.items():
            if source.get(key) != expected:
                errors.append(f"deadline freeze source hash mismatch: {key}")
        if source.get("expected_input_hashes") != _deadline_expected_input_hashes(root):
            errors.append("deadline freeze oracle execution bundle is no longer current")
        entries = overlay.get("entries", [])
        reconstructed = _derive_deadline_sidecar(base, entries)
        if reconstructed != derived:
            errors.append("derived sidecar differs from exact deadline-only reconstruction")
        derived_hash = sha256_file(derived_path)
        if overlay.get("derived_sidecar", {}).get("sha256") != derived_hash:
            errors.append("derived sidecar hash does not match deadline freeze overlay")
        if overlay.get("required_runs") != 768 or overlay.get("valid_runs") != 768:
            errors.append("deadline freeze does not bind all 768 witnesses")
        if overlay.get("expected_arms") != 256 or overlay.get("calibrated_arms") != 256:
            errors.append("deadline freeze does not bind all 256 arms")
        witnesses = overlay.get("witnesses", [])
        if not isinstance(witnesses, list) or len(witnesses) != 768:
            errors.append("deadline freeze witness provenance must contain 768 rows")
        elif verify_witnesses:
            results_dir = Path(str(source.get("results_dir", "")))
            expected_bundle = source.get("expected_input_hashes")
            seen_runs: set[str] = set()
            for witness in witnesses:
                run_id = str(witness.get("run_id", ""))
                if not run_id or run_id in seen_runs:
                    errors.append(f"invalid or duplicate frozen witness run_id: {run_id!r}")
                    continue
                seen_runs.add(run_id)
                run_dir = results_dir / run_id
                live_paths = {
                    "result": run_dir / "contract_result.json",
                    "trace": run_dir / "trace.jsonl",
                    "runner_command": run_dir / "runner_command.json",
                    "run_meta": run_dir / "run_meta.json",
                }
                for name, path in live_paths.items():
                    expected_hash = witness.get("sha256", {}).get(name)
                    if not path.is_file() or expected_hash != sha256_file(path):
                        errors.append(f"{run_id}: frozen witness hash mismatch for {name}")
                try:
                    command = load_json(live_paths["runner_command"])
                    if command.get("input_hashes") != expected_bundle:
                        errors.append(f"{run_id}: frozen witness execution bundle mismatch")
                    run_meta = load_json(live_paths["run_meta"])
                    provenance = run_meta.get("private_capture_provenance", {})
                    if provenance.get("private_sidecar_sha256") != source.get("base_sidecar_sha256"):
                        errors.append(f"{run_id}: frozen witness base sidecar mismatch")
                except (OSError, ValueError, TypeError):
                    errors.append(f"{run_id}: frozen witness metadata is unreadable")
        expected_freeze_id = stable_hash({
            "formula": overlay.get("formula"),
            "source": source,
            "entries": entries,
            "witnesses": witnesses,
            "derived_sidecar_sha256": derived_hash,
            "new_deadline_rescore": overlay.get("new_deadline_rescore"),
        })
        if overlay.get("freeze_id") != expected_freeze_id:
            errors.append("deadline freeze_id is not content-derived")
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
        overlay = {}
    return {
        "status": "frozen" if not errors else "invalid",
        "valid": not errors,
        "errors": errors,
        "artifacts": present,
        "freeze_id": overlay.get("freeze_id"),
        "overlay": str(overlay_path),
        "derived_sidecar": str(derived_path),
    }


def active_core_sidecar_path(project_root: Path) -> Path:
    """Resolve the base or hash-validated deadline-frozen core sidecar."""

    base_path, derived_path, _ = _deadline_artifact_paths(project_root)
    status = deadline_freeze_status(project_root, verify_witnesses=False)
    if status["status"] == "not_frozen":
        return base_path
    if not status["valid"]:
        raise ValueError("invalid deadline freeze: " + "; ".join(status["errors"]))
    return derived_path


def _current_runtime_asset_authority() -> dict[str, Any]:
    manifest = validate_carla_runtime_manifest(
        CARLA_RUNTIME_MANIFEST, CARLA_RUNTIME_ROOT,
    )
    provenance = validate_carla_runtime_provenance(
        CARLA_RUNTIME_PROVENANCE,
        manifest_path=CARLA_RUNTIME_MANIFEST,
        carla_root=CARLA_RUNTIME_ROOT,
        generator_path=Path(carla_runtime_assets_module.__file__),
    )
    try:
        account = pwd.getpwnam(CARLA_SERVER_ACCOUNT)
        server_gids = set(os.getgrouplist(account.pw_name, account.pw_gid))
    except (KeyError, OSError) as error:
        raise ValueError("dedicated CARLA server account is unavailable") from error
    permissions = validate_server_readonly_permissions(
        CARLA_RUNTIME_MANIFEST,
        CARLA_RUNTIME_ROOT,
        logical_root=CARLA_RUNTIME_ROOT,
        server_uid=account.pw_uid,
        server_gids=server_gids,
    )
    return {
        **manifest,
        **provenance,
        "server_readonly_permissions_sha256": permissions["entries_sha256"],
        "server_readonly_permission_entry_count": permissions["entry_count"],
        "server_uid": permissions["server_uid"],
        "server_gids": permissions["server_gids"],
    }


def _current_session_runtime_asset_authority() -> dict[str, Any]:
    """Load the freshly frozen CARLA declaration without rehashing 21.9 GB.

    The experiment launcher performs one strict ``freeze-runtime-assets`` pass
    before starting the seven release servers.  Session workers then bind that
    immutable manifest, current boot provenance, Shipping binary, and account
    identity.  They do not independently reread all 34,934 CARLA files.
    """

    try:
        manifest = load_json(CARLA_RUNTIME_MANIFEST)
        provenance = load_json(CARLA_RUNTIME_PROVENANCE)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"session CARLA authority is unreadable: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "language2drive.carla_runtime_asset_manifest.v1"
        or not isinstance(provenance, dict)
        or provenance.get("schema_version")
        != "language2drive.carla_runtime_asset_manifest_provenance.v1"
    ):
        raise ValueError("session CARLA manifest/provenance schema differs")
    resolved_root = str(CARLA_RUNTIME_ROOT.resolve(strict=True))
    if (
        manifest.get("resolved_carla_root") != resolved_root
        or provenance.get("resolved_carla_root") != resolved_root
        or provenance.get("manifest_sha256")
        != _current_file_sha256(CARLA_RUNTIME_MANIFEST)
        or provenance.get("generator_sha256")
        != _current_file_sha256(Path(carla_runtime_assets_module.__file__))
    ):
        raise ValueError("session CARLA manifest/provenance identity differs")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    if provenance.get("boot_id") != boot_id:
        raise ValueError("session CARLA provenance belongs to another boot")
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    freeze_ticks = provenance.get("freeze_monotonic_ticks")
    if (
        provenance.get("clock_ticks_per_second") != clock_ticks
        or type(freeze_ticks) is not int
        or freeze_ticks <= 0
    ):
        raise ValueError("session CARLA provenance clock differs")
    witness = provenance.get("available_maps_witness")
    if not isinstance(witness, dict):
        raise ValueError("session CARLA available-map witness is missing")
    maps = witness.get("available_maps")
    if (
        not isinstance(maps, list)
        or not maps
        or not all(isinstance(item, str) and item for item in maps)
        or maps != sorted(set(maps))
    ):
        raise ValueError("session CARLA available maps are invalid")
    for key in ("client_version", "server_version"):
        if not isinstance(witness.get(key), str) or not witness[key]:
            raise ValueError(f"session CARLA {key} is invalid")
    try:
        account = pwd.getpwnam(CARLA_SERVER_ACCOUNT)
        server_gids = sorted(set(os.getgrouplist(account.pw_name, account.pw_gid)))
    except (KeyError, OSError) as error:
        raise ValueError("dedicated CARLA server account is unavailable") from error
    file_count = manifest.get("file_count")
    directory_count = manifest.get("directory_count")
    total_size = manifest.get("total_size_bytes")
    if any(
        type(value) is not int or value < 0
        for value in (file_count, directory_count, total_size)
    ):
        raise ValueError("session CARLA manifest counts are invalid")
    shipping = CARLA_RUNTIME_ROOT / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    permission_declaration = {
        "schema_version": "language2drive.session_permission_declaration.v1",
        "manifest_inventory_sha256": manifest.get("inventory_sha256"),
        "server_uid": account.pw_uid,
        "server_gids": server_gids,
    }
    return {
        "resolved_carla_root": resolved_root,
        "manifest_sha256": provenance["manifest_sha256"],
        "inventory_sha256": manifest.get("inventory_sha256"),
        "files_sha256": manifest.get("files_sha256"),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "shipping_binary_sha256": _current_file_sha256(shipping),
        "provenance_sha256": _current_file_sha256(CARLA_RUNTIME_PROVENANCE),
        "available_maps_sha256": stable_hash(maps),
        "client_version": witness["client_version"],
        "server_version": witness["server_version"],
        "available_map_count": len(maps),
        "boot_id": boot_id,
        "freeze_monotonic_ticks": freeze_ticks,
        "clock_ticks_per_second": clock_ticks,
        "server_readonly_permissions_sha256": stable_hash(permission_declaration),
        "server_readonly_permission_entry_count": file_count + directory_count,
        "server_uid": account.pw_uid,
        "server_gids": server_gids,
    }


def expected_input_hashes(
    *,
    sidecar: Path | None,
    route_xml: Path,
    agent_path: Path,
    agent_config: Path,
    benchmark_track: str,
    evaluator_track: str,
    workload: str,
    dependency_profile: str | None = None,
    wall_timeout_policy: dict[str, Any] | None = None,
    runtime_asset_authority: dict[str, Any] | None = None,
    live_process_authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact release input bundle used for result recovery."""

    paths = {
        "route_xml_sha256": Path(route_xml),
        "agent_sha256": Path(agent_path),
        "agent_config_sha256": Path(agent_config),
        "evaluator_sha256": DEFAULT_EVALUATOR,
    }
    if sidecar is not None:
        paths["private_sidecar_sha256"] = Path(sidecar)
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"execution bundle input does not exist ({label}): {path}")
    resolved_profile = dependency_profile or "base-runtime.v1"
    dependency_hashes = _execution_dependency_hashes(resolved_profile)
    runtime_assets = runtime_asset_authority or _current_runtime_asset_authority()
    expected_runtime_keys = {
        "resolved_carla_root", "manifest_sha256", "inventory_sha256",
        "files_sha256", "file_count", "total_size_bytes",
        "shipping_binary_sha256",
        "provenance_sha256", "available_maps_sha256", "client_version",
        "server_version", "available_map_count", "boot_id",
        "freeze_monotonic_ticks", "clock_ticks_per_second",
        "server_readonly_permissions_sha256",
        "server_readonly_permission_entry_count", "server_uid", "server_gids",
    }
    if not isinstance(runtime_assets, dict) or set(runtime_assets) != expected_runtime_keys:
        raise ValueError("CARLA runtime asset authority shape differs")
    if runtime_assets["resolved_carla_root"] != str(CARLA_RUNTIME_ROOT.resolve(strict=True)):
        raise ValueError("CARLA runtime asset authority root differs")
    bundle = {
        **{key: _current_file_sha256(path) for key, path in paths.items()},
        "schema_version": "language2drive.execution_bundle.v2",
        "workload": workload,
        "benchmark_track": benchmark_track,
        "evaluator_track": evaluator_track,
        "dependency_profile": resolved_profile,
        "dependency_hashes": dependency_hashes,
        "dependency_manifest_sha256": stable_hash(dependency_hashes),
        "execution_environment_sha256": stable_hash(
            evaluator_environment(
                _EVALUATOR_ENVIRONMENT_TEMPLATE_ROOT, prepare=False,
            )
        ),
        "carla_runtime_root": runtime_assets["resolved_carla_root"],
        "carla_runtime_asset_manifest_sha256": runtime_assets["manifest_sha256"],
        "carla_runtime_asset_inventory_sha256": runtime_assets["inventory_sha256"],
        "carla_runtime_asset_files_sha256": runtime_assets["files_sha256"],
        "carla_runtime_asset_file_count": runtime_assets["file_count"],
        "carla_runtime_asset_total_size_bytes": runtime_assets["total_size_bytes"],
        "carla_runtime_shipping_binary_sha256": runtime_assets[
            "shipping_binary_sha256"
        ],
        "carla_runtime_asset_provenance_sha256": runtime_assets["provenance_sha256"],
        "carla_runtime_available_maps_sha256": runtime_assets["available_maps_sha256"],
        "carla_runtime_available_map_count": runtime_assets["available_map_count"],
        "carla_runtime_client_version": runtime_assets["client_version"],
        "carla_runtime_server_version": runtime_assets["server_version"],
        "carla_runtime_boot_id": runtime_assets["boot_id"],
        "carla_runtime_freeze_monotonic_ticks": runtime_assets[
            "freeze_monotonic_ticks"
        ],
        "carla_runtime_clock_ticks_per_second": runtime_assets[
            "clock_ticks_per_second"
        ],
        "carla_runtime_asset_validator_sha256": _current_file_sha256(
            Path(carla_runtime_assets_module.__file__)
        ),
        "carla_runtime_server_readonly_permissions_sha256": runtime_assets[
            "server_readonly_permissions_sha256"
        ],
        "carla_runtime_server_readonly_permission_entry_count": runtime_assets[
            "server_readonly_permission_entry_count"
        ],
        "carla_runtime_server_uid": runtime_assets["server_uid"],
        "carla_runtime_server_gids": runtime_assets["server_gids"],
    }
    if wall_timeout_policy is not None:
        bundle["wall_timeout_policy_sha256"] = stable_hash(wall_timeout_policy)
    bundle["storage_policy_sha256"] = stable_hash(_storage_policy())
    if live_process_authority is not None:
        if (
            not isinstance(live_process_authority, dict)
            or set(live_process_authority) != _LISTENER_RECEIPT_KEYS
        ):
            raise ValueError("CARLA live process authority shape differs")
        if live_process_authority["boot_id"] != runtime_assets["boot_id"]:
            raise ValueError("CARLA live process belongs to another boot")
        if (
            live_process_authority["clock_ticks_per_second"]
            != runtime_assets["clock_ticks_per_second"]
        ):
            raise ValueError("CARLA live process clock frequency differs")
        if (
            live_process_authority["starttime_ticks"]
            <= runtime_assets["freeze_monotonic_ticks"]
        ):
            raise ValueError("CARLA live process predates the frozen runtime assets")
        bundle["carla_server_process_receipt"] = live_process_authority
        bundle["carla_server_process_receipt_sha256"] = stable_hash(
            live_process_authority
        )
    return bundle


def resolve_plan_inputs(
    *,
    project_root: Path,
    build_dir: Path,
    track: str,
    agent: str,
    agent_path: Path | None = None,
    agent_config: Path | None = None,
    evaluator_track: str | None = None,
    route_xml: Path | None = None,
    sidecar: Path | None = None,
    workload: str = "core",
    host: str = "127.0.0.1",
    port: int = 2000,
    gpu_rank: int = 0,
) -> dict[str, Any]:
    """Resolve once so selection and plan construction share one bundle."""

    root = Path(project_root)
    build = Path(build_dir)
    selected_agent, selected_config = _agent_paths(agent, agent_path, agent_config)
    actual_evaluator_track = evaluator_track or ("native" if track == "oracle" else track)
    if sidecar is not None:
        actual_sidecar = Path(sidecar)
    elif track == "oracle" and build.resolve() == (root / "build/suite").resolve():
        # Calibration witnesses forever bind the immutable base suite.  The
        # derived sidecar is for post-freeze systems being evaluated.
        actual_sidecar = build / "sidecar.private.json"
    elif build.resolve() == (root / "build/suite").resolve():
        actual_sidecar = active_core_sidecar_path(root)
    else:
        actual_sidecar = build / "sidecar.private.json"
    actual_route_xml = (
        Path(route_xml)
        if route_xml is not None
        else build / f"routes_{actual_evaluator_track}.xml"
    )
    wall_timeout_policy = _wall_timeout_policy(agent=agent, workload=workload)
    storage_policy = _storage_policy()
    dependency_profile = (
        "autovla-runtime.v1"
        if agent == "autovla" and agent_path is None
        else "session-runtime.v1"
    )
    runtime_assets = (
        _current_session_runtime_asset_authority()
        if dependency_profile == "session-runtime.v1"
        else _current_runtime_asset_authority()
    )
    live_process = _attest_carla_listener_process(
        host, port,
        gpu_rank=gpu_rank,
        expected_sha256=runtime_assets["shipping_binary_sha256"],
    )
    bundle = expected_input_hashes(
        sidecar=actual_sidecar,
        route_xml=actual_route_xml,
        agent_path=selected_agent,
        agent_config=selected_config,
        benchmark_track=track,
        evaluator_track=actual_evaluator_track,
        workload=workload,
        dependency_profile=dependency_profile,
        wall_timeout_policy=wall_timeout_policy,
        runtime_asset_authority=runtime_assets,
        live_process_authority=live_process,
    )
    return {
        "agent_path": selected_agent,
        "agent_config": selected_config,
        "evaluator_track": actual_evaluator_track,
        "route_xml": actual_route_xml,
        "sidecar": actual_sidecar,
        "input_hashes": bundle,
        "wall_timeout_policy": wall_timeout_policy,
        "storage_policy": storage_policy,
        "runtime_asset_authority": runtime_assets,
        "live_process_authority": live_process,
    }


def resolve_fdq_inputs(
    *,
    agent: str,
    agent_path: Path | None = None,
    agent_config: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 2000,
    gpu_rank: int = 0,
) -> dict[str, Any]:
    """Resolve the sidecar-free FDQ execution bundle once."""

    selected_agent, selected_config = _agent_paths(agent, agent_path, agent_config)
    wall_timeout_policy = _wall_timeout_policy(agent=agent, workload="fdq")
    storage_policy = _storage_policy()
    dependency_profile = (
        "autovla-runtime.v1"
        if agent == "autovla" and agent_path is None
        else "session-runtime.v1"
    )
    runtime_assets = (
        _current_session_runtime_asset_authority()
        if dependency_profile == "session-runtime.v1"
        else _current_runtime_asset_authority()
    )
    live_process = _attest_carla_listener_process(
        host, port,
        gpu_rank=gpu_rank,
        expected_sha256=runtime_assets["shipping_binary_sha256"],
    )
    bundle = expected_input_hashes(
        sidecar=None,
        route_xml=DEFAULT_FDQ_ROUTES,
        agent_path=selected_agent,
        agent_config=selected_config,
        benchmark_track="fdq",
        evaluator_track="native",
        workload="fdq",
        dependency_profile=dependency_profile,
        wall_timeout_policy=wall_timeout_policy,
        runtime_asset_authority=runtime_assets,
        live_process_authority=live_process,
    )
    return {
        "agent_path": selected_agent,
        "agent_config": selected_config,
        "input_hashes": bundle,
        "wall_timeout_policy": wall_timeout_policy,
        "storage_policy": storage_policy,
        "runtime_asset_authority": runtime_assets,
        "live_process_authority": live_process,
    }


def _runtime_bundle_fields(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "carla_runtime_root": authority["resolved_carla_root"],
        "carla_runtime_asset_manifest_sha256": authority["manifest_sha256"],
        "carla_runtime_asset_inventory_sha256": authority["inventory_sha256"],
        "carla_runtime_asset_files_sha256": authority["files_sha256"],
        "carla_runtime_asset_file_count": authority["file_count"],
        "carla_runtime_asset_total_size_bytes": authority["total_size_bytes"],
        "carla_runtime_shipping_binary_sha256": authority["shipping_binary_sha256"],
        "carla_runtime_asset_provenance_sha256": authority["provenance_sha256"],
        "carla_runtime_available_maps_sha256": authority["available_maps_sha256"],
        "carla_runtime_available_map_count": authority["available_map_count"],
        "carla_runtime_client_version": authority["client_version"],
        "carla_runtime_server_version": authority["server_version"],
        "carla_runtime_boot_id": authority["boot_id"],
        "carla_runtime_freeze_monotonic_ticks": authority[
            "freeze_monotonic_ticks"
        ],
        "carla_runtime_clock_ticks_per_second": authority[
            "clock_ticks_per_second"
        ],
        "carla_runtime_asset_validator_sha256": _current_file_sha256(
            Path(carla_runtime_assets_module.__file__)
        ),
        "carla_runtime_server_readonly_permissions_sha256": authority[
            "server_readonly_permissions_sha256"
        ],
        "carla_runtime_server_readonly_permission_entry_count": authority[
            "server_readonly_permission_entry_count"
        ],
        "carla_runtime_server_uid": authority["server_uid"],
        "carla_runtime_server_gids": authority["server_gids"],
    }


def _validate_plan_runtime_asset_authority(plan: dict[str, Any]) -> None:
    """Validate only the runtime sub-bundle (full execute uses the stricter gate)."""

    input_hashes = plan.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("plan input_hashes are required for CARLA runtime verification")
    expected_runtime = _runtime_bundle_fields(_current_runtime_asset_authority())
    observed_runtime = {key: input_hashes.get(key) for key in expected_runtime}
    if observed_runtime != expected_runtime:
        raise ValueError("plan CARLA runtime asset authority is stale or incomplete")


def _runtime_asset_authority_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "resolved_carla_root": "carla_runtime_root",
        "manifest_sha256": "carla_runtime_asset_manifest_sha256",
        "inventory_sha256": "carla_runtime_asset_inventory_sha256",
        "files_sha256": "carla_runtime_asset_files_sha256",
        "file_count": "carla_runtime_asset_file_count",
        "total_size_bytes": "carla_runtime_asset_total_size_bytes",
        "shipping_binary_sha256": "carla_runtime_shipping_binary_sha256",
        "provenance_sha256": "carla_runtime_asset_provenance_sha256",
        "available_maps_sha256": "carla_runtime_available_maps_sha256",
        "client_version": "carla_runtime_client_version",
        "server_version": "carla_runtime_server_version",
        "available_map_count": "carla_runtime_available_map_count",
        "boot_id": "carla_runtime_boot_id",
        "freeze_monotonic_ticks": "carla_runtime_freeze_monotonic_ticks",
        "clock_ticks_per_second": "carla_runtime_clock_ticks_per_second",
        "server_readonly_permissions_sha256": (
            "carla_runtime_server_readonly_permissions_sha256"
        ),
        "server_readonly_permission_entry_count": (
            "carla_runtime_server_readonly_permission_entry_count"
        ),
        "server_uid": "carla_runtime_server_uid",
        "server_gids": "carla_runtime_server_gids",
    }
    value = {key: bundle.get(source) for key, source in mapping.items()}
    if any(item is None for item in value.values()):
        raise ValueError("execution plan runtime authority bundle is incomplete")
    return value


def _validate_nonasset_execution_authority(plan: dict[str, Any]) -> None:
    """Rehash code/config/Python before each new evaluator process.

    The 21.6 GiB CARLA content tree is validated once under the batch lock and
    again after the batch.  This per-job gate intentionally reuses only that
    immutable manifest declaration while freshly checking every byte a new
    Python evaluator can import or execute.
    """

    authority = plan.get("execution_authority")
    bundle = plan.get("input_hashes")
    if not isinstance(authority, dict) or not isinstance(bundle, dict):
        raise ValueError("execution plan non-asset authority is missing")
    runtime = _runtime_asset_authority_from_bundle(bundle)
    process = bundle.get("carla_server_process_receipt")
    expected = expected_input_hashes(
        sidecar=(None if authority["sidecar"] is None else Path(authority["sidecar"])),
        route_xml=Path(authority["route_xml"]),
        agent_path=Path(authority["agent_path"]),
        agent_config=Path(authority["agent_config"]),
        benchmark_track=authority["benchmark_track"],
        evaluator_track=authority["evaluator_track"],
        workload=authority["workload"],
        dependency_profile=authority["dependency_profile"],
        wall_timeout_policy=plan.get("wall_timeout_policy"),
        runtime_asset_authority=runtime,
        live_process_authority=process,
    )
    if expected != bundle:
        raise ValueError("execution plan code/Python authority is stale or incomplete")


def _validate_fast_session_nonasset_authority(plan: dict[str, Any]) -> None:
    """Check route-local inputs without rescanning the whole Python prefix.

    ``execute_plan(..., validation_mode="session")`` performs the complete
    dependency and runtime attestation once before it enters the batch lock.
    Between routes, this guard rehashes only the small files that an operator
    can reasonably change during a live experiment.  A restarted worker must
    create/validate a new plan and therefore receives another full scan.
    """

    authority = plan.get("execution_authority")
    bundle = plan.get("input_hashes")
    if not isinstance(authority, dict) or not isinstance(bundle, dict):
        raise ValueError("execution plan fast-session authority is missing")
    paths: dict[str, Path] = {
        "route_xml_sha256": Path(authority["route_xml"]),
        "agent_sha256": Path(authority["agent_path"]),
        "agent_config_sha256": Path(authority["agent_config"]),
        "evaluator_sha256": DEFAULT_EVALUATOR,
    }
    if authority.get("sidecar") is not None:
        paths["private_sidecar_sha256"] = Path(authority["sidecar"])
    for key, path in paths.items():
        if not path.is_file() or _current_file_sha256(path) != bundle.get(key):
            raise ValueError(f"fast-session input changed ({key})")
    if bundle.get("wall_timeout_policy_sha256") != stable_hash(
        plan.get("wall_timeout_policy")
    ):
        raise ValueError("fast-session wall-timeout policy changed")
    if bundle.get("storage_policy_sha256") != stable_hash(_storage_policy()):
        raise ValueError("fast-session storage policy changed")


_PLAN_EXECUTION_AUTHORITY_KEYS = {
    "schema_version", "project_root", "build_dir", "route_xml", "sidecar",
    "results_dir", "agent", "agent_path", "agent_config", "benchmark_track",
    "evaluator_track", "workload", "dependency_profile", "host", "port",
    "traffic_manager_port", "gpu_rank",
}


def _plan_execution_authority(
    *, project_root: Path, build_dir: Path, route_xml: Path,
    sidecar: Path | None, results_dir: Path, agent: str,
    agent_path: Path, agent_config: Path,
    benchmark_track: str, evaluator_track: str, workload: str,
    dependency_profile: str, host: str, port: int,
    traffic_manager_port: int, gpu_rank: int,
) -> dict[str, Any]:
    return {
        "schema_version": "language2drive.plan_execution_authority.v1",
        "project_root": str(Path(project_root).resolve()),
        "build_dir": str(Path(build_dir).resolve()),
        "route_xml": str(Path(route_xml).resolve()),
        "sidecar": None if sidecar is None else str(Path(sidecar).resolve()),
        "results_dir": str(Path(results_dir).resolve()),
        "agent": agent,
        "agent_path": str(Path(agent_path).resolve()),
        "agent_config": str(Path(agent_config).resolve()),
        "benchmark_track": benchmark_track,
        "evaluator_track": evaluator_track,
        "workload": workload,
        "dependency_profile": dependency_profile,
        "host": host,
        "port": int(port),
        "traffic_manager_port": int(traffic_manager_port),
        "gpu_rank": int(gpu_rank),
    }


def _validate_plan_sha256(plan: dict[str, Any]) -> None:
    observed = plan.get("plan_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError("execution plan_sha256 is missing or invalid")
    expected = stable_hash({
        key: value for key, value in plan.items() if key != "plan_sha256"
    })
    if observed != expected:
        raise ValueError("execution plan bytes differ from plan_sha256")


def _require_safe_run_id(value: Any, label: str = "run_id") -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", value)
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _validate_plan_execution_authority(
    plan: dict[str, Any], *, project_root: Path, results_dir: Path,
) -> None:
    authority = plan.get("execution_authority")
    if not isinstance(authority, dict) or set(authority) != _PLAN_EXECUTION_AUTHORITY_KEYS:
        raise ValueError("execution plan authority shape differs")
    if authority["schema_version"] != "language2drive.plan_execution_authority.v1":
        raise ValueError("execution plan authority schema differs")
    for key in (
        "project_root", "build_dir", "route_xml", "results_dir", "agent_path",
        "agent_config",
    ):
        if not isinstance(authority[key], str) or not Path(authority[key]).is_absolute():
            raise ValueError(f"execution plan authority {key} must be an absolute path")
    if authority["sidecar"] is not None and (
        not isinstance(authority["sidecar"], str)
        or not Path(authority["sidecar"]).is_absolute()
    ):
        raise ValueError("execution plan authority sidecar must be null or absolute")
    for key in (
        "agent", "benchmark_track", "evaluator_track", "workload",
        "dependency_profile", "host",
    ):
        if not isinstance(authority[key], str) or not authority[key]:
            raise ValueError(f"execution plan authority {key} must be nonempty")
    for key in ("port", "traffic_manager_port"):
        if type(authority[key]) is not int or not 1 <= authority[key] <= 65535:
            raise ValueError(f"execution plan authority {key} must be a TCP port")
    if type(authority["gpu_rank"]) is not int or authority["gpu_rank"] < 0:
        raise ValueError("execution plan authority gpu_rank must be non-negative")
    if authority["project_root"] != str(Path(project_root).resolve()):
        raise ValueError("execution plan authority project root differs")
    if authority["results_dir"] != str(Path(results_dir).resolve()):
        raise ValueError("execution plan authority results directory differs")
    if authority["benchmark_track"] != plan.get("track"):
        raise ValueError("execution plan authority benchmark track differs")
    if authority["evaluator_track"] != plan.get("evaluator_track"):
        raise ValueError("execution plan authority evaluator track differs")
    if authority["workload"] != plan.get("workload"):
        raise ValueError("execution plan authority workload differs")
    if authority["agent"] != plan.get("agent"):
        raise ValueError("execution plan authority agent differs")
    if authority["agent_path"] != plan.get("agent_path"):
        raise ValueError("execution plan authority agent path differs")
    if authority["agent_config"] != plan.get("agent_config"):
        raise ValueError("execution plan authority agent config differs")
    policy = _wall_timeout_policy(
        agent=str(plan.get("agent", "")), workload=authority["workload"],
    )
    if plan.get("wall_timeout_policy") != policy:
        raise ValueError("execution plan wall timeout policy differs")
    runtime = (
        _current_session_runtime_asset_authority()
        if authority["dependency_profile"] == "session-runtime.v1"
        else _current_runtime_asset_authority()
    )
    expected = expected_input_hashes(
        sidecar=(None if authority["sidecar"] is None else Path(authority["sidecar"])),
        route_xml=Path(authority["route_xml"]),
        agent_path=Path(authority["agent_path"]),
        agent_config=Path(authority["agent_config"]),
        benchmark_track=authority["benchmark_track"],
        evaluator_track=authority["evaluator_track"],
        workload=authority["workload"],
        dependency_profile=authority["dependency_profile"],
        wall_timeout_policy=policy,
        runtime_asset_authority=runtime,
        live_process_authority=_attest_carla_listener_process(
            authority["host"], authority["port"],
            gpu_rank=authority["gpu_rank"],
            expected_sha256=runtime["shipping_binary_sha256"],
        ),
    )
    if plan.get("input_hashes") != expected:
        raise ValueError("execution plan input bundle is stale or incomplete")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("execution plan jobs must be a list")
    if type(plan.get("job_count")) is not int or plan["job_count"] != len(jobs):
        raise ValueError("execution plan job count differs")
    deadline_by_route: dict[int, float] = {}
    route_by_id: dict[int, dict[str, Any]] = {}
    if authority["sidecar"] is not None:
        private_value = load_json(Path(authority["sidecar"]))
        routes = private_value.get("routes", []) if isinstance(private_value, dict) else []
        if not isinstance(routes, list):
            raise ValueError("execution plan sidecar routes must be a list")
        for route in routes:
            if (
                not isinstance(route, dict)
                or type(route.get("route_entry_id")) is not int
                or type(route.get("deadline_seconds")) not in {int, float}
                or isinstance(route.get("deadline_seconds"), bool)
                or not math.isfinite(float(route["deadline_seconds"]))
            ):
                raise ValueError("execution plan sidecar route identity/deadline is invalid")
            route_id = route["route_entry_id"]
            if route_id in route_by_id:
                raise ValueError("execution plan sidecar route identity is duplicated")
            route_by_id[route_id] = route
            deadline_by_route[route_id] = float(route["deadline_seconds"])
    storage_policy = _storage_policy()
    seen_run_ids: set[str] = set()
    seen_fdq_identities: set[tuple[str, int]] = set()
    if authority["workload"] == "fdq":
        expected_fdq_scope = {
            "schema_version": "language2drive.fdq_execution_scope.v1",
            "qualification_claim": False,
            "scope": "selected_recovery_subset_not_full_fdq_qualification",
            "selected_identities": [
                {"ability": job.get("run", {}).get("ability"),
                 "seed": job.get("run", {}).get("seed")}
                for job in jobs
            ],
        }
        if (
            authority["route_xml"] != str(DEFAULT_FDQ_ROUTES.resolve())
            or authority["build_dir"] != str((L2D_ROOT / "build/suite").resolve())
            or plan.get("routes_xml") != str(DEFAULT_FDQ_ROUTES)
            or plan.get("route_bindings") != _canonical_fdq_route_bindings()
            or plan.get("fdq_execution_scope") != expected_fdq_scope
        ):
            raise ValueError("execution FDQ route authority differs")
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or job.get("input_hashes") != expected:
            raise ValueError("execution plan job input bundle differs from current authority")
        expected_job_keys = {
            "run", "command", "command_display", "result_dir", "input_hashes",
            "default_wall_timeout_seconds", "storage_estimate",
        }
        if authority["workload"] != "fdq":
            expected_job_keys.add("deadline_seconds")
        if set(job) != expected_job_keys:
            raise ValueError(f"execution plan job {index} shape differs")
        run = job.get("run")
        if not isinstance(run, dict):
            raise ValueError(f"execution plan job {index} run is invalid")
        if authority["workload"] == "fdq":
            if set(run) != {"ability", "run_id", "seed", "status", "route_binding"}:
                raise ValueError(f"execution FDQ job {index} run shape differs")
            ability = run.get("ability")
            seed = run.get("seed")
            if (
                ability not in _canonical_fdq_route_bindings()
                or type(seed) is not int or seed not in {1, 2, 3}
                or run.get("status") != "expected"
                or run.get("run_id") != f"fdq__{ability}__s{seed}"
                or run.get("route_binding")
                != _canonical_fdq_route_bindings()[ability]
                or (ability, seed) in seen_fdq_identities
            ):
                raise ValueError(f"execution FDQ job {index} identity differs")
            seen_fdq_identities.add((ability, seed))
            expected_command = build_fdq_command(
                run,
                results_dir=Path(authority["results_dir"]),
                agent_path=Path(authority["agent_path"]),
                agent_config=Path(authority["agent_config"]),
                host=authority["host"],
                port=authority["port"],
                traffic_manager_port=authority["traffic_manager_port"],
                gpu_rank=authority["gpu_rank"],
            )
        else:
            base_run_keys = {
                "arm_id", "layer", "paraphrase", "pcu_id", "route_entry_id",
                "run_id", "seed", "split", "status", "track",
            }
            if authority["workload"].startswith("baseline:"):
                expected_run_keys = base_run_keys | {"baseline_id", "evaluator_track"}
            elif authority["workload"] == "safety_priority":
                expected_run_keys = (base_run_keys - {"layer"}) | {
                    "benchmark_track", "hazard_class", "response_mode", "safety_scope",
                }
            else:
                expected_run_keys = base_run_keys
            if set(run) != expected_run_keys:
                raise ValueError(f"execution plan job {index} run shape differs")
            if type(run.get("route_entry_id")) is not int:
                raise ValueError(f"execution plan job {index} route identity is invalid")
            route = route_by_id.get(run["route_entry_id"])
            if route is None:
                raise ValueError(f"execution plan job {index} route identity is absent")
            seed = run.get("seed")
            if type(seed) is not int or seed <= 0:
                raise ValueError(f"execution plan job {index} seed is invalid")
            run_prefix = authority["benchmark_track"]
            expected_core_run: dict[str, Any] = {
                "arm_id": route.get("arm_id"),
                "layer": route.get("layer"),
                "paraphrase": route.get("paraphrase"),
                "pcu_id": route.get("pcu_id"),
                "route_entry_id": route["route_entry_id"],
                "run_id": "",
                "seed": seed,
                "split": route.get("split"),
                "status": "expected",
                "track": authority["benchmark_track"],
            }
            if authority["workload"].startswith("baseline:"):
                baseline_id = authority["workload"].split(":", 1)[1]
                run_prefix = f"baseline__{baseline_id}"
                expected_core_run.update({
                    "baseline_id": baseline_id,
                    "evaluator_track": authority["evaluator_track"],
                })
            elif authority["workload"] == "safety_priority":
                expected_core_run.pop("layer")
                expected_core_run.update({
                    "benchmark_track": "safety_priority",
                    "hazard_class": route.get("hazard_class"),
                    "response_mode": route.get("response_mode"),
                    "safety_scope": route.get("safety_scope"),
                })
            expected_core_run["run_id"] = (
                f"{run_prefix}__{route.get('pcu_id')}__{route.get('arm_id')}__"
                f"p{route.get('paraphrase')}__s{seed}"
            )
            if run != expected_core_run:
                raise ValueError(f"execution plan job {index} route/run identity differs")
            expected_command = build_command(
                run,
                project_root=Path(authority["project_root"]),
                build_dir=Path(authority["build_dir"]),
                results_dir=Path(authority["results_dir"]),
                track=authority["benchmark_track"],
                agent_path=Path(authority["agent_path"]),
                agent_config=Path(authority["agent_config"]),
                host=authority["host"],
                port=authority["port"],
                traffic_manager_port=authority["traffic_manager_port"],
                gpu_rank=authority["gpu_rank"],
                evaluator_track=authority["evaluator_track"],
                route_xml=Path(authority["route_xml"]),
                sidecar=(None if authority["sidecar"] is None else Path(authority["sidecar"])),
            )
        if job.get("command") != expected_command:
            raise ValueError(f"execution plan job {index} command differs")
        if job.get("command_display") != shlex.join(expected_command):
            raise ValueError(f"execution plan job {index} displayed command differs")
        run_id = run.get("run_id")
        try:
            run_id = _require_safe_run_id(run_id)
        except ValueError as error:
            raise ValueError(f"execution plan job {index} run_id is invalid") from error
        if run_id in seen_run_ids:
            raise ValueError(f"execution plan job {index} run_id is duplicated")
        seen_run_ids.add(run_id)
        expected_result_dir = str(Path(authority["results_dir"]) / run_id)
        if job.get("result_dir") != expected_result_dir:
            raise ValueError(f"execution plan job {index} result directory differs")
        if authority["workload"] != "fdq":
            try:
                route_entry_id = int(run["route_entry_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"execution plan job {index} route identity is invalid") from error
            if job.get("deadline_seconds") != deadline_by_route.get(route_entry_id):
                raise ValueError(f"execution plan job {index} deadline differs")
        if job.get("default_wall_timeout_seconds") != _job_wall_timeout(
            job, None, policy,
        ):
            raise ValueError(f"execution plan job {index} wall timeout differs")
        if job.get("storage_estimate") != _job_storage_estimate(job, storage_policy):
            raise ValueError(f"execution plan job {index} storage estimate differs")


def _query_live_carla_runtime_witness(host: str, port: int) -> dict[str, Any]:
    """Read version/map identity from the active endpoint without lifecycle writes."""

    script = r"""
import json
import sys
import carla

client = carla.Client(sys.argv[1], int(sys.argv[2]))
client.set_timeout(15.0)
maps = client.get_available_maps()
if not isinstance(maps, list) or not all(isinstance(item, str) and item for item in maps):
    raise RuntimeError("CARLA get_available_maps returned invalid values")
if len(set(maps)) != len(maps):
    raise RuntimeError("CARLA get_available_maps returned duplicates")
print(json.dumps({
    "client_version": str(client.get_client_version()),
    "server_version": str(client.get_server_version()),
    "available_maps": sorted(maps),
}, sort_keys=True, separators=(",", ":")))
"""
    try:
        with _temporary_evaluator_environment("carla-rpc") as environment:
            completed = subprocess.run(
                [str(CONDA_PREFIX / "bin/python"), "-c", script, host, str(port)],
                cwd=str(EVALUATOR_ROOT),
                env=environment,
                text=True,
                capture_output=True,
                timeout=25,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"CARLA runtime witness query failed: {error}") from error
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()[-1000:]
        raise RuntimeError(
            f"CARLA runtime witness query failed ({completed.returncode}): {diagnostic}"
        )
    try:
        value = _strict_json_loads(completed.stdout, "CARLA runtime witness query")
    except ValueError as error:
        raise RuntimeError("CARLA runtime witness query returned invalid JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "client_version", "server_version", "available_maps",
    }:
        raise RuntimeError("CARLA runtime witness query shape differs")
    maps = value["available_maps"]
    if (
        not isinstance(maps, list)
        or not all(isinstance(item, str) and item for item in maps)
        or maps != sorted(set(maps))
    ):
        raise RuntimeError("CARLA runtime witness maps are not sorted unique strings")
    for key in ("client_version", "server_version"):
        if not isinstance(value[key], str) or not value[key]:
            raise RuntimeError(f"CARLA runtime witness {key} is invalid")
    return {
        **value,
        "available_map_count": len(maps),
        "available_maps_sha256": stable_hash(maps),
    }


def _listener_socket_rows(
    port: int, proc_root: Path = Path("/proc"),
) -> list[dict[str, Any]]:
    if type(port) is not int or not 1 <= port <= 65535:
        raise RuntimeError("CARLA listener port is invalid")
    rows: list[dict[str, Any]] = []
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        try:
            lines = path.read_text(encoding="ascii").splitlines()[1:]
        except (OSError, UnicodeError) as error:
            raise RuntimeError(f"cannot read Linux listener table {path}: {error}") from error
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                raise RuntimeError(f"invalid Linux listener table row in {path}")
            try:
                address, encoded_port = fields[1].rsplit(":", 1)
                local_port = int(encoded_port, 16)
                uid = int(fields[7])
                inode = int(fields[9])
            except (IndexError, ValueError) as error:
                raise RuntimeError(f"invalid Linux listener address in {path}") from error
            if local_port == int(port) and fields[3] == "0A":
                if uid < 0 or inode <= 0 or not address:
                    raise RuntimeError(f"invalid Linux listener identity in {path}")
                rows.append({
                    "family": name,
                    "address": address,
                    "port": local_port,
                    "state": fields[3],
                    "uid": uid,
                    "inode": inode,
                })
    rows.sort(key=lambda row: (row["family"], row["address"], row["inode"]))
    if not rows:
        raise RuntimeError(f"no Linux listening socket owns CARLA port {port}")
    if len({row["inode"] for row in rows}) != len(rows):
        raise RuntimeError("CARLA listener table contains duplicate socket identities")
    if len({row["uid"] for row in rows}) != 1:
        raise RuntimeError("CARLA listener sockets have multiple owning UIDs")
    return rows


_LISTENER_RECEIPT_KEYS = {
    "uid", "gid", "pid", "starttime_ticks", "clock_ticks_per_second",
    "socket_inodes", "executable", "executable_dev", "executable_ino",
    "executable_size", "executable_sha256", "cmdline", "cmdline_sha256",
    "cwd", "cwd_device", "cwd_inode", "root", "root_device", "root_inode",
    "mount_namespace_inode", "boot_id",
    "environment", "environment_sha256",
    "server_home_inventory", "server_runtime_inventories",
    "server_runtime_inventories_sha256",
    "mapped_runtime_files", "mapped_runtime_files_sha256",
    "launch_receipt", "launch_receipt_sha256", "port",
}

_LAUNCH_RECEIPT_KEYS = {
    "schema_version", "boot_id", "prelaunch_monotonic_ticks",
    "clock_ticks_per_second", "server_runtime_root",
    "runtime_asset_manifest_sha256", "runtime_asset_provenance_sha256",
    "launcher_sha256", "shipping_binary_sha256",
}


def _load_carla_server_launch_receipt(
    shipping: Path, *, port: int = 2000, proc_root: Path = Path("/proc"),
) -> tuple[dict[str, Any], str]:
    server_runtime_root = carla_server_runtime_root(port)
    path = server_runtime_root / "launch-receipt.json"
    metadata = path.lstat()
    parent_metadata = path.parent.lstat()
    if (
        path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_uid != 0 or metadata.st_gid != 0
        or path.parent.is_symlink() or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != 0 or parent_metadata.st_gid != 0
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise RuntimeError(
            "CARLA launch receipt is not root-owned immutable regular evidence"
        )
    try:
        value = _strict_json_loads(
            path.read_text(encoding="utf-8"), "CARLA launch receipt",
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"CARLA launch receipt is invalid: {error}") from error
    if not isinstance(value, dict) or set(value) != _LAUNCH_RECEIPT_KEYS:
        raise RuntimeError("CARLA launch receipt shape differs")
    if value["schema_version"] != "language2drive.carla_server_launch_receipt.v1":
        raise RuntimeError("CARLA launch receipt schema differs")
    for key in ("prelaunch_monotonic_ticks", "clock_ticks_per_second"):
        if type(value[key]) is not int or value[key] <= 0:
            raise RuntimeError(f"CARLA launch receipt {key} is invalid")
    if value["server_runtime_root"] != str(server_runtime_root.resolve(strict=True)):
        raise RuntimeError("CARLA launch receipt runtime root differs")
    boot_id = (Path(proc_root) / "sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    if value["boot_id"] != boot_id:
        raise RuntimeError("CARLA launch receipt belongs to another boot")
    expected_hashes = {
        "runtime_asset_manifest_sha256": _current_file_sha256(CARLA_RUNTIME_MANIFEST),
        "runtime_asset_provenance_sha256": _current_file_sha256(CARLA_RUNTIME_PROVENANCE),
        "launcher_sha256": _current_file_sha256(
            L2D_ROOT / "tools/start_external_carla.sh"
        ),
        "shipping_binary_sha256": _current_file_sha256(shipping),
    }
    for key, expected in expected_hashes.items():
        try:
            observed = _require_sha256(value[key], f"CARLA launch receipt {key}")
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        if observed != expected:
            raise RuntimeError(f"CARLA launch receipt {key} differs")
    return value, _current_file_sha256(path)


def _mapped_process_runtime_files(process: Path) -> list[dict[str, Any]]:
    paths: dict[Path, tuple[str, int]] = {}
    for line in (process / "maps").read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw = fields[5]
        if raw.endswith(" (deleted)"):
            # NVIDIA's GL stack creates anonymous executable memfd mappings.
            # They have no persistent filesystem bytes to bind and appear in
            # /proc/PID/maps as /memfd:/.glXXXXXX (deleted) on device 00:01.
            # Ignore only that exact anonymous class; a deleted filesystem
            # mapping remains a hard failure below.
            try:
                anonymous_inode = int(fields[4])
            except ValueError as error:
                raise RuntimeError("CARLA process maps an invalid deleted file") from error
            if (
                fields[3] == "00:01"
                and anonymous_inode > 0
                and re.fullmatch(r"/memfd:/\.gl[A-Za-z0-9._-]* \(deleted\)", raw)
            ):
                continue
            raise RuntimeError("CARLA process maps a deleted runtime file")
        try:
            major_raw, minor_raw = fields[3].split(":", 1)
            int(major_raw, 16)
            int(minor_raw, 16)
            mapped_inode = int(fields[4])
        except (ValueError, OSError) as error:
            raise RuntimeError("CARLA process maps an invalid runtime file") from error
        if mapped_inode <= 0:
            raise RuntimeError("CARLA process maps a runtime file without an inode")
        path = Path(os.path.realpath(raw))
        if path.is_file():
            identity = (fields[3].lower(), mapped_inode)
            if path in paths and paths[path] != identity:
                raise RuntimeError("CARLA process maps one path from multiple inodes")
            paths[path] = identity
    rows = []
    for path in sorted(paths, key=str):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"CARLA mapped runtime file is non-regular: {path}")
        metadata = path.stat()
        mapped_device, mapped_inode = paths[path]
        # Container/overlay procfs may expose the underlying maps device
        # number while stat(2) exposes the namespace device number.  The inode
        # must nevertheless be the exact mapped inode; record both device
        # authorities rather than pretending they are interchangeable.
        if int(metadata.st_ino) != mapped_inode:
            raise RuntimeError(
                f"CARLA mapped runtime inode differs from its current path: {path}"
            )
        rows.append({
            "path": str(path), "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino), "size_bytes": int(metadata.st_size),
            "maps_device": mapped_device,
            "sha256": _current_file_sha256(path),
        })
    return rows


def _server_runtime_directory_identity(
    directory: Path, *, runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Bind one mutable server cache by its directory, not its contents.

    Unreal and the graphics stack legitimately create configuration and shader
    cache files while a route is running.  Those bytes are explicitly
    ephemeral and must not become a false immutable-input claim.  The owning
    directory itself is still an authority boundary: it may not be a symlink,
    be replaced, escape the root-owned runtime parent, or become writable by a
    different account.
    """

    root = Path(directory)
    metadata = root.lstat()
    resolved = root.resolve(strict=True)
    runtime_root = Path(
        CARLA_SERVER_RUNTIME_ROOT if runtime_root is None else runtime_root
    ).resolve(strict=True)
    try:
        resolved.relative_to(runtime_root)
    except ValueError as error:
        raise RuntimeError(
            "CARLA server runtime directory escapes its release root"
        ) from error
    if (
        root.is_symlink()
        or resolved != root.absolute()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_nlink < 2
    ):
        raise RuntimeError("CARLA server runtime directory is not a real directory")
    return {
        "root": str(resolved),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "uid": int(metadata.st_uid),
        "gid": int(metadata.st_gid),
        "mode": int(stat.S_IMODE(metadata.st_mode)),
    }


def _server_home_inventory(
    home: Path, *, runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Compatibility name for the structural HOME directory authority."""

    return _server_runtime_directory_identity(home, runtime_root=runtime_root)


def _listener_static_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Return the process fields that must remain byte-exact for all jobs."""

    return {
        key: value for key, value in receipt.items()
        if key not in {"mapped_runtime_files", "mapped_runtime_files_sha256"}
    }


def _mapped_runtime_rows_compatible(
    current: list[dict[str, Any]], frozen: list[dict[str, Any]],
    expected_manifest_sha256: str,
) -> bool:
    """Allow lazy CARLA mappings without accepting an unfrozen dependency.

    System/Python libraries must already occur byte-exactly in the process
    receipt captured by the plan.  A later mapping may additionally come from
    the frozen CARLA installation manifest (maps and plugins are commonly
    loaded lazily).  No other new path is accepted.
    """

    if not isinstance(current, list) or not isinstance(frozen, list):
        return False
    frozen_by_path = {
        row.get("path"): row for row in frozen if isinstance(row, dict)
    }
    current_by_path = {
        row.get("path"): row for row in current if isinstance(row, dict)
    }
    if (
        len(frozen_by_path) != len(frozen)
        or len(current_by_path) != len(current)
        or any(not isinstance(path, str) or not path for path in frozen_by_path)
        or any(not isinstance(path, str) or not path for path in current_by_path)
        or any(current_by_path.get(path) != row for path, row in frozen_by_path.items())
    ):
        # Every mapping captured when the controlled server was frozen must
        # remain present byte-for-byte.  Only additional CARLA-manifest rows
        # may appear lazily; unloading/replacing an initial native dependency
        # is an authority change, not a lazy-load event.
        return False
    # The common case needs no manifest lookup: every mapped row is exactly
    # the one captured in the plan.  This also keeps bounded synthetic process
    # receipts independent of the host release manifest.
    if current == frozen:
        return True
    try:
        if (
            _require_sha256(expected_manifest_sha256, "CARLA mapped manifest hash")
            != _current_file_sha256(CARLA_RUNTIME_MANIFEST)
        ):
            return False
        manifest = _strict_json_loads(
            CARLA_RUNTIME_MANIFEST.read_text(encoding="utf-8"),
            "CARLA runtime asset manifest",
        )
    except (OSError, UnicodeError, ValueError):
        return False
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return False
    carla_root = CARLA_RUNTIME_ROOT.resolve(strict=True)
    manifest_by_path = {
        str(carla_root / row["path"]): row
        for row in manifest["files"]
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and isinstance(row.get("sha256"), str)
        and type(row.get("size_bytes")) is int
    }
    if len(manifest_by_path) != len(manifest["files"]):
        return False
    seen: set[str] = set()
    for row in current:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            return False
        path = row["path"]
        if path in seen:
            return False
        seen.add(path)
        if frozen_by_path.get(path) == row:
            continue
        manifest_row = manifest_by_path.get(path)
        if (
            manifest_row is None
            or row.get("sha256") != manifest_row.get("sha256")
            or row.get("size_bytes") != manifest_row.get("size_bytes")
        ):
            return False
    return True


def _read_proc_process_snapshot(
    process: Path, expected_inodes: set[int], port: int,
) -> dict[str, Any] | None:
    """Read one process twice-consistently enough for a parent pre/post gate."""

    quick_owned: set[int] = set()
    for descriptor in (process / "fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            # CARLA continuously creates and closes unrelated descriptors.
            # An entry returned by getdents may therefore disappear before
            # readlink.  Ignore only that vanished entry; the listener socket
            # itself is still required below, in both complete snapshots.
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            quick_owned.add(int(target[8:-1]))
    if not expected_inodes.issubset(quick_owned):
        return None

    def snapshot() -> dict[str, Any]:
        owned: set[int] = set()
        for descriptor in sorted((process / "fd").iterdir(), key=lambda path: path.name):
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                # See the quick scan above.  ENOENT is safe to tolerate for
                # an unrelated, already-closed fd because expected_inodes is
                # checked after both full snapshots.  Permission and other
                # I/O failures remain fail-closed.
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                try:
                    owned.add(int(target[8:-1]))
                except ValueError as error:
                    raise RuntimeError("invalid socket fd identity") from error
        raw_stat = (process / "stat").read_text(encoding="ascii")
        close = raw_stat.rfind(")")
        if close < 0:
            raise RuntimeError("invalid process stat row")
        stat_fields = raw_stat[close + 2:].split()
        if len(stat_fields) <= 19:
            raise RuntimeError("process stat row is truncated")
        starttime_ticks = int(stat_fields[19])
        status: dict[str, list[int]] = {}
        for line in (process / "status").read_text(encoding="ascii").splitlines():
            if line.startswith(("Uid:", "Gid:")):
                key, raw_values = line.split(":", 1)
                status[key] = [int(value) for value in raw_values.split()]
        if set(status) != {"Uid", "Gid"} or any(len(values) != 4 for values in status.values()):
            raise RuntimeError("process status UID/GID shape differs")
        if len(set(status["Uid"])) != 1 or len(set(status["Gid"])) != 1:
            raise RuntimeError("process has changing UID/GID credentials")
        executable = (process / "exe").resolve(strict=True)
        executable_stat = executable.stat()
        raw_command = (process / "cmdline").read_bytes()
        command = [
            part.decode("utf-8", "strict")
            for part in raw_command.split(b"\0") if part
        ]
        namespace = (process / "ns/mnt").stat()
        process_cwd = (process / "cwd").resolve(strict=True)
        process_cwd_metadata = process_cwd.stat()
        if not stat.S_ISDIR(process_cwd_metadata.st_mode):
            raise RuntimeError("CARLA process cwd is not a directory")
        process_root = (process / "root").resolve(strict=True)
        process_root_metadata = process_root.stat()
        if not stat.S_ISDIR(process_root_metadata.st_mode):
            raise RuntimeError("CARLA process root is not a directory")
        environment: dict[str, str] = {}
        for item in (process / "environ").read_bytes().split(b"\0"):
            if not item:
                continue
            if b"=" not in item:
                raise RuntimeError("process environment entry has no equals sign")
            raw_key, raw_value = item.split(b"=", 1)
            key = raw_key.decode("utf-8", "strict")
            if not key or key in environment:
                raise RuntimeError("process environment contains an invalid/duplicate key")
            environment[key] = raw_value.decode("utf-8", "strict")
        home_value = environment.get("HOME")
        if not home_value or not Path(home_value).is_absolute():
            raise RuntimeError("CARLA server HOME environment is invalid")
        server_runtime_root = carla_server_runtime_root(port)
        home_inventory = _server_home_inventory(
            Path(home_value), runtime_root=server_runtime_root,
        )
        runtime_directory_keys = (
            "HOME", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
            "CARLA_CACHE_DIR", "UE-SharedDataCachePath",
        )
        runtime_inventories = {
            key: _server_home_inventory(
                Path(environment[key]), runtime_root=server_runtime_root,
            )
            for key in runtime_directory_keys
            if isinstance(environment.get(key), str)
            and Path(environment[key]).is_absolute()
        }
        mapped_runtime_files = _mapped_process_runtime_files(process)
        boot_id = (process.parent / "sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        if not boot_id:
            raise RuntimeError("Linux boot identity is empty")
        return {
            "uid": status["Uid"][0],
            "gid": status["Gid"][0],
            "pid": int(process.name),
            "starttime_ticks": starttime_ticks,
            "clock_ticks_per_second": int(os.sysconf("SC_CLK_TCK")),
            "socket_inodes": sorted(owned & expected_inodes),
            "executable": str(executable),
            "executable_dev": int(executable_stat.st_dev),
            "executable_ino": int(executable_stat.st_ino),
            "executable_size": int(executable_stat.st_size),
            "executable_sha256": sha256_file(executable),
            "cmdline": command,
            "cmdline_sha256": stable_hash(command),
            "cwd": str(process_cwd),
            "cwd_device": int(process_cwd_metadata.st_dev),
            "cwd_inode": int(process_cwd_metadata.st_ino),
            "root": str(process_root),
            "root_device": int(process_root_metadata.st_dev),
            "root_inode": int(process_root_metadata.st_ino),
            "mount_namespace_inode": int(namespace.st_ino),
            "boot_id": boot_id,
            "environment": dict(sorted(environment.items())),
            "environment_sha256": stable_hash(dict(sorted(environment.items()))),
            "server_home_inventory": home_inventory,
            "server_runtime_inventories": runtime_inventories,
            "server_runtime_inventories_sha256": stable_hash(runtime_inventories),
            "mapped_runtime_files": mapped_runtime_files,
            "mapped_runtime_files_sha256": stable_hash(mapped_runtime_files),
            "port": int(port),
        }

    before = snapshot()
    after = snapshot()
    if _listener_static_receipt(before) != _listener_static_receipt(after):
        raise RuntimeError("process identity changed while being attested")
    if set(before["socket_inodes"]) != expected_inodes:
        return None
    return before


def _collect_listener_process_candidates(
    proc_root: Path, expected_inodes: set[int], port: int,
) -> list[dict[str, Any]]:
    candidates = []
    for process in sorted(
        (path for path in Path(proc_root).iterdir() if path.name.isdecimal()),
        key=lambda path: int(path.name),
    ):
        try:
            candidate = _read_proc_process_snapshot(process, expected_inodes, port)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _same_uid_listener_candidates(
    proc_root: Path, uid: int, inodes: set[int], port: int,
) -> list[dict[str, Any]]:
    if Path(proc_root).resolve() != Path("/proc"):
        return _collect_listener_process_candidates(proc_root, inodes, port)
    try:
        account = pwd.getpwuid(uid)
    except KeyError as error:
        raise RuntimeError(f"CARLA listener UID {uid} has no local account") from error
    helper = (
        "import json,sys;"
        f"sys.path.insert(0,{str((L2D_ROOT / 'src').resolve())!r});"
        "from language2drive.runner import _collect_listener_process_candidates;"
        "v=_collect_listener_process_candidates(__import__('pathlib').Path(sys.argv[1]),"
        "set(json.loads(sys.argv[2])),int(sys.argv[3]));"
        "print(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False))"
    )
    try:
        completed = subprocess.run(
            [
                "/usr/sbin/runuser", "-u", account.pw_name, "--",
                str(CONDA_PREFIX / "bin/python"), "-I", "-c", helper,
                str(proc_root), json.dumps(sorted(inodes)), str(port),
            ],
            cwd="/",
            env={"PATH": "/usr/sbin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"same-UID CARLA listener helper failed: {error}") from error
    if completed.returncode != 0 or completed.stderr:
        diagnostic = (completed.stderr or completed.stdout).strip()[-1000:]
        raise RuntimeError(
            f"same-UID CARLA listener helper failed ({completed.returncode}): {diagnostic}"
        )
    try:
        value = _strict_json_loads(completed.stdout, "same-UID CARLA listener helper")
    except ValueError as error:
        raise RuntimeError("same-UID CARLA listener helper returned invalid JSON") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError("same-UID CARLA listener helper shape differs")
    return value


def _attest_carla_listener_process(
    host: str,
    port: int,
    *,
    proc_root: Path = Path("/proc"),
    gpu_rank: int = 0,
    expected_shipping: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind a local listening endpoint to the frozen Shipping executable."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("CARLA runtime process attestation requires a localhost endpoint")
    shipping_argument = (
        Path(expected_shipping).absolute()
        if expected_shipping is not None else
        CARLA_RUNTIME_ROOT / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    )
    shipping = shipping_argument.resolve(strict=True)
    try:
        shipping_sha = _require_sha256(expected_sha256, "expected Shipping binary hash")
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    rows = _listener_socket_rows(port, proc_root)
    uid = rows[0]["uid"]
    inodes = {row["inode"] for row in rows}
    candidates = _same_uid_listener_candidates(proc_root, uid, inodes, port)
    if len(candidates) != 1:
        raise RuntimeError(
            f"CARLA port {port} must have exactly one attestable owner; found {len(candidates)}"
        )
    receipt = candidates[0]
    if type(gpu_rank) is not int or gpu_rank < 0:
        raise RuntimeError("CARLA GPU rank is invalid")
    server_runtime_root = carla_server_runtime_root(port)
    launch_receipt, launch_receipt_sha256 = _load_carla_server_launch_receipt(
        shipping, port=port, proc_root=proc_root,
    )
    receipt = {
        **receipt,
        "launch_receipt": launch_receipt,
        "launch_receipt_sha256": launch_receipt_sha256,
    }
    if set(receipt) != _LISTENER_RECEIPT_KEYS:
        raise RuntimeError("CARLA listener process receipt shape differs")
    for key in (
        "uid", "gid", "pid", "starttime_ticks", "clock_ticks_per_second",
        "executable_dev", "executable_ino", "executable_size",
        "cwd_device", "cwd_inode", "mount_namespace_inode", "port",
        "root_device", "root_inode",
    ):
        if type(receipt[key]) is not int or receipt[key] < 0:
            raise RuntimeError(f"CARLA listener process {key} is invalid")
    if not isinstance(receipt["boot_id"], str) or not receipt["boot_id"]:
        raise RuntimeError("CARLA listener process boot_id is invalid")
    environment = receipt["environment"]
    if (
        not isinstance(environment, dict)
        or not all(isinstance(key, str) and key and isinstance(value, str) for key, value in environment.items())
        or receipt["environment_sha256"] != stable_hash(environment)
    ):
        raise RuntimeError("CARLA listener process environment is invalid")
    try:
        server_account = pwd.getpwnam(CARLA_SERVER_ACCOUNT)
    except KeyError as error:
        raise RuntimeError(
            f"required CARLA server account {CARLA_SERVER_ACCOUNT} is missing"
        ) from error
    if receipt["uid"] != server_account.pw_uid or receipt["gid"] != server_account.pw_gid:
        raise RuntimeError("CARLA listener process account differs from release account")
    required_environment = {
        "PATH": (
            "/usr/local/bin:/usr/local/nvidia/bin:"
            "/usr/local/cuda/bin:/usr/local/sbin:/usr/bin:/sbin:/bin"
        ),
        "HOME": str((server_runtime_root / "home").resolve()),
        "TMPDIR": str((server_runtime_root / "tmp").resolve()),
        "XDG_CACHE_HOME": str((server_runtime_root / "xdg-cache").resolve()),
        "CUDA_CACHE_PATH": str((server_runtime_root / "cuda-cache").resolve()),
        "CARLA_CACHE_DIR": str((server_runtime_root / "carla-cache").resolve()),
        "UE-SharedDataCachePath": str((server_runtime_root / "user-dir").resolve()),
        "USER": server_account.pw_name,
        "LOGNAME": server_account.pw_name,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if environment != required_environment:
        raise RuntimeError("CARLA listener process environment differs from release profile")
    home_inventory = receipt["server_home_inventory"]
    directory_identity_keys = {
        "root", "device", "inode", "uid", "gid", "mode",
    }
    if (
        not isinstance(home_inventory, dict)
        or set(home_inventory) != directory_identity_keys
        or home_inventory["root"] != required_environment["HOME"]
    ):
        raise RuntimeError("CARLA server HOME inventory shape/root differs")
    for key in ("device", "inode", "uid", "gid", "mode"):
        if type(home_inventory[key]) is not int or home_inventory[key] < 0:
            raise RuntimeError(f"CARLA server HOME {key} is invalid")
    runtime_inventories = receipt["server_runtime_inventories"]
    expected_runtime_roots = {
        key: _server_home_inventory(Path(value), runtime_root=server_runtime_root)
        for key, value in required_environment.items()
        if key in {
            "HOME", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
            "CARLA_CACHE_DIR", "UE-SharedDataCachePath",
        }
    }
    if (
        runtime_inventories != expected_runtime_roots
        or receipt["server_runtime_inventories_sha256"]
        != stable_hash(expected_runtime_roots)
    ):
        raise RuntimeError("CARLA server runtime directory inventories differ")
    if set(runtime_inventories) != {
        "HOME", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
        "CARLA_CACHE_DIR", "UE-SharedDataCachePath",
    }:
        raise RuntimeError("CARLA server runtime directory set differs")
    for key, identity in runtime_inventories.items():
        if (
            not isinstance(identity, dict)
            or set(identity) != directory_identity_keys
            or identity["root"] != required_environment[key]
            or identity["uid"] != server_account.pw_uid
            or identity["gid"] != server_account.pw_gid
            or identity["mode"] & 0o077
        ):
            raise RuntimeError(
                f"CARLA server runtime directory authority differs ({key})"
            )
    if receipt["uid"] != uid or receipt["port"] != port:
        raise RuntimeError("CARLA listener process UID/port differs")
    if receipt["socket_inodes"] != sorted(inodes):
        raise RuntimeError("CARLA listener process socket set differs")
    if receipt["executable"] != str(shipping):
        raise RuntimeError("CARLA listening process executable is outside the frozen runtime root")
    if receipt["executable_sha256"] != shipping_sha:
        raise RuntimeError("CARLA listening process Shipping binary hash differs")
    expected_command = [
        str(shipping_argument), "CarlaUE4", "-RenderOffScreen", "-nosound",
        f"-carla-port={int(port)}",
    ]
    if port != 2000 or gpu_rank != 0:
        expected_command.append(f"-graphicsadapter={gpu_rank}")
    expected_command.append("-quality-level=Epic")
    if receipt["cmdline"] != expected_command:
        raise RuntimeError("CARLA listening process command line differs from release profile")
    if receipt["cmdline_sha256"] != stable_hash(expected_command):
        raise RuntimeError("CARLA listening process command hash differs")
    # UE switches its working directory to the Shipping binary directory
    # during bootstrap even when the launcher starts in the install root.
    expected_cwd = shipping.parent.resolve(strict=True)
    expected_cwd_metadata = expected_cwd.stat()
    if (
        receipt["cwd"] != str(expected_cwd)
        or receipt["cwd_device"] != expected_cwd_metadata.st_dev
        or receipt["cwd_inode"] != expected_cwd_metadata.st_ino
    ):
        raise RuntimeError("CARLA listener process working directory differs")
    expected_root = Path("/").resolve(strict=True)
    expected_root_metadata = expected_root.stat()
    if (
        receipt["root"] != str(expected_root)
        or receipt["root_device"] != expected_root_metadata.st_dev
        or receipt["root_inode"] != expected_root_metadata.st_ino
    ):
        raise RuntimeError("CARLA listener process uses another filesystem root")
    namespace_path = (
        Path("/proc/self/ns/mnt")
        if Path(proc_root).resolve() == Path("/proc")
        else Path(proc_root) / "self/ns/mnt"
    )
    if receipt["mount_namespace_inode"] != namespace_path.stat().st_ino:
        raise RuntimeError("CARLA listener process uses another mount namespace")
    if (
        launch_receipt["boot_id"] != receipt["boot_id"]
        or launch_receipt["clock_ticks_per_second"]
        != receipt["clock_ticks_per_second"]
        or receipt["starttime_ticks"] <= launch_receipt["prelaunch_monotonic_ticks"]
    ):
        raise RuntimeError("CARLA listener process predates its clean launch receipt")
    mapped = receipt["mapped_runtime_files"]
    if (
        not isinstance(mapped, list) or not mapped
        or not all(
            isinstance(row, dict)
            and set(row) == {
                "path", "device", "maps_device", "inode", "size_bytes", "sha256",
            }
            and isinstance(row["path"], str) and Path(row["path"]).is_absolute()
            and isinstance(row["maps_device"], str)
            and re.fullmatch(r"[0-9a-f]+:[0-9a-f]+", row["maps_device"])
            and all(
                type(row[key]) is int and row[key] >= 0
                for key in ("device", "inode", "size_bytes")
            )
            and isinstance(row["sha256"], str)
            and len(row["sha256"]) == 64
            and all(character in _LOWER_HEX for character in row["sha256"])
            for row in mapped
        )
        or [row["path"] for row in mapped] != sorted({row["path"] for row in mapped})
        or receipt["mapped_runtime_files_sha256"] != stable_hash(mapped)
    ):
        raise RuntimeError("CARLA mapped runtime file authority is invalid")
    return receipt


def _validate_live_carla_runtime_authority(
    plan: dict[str, Any], host: str, port: int,
) -> dict[str, Any]:
    bundle = plan.get("input_hashes", {})
    authority = plan.get("execution_authority", {})
    gpu_rank = authority.get("gpu_rank", 0) if isinstance(authority, dict) else 0
    before = _attest_carla_listener_process(
        host,
        port,
        gpu_rank=gpu_rank,
        expected_sha256=bundle.get("carla_runtime_shipping_binary_sha256"),
    )
    expected_process = bundle.get("carla_server_process_receipt")
    if (
        not isinstance(expected_process, dict)
        or set(expected_process) != _LISTENER_RECEIPT_KEYS
        or bundle.get("carla_server_process_receipt_sha256")
        != stable_hash(expected_process)
        or _listener_static_receipt(expected_process)
        != _listener_static_receipt(before)
        or not _mapped_runtime_rows_compatible(
            before["mapped_runtime_files"],
            expected_process["mapped_runtime_files"],
            expected_process["launch_receipt"][
                "runtime_asset_manifest_sha256"
            ],
        )
    ):
        raise RuntimeError("active CARLA listener process differs from plan")
    observed = _query_live_carla_runtime_witness(host, port)
    after = _attest_carla_listener_process(
        host,
        port,
        gpu_rank=gpu_rank,
        expected_sha256=bundle.get("carla_runtime_shipping_binary_sha256"),
    )
    if (
        _listener_static_receipt(before) != _listener_static_receipt(after)
        or not _mapped_runtime_rows_compatible(
            after["mapped_runtime_files"],
            expected_process["mapped_runtime_files"],
            expected_process["launch_receipt"][
                "runtime_asset_manifest_sha256"
            ],
        )
    ):
        raise RuntimeError("CARLA listener process changed during RPC attestation")
    expected = {
        "client_version": bundle.get("carla_runtime_client_version"),
        "server_version": bundle.get("carla_runtime_server_version"),
        "available_map_count": bundle.get("carla_runtime_available_map_count"),
        "available_maps_sha256": bundle.get("carla_runtime_available_maps_sha256"),
    }
    actual = {key: observed[key] for key in expected}
    if actual != expected:
        raise RuntimeError("active CARLA endpoint runtime witness differs from plan")
    return {**observed, "listener_process": before}


def _validate_fast_session_runtime_authority(
    plan: dict[str, Any], host: str, port: int,
) -> dict[str, Any]:
    """Cheaply prove that the fully attested batch CARLA process is unchanged.

    The expensive mapped-file and runtime-directory inventories are already
    bound into ``input_hashes`` by the full pre-batch validation.  This guard
    checks the kernel process/socket identity between routes, which detects a
    crash, replacement, port takeover, executable swap, or command change
    without hashing every mapped file again.
    """

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("fast-session CARLA validation requires localhost")
    bundle = plan.get("input_hashes")
    authority = plan.get("execution_authority")
    if not isinstance(bundle, dict) or not isinstance(authority, dict):
        raise RuntimeError("fast-session CARLA authority is missing")
    expected = bundle.get("carla_server_process_receipt")
    if not isinstance(expected, dict) or set(expected) != _LISTENER_RECEIPT_KEYS:
        raise RuntimeError("fast-session CARLA receipt is missing")
    if expected.get("port") != port or authority.get("port") != port:
        raise RuntimeError("fast-session CARLA port differs")

    rows = _listener_socket_rows(port)
    observed_inodes = sorted(row["inode"] for row in rows)
    if (
        observed_inodes != expected.get("socket_inodes")
        or rows[0]["uid"] != expected.get("uid")
    ):
        raise RuntimeError("fast-session CARLA listener socket changed")

    pid = expected.get("pid")
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("fast-session CARLA PID is invalid")
    process = Path("/proc") / str(pid)
    try:
        raw_stat = (process / "stat").read_text(encoding="ascii")
        close = raw_stat.rfind(")")
        fields = raw_stat[close + 2:].split()
        starttime_ticks = int(fields[19])
        command = [
            part.decode("utf-8", "strict")
            for part in (process / "cmdline").read_bytes().split(b"\0")
            if part
        ]
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise RuntimeError(f"fast-session CARLA process cannot be read: {error}") from error
    if (
        starttime_ticks != expected.get("starttime_ticks")
        or command != expected.get("cmdline")
    ):
        raise RuntimeError("fast-session CARLA kernel process identity changed")
    if not carla_server_available(host, port):
        raise RuntimeError("fast-session CARLA endpoint is unavailable")
    return {
        "schema_version": "language2drive.fast_session_runtime_witness.v1",
        "validation_scope": "kernel_process_socket_identity_after_full_batch_attestation",
        "port": port,
        "pid": pid,
        "starttime_ticks": starttime_ticks,
        "socket_inodes": observed_inodes,
        "executable": expected.get("executable"),
        "cmdline_sha256": stable_hash(command),
        "initial_full_receipt_sha256": bundle.get(
            "carla_server_process_receipt_sha256"
        ),
        "checked_at": _utc_now(),
    }


def evaluator_environment(
    runtime_root: Path | None = None, *, prepare: bool = False,
    gpu_rank: int | None = None,
) -> dict[str, str]:
    """Return a closed evaluator environment, independent of the caller shell.

    In particular, an ambient ``PYTHONPATH``/``LD_PRELOAD`` must never change
    which evaluator code is executed.  Source-tree ``__pycache__`` entries are
    also excluded from import resolution by combining a dedicated empty cache
    prefix with ``PYTHONDONTWRITEBYTECODE``.
    """

    if gpu_rank is not None and (type(gpu_rank) is not int or gpu_rank < 0):
        raise ValueError("evaluator gpu_rank must be a non-negative integer")
    carla_root = CARLA_RUNTIME_ROOT
    python_paths = [
        L2D_ROOT,
        L2D_ROOT / "team_code",
        MODEL_ZOO_ROOT,
        MODEL_ZOO_ROOT / "autovla/code",
        # ScenarioRunner imports ``agents.navigation``.  The ``agents``
        # package lives directly below PythonAPI, while ``carla`` itself is
        # below PythonAPI/carla; both roots are required in the closed
        # evaluator environment.
        carla_root / "PythonAPI",
        carla_root / "PythonAPI/carla",
        BENCHMARK_ROOT / "leaderboard",
        BENCHMARK_ROOT / "scenario_runner",
    ]
    pillow_libs = CONDA_PREFIX / "lib/python3.11/site-packages/pillow.libs"
    runtime_root = Path(runtime_root or _EVALUATOR_ENVIRONMENT_TEMPLATE_ROOT)
    if not runtime_root.is_absolute():
        raise ValueError("evaluator runtime root must be absolute")
    pycache_prefix = runtime_root / "pycache-disabled"
    cache_paths = {
        "TMPDIR": runtime_root / "tmp",
        "XDG_CACHE_HOME": runtime_root / "xdg-cache",
        "HF_HOME": runtime_root / "huggingface",
        "TRANSFORMERS_CACHE": runtime_root / "huggingface/transformers",
        "HF_DATASETS_CACHE": runtime_root / "huggingface/datasets",
        "TORCH_HOME": runtime_root / "torch",
        "CUDA_CACHE_PATH": runtime_root / "cuda-cache",
        "CARLA_CACHE_DIR": runtime_root / "carla-cache",
    }
    home = runtime_root / "evaluator-home"
    if prepare:
        if runtime_root.exists():
            metadata = runtime_root.lstat()
            if runtime_root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("evaluator runtime root is not a real directory")
            if any(runtime_root.iterdir()):
                raise ValueError("evaluator runtime root must be empty before an attempt")
        else:
            runtime_root.mkdir(parents=False, mode=0o700)
        directories = {
            home, pycache_prefix, *cache_paths.values(),
        }
        # Nested cache paths require parent-first construction.
        for path in sorted(directories, key=lambda item: (len(item.parts), str(item))):
            path.mkdir(parents=True, exist_ok=False)
        os.chmod(pycache_prefix, 0o555)
    environment = {
        "PATH": (
            f"{CONDA_PREFIX}/bin:/usr/local/bin:"
            "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:"
            "/usr/bin:/sbin:/bin"
        ),
        "HOME": str(home),
        "USER": "root",
        "LOGNAME": "root",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TZ": "Asia/Shanghai",
        "OMP_NUM_THREADS": "12",
        "MKL_NUM_THREADS": "12",
        # The embedded evaluator keeps its data files below
        # ``language2drive/leaderboard``.  WORK_DIR is the evaluator root;
        # project/model paths are resolved independently from L2D_ROOT.
        "WORK_DIR": str(BENCHMARK_ROOT),
        "CARLA_ROOT": str(carla_root),
        "SCENARIO_RUNNER_ROOT": str(BENCHMARK_ROOT / "scenario_runner"),
        "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
        "LD_LIBRARY_PATH": os.pathsep.join(
            [str(pillow_libs), str(CONDA_PREFIX / "lib")]
        ),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPYCACHEPREFIX": str(pycache_prefix),
        "TOKENIZERS_PARALLELISM": "false",
        # All release model assets are local and execution must never depend
        # on hub reachability or populate an untracked cache.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **{key: str(path) for key, path in cache_paths.items()},
    }
    # The evaluator environment is intentionally closed rather than inherited
    # from the worker shell.  Bind the plan's audited physical rank explicitly
    # so concurrent evaluators cannot all fall back to CUDA device zero.
    if gpu_rank is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_rank)
    return environment


def _new_opaque_evaluator_attempt_root() -> Path:
    """Reserve a non-semantic, unpredictable root for one evaluator attempt."""

    parent = _EVALUATOR_ENVIRONMENT_PARENT
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = parent.lstat()
    if (
        parent.is_symlink() or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ValueError("evaluator environment parent is not private real storage")
    for _ in range(16):
        candidate = parent / f"attempt-{secrets.token_hex(24)}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError("cannot allocate a unique opaque evaluator environment")


def _validate_evaluator_environment_non_disclosure(
    environment: dict[str, str], run: dict[str, Any],
) -> None:
    """Ensure environment values do not reveal standardized route identity."""

    exact_substrings = {
        value
        for key in ("run_id", "pcu_id")
        if isinstance((value := run.get(key)), str) and value
    }
    if any(
        token in value
        for token in exact_substrings
        for value in environment.values()
    ):
        raise RuntimeError("evaluator environment discloses route identity")
    arm_id = run.get("arm_id")
    if isinstance(arm_id, str) and arm_id:
        # Arm labels can be short ordinary words (for example ``low``), so a
        # raw substring test falsely classifies unrelated dependencies such
        # as ``pillow.libs`` as an identity leak.  Reject the label only as a
        # complete semantic token: path/underscore/hyphen delimiters still
        # catch ``.../__low__/...`` and a full run-id, while ``pillow`` does
        # not reveal that this run selected the ``low`` arm.
        arm_pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(arm_id)}(?![A-Za-z0-9])"
        )
        if any(arm_pattern.search(value) for value in environment.values()):
            raise RuntimeError("evaluator environment discloses route identity")


@contextmanager
def _temporary_evaluator_environment(label: str) -> Iterator[dict[str, str]]:
    """Yield a fresh throwaway environment for authority-only probe imports."""

    runtime_parent = L2D_ROOT / ".runtime/probe-environments"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{label}-", dir=str(runtime_parent),
    ) as temporary:
        root = Path(temporary)
        yield evaluator_environment(root, prepare=True)


def build_command(
    run: dict[str, Any],
    *,
    project_root: Path,
    build_dir: Path,
    results_dir: Path,
    track: str,
    agent_path: Path,
    agent_config: Path,
    host: str = "127.0.0.1",
    port: int = 2000,
    traffic_manager_port: int = 8000,
    gpu_rank: int = 0,
    evaluator_track: str | None = None,
    route_xml: Path | None = None,
    sidecar: Path | None = None,
) -> list[str]:
    evaluator_track = evaluator_track or ("native" if track == "oracle" else track)
    route_xml = Path(route_xml) if route_xml is not None else build_dir / f"routes_{evaluator_track}.xml"
    sidecar = Path(sidecar) if sidecar is not None else build_dir / "sidecar.private.json"
    for label, path in (("route XML", route_xml), ("private sidecar", sidecar), ("evaluator", DEFAULT_EVALUATOR)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    route_entry_id = int(run["route_entry_id"])
    seed = int(run["seed"])
    run_dir = results_dir / run["run_id"]
    evaluator_record_root = run_dir / "_evaluator_record"
    command = [
        str(CONDA_PREFIX / "bin/python"),
        str(DEFAULT_EVALUATOR),
        "--routes", str(route_xml),
        "--routes-subset", str(route_entry_id),
        "--agent", str(agent_path),
        "--agent-config", str(agent_config),
        "--checkpoint", str(run_dir / "bench2drive_result.json"),
        "--debug-checkpoint", str(run_dir / "bench2drive_live.txt"),
        "--track", "SENSORS",
        "--host", host,
        "--port", str(port),
        "--traffic-manager-port", str(traffic_manager_port),
        "--traffic-manager-seed", str(seed),
        "--gpu-rank", str(gpu_rank),
        "--record_eval_data",
        "--record_dir", str(evaluator_record_root),
        "--record_frame_interval", "1",
        "--l2d-external-carla",
        "--l2d-track", evaluator_track,
        "--l2d-seed", str(seed),
        "--l2d-private-sidecar", str(sidecar),
        "--l2d-route-entry-id", str(route_entry_id),
        "--l2d-v1-root", str(project_root),
    ]
    # Rendering is unnecessary for the dedicated privileged witness and is a
    # substantial throughput cost across 768 rollouts.  The evaluator accepts
    # this flag only for its exact versioned oracle capability on the private
    # Native track, and explicitly restores rendering for every other route.
    # Camera-based AutoVLA, Safety, controls and FDQ must never receive it.
    if track == "oracle":
        command.append("--l2d-no-rendering")
    return command


def build_fdq_command(
    run: dict[str, Any],
    *,
    results_dir: Path,
    agent_path: Path,
    agent_config: Path,
    host: str = "127.0.0.1",
    port: int = 2000,
    traffic_manager_port: int = 8000,
    gpu_rank: int = 0,
) -> list[str]:
    for label, path in (("FDQ routes", DEFAULT_FDQ_ROUTES), ("evaluator", DEFAULT_EVALUATOR)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    ability = str(run["ability"])
    binding = _canonical_fdq_route_bindings()[ability]
    seed = int(run["seed"])
    run_dir = results_dir / run["run_id"]
    evaluator_record_root = run_dir / "_evaluator_record"
    return [
        str(CONDA_PREFIX / "bin/python"),
        str(DEFAULT_EVALUATOR),
        "--routes", str(DEFAULT_FDQ_ROUTES),
        "--routes-subset", binding["route_id"],
        "--agent", str(agent_path),
        "--agent-config", str(agent_config),
        "--checkpoint", str(run_dir / "bench2drive_result.json"),
        "--debug-checkpoint", str(run_dir / "bench2drive_live.txt"),
        "--track", "SENSORS",
        "--host", host,
        "--port", str(port),
        "--traffic-manager-port", str(traffic_manager_port),
        "--traffic-manager-seed", str(seed),
        "--gpu-rank", str(gpu_rank),
        "--record_eval_data",
        "--record_dir", str(evaluator_record_root),
        "--record_frame_interval", "1",
        "--l2d-external-carla",
        "--l2d-track", "native",
        "--l2d-workload", "fdq",
        "--l2d-seed", str(seed),
    ]


def make_plan(
    runs: list[dict[str, Any]],
    *,
    project_root: Path,
    build_dir: Path,
    results_dir: Path,
    track: str,
    agent: str,
    agent_path: Path | None = None,
    agent_config: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 2000,
    traffic_manager_port: int = 8000,
    gpu_rank: int = 0,
    evaluator_track: str | None = None,
    route_xml: Path | None = None,
    sidecar: Path | None = None,
    workload: str = "core",
    resolved_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolved_inputs
    if resolved is None:
        resolved = resolve_plan_inputs(
            project_root=project_root,
            build_dir=build_dir,
            track=track,
            agent=agent,
            agent_path=agent_path,
            agent_config=agent_config,
            evaluator_track=evaluator_track,
            route_xml=route_xml,
            sidecar=sidecar,
            workload=workload,
            host=host,
            port=port,
            gpu_rank=gpu_rank,
        )
    selected_agent = resolved["agent_path"]
    selected_config = resolved["agent_config"]
    actual_evaluator_track = resolved["evaluator_track"]
    actual_sidecar = resolved["sidecar"]
    actual_route_xml = resolved["route_xml"]
    input_hashes = resolved["input_hashes"]
    wall_timeout_policy = resolved["wall_timeout_policy"]
    storage_policy = resolved["storage_policy"]
    deadline_by_route: dict[int, float] = {}
    if actual_sidecar.is_file():
        private_value = load_json(actual_sidecar)
        deadline_by_route = {
            int(route["route_entry_id"]): float(route["deadline_seconds"])
            for route in private_value.get("routes", [])
            if isinstance(route, dict) and isinstance(route.get("deadline_seconds"), (int, float))
        }
    jobs = []
    for run in runs:
        command = build_command(
            run,
            project_root=project_root,
            build_dir=build_dir,
            results_dir=results_dir,
            track=track,
            agent_path=selected_agent,
            agent_config=selected_config,
            host=host,
            port=port,
            traffic_manager_port=traffic_manager_port,
            gpu_rank=gpu_rank,
            evaluator_track=actual_evaluator_track,
            route_xml=actual_route_xml,
            sidecar=actual_sidecar,
        )
        job = {
            "run": run,
            "command": command,
            "command_display": shlex.join(command),
            "result_dir": str(results_dir / run["run_id"]),
            "deadline_seconds": deadline_by_route.get(int(run["route_entry_id"])),
            "input_hashes": input_hashes,
        }
        job["default_wall_timeout_seconds"] = _job_wall_timeout(
            job, None, wall_timeout_policy
        )
        job["storage_estimate"] = _job_storage_estimate(job, storage_policy)
        jobs.append(job)
    plan_storage = _plan_storage_estimate(jobs)
    execution_authority = _plan_execution_authority(
        project_root=project_root,
        build_dir=build_dir,
        route_xml=actual_route_xml,
        sidecar=actual_sidecar,
        results_dir=results_dir,
        agent=agent,
        agent_path=selected_agent,
        agent_config=selected_config,
        benchmark_track=track,
        evaluator_track=actual_evaluator_track,
        workload=workload,
        dependency_profile=input_hashes["dependency_profile"],
        host=host,
        port=port,
        traffic_manager_port=traffic_manager_port,
        gpu_rank=gpu_rank,
    )
    payload = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "track": track,
        "evaluator_track": actual_evaluator_track,
        "workload": workload,
        "agent": agent,
        "agent_path": str(selected_agent),
        "agent_config": str(selected_config),
        "input_hashes": input_hashes,
        "execution_authority": execution_authority,
        "wall_timeout_policy": wall_timeout_policy,
        "storage_policy": storage_policy,
        **plan_storage,
        "carla_server_prerequisite": (
            f"Externally managed CARLA must already be listening on {host}:{port}; "
            "every evaluator job is forced to reuse it and must not spawn CarlaUE4"
        ),
        "carla_lifecycle": {
            "mode": "external_required",
            "evaluator_flag": "--l2d-external-carla",
            "runner_starts_or_stops_server": False,
            "threat_and_storage_note": (
                "The evaluator is explicitly prohibited from spawning an implicit CARLA process. "
                "This avoids lifecycle races and prevents an evaluator-owned server/cache from "
                "writing to the system disk."
            ),
        },
        "standardized_information_policy": (
            "instruction+cameras+vehicle_state; no arm-specific global/dense route"
            if actual_evaluator_track == "standardized"
            else (
                "privileged oracle consumes evaluator-private native route; results use oracle namespace"
                if track == "oracle"
                else "agent-native full route; separately ranked"
            )
        ),
        "private_sidecar_visibility": (
            "Trusted same-process engineering API boundary: the evaluator validates the private "
            "sidecar and forwards only public_context to the agent. This is interface discipline, "
            "not OS/process security isolation."
        ),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    payload["plan_sha256"] = stable_hash({key: value for key, value in payload.items() if key != "plan_sha256"})
    return payload


def make_fdq_plan(
    runs: list[dict[str, Any]],
    *,
    results_dir: Path,
    agent: str,
    agent_path: Path | None = None,
    agent_config: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 2000,
    traffic_manager_port: int = 8000,
    gpu_rank: int = 0,
    resolved_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolved_inputs
    if resolved is None:
        resolved = resolve_fdq_inputs(
            agent=agent, agent_path=agent_path, agent_config=agent_config,
            host=host, port=port, gpu_rank=gpu_rank,
        )
    selected_agent = resolved["agent_path"]
    selected_config = resolved["agent_config"]
    input_hashes = resolved["input_hashes"]
    wall_timeout_policy = resolved["wall_timeout_policy"]
    storage_policy = resolved["storage_policy"]
    jobs = []
    for run in runs:
        command = build_fdq_command(
            run,
            results_dir=results_dir,
            agent_path=selected_agent,
            agent_config=selected_config,
            host=host,
            port=port,
            traffic_manager_port=traffic_manager_port,
            gpu_rank=gpu_rank,
        )
        binding = _canonical_fdq_route_bindings()[str(run["ability"])]
        job = {
            "run": {**run, "route_binding": copy.deepcopy(binding)},
            "command": command,
            "command_display": shlex.join(command),
            "result_dir": str(results_dir / run["run_id"]),
            "input_hashes": input_hashes,
        }
        job["default_wall_timeout_seconds"] = _job_wall_timeout(
            job, None, wall_timeout_policy
        )
        job["storage_estimate"] = _job_storage_estimate(job, storage_policy)
        jobs.append(job)
    plan_storage = _plan_storage_estimate(jobs)
    execution_authority = _plan_execution_authority(
        project_root=L2D_ROOT,
        build_dir=L2D_ROOT / "build/suite",
        route_xml=DEFAULT_FDQ_ROUTES,
        sidecar=None,
        results_dir=results_dir,
        agent=agent,
        agent_path=selected_agent,
        agent_config=selected_config,
        benchmark_track="fdq",
        evaluator_track="native",
        workload="fdq",
        dependency_profile=input_hashes["dependency_profile"],
        host=host,
        port=port,
        traffic_manager_port=traffic_manager_port,
        gpu_rank=gpu_rank,
    )
    payload = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "workload": "fdq",
        "track": "fdq",
        "evaluator_track": "native",
        "agent": agent,
        "agent_path": str(selected_agent),
        "agent_config": str(selected_config),
        "input_hashes": input_hashes,
        "execution_authority": execution_authority,
        "wall_timeout_policy": wall_timeout_policy,
        "storage_policy": storage_policy,
        **plan_storage,
        "carla_server_prerequisite": (
            f"Externally managed CARLA must already be listening on {host}:{port}; "
            "every evaluator job is forced to reuse it and must not spawn CarlaUE4"
        ),
        "carla_lifecycle": {
            "mode": "external_required",
            "evaluator_flag": "--l2d-external-carla",
            "runner_starts_or_stops_server": False,
            "threat_and_storage_note": (
                "The evaluator is explicitly prohibited from spawning an implicit CARLA process. "
                "This avoids lifecycle races and prevents an evaluator-owned server/cache from "
                "writing to the system disk."
            ),
        },
        "routes_xml": str(DEFAULT_FDQ_ROUTES),
        "route_bindings": _canonical_fdq_route_bindings(),
        "fdq_execution_scope": {
            "schema_version": "language2drive.fdq_execution_scope.v1",
            "qualification_claim": False,
            "scope": "selected_recovery_subset_not_full_fdq_qualification",
            "selected_identities": [
                {"ability": job["run"]["ability"], "seed": job["run"]["seed"]}
                for job in jobs
            ],
        },
        "score_policy": (
            "This plan executes only the selected recovery subset; full FDQ "
            "qualification is computed separately from the exact 4x3 ledger."
        ),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    payload["plan_sha256"] = stable_hash({key: value for key, value in payload.items() if key != "plan_sha256"})
    return payload


def carla_server_available(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_seconds)):
            return True
    except OSError:
        return False


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "1.0", "runs": {}}
    try:
        value = load_json(path)
    except ValueError:
        return {"schema_version": "1.0", "runs": {}}
    if not isinstance(value, dict) or not isinstance(value.get("runs"), dict):
        return {"schema_version": "1.0", "runs": {}}
    return value

def _fdq_identity(run: dict[str, Any]) -> dict[str, Any]:
    binding = _canonical_fdq_route_bindings()[str(run["ability"])]
    return {
        "run_id": run["run_id"],
        "ability": run["ability"],
        "seed": run["seed"],
        "route_id": binding["route_id"],
        "town": binding["town"],
        "scenario": binding["scenario"],
    }


def _core_terminal_result(
    run: dict[str, Any], status: str, reason: str, **details: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "language2drive.contract_result.v1",
        "status": status,
        "success": False,
        "semantic_success": False,
        "capture_complete": False,
        "run_id": run.get("run_id"),
        "pcu_id": run.get("pcu_id"),
        "arm_id": run.get("arm_id"),
        "paraphrase": run.get("paraphrase"),
        "seed": run.get("seed"),
        "track": run.get("track"),
        "route_entry_id": run.get("route_entry_id"),
        "own_contract": False,
        "opposite_contract": False,
        "safe_legal": False,
        "progress": False,
        "driving_score": None,
        "deadline": False,
        "stability": False,
        "reason": reason,
        **details,
    }


def _fdq_terminal_result(
    run: dict[str, Any], status: str, reason: str, **details: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "language2drive.fdq_result.v1",
        "status": status,
        "success": False,
        "capture_complete": False,
        **_fdq_identity(run),
        "reason": reason,
        **details,
    }


def _finalize_fdq_result(run: dict[str, Any], run_dir: Path, returncode: int) -> Path:
    """Normalize one Bench2Drive route record into the frozen FDQ schema."""

    output = run_dir / "fdq_result.json"
    if returncode != 0:
        atomic_json(output, _fdq_terminal_result(
            run, "crash", "evaluator_nonzero_exit",
            runner_returncode=returncode,
        ))
        return output
    checkpoint = run_dir / "bench2drive_result.json"
    try:
        payload = _strict_json_object_file(checkpoint, "Bench2Drive checkpoint")
        records = payload["_checkpoint"]["records"]
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
            raise ValueError("expected exactly one Bench2Drive route record")
        record = records[0]
        scores = record.get("scores", {})
        infractions = record.get("infractions", {})
        if not isinstance(scores, dict) or not isinstance(infractions, dict):
            raise ValueError("route record has invalid scores/infractions")
        binding = _canonical_fdq_route_bindings()[str(run["ability"])]
        expected_record_identity = {
            "route_id": f"RouteScenario_{binding['route_id']}_rep0",
            "town_name": binding["town"],
            "scenario_name": f"{binding['scenario']}_1",
        }
        if any(record.get(key) != value for key, value in expected_record_identity.items()):
            raise ValueError("Bench2Drive route record identity differs")
        if record.get("status") not in _FDQ_RECORD_STATUSES:
            raise ValueError("Bench2Drive route record status differs")
        canonical_infraction_keys = {
            "collisions_layout", "collisions_pedestrian", "collisions_vehicle",
            "red_light", "stop_infraction", "outside_route_lanes",
            "min_speed_infractions", "yield_emergency_vehicle_infractions",
            "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout",
        }
        if (
            set(infractions) != canonical_infraction_keys
            or any(not isinstance(value, list) for value in infractions.values())
            or type(record.get("num_infractions")) is not int
            or record["num_infractions"] < 0
            or record["num_infractions"]
            != sum(len(value) for value in infractions.values())
        ):
            raise ValueError("Bench2Drive route infractions must be lists")
        metrics: dict[str, float] = {}
        metric_bounds = {
            "score_composed": (0.0, 100.0),
            "score_route": (0.0, 100.0),
            "score_penalty": (0.0, 1.0),
        }
        for name, (minimum, maximum) in metric_bounds.items():
            raw = scores.get(name)
            if (
                not isinstance(raw, (int, float))
                or isinstance(raw, bool)
                or not math.isfinite(float(raw))
                or not minimum <= float(raw) <= maximum
            ):
                raise ValueError(f"Bench2Drive route metric {name} is invalid")
            metrics[name] = float(raw)
        infraction_free = all(not value for value in infractions.values())
        success = record["status"] in {"Completed", "Perfect"} and infraction_free
        if success and not (
            metrics["score_route"] == 100.0
            and metrics["score_penalty"] == 1.0
            and metrics["score_composed"] == 100.0
        ):
            raise ValueError("successful Bench2Drive record scores are inconsistent")
        result = {
            "schema_version": "language2drive.fdq_result.v1",
            "status": "complete",
            "success": success,
            "capture_complete": True,
            **_fdq_identity(run),
            "record_status": record.get("status"),
            "DS": metrics["score_composed"],
            "RC": metrics["score_route"],
            "IP": metrics["score_penalty"],
            "infraction_free": infraction_free,
            "source": str(checkpoint.resolve()),
            "source_sha256": _current_file_sha256(checkpoint),
            "checkpoint_record_sha256": stable_hash(record),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = _fdq_terminal_result(
            run, "invalid_result", f"{type(error).__name__}: {error}",
            source=str(checkpoint.resolve()),
        )
    atomic_json(output, result)
    if (
        result.get("status") == "complete"
        and result.get("capture_complete") is True
        and (run_dir / "trace.jsonl").is_file()
        and (run_dir / "capture_capabilities.json").is_file()
    ):
        try:
            from .fdq_metrics import score_fdq_metrics, validate_fdq_metrics

            supplemental = score_fdq_metrics(run, run_dir)
            if not validate_fdq_metrics(supplemental, run):
                raise ValueError("supplemental FDQ metric artifact is non-canonical")
            atomic_json(run_dir / "fdq_metrics.json", supplemental)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
            atomic_json(output, _fdq_terminal_result(
                run, "invalid_result", "supplemental_fdq_metric_failure",
                detail=f"{type(error).__name__}: {error}",
                source=str(checkpoint.resolve()),
            ))
    return output


def _storage_policy() -> dict[str, Any]:
    """Return the versioned conservative result-storage contract."""

    return {
        "schema_version": "language2drive.storage_policy.v1",
        "formula": (
            "fixed_overhead_bytes=bounded_evaluator_log_bytes+"
            "nontrace_fixed_overhead_bytes; retained=fixed_overhead_bytes+"
            "ceil(simulator_seconds_basis*retained_upper_bytes_per_sim_second); "
            "transient=fixed_overhead_bytes+ceil(simulator_seconds_basis*"
            "historical_upper_bytes_per_sim_second)"
        ),
        "historical_upper_bytes_per_sim_second": STORAGE_UPPER_BYTES_PER_SIM_SECOND,
        "historical_basis": {
            "sample": "12 canonical 2026-08-12 Oracle result directories",
            "observed_p90_total_bytes_per_sim_second": 72_721,
            "observed_max_total_bytes_per_sim_second": 128_769,
            "selected_conservative_upper_bytes_per_sim_second": STORAGE_UPPER_BYTES_PER_SIM_SECOND,
        },
        "simulator_seconds_basis": "private sidecar deadline; 900s when no deadline exists (FDQ)",
        "no_deadline_simulator_seconds": STORAGE_NO_DEADLINE_SIM_SECONDS,
        "minimum_free_reserve_bytes": STORAGE_RESERVE_BYTES,
        "bounded_evaluator_log_bytes": EVALUATOR_LOG_MAX_BYTES,
        "nontrace_fixed_overhead_bytes": STORAGE_NONTRACE_FIXED_OVERHEAD_BYTES,
        "fixed_overhead_bytes": (
            EVALUATOR_LOG_MAX_BYTES + STORAGE_NONTRACE_FIXED_OVERHEAD_BYTES
        ),
        "nontrace_overhead_basis": {
            "sample": "12 canonical 2026-08-12 Oracle result directories",
            "observed_median_bytes": 27_674,
            "observed_p90_bytes": 37_977,
            "observed_max_bytes": 62_859,
            "selected_fixed_nontrace_bytes": STORAGE_NONTRACE_FIXED_OVERHEAD_BYTES,
            "bounded_stdout_budgeted_separately": True,
        },
        "retained_upper_bytes_per_sim_second": STORAGE_HARDLINK_PROJECTION_BYTES_PER_SIM_SECOND,
        "active_trace_layout": "evaluator_may_materialize_duplicate_then_runner_verified_hardlink",
        "hardlink_projection": {
            "active": True,
            "requires_runner_verification": (
                "trace.jsonl and frames.jsonl must have equal size and SHA-256 before "
                "frames.jsonl is atomically replaced by a same-filesystem hardlink"
            ),
            "projected_bytes_per_sim_second": STORAGE_HARDLINK_PROJECTION_BYTES_PER_SIM_SECOND,
            "projection_multiplier": 0.5,
        },
    }


def _job_storage_estimate(
    job: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    if policy != _storage_policy():
        raise ValueError("unsupported or stale result storage policy")
    deadline = job.get("deadline_seconds")
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool):
        if not math.isfinite(float(deadline)) or float(deadline) <= 0:
            raise ValueError("job deadline_seconds must be a finite positive number")
        simulator_seconds = float(deadline)
        basis = "private_sidecar_deadline"
    elif deadline is None:
        simulator_seconds = float(policy["no_deadline_simulator_seconds"])
        basis = "no_deadline_conservative_default"
    else:
        raise ValueError("job deadline_seconds must be numeric or null")
    transient = int(math.ceil(
        simulator_seconds * int(policy["historical_upper_bytes_per_sim_second"])
    ))
    retained = int(math.ceil(
        simulator_seconds
        * int(policy["retained_upper_bytes_per_sim_second"])
    ))
    fixed = int(policy["fixed_overhead_bytes"])
    transient += fixed
    retained += fixed
    return {
        "simulator_seconds_basis": simulator_seconds,
        "basis": basis,
        "estimated_bytes": retained,
        "estimated_transient_capture_bytes": transient,
        "fixed_overhead_bytes": fixed,
        "estimated_bytes_without_trace_hardlink": transient,
        "estimated_bytes_with_trace_hardlink_projection": retained,
    }


def _plan_storage_estimate(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    retained = sum(int(job["storage_estimate"]["estimated_bytes"]) for job in jobs)
    transient = sum(
        int(job["storage_estimate"]["estimated_transient_capture_bytes"])
        for job in jobs
    )
    hardlink = sum(
        int(job["storage_estimate"]["estimated_bytes_with_trace_hardlink_projection"])
        for job in jobs
    )
    return {
        "estimated_plan_storage_bytes": retained,
        "estimated_plan_storage_bytes_without_trace_hardlink": transient,
        "estimated_plan_storage_bytes_with_trace_hardlink_projection": hardlink,
        "maximum_current_job_transient_storage_bytes": max(
            (int(job["storage_estimate"]["estimated_transient_capture_bytes"]) for job in jobs),
            default=0,
        ),
    }


def _wall_timeout_policy(*, agent: str, workload: str) -> dict[str, Any]:
    """Version the infrastructure ceiling separately from benchmark time."""

    if agent == "autovla":
        return {
            "schema_version": "language2drive.wall_timeout_policy.v1",
            "agent_class": "autovla",
            "workload": workload,
            "formula": "max(900s, 12 * simulator_deadline + 300s)",
            "sim_time_multiplier": 12.0,
            "startup_and_world_load_grace_seconds": 300.0,
            "minimum_wall_seconds": 900.0,
            "no_deadline_wall_seconds": 900.0,
            "provenance": {
                "source": "2026-08-12 local AutoVLA closed-loop runtime audit",
                "observed_system_to_sim_time_ratios": [10.70, 11.05],
                "selected_multiplier": 12.0,
                "purpose": "infrastructure ceiling only; benchmark deadline remains sidecar-authoritative",
            },
        }
    return {
        "schema_version": "language2drive.wall_timeout_policy.v1",
        "agent_class": "lightweight_or_operator_managed",
        "workload": workload,
        "formula": "max(300s, simulator_deadline + 300s)",
        "sim_time_multiplier": 1.0,
        "startup_and_world_load_grace_seconds": 300.0,
        "minimum_wall_seconds": 300.0,
        "no_deadline_wall_seconds": 900.0,
        "provenance": {
            "source": "legacy lightweight oracle/fixed-policy runner ceiling",
            "purpose": "infrastructure ceiling only; benchmark deadline remains sidecar-authoritative",
        },
    }


def _job_wall_timeout(
    job: dict[str, Any],
    requested: float | None,
    policy: dict[str, Any] | None = None,
) -> float:
    if requested is not None:
        if not math.isfinite(float(requested)) or float(requested) <= 0:
            raise ValueError("wall_timeout_seconds must be a finite positive number")
        return float(requested)
    policy = policy or _wall_timeout_policy(agent="unknown", workload="unknown")
    if policy.get("schema_version") != "language2drive.wall_timeout_policy.v1":
        raise ValueError("unsupported wall timeout policy")
    numeric = {}
    for key in (
        "sim_time_multiplier",
        "startup_and_world_load_grace_seconds",
        "minimum_wall_seconds",
        "no_deadline_wall_seconds",
    ):
        value = policy.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"invalid wall timeout policy field: {key}")
        numeric[key] = float(value)
    deadline = job.get("deadline_seconds")
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool):
        if not math.isfinite(float(deadline)) or float(deadline) <= 0:
            raise ValueError("job deadline_seconds must be a finite positive number")
        return max(
            numeric["minimum_wall_seconds"],
            numeric["sim_time_multiplier"] * float(deadline)
            + numeric["startup_and_world_load_grace_seconds"],
        )
    # FDQ has no simulator deadline and keeps a conservative fifteen-minute
    # infrastructure ceiling.  This never changes the benchmark success rule.
    return numeric["no_deadline_wall_seconds"]


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether any process still belongs to ``process_group_id``."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A group that exists but cannot be signalled is still not clean.
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Terminate the evaluator session and reap its direct child leader."""

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    # Waiting only for the Popen leader is insufficient: it may honor TERM
    # while an evaluator descendant ignores it and keeps the CARLA endpoint
    # busy after the runner releases its flock.  Give the whole process group
    # one bounded grace period before escalating.
    deadline = time.monotonic() + EVALUATOR_TERMINATE_GRACE_SECONDS
    while True:
        process.poll()
        if not _process_group_exists(process_group_id):
            process.wait()
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(0.05, remaining))

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait()
    reap_deadline = time.monotonic() + EVALUATOR_TERMINATE_GRACE_SECONDS
    while _process_group_exists(process_group_id):
        if time.monotonic() >= reap_deadline:
            raise RuntimeError("evaluator process group survived SIGKILL cleanup")
        time.sleep(0.01)


class _StreamingBoundedLog:
    """Incrementally hash stdout while retaining only a bounded head/tail."""

    def __init__(self, path: Path, *, max_bytes: int = EVALUATOR_LOG_MAX_BYTES):
        if int(max_bytes) < 1024:
            raise ValueError("bounded evaluator log size must be at least 1024 bytes")
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self._head_limit = self.max_bytes // 2
        self._tail_limit = self.max_bytes - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._digest = hashlib.sha256()
        self._original_size = 0
        self._metadata: dict[str, Any] | None = None

    def write(self, value: bytes | str) -> int:
        if self._metadata is not None:
            raise ValueError("cannot append to a finalized bounded evaluator log")
        data = value.encode("utf-8", errors="replace") if isinstance(value, str) else bytes(value)
        self._digest.update(data)
        self._original_size += len(data)
        missing_head = self._head_limit - len(self._head)
        if missing_head > 0:
            self._head.extend(data[:missing_head])
        self._tail.extend(data)
        if len(self._tail) > self._tail_limit:
            del self._tail[: len(self._tail) - self._tail_limit]
        return len(data)

    def flush(self) -> None:
        """File-like compatibility for unit fakes; capture is memory-resident."""

    def _exact_payload(self) -> bytes:
        if self._original_size <= len(self._head):
            return bytes(self._head[: self._original_size])
        overlap = len(self._head) + len(self._tail) - self._original_size
        return bytes(self._head) + bytes(self._tail[max(0, overlap) :])

    def finalize(self) -> dict[str, Any]:
        if self._metadata is not None:
            return dict(self._metadata)
        original_sha256 = self._digest.hexdigest()
        compacted = self._original_size > self.max_bytes
        if not compacted:
            payload = self._exact_payload()
            retained_head = len(payload)
            retained_tail = 0
        else:
            prefix = (
                "LANGUAGE2DRIVE_BOUNDED_LOG_V1\n"
                f"original_size_bytes={self._original_size}\n"
                f"original_sha256={original_sha256}\n"
                "--- BEGIN RETAINED HEAD ---\n"
            ).encode("ascii")
            middle = b"\n--- OMITTED MIDDLE; SEE DIGEST ABOVE ---\n"
            suffix = b"\n--- END RETAINED TAIL ---\n"
            payload_budget = self.max_bytes - len(prefix) - len(middle) - len(suffix)
            if payload_budget <= 0:
                raise ValueError("bounded evaluator log size is too small for metadata")
            retained_head = payload_budget // 2
            retained_tail = payload_budget - retained_head
            payload = (
                prefix
                + bytes(self._head[:retained_head])
                + middle
                + bytes(self._tail[-retained_tail:])
                + suffix
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.stream.", dir=str(self.path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        metadata = {
            "schema_version": "language2drive.bounded_log.v1",
            "capture_mode": "streaming_pipe_incremental_sha256_head_tail",
            "compacted": compacted,
            "original_size_bytes": self._original_size,
            "original_sha256": original_sha256,
            "retained_head_bytes": retained_head,
            "retained_tail_bytes": retained_tail,
            "stored_size_bytes": len(payload),
            "maximum_size_bytes": self.max_bytes,
        }
        atomic_json(self.path.with_name(self.path.name + ".bounded.json"), metadata)
        self._metadata = metadata
        return dict(metadata)


def _drain_evaluator_pipe(file_descriptor: int, log: _StreamingBoundedLog) -> bool:
    """Drain all currently readable bytes; return true after pipe EOF."""

    # Return to the deadline loop even if a pathological child emits stdout
    # continuously.  Sixteen MiB per pass is far above a pipe's normal buffer
    # but keeps the wall-time watchdog from being starved by output flooding.
    for _ in range(16):
        try:
            chunk = os.read(file_descriptor, 1024 * 1024)
        except BlockingIOError:
            return False
        if not chunk:
            return True
        log.write(chunk)
    return False


def _run_evaluator(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: _StreamingBoundedLog,
    wall_timeout: float,
) -> tuple[int | None, bool]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if process.stdout is None:
        _terminate_process_group(process)
        raise OSError("evaluator stdout pipe was not created")
    file_descriptor = process.stdout.fileno()
    os.set_blocking(file_descriptor, False)
    deadline = time.monotonic() + float(wall_timeout)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and process.poll() is None:
                _terminate_process_group(process)
                _drain_evaluator_pipe(file_descriptor, log)
                return None, True
            try:
                readable, _, _ = select.select(
                    [file_descriptor], [], [], max(0.0, min(0.1, remaining))
                )
            except InterruptedError:
                continue
            if readable:
                _drain_evaluator_pipe(file_descriptor, log)
            returncode = process.poll()
            if returncode is not None:
                _drain_evaluator_pipe(file_descriptor, log)
                # A successful leader is not sufficient: it may have left a
                # child in the evaluator session.  Cleaning that child after
                # it already influenced CARLA/artifacts does not make the run
                # valid: clean it up, then reject the attempt explicitly.
                if _process_group_exists(process.pid):
                    _terminate_process_group(process)
                    raise RuntimeError(
                        "evaluator leader exited with a surviving process-group member"
                    )
                process.wait()
                return returncode, False
    except BaseException as error:
        try:
            _terminate_process_group(process)
        except BaseException as cleanup_error:
            # Preserve the operator/system exception exactly.  Cleanup is
            # best-effort only if it is itself interrupted or the OS refuses
            # process control; recording the secondary failure keeps that
            # exceptional diagnostic without replacing the original cause.
            error.add_note(
                "evaluator process-group cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    finally:
        process.stdout.close()


def _require_external_carla_flag(command: list[str]) -> None:
    """Reject stale/tampered evaluator plans that could spawn CARLA."""

    if str(DEFAULT_EVALUATOR) in command and command.count("--l2d-external-carla") != 1:
        raise ValueError(
            "Language2Drive evaluator commands must contain --l2d-external-carla exactly once"
        )


def _command_option(command: list[str], option: str, default: str) -> str:
    separate = [index for index, value in enumerate(command) if value == option]
    inline = [value.removeprefix(f"{option}=") for value in command if value.startswith(f"{option}=")]
    if len(separate) + len(inline) > 1:
        raise ValueError(f"plan command option {option} must appear at most once")
    if inline:
        if not inline[0]:
            raise ValueError(f"plan command option {option} requires a value")
        return inline[0]
    if not separate:
        return default
    index = separate[0]
    if index + 1 >= len(command):
        raise ValueError(f"plan command option {option} requires a value")
    return command[index + 1]


def _plan_carla_endpoint(plan: dict[str, Any]) -> tuple[str, int, int]:
    """Return the immutable authority endpoint, never one parsed from a job."""

    authority = plan.get("execution_authority")
    if not isinstance(authority, dict) or set(authority) != _PLAN_EXECUTION_AUTHORITY_KEYS:
        raise ValueError("execution plan authority shape differs")
    host = authority.get("host")
    port = authority.get("port")
    traffic_manager_port = authority.get("traffic_manager_port")
    if not isinstance(host, str) or not host:
        raise ValueError("execution plan authority host is invalid")
    if any(type(value) is not int or not 1 <= value <= 65535 for value in (port, traffic_manager_port)):
        raise ValueError("execution plan authority CARLA ports are invalid")
    return host, port, traffic_manager_port


def _carla_lock_path(project_root: Path, host: str, port: int, traffic_manager_port: int) -> Path:
    # Network host names cannot legitimately contain path separators.  Replace
    # any other filename-hostile character so a tampered plan cannot escape
    # the project-local lock directory.
    safe_host = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in host
    ) or "_"
    return (
        Path(project_root)
        / ".runtime/locks"
        / f"carla-{safe_host}-{port}-tm{traffic_manager_port}.lock"
    )


@contextmanager
def _carla_execution_lock(
    *,
    project_root: Path,
    host: str,
    port: int,
    traffic_manager_port: int,
    plan_sha256: Any,
) -> Iterator[Path]:
    """Fail-fast process lock for one externally managed CARLA endpoint."""

    lock_path = _carla_lock_path(project_root, host, port, traffic_manager_port)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            stream.seek(0)
            owner = stream.read().strip() or "owner metadata unavailable"
            raise RuntimeError(
                f"CARLA endpoint {host}:{port} (TrafficManager {traffic_manager_port}) "
                f"is already owned by another runner; lock={lock_path}; owner={owner}"
            ) from None

        stream.seek(0)
        stream.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "plan_sha256": plan_sha256,
                "started_at": _utc_now(),
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield lock_path
    finally:
        try:
            if acquired:
                try:
                    stream.seek(0)
                    stream.truncate()
                    json.dump(
                        {
                            "pid": os.getpid(),
                            "status": "released",
                            "released_at": _utc_now(),
                        },
                        stream,
                        sort_keys=True,
                    )
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                except OSError:
                    # The kernel lock is authoritative; release it even if
                    # best-effort operator metadata cannot be refreshed.
                    pass
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            # Closing also releases a process lock if explicit unlock itself
            # is interrupted, preserving fail-safe exceptional cleanup.
            stream.close()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).relative_to(Path(root))
    except ValueError:
        return False
    return True


def _validate_results_dir(project_root: Path, results_dir: Path) -> Path:
    """Keep evaluator writes on the project/data disk, never the system disk."""

    project = Path(project_root).resolve()
    resolved = Path(results_dir).resolve()
    canonical_project_results = (project / "results").resolve()
    if not _path_is_within(resolved, canonical_project_results):
        raise ValueError(
            "results_dir must stay under the project-owned results root; "
            f"refusing out-of-scope path: {resolved}"
        )
    return resolved


def _validated_state_path(
    results_dir: Path, state_path: Path | None, *, plan_sha256: str,
) -> Path:
    root = Path(results_dir).resolve()
    if (
        not isinstance(plan_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}|lock-test-[0-9]+-[0-9]+|[A-Za-z0-9_.-]{1,128}", plan_sha256)
    ):
        raise ValueError("runner state plan identity is invalid")
    state_root = root / ".runner-states"
    expected_name = f"{plan_sha256}.json"
    candidate = Path(state_path) if state_path is not None else state_root / expected_name
    if not candidate.is_absolute():
        candidate = state_root / candidate
    resolved_parent = candidate.parent.resolve()
    resolved = resolved_parent / candidate.name
    if (
        resolved_parent != state_root.resolve()
        or candidate.name != expected_name
        or resolved.exists() and resolved.is_symlink()
    ):
        raise ValueError("runner state path must be the canonical per-plan results ledger")
    return resolved


def _storage_available_bytes(path: Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _validate_storage_plan(plan: dict[str, Any]) -> dict[str, Any]:
    policy = plan.get("storage_policy")
    if policy != _storage_policy():
        raise ValueError(
            "execution plan is missing the current language2drive.storage_policy.v1; "
            "regenerate it with plan-only CLI before execution"
        )
    jobs = plan.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("execution plan jobs must be a list")
    expected_rows = [_job_storage_estimate(job, policy) for job in jobs]
    for index, (job, expected) in enumerate(zip(jobs, expected_rows)):
        if job.get("storage_estimate") != expected:
            raise ValueError(f"execution plan job {index} has a stale storage estimate")
    expected_plan = _plan_storage_estimate(jobs)
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise ValueError(f"execution plan has a stale {key}")
    return policy


def _storage_snapshot(
    results_dir: Path,
    job: dict[str, Any] | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    estimate = (
        int(_job_storage_estimate(job, policy)["estimated_transient_capture_bytes"])
        if job is not None
        else 0
    )
    available = _storage_available_bytes(results_dir)
    reserve = int(policy["minimum_free_reserve_bytes"])
    required = reserve + estimate
    return {
        "available_bytes": available,
        "required_bytes": required,
        "minimum_free_reserve_bytes": reserve,
        "current_job_estimated_transient_bytes": estimate,
        "sufficient": available >= required,
    }


def _compact_evaluator_log(
    path: Path, *, max_bytes: int = EVALUATOR_LOG_MAX_BYTES
) -> dict[str, Any]:
    """Atomically retain a bounded diagnostic head/tail with source digest."""

    log_path = Path(path)
    original_size = log_path.stat().st_size
    if original_size <= int(max_bytes):
        return {
            "schema_version": "language2drive.bounded_log.v1",
            "compacted": False,
            "original_size_bytes": original_size,
            "stored_size_bytes": original_size,
        }
    if int(max_bytes) < 1024:
        raise ValueError("bounded evaluator log size must be at least 1024 bytes")
    original_sha256 = sha256_file(log_path)
    prefix = (
        "LANGUAGE2DRIVE_BOUNDED_LOG_V1\n"
        f"original_size_bytes={original_size}\n"
        f"original_sha256={original_sha256}\n"
        "--- BEGIN RETAINED HEAD ---\n"
    ).encode("ascii")
    middle = b"\n--- OMITTED MIDDLE; SEE DIGEST ABOVE ---\n"
    suffix = b"\n--- END RETAINED TAIL ---\n"
    payload_budget = int(max_bytes) - len(prefix) - len(middle) - len(suffix)
    if payload_budget <= 0:
        raise ValueError("bounded evaluator log size is too small for metadata")
    head_bytes = payload_budget // 2
    tail_bytes = payload_budget - head_bytes
    with log_path.open("rb") as source:
        head = source.read(head_bytes)
        source.seek(max(0, original_size - tail_bytes))
        tail = source.read(tail_bytes)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{log_path.name}.bounded.", dir=str(log_path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(prefix)
            destination.write(head)
            destination.write(middle)
            destination.write(tail)
            destination.write(suffix)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary, log_path.stat().st_mode & 0o777)
        os.replace(temporary, log_path)
        _fsync_directory(log_path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    metadata = {
        "schema_version": "language2drive.bounded_log.v1",
        "compacted": True,
        "original_size_bytes": original_size,
        "original_sha256": original_sha256,
        "retained_head_bytes": len(head),
        "retained_tail_bytes": len(tail),
        "stored_size_bytes": log_path.stat().st_size,
        "maximum_size_bytes": int(max_bytes),
    }
    atomic_json(log_path.with_name(log_path.name + ".bounded.json"), metadata)
    return metadata


def _merge_core_evaluator_capture(
    alias_root: Path,
    run_dir: Path,
    run: dict[str, Any],
    *,
    evaluator_track: str,
    private_sidecar_sha256: str,
) -> None:
    """Validate the evaluator namespace/provenance before publishing files."""

    root_metadata = alias_root.lstat()
    if alias_root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("evaluator record root is not a real directory")
    children = sorted(alias_root.iterdir(), key=lambda path: path.name)
    expected_child_name = (
        f"{evaluator_track}__{run['pcu_id']}__{run['arm_id']}__"
        f"p{run['paraphrase']}__s{run['seed']}"
    )
    if len(children) != 1 or children[0].name != expected_child_name:
        raise RuntimeError("evaluator record namespace differs from route authority")
    child = children[0]
    child_metadata = child.lstat()
    if child.is_symlink() or not stat.S_ISDIR(child_metadata.st_mode):
        raise RuntimeError("evaluator record namespace is not a real directory")
    metadata_values = []
    for name in ("run_meta.json", "route_meta.json"):
        path = child / name
        file_metadata = path.lstat()
        if (
            path.is_symlink() or not stat.S_ISREG(file_metadata.st_mode)
            or file_metadata.st_nlink != 1
        ):
            raise RuntimeError(f"evaluator {name} is not regular unique evidence")
        try:
            value = _strict_json_loads(
                path.read_text(encoding="utf-8"), f"evaluator {name}",
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise RuntimeError(f"evaluator {name} is invalid") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"evaluator {name} shape differs")
        metadata_values.append(value)
    if metadata_values[0] != metadata_values[1]:
        raise RuntimeError("evaluator run/route metadata differ")
    metadata = metadata_values[0]
    public = metadata.get("language2drive_public_context")
    provenance = metadata.get("private_capture_provenance")
    expected_route_entry = run["route_entry_id"]
    if (
        metadata.get("run_id") != expected_child_name
        or metadata.get("route_entry_id") != expected_route_entry
        or metadata.get("route_id") != f"RouteScenario_{expected_route_entry}"
        or metadata.get("scenario_name") != "Language2DriveContract"
        or not _public_context_matches_evaluator_track(
            public, evaluator_track, run["seed"],
        )
        or provenance != {
            "route_entry_id": expected_route_entry,
            "private_sidecar_sha256": private_sidecar_sha256,
        }
    ):
        raise RuntimeError("evaluator record metadata identity/provenance differs")

    def jsonl_rows(name: str, *, require_nonempty: bool) -> list[dict[str, Any]]:
        source = child / name
        rows: list[dict[str, Any]] = []
        try:
            for line_number, line in enumerate(
                source.read_text(encoding="utf-8").splitlines(), 1,
            ):
                if not line.strip():
                    continue
                row = _strict_json_loads(line, f"evaluator {name}:{line_number}")
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append(row)
        except (OSError, UnicodeError, ValueError) as error:
            raise RuntimeError(f"evaluator {name} is invalid") from error
        if require_nonempty and not rows:
            raise RuntimeError(f"evaluator {name} is empty")
        return rows

    artifacts = sorted(child.iterdir(), key=lambda path: path.name)
    required = {
        "agent_plans.jsonl", "events.jsonl", "frames.jsonl",
        "route_meta.json", "run_meta.json", "trace.jsonl",
        "route_summary.json", "capture_capabilities.json",
        "contract_result.json",
    }
    optional_json = {"checkpoint_load_report.json", "metric_info.json"}
    optional_binary = {"front_frames.zip"}
    optional = optional_json | optional_binary
    names = {artifact.name for artifact in artifacts}
    if not required <= names or names - required - optional:
        raise RuntimeError("evaluator record contains an unexpected artifact")
    for artifact in artifacts:
        artifact_metadata = artifact.lstat()
        destination = run_dir / artifact.name
        if (
            artifact.is_symlink() or not stat.S_ISREG(artifact_metadata.st_mode)
            or artifact_metadata.st_nlink != 1
            or destination.exists() or destination.is_symlink()
        ):
            raise RuntimeError("evaluator record artifact is unsafe or collides")
    trace_rows = jsonl_rows("trace.jsonl", require_nonempty=True)
    jsonl_rows("frames.jsonl", require_nonempty=True)
    jsonl_rows("events.jsonl", require_nonempty=False)
    # Fixed-policy baseline agents deliberately emit an explicit empty plan
    # stream.  Its presence, regular-file identity and strict JSONL syntax are
    # capture evidence; non-emptiness is not a universal agent capability.
    jsonl_rows("agent_plans.jsonl", require_nonempty=False)
    trace_path = child / "trace.jsonl"
    frames_path = child / "frames.jsonl"
    trace_stat = trace_path.lstat()
    frames_stat = frames_path.lstat()
    def evidence_identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            int(value.st_dev), int(value.st_ino), int(value.st_size),
            int(value.st_mtime_ns), int(value.st_ctime_ns), int(value.st_nlink),
        )
    trace_identity = evidence_identity(trace_stat)
    frames_identity = evidence_identity(frames_stat)
    if (
        trace_stat.st_nlink != 1 or frames_stat.st_nlink != 1
        or trace_stat.st_size != frames_stat.st_size
        or _current_file_sha256(trace_path) != _current_file_sha256(frames_path)
        or evidence_identity(trace_path.lstat()) != trace_identity
        or evidence_identity(frames_path.lstat()) != frames_identity
    ):
        raise RuntimeError("evaluator trace/frames compatibility evidence differs")
    try:
        result = _strict_json_object_file(child / "contract_result.json", "contract result")
        summary = _strict_json_object_file(child / "route_summary.json", "route summary")
        capabilities = _strict_json_object_file(
            child / "capture_capabilities.json", "capture capabilities",
        )
        for name in optional_json & names:
            _strict_json_object_file(child / name, f"evaluator {name}")
        for name in optional_binary & names:
            archive_path = child / name
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                if not members or archive.testzip() is not None:
                    raise ValueError("front frame archive is empty or corrupt")
                if any(
                    member.is_dir()
                    or member.filename.startswith(("/", "../"))
                    or "/../" in member.filename
                    or member.file_size > 16 * 1024 * 1024
                    for member in members
                ):
                    raise ValueError("front frame archive has an unsafe member")
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError("evaluator structured capture is invalid") from error
    if not _core_result_is_final(result, child / "contract_result.json", run):
        raise RuntimeError("evaluator contract result is not canonical/final")
    if (
        type(summary.get("total_frames_logged")) is not int
        or summary["total_frames_logged"] != len(trace_rows)
        or type(summary.get("frame_interval")) is not int
        or summary["frame_interval"] != 1
        or summary.get("logger_error_count") != 0
        or summary.get("capture_complete") is not True
        or summary.get("contract_success") is not result["success"]
        or capabilities.get("schema_version") != "language2drive.trace_capture.v1"
        or capabilities.get("private_context_loaded") is not True
        or capabilities.get("logger_error_count") != 0
        or capabilities.get("unresolved_actor_ids") != []
        or capabilities.get("unresolved_gate_ids") != []
        or capabilities.get("unresolved_dwell_ids") != []
        or capabilities.get("trace_cadence_complete") is not True
        or capabilities.get("trace_cadence_errors") != []
    ):
        raise RuntimeError("evaluator capture summary/capability authority differs")
    for artifact in artifacts:
        os.replace(artifact, run_dir / artifact.name)
    child.rmdir()
    alias_root.rmdir()


def _merge_fdq_evaluator_capture(
    alias_root: Path, run_dir: Path, run: dict[str, Any],
) -> None:
    """Publish only the exact upstream FDQ logger namespace/artifact set."""

    root_metadata = alias_root.lstat()
    if alias_root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("FDQ evaluator record root is unsafe")
    children = sorted(alias_root.iterdir(), key=lambda path: path.name)
    binding = _canonical_fdq_route_bindings()[run["ability"]]
    expected_child_name = f"RouteScenario_{binding['route_id']}_rep0"
    if len(children) != 1 or children[0].name != expected_child_name:
        raise RuntimeError("FDQ evaluator record namespace differs")
    child = children[0]
    child_metadata = child.lstat()
    if child.is_symlink() or not stat.S_ISDIR(child_metadata.st_mode):
        raise RuntimeError("FDQ evaluator record namespace is unsafe")
    required = {
        "events.jsonl", "frames.jsonl", "route_meta.json", "run_meta.json",
        "trace.jsonl", "route_summary.json", "capture_capabilities.json",
    }
    optional = {
        "agent_plans.jsonl", "model_plans.jsonl",
        "checkpoint_load_report.json", "metric_info.json",
        "alpamayo_load_report.json", "alpamayo_agent_summary.json",
    }
    artifacts = sorted(child.iterdir(), key=lambda path: path.name)
    names = {path.name for path in artifacts}
    if not required <= names or names - required - optional:
        raise RuntimeError("FDQ evaluator record contains an unexpected artifact")
    for artifact in artifacts:
        metadata = artifact.lstat()
        destination = run_dir / artifact.name
        expected_links = 2 if artifact.name in {"trace.jsonl", "frames.jsonl"} else 1
        if (
            artifact.is_symlink() or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != expected_links
            or destination.exists() or destination.is_symlink()
        ):
            raise RuntimeError("FDQ evaluator record artifact is unsafe or collides")
    try:
        route_meta = _strict_json_object_file(child / "route_meta.json", "FDQ route metadata")
        run_meta = _strict_json_object_file(child / "run_meta.json", "FDQ run metadata")
        summary = _strict_json_object_file(child / "route_summary.json", "FDQ route summary")
        capabilities = _strict_json_object_file(
            child / "capture_capabilities.json", "FDQ capture capabilities",
        )
        for name in optional & names - {"agent_plans.jsonl", "model_plans.jsonl"}:
            _strict_json_object_file(child / name, f"FDQ {name}")
        rows_by_name: dict[str, list[dict[str, Any]]] = {}
        for name, require_nonempty in (
            ("trace.jsonl", True), ("frames.jsonl", True),
            ("events.jsonl", False), ("agent_plans.jsonl", False),
            ("model_plans.jsonl", False),
        ):
            if name not in names:
                continue
            rows = []
            for line_number, line in enumerate(
                (child / name).read_text(encoding="utf-8").splitlines(), 1,
            ):
                if not line.strip():
                    continue
                row = _strict_json_loads(line, f"FDQ {name}:{line_number}")
                if not isinstance(row, dict):
                    raise ValueError("FDQ JSONL row is not an object")
                rows.append(row)
            if require_nonempty and not rows:
                raise ValueError(f"FDQ {name} is empty")
            rows_by_name[name] = rows
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError("FDQ evaluator structured capture is invalid") from error
    expected_route_entry = int(binding["route_id"])
    public = route_meta.get("language2drive_public_context")
    if (
        route_meta != run_meta
        or route_meta.get("run_id") != expected_child_name
        or route_meta.get("route_entry_id") != expected_route_entry
        or route_meta.get("route_id") != f"RouteScenario_{binding['route_id']}"
        or route_meta.get("town_name") != binding["town"]
        or route_meta.get("scenario_name") != f"{binding['scenario']}_1"
        or not isinstance(public, dict)
        or public.get("track") != "native"
        or public.get("seed") != run["seed"]
        or route_meta.get("private_capture_provenance") not in ({}, None)
        or type(summary.get("total_frames_logged")) is not int
        or summary["total_frames_logged"] != len(rows_by_name["trace.jsonl"])
        or summary.get("frame_interval") != 1
        or summary.get("logger_error_count") != 0
        or "capture_complete" in summary
        or "contract_success" in summary
        or capabilities.get("schema_version") != "language2drive.trace_capture.v1"
        or capabilities.get("private_context_loaded") is not False
        or capabilities.get("logger_error_count") != 0
    ):
        raise RuntimeError("FDQ evaluator metadata/capture authority differs")
    trace_stat = (child / "trace.jsonl").stat()
    frames_stat = (child / "frames.jsonl").stat()
    if (
        trace_stat.st_dev != frames_stat.st_dev
        or trace_stat.st_ino != frames_stat.st_ino
        or trace_stat.st_size != frames_stat.st_size
    ):
        raise RuntimeError("FDQ trace/frames compatibility evidence differs")
    for artifact in artifacts:
        os.replace(artifact, run_dir / artifact.name)
    child.rmdir()
    alias_root.rmdir()


def _deduplicate_trace_alias(run_dir: Path) -> dict[str, Any]:
    """Publish frames.jsonl as trace.jsonl's hardlink only after exact proof."""

    root = Path(run_dir)
    trace_path = root / "trace.jsonl"
    frames_path = root / "frames.jsonl"
    metadata_path = root / "trace_storage.json"
    metadata: dict[str, Any] = {
        "schema_version": "language2drive.trace_storage.v1",
        "canonical": "trace.jsonl",
        "compatibility_alias": "frames.jsonl",
        "verified_at": _utc_now(),
        "projection_satisfied": False,
        "bytes_saved": 0,
    }
    if not trace_path.is_file():
        metadata.update({"status": "canonical_trace_missing"})
        atomic_json(metadata_path, metadata)
        return metadata
    trace_stat = trace_path.stat()
    trace_hash = sha256_file(trace_path)
    metadata.update({
        "trace_size_bytes": trace_stat.st_size,
        "trace_sha256": trace_hash,
    })
    if not frames_path.is_file():
        metadata.update({
            "status": "single_canonical_stream",
            "projection_satisfied": True,
            "physical_trace_bytes": trace_stat.st_size,
        })
        atomic_json(metadata_path, metadata)
        return metadata
    frames_stat = frames_path.stat()
    if (
        trace_stat.st_dev == frames_stat.st_dev
        and trace_stat.st_ino == frames_stat.st_ino
    ):
        metadata.update({
            "status": "already_hardlinked",
            "projection_satisfied": True,
            "frames_sha256": trace_hash,
            "frames_size_bytes": frames_stat.st_size,
            "physical_trace_bytes": trace_stat.st_size,
        })
        atomic_json(metadata_path, metadata)
        return metadata
    metadata["frames_size_bytes"] = frames_stat.st_size
    if trace_stat.st_size != frames_stat.st_size:
        metadata.update({
            "status": "not_deduplicated_size_mismatch",
            "physical_trace_bytes": trace_stat.st_size + frames_stat.st_size,
        })
        atomic_json(metadata_path, metadata)
        return metadata
    frames_hash = sha256_file(frames_path)
    metadata["frames_sha256"] = frames_hash
    if frames_hash != trace_hash:
        metadata.update({
            "status": "not_deduplicated_hash_mismatch",
            "physical_trace_bytes": trace_stat.st_size + frames_stat.st_size,
        })
        atomic_json(metadata_path, metadata)
        return metadata

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".frames.jsonl.link.", dir=str(root)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        os.link(trace_path, temporary)
        linked = temporary.stat()
        if (
            linked.st_dev != trace_stat.st_dev
            or linked.st_ino != trace_stat.st_ino
            or linked.st_size != trace_stat.st_size
        ):
            raise OSError("temporary trace compatibility hardlink identity mismatch")
        os.replace(temporary, frames_path)
        _fsync_directory(root)
        published = frames_path.stat()
        if (
            published.st_dev != trace_stat.st_dev
            or published.st_ino != trace_stat.st_ino
            or published.st_size != trace_stat.st_size
        ):
            raise OSError("published trace compatibility hardlink identity mismatch")
    except OSError as error:
        temporary.unlink(missing_ok=True)
        metadata.update({
            "status": "hardlink_error",
            "error": f"{type(error).__name__}: {error}",
            "physical_trace_bytes": trace_stat.st_size + frames_stat.st_size,
        })
        atomic_json(metadata_path, metadata)
        return metadata
    metadata.update({
        "status": "deduplicated_verified_equal",
        "projection_satisfied": True,
        "bytes_saved": frames_stat.st_size,
        "physical_trace_bytes": trace_stat.st_size,
    })
    atomic_json(metadata_path, metadata)
    return metadata



def execute_plan(
    plan: dict[str, Any],
    *,
    project_root: Path = L2D_ROOT,
    results_dir: Path,
    state_path: Path | None = None,
    require_server: bool = True,
    wall_timeout_seconds: float | None = None,
    validation_mode: str = "strict",
) -> dict[str, Any]:
    """Execute jobs serially and persist state after every attempt."""

    if validation_mode not in {"strict", "session"}:
        raise ValueError("validation_mode must be 'strict' or 'session'")
    _validate_plan_sha256(plan)
    input_hashes = plan.get("input_hashes", {})
    results_dir = _validate_results_dir(project_root, results_dir)
    state_path = _validated_state_path(
        results_dir, state_path, plan_sha256=str(plan.get("plan_sha256", "")),
    )
    storage_policy = _validate_storage_plan(plan)
    seen_run_ids: set[str] = set()
    for job in plan.get("jobs", []):
        if input_hashes and job.get("input_hashes") != input_hashes:
            raise ValueError("plan job execution bundle differs from plan authority")
        command = job.get("command", [])
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("each plan job command must be a list of strings")
        _require_external_carla_flag(command)
        run = job.get("run")
        if not isinstance(run, dict):
            raise ValueError("each plan job run must be an object")
        run_id = _require_safe_run_id(run.get("run_id"), "plan job run_id")
        if run_id in seen_run_ids:
            raise ValueError("plan job run_id is duplicated")
        seen_run_ids.add(run_id)

    host, port, traffic_manager_port = _plan_carla_endpoint(plan)
    with _carla_execution_lock(
        project_root=project_root,
        host=host,
        port=port,
        traffic_manager_port=traffic_manager_port,
        plan_sha256=plan.get("plan_sha256"),
    ):
        # Rebuild the entire bundle under the execution lock immediately
        # before any run artifact is written.  This catches drift in the XML,
        # sidecar, agent/config, evaluator, transitive Python/data closure,
        # CARLA client, and packaged runtime assets—not merely the runtime
        # manifest summary.
        _validate_plan_execution_authority(
            plan, project_root=project_root, results_dir=results_dir,
        )
        summary = _execute_plan_locked(
            plan,
            results_dir=results_dir,
            state_path=state_path,
            require_server=require_server,
            wall_timeout_seconds=wall_timeout_seconds,
            host=host,
            port=port,
            traffic_manager_port=traffic_manager_port,
            storage_policy=storage_policy,
            validation_mode=validation_mode,
        )
        if summary.get("status") == "completed":
            try:
                if validation_mode == "strict":
                    _validate_plan_execution_authority(
                        plan, project_root=project_root, results_dir=results_dir,
                    )
                else:
                    _validate_fast_session_nonasset_authority(plan)
                    _validate_fast_session_runtime_authority(plan, host, port)
            except (RuntimeError, ValueError, OSError) as error:
                final_state_path = state_path
                state = _load_state(final_state_path)
                state["execution"] = {
                    "status": "execution_halted",
                    "reason": "final_authority_mismatch",
                    "detail": str(error),
                    "halted_runs": 0,
                    "updated_at": _utc_now(),
                }
                state["updated_at"] = _utc_now()
                atomic_json(final_state_path, state)
                return {
                    **summary, "status": "execution_halted",
                    "reason": "final_authority_mismatch",
                    "state_path": str(final_state_path),
                }
        return summary


def _execute_plan_locked(
    plan: dict[str, Any],
    *,
    results_dir: Path,
    state_path: Path | None,
    require_server: bool,
    wall_timeout_seconds: float | None,
    host: str,
    port: int,
    traffic_manager_port: int,
    storage_policy: dict[str, Any],
    validation_mode: str,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    results_metadata = results_dir.lstat()
    if results_dir.is_symlink() or not stat.S_ISDIR(results_metadata.st_mode):
        raise ValueError("results root must be a real directory")
    state_path = _validated_state_path(
        results_dir, state_path, plan_sha256=str(plan.get("plan_sha256", "")),
    )
    state_path.parent.mkdir(parents=False, exist_ok=True, mode=0o700)
    state_root_metadata = state_path.parent.lstat()
    if (
        state_path.parent.is_symlink()
        or not stat.S_ISDIR(state_root_metadata.st_mode)
        or state_root_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(state_root_metadata.st_mode) & 0o077
    ):
        raise ValueError("runner state directory must be private real storage")
    jobs = plan.get("jobs", [])
    current_run_ids = {job["run"]["run_id"] for job in jobs}
    if state_path.exists():
        try:
            state = _strict_json_object_file(state_path, "runner state")
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError("existing runner state is invalid") from error
        if (
            state.get("schema_version") != "1.0"
            or state.get("plan_sha256") != plan.get("plan_sha256")
            or not isinstance(state.get("runs"), dict)
            or not set(state["runs"]) <= current_run_ids
        ):
            raise ValueError("existing runner state belongs to another plan")
    else:
        state = {"schema_version": "1.0", "runs": {}}
    state.update({"updated_at": _utc_now(), "plan_sha256": plan.get("plan_sha256")})
    initial_storage = _storage_snapshot(
        results_dir, jobs[0] if jobs else None, storage_policy
    )
    if not initial_storage["sufficient"]:
        state["execution"] = {
            "status": "execution_halted",
            "reason": "storage_budget_exhausted",
            "phase": "pre_execution",
            "halted_runs": len(jobs),
            "storage": initial_storage,
            "updated_at": _utc_now(),
        }
        state["updated_at"] = _utc_now()
        atomic_json(state_path, state)
        return {
            "planned": len(jobs),
            "completed": 0,
            "failed": 0,
            "halted": len(jobs),
            "status": "execution_halted",
            "reason": "storage_budget_exhausted",
            "storage": initial_storage,
            "state_path": str(state_path),
        }
    if require_server:
        prerequisite = plan.get("carla_server_prerequisite", "")
        if not carla_server_available(host, port):
            raise RuntimeError(
                f"{prerequisite}. Start the existing CarlaUE4 server before using --execute."
            )
    state["execution"] = {
        "status": "running",
        "reason": None,
        "host": host,
        "port": port,
        "traffic_manager_port": traffic_manager_port,
        "validation_mode": validation_mode,
        "started_at": _utc_now(),
    }
    completed = failed = 0
    wall_timeout_policy = plan.get("wall_timeout_policy")
    if wall_timeout_policy is None:
        # Backward-compatible execution of an older saved plan still receives
        # the safe policy inferred from its recorded agent/workload.
        wall_timeout_policy = _wall_timeout_policy(
            agent=str(plan.get("agent", "unknown")),
            workload=str(plan.get("workload", "unknown")),
    )
    for job_index, job in enumerate(jobs):
        try:
            if validation_mode == "strict":
                _validate_nonasset_execution_authority(plan)
            else:
                _validate_fast_session_nonasset_authority(plan)
        except (RuntimeError, ValueError, OSError) as error:
            halted = len(jobs) - job_index
            state["execution"] = {
                "status": "execution_halted",
                "reason": "code_authority_mismatch",
                "detail": str(error), "halted_runs": halted,
                "updated_at": _utc_now(),
            }
            state["updated_at"] = _utc_now()
            atomic_json(state_path, state)
            return {
                "planned": len(jobs), "completed": completed, "failed": failed,
                "halted": halted, "status": "execution_halted",
                "reason": "code_authority_mismatch", "state_path": str(state_path),
            }
        if require_server and not carla_server_available(host, port):
            halted = len(jobs) - job_index
            state["execution"] = {
                "status": "execution_halted",
                "reason": "server_unavailable",
                "host": host,
                "port": port,
                "traffic_manager_port": traffic_manager_port,
                "halted_runs": halted,
                "updated_at": _utc_now(),
            }
            state["updated_at"] = _utc_now()
            atomic_json(state_path, state)
            return {
                "planned": len(jobs),
                "completed": completed,
                "failed": failed,
                "halted": halted,
                "status": "execution_halted",
                "reason": "server_unavailable",
                "state_path": str(state_path),
            }
        try:
            if validation_mode == "strict":
                live_runtime_witness = _validate_live_carla_runtime_authority(
                    plan, host, port,
                )
            else:
                live_runtime_witness = _validate_fast_session_runtime_authority(
                    plan, host, port,
                )
        except (RuntimeError, ValueError) as error:
            halted = len(jobs) - job_index
            state["execution"] = {
                "status": "execution_halted",
                "reason": "runtime_asset_authority_mismatch",
                "detail": str(error),
                "host": host,
                "port": port,
                "traffic_manager_port": traffic_manager_port,
                "halted_runs": halted,
                "updated_at": _utc_now(),
            }
            state["updated_at"] = _utc_now()
            atomic_json(state_path, state)
            return {
                "planned": len(jobs),
                "completed": completed,
                "failed": failed,
                "halted": halted,
                "status": "execution_halted",
                "reason": "runtime_asset_authority_mismatch",
                "state_path": str(state_path),
            }
        run = job["run"]
        run_id = run["run_id"]
        job_storage = _storage_snapshot(results_dir, job, storage_policy)
        if not job_storage["sufficient"]:
            halted = len(jobs) - job_index
            state["execution"] = {
                "status": "execution_halted",
                "reason": "storage_budget_exhausted",
                "phase": "before_job",
                "next_run_id": run_id,
                "halted_runs": halted,
                "storage": job_storage,
                "updated_at": _utc_now(),
            }
            state["updated_at"] = _utc_now()
            atomic_json(state_path, state)
            return {
                "planned": len(jobs),
                "completed": completed,
                "failed": failed,
                "halted": halted,
                "status": "execution_halted",
                "reason": "storage_budget_exhausted",
                "storage": job_storage,
                "state_path": str(state_path),
            }
        run_dir = results_dir / run_id
        if run_dir.exists() or run_dir.is_symlink():
            halted = len(jobs) - job_index
            state["execution"] = {
                "status": "execution_halted",
                "reason": "existing_run_directory_requires_archive",
                "next_run_id": run_id, "halted_runs": halted,
                "updated_at": _utc_now(),
            }
            state["updated_at"] = _utc_now()
            atomic_json(state_path, state)
            return {
                "planned": len(jobs), "completed": completed, "failed": failed,
                "halted": halted, "status": "execution_halted",
                "reason": "existing_run_directory_requires_archive",
                "state_path": str(state_path),
            }
        run_dir.mkdir(parents=False, exist_ok=False, mode=0o700)
        environment_root = _new_opaque_evaluator_attempt_root()
        environment = evaluator_environment(
            environment_root,
            prepare=True,
            gpu_rank=plan["execution_authority"]["gpu_rank"],
        )
        _validate_evaluator_environment_non_disclosure(environment, run)
        wall_timeout = _job_wall_timeout(
            job, wall_timeout_seconds, wall_timeout_policy
        )
        atomic_json(run_dir / "runner_command.json", {
            "run": run,
            "command": job["command"],
            "started_at": _utc_now(),
            "plan_sha256": plan.get("plan_sha256"),
            "input_hashes": job.get("input_hashes", {}),
            "wall_timeout_policy": wall_timeout_policy,
            "wall_timeout_seconds": wall_timeout,
            "wall_timeout_operator_override": wall_timeout_seconds is not None,
            "storage_policy": storage_policy,
            "storage_estimate": job["storage_estimate"],
            "storage_before_job": job_storage,
            "runtime_asset_live_witness": live_runtime_witness,
            "validation_mode": validation_mode,
            "execution_environment_root": str(environment_root),
            "execution_environment": environment,
            "execution_environment_sha256": stable_hash(environment),
        })
        state["execution"]["runtime_asset_live_witness"] = live_runtime_witness
        state["runs"][run_id] = {"status": "running", "updated_at": _utc_now()}
        atomic_json(state_path, state)
        launch_error: str | None = None
        evaluator_log_path = run_dir / "evaluator.stdout.log"
        log = _StreamingBoundedLog(evaluator_log_path)
        try:
            try:
                returncode, timed_out = _run_evaluator(
                    job["command"],
                    cwd=EVALUATOR_ROOT,
                    environment=environment,
                    log=log,
                    wall_timeout=wall_timeout,
                )
            except (OSError, RuntimeError) as error:
                returncode, timed_out = None, False
                launch_error = f"{type(error).__name__}: {error}"
                log.write(f"runner failed to launch evaluator: {launch_error}\n")
        finally:
            evaluator_log = log.finalize()
        post_authority_error: str | None = None
        try:
            if validation_mode == "strict":
                _validate_nonasset_execution_authority(plan)
                _validate_live_carla_runtime_authority(plan, host, port)
            else:
                _validate_fast_session_nonasset_authority(plan)
                _validate_fast_session_runtime_authority(plan, host, port)
        except (RuntimeError, ValueError, OSError) as error:
            post_authority_error = f"{type(error).__name__}: {error}"
        if post_authority_error is not None:
            result_path = run_dir / (
                "fdq_result.json" if plan.get("workload") == "fdq"
                else "contract_result.json"
            )
            if plan.get("workload") == "fdq":
                authority_result = _fdq_terminal_result(
                    run, "runtime_authority_failure",
                    "post_evaluator_authority_mismatch",
                    detail=post_authority_error,
                )
            else:
                authority_result = _core_terminal_result(
                    run, "runtime_authority_failure",
                    "post_evaluator_authority_mismatch",
                    detail=post_authority_error,
                )
            atomic_json(result_path, authority_result)
            failed += 1
            state["runs"][run_id] = {
                "status": "failed", "returncode": returncode,
                "timed_out": timed_out, "launch_error": launch_error,
                "result_path": str(result_path), "result_finalized": False,
                "updated_at": _utc_now(),
            }
            halted = len(jobs) - job_index - 1
            state["execution"] = {
                "status": "execution_halted",
                "reason": "post_evaluator_authority_mismatch",
                "detail": post_authority_error, "halted_runs": halted,
                "updated_at": _utc_now(),
            }
            state["updated_at"] = _utc_now()
            atomic_json(state_path, state)
            return {
                "planned": len(jobs), "completed": completed, "failed": failed,
                "halted": halted, "status": "execution_halted",
                "reason": "post_evaluator_authority_mismatch",
                "state_path": str(state_path),
            }
        # The evaluator derives its record directory from its information
        # track.  Oracle execution intentionally uses the native route track
        # internally, but the benchmark ledger namespace remains ``oracle``.
        # A per-job alias root makes the merge unambiguous and keeps it separate
        # from genuine Native-System smoke output.
        alias_root = run_dir / "_evaluator_record"
        capture_merge_error: str | None = None
        if (
            plan.get("workload") != "fdq"
            and returncode == 0
            and not timed_out
            and launch_error is None
        ):
            try:
                _merge_core_evaluator_capture(
                    alias_root, run_dir, run,
                    evaluator_track=plan["execution_authority"]["evaluator_track"],
                    private_sidecar_sha256=plan["input_hashes"][
                        "private_sidecar_sha256"
                    ],
                )
            except (OSError, UnicodeError, ValueError, KeyError, RuntimeError) as error:
                capture_merge_error = f"{type(error).__name__}: {error}"
        elif (
            plan.get("workload") == "fdq"
            and returncode == 0
            and not timed_out
            and launch_error is None
        ):
            try:
                _merge_fdq_evaluator_capture(alias_root, run_dir, run)
            except (OSError, UnicodeError, ValueError, KeyError, RuntimeError) as error:
                capture_merge_error = f"{type(error).__name__}: {error}"
        trace_storage = _deduplicate_trace_alias(run_dir)
        if plan.get("workload") == "fdq":
            if launch_error is not None:
                result_path = run_dir / "fdq_result.json"
                atomic_json(result_path, _fdq_terminal_result(
                    run, "crash", "runner_launch_error",
                    launch_error=launch_error,
                ))
            elif timed_out:
                result_path = run_dir / "fdq_result.json"
                atomic_json(result_path, _fdq_terminal_result(
                    run, "timeout", "runner_wall_timeout",
                    wall_timeout_seconds=wall_timeout,
                ))
            elif capture_merge_error is not None:
                result_path = run_dir / "fdq_result.json"
                atomic_json(result_path, _fdq_terminal_result(
                    run, "invalid_result", "evaluator_capture_identity_mismatch",
                    detail=capture_merge_error,
                ))
            else:
                result_path = _finalize_fdq_result(run, run_dir, int(returncode))
        else:
            result_path = run_dir / "contract_result.json"
            if launch_error is not None:
                atomic_json(result_path, _core_terminal_result(
                    run, "crash", "runner_launch_error",
                    launch_error=launch_error,
                ))
            elif timed_out:
                atomic_json(result_path, _core_terminal_result(
                    run, "timeout", "runner_wall_timeout",
                    wall_timeout_seconds=wall_timeout,
                ))
            elif capture_merge_error is not None:
                atomic_json(result_path, _core_terminal_result(
                    run, "capture_failure", "evaluator_capture_identity_mismatch",
                    detail=capture_merge_error,
                ))
            elif returncode != 0 and not _result_is_final(result_path, run):
                atomic_json(result_path, _core_terminal_result(
                    run, "crash", "evaluator_nonzero_exit",
                    runner_returncode=returncode,
                ))
            elif not result_path.is_file():
                atomic_json(result_path, _core_terminal_result(
                    run, "capture_failure",
                    "evaluator_zero_exit_without_contract_result",
                ))
        finalized = _result_is_final(result_path, run)
        status = "completed" if finalized else "failed"
        completed += status == "completed"
        failed += status == "failed"
        state["runs"][run_id] = {
            "status": status,
            "returncode": returncode,
            "wall_timeout_seconds": wall_timeout,
            "timed_out": timed_out,
            "launch_error": launch_error,
            "evaluator_log": evaluator_log,
            "trace_storage": trace_storage,
            "result_path": str(result_path),
            "result_finalized": finalized,
            "updated_at": _utc_now(),
        }
        state["updated_at"] = _utc_now()
        atomic_json(state_path, state)
    state["execution"] = {
        "status": "completed",
        "reason": None,
        "host": host,
        "port": port,
        "traffic_manager_port": traffic_manager_port,
        "validation_mode": validation_mode,
        "completed_runs": completed,
        "failed_runs": failed,
        "halted_runs": 0,
        "updated_at": _utc_now(),
    }
    state["updated_at"] = _utc_now()
    atomic_json(state_path, state)
    return {
        "planned": len(jobs),
        "completed": completed,
        "failed": failed,
        "halted": 0,
        "status": "completed",
        "state_path": str(state_path),
    }


def _finite_point(point: Any) -> bool:
    return isinstance(point, dict) and all(
        isinstance(point.get(axis, 0), (int, float))
        and not isinstance(point.get(axis, 0), bool)
        and math.isfinite(float(point.get(axis, 0)))
        for axis in ("x", "y", "z")
    )


def _deadline_expected_input_hashes(project_root: Path) -> dict[str, Any]:
    root = Path(project_root)
    return expected_input_hashes(
        sidecar=root / "build/suite/sidecar.private.json",
        route_xml=root / "build/suite/routes_native.xml",
        agent_path=DEFAULT_ORACLE_AGENT,
        agent_config=root / "config/defaults.json",
        benchmark_track="oracle",
        evaluator_track="native",
        workload="core",
        dependency_profile="base-runtime.v1",
        wall_timeout_policy=_wall_timeout_policy(agent="oracle", workload="core"),
    )


def _deadline_witness_provenance(
    run: dict[str, Any],
    *,
    results_dir: Path,
    expected_bundle: dict[str, Any],
    base_sidecar_sha256: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    run_id = str(run["run_id"])
    run_dir = Path(results_dir) / run_id
    paths = {
        "result": run_dir / "contract_result.json",
        "trace": run_dir / "trace.jsonl",
        "runner_command": run_dir / "runner_command.json",
        "run_meta": run_dir / "run_meta.json",
    }
    errors: list[str] = []
    values: dict[str, Any] = {}
    for name in ("result", "runner_command", "run_meta"):
        try:
            values[name] = load_json(paths[name])
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{run_id}: invalid or missing {name}: {type(error).__name__}")
    if not paths["trace"].is_file() or paths["trace"].stat().st_size <= 0:
        errors.append(f"{run_id}: missing non-empty trace.jsonl")
    command = values.get("runner_command", {})
    if command.get("input_hashes") != expected_bundle:
        errors.append(f"{run_id}: runner input bundle does not match the frozen oracle bundle")
    command_run = command.get("run", {})
    for key in ("run_id", "route_entry_id", "pcu_id", "arm_id", "paraphrase", "seed", "track"):
        if command_run.get(key) != run.get(key):
            errors.append(f"{run_id}: runner_command run identity mismatch for {key}")
    result = values.get("result", {})
    if result.get("status") != "complete" or result.get("success") is not True:
        errors.append(f"{run_id}: witness result is not a successful complete verdict")
    for key in ("route_entry_id", "pcu_id", "arm_id"):
        if key in result and result.get(key) != run.get(key):
            errors.append(f"{run_id}: result identity mismatch for {key}")
    provenance = values.get("run_meta", {}).get("private_capture_provenance", {})
    if provenance.get("route_entry_id") != run.get("route_entry_id"):
        errors.append(f"{run_id}: run_meta route_entry_id mismatch")
    if provenance.get("private_sidecar_sha256") != base_sidecar_sha256:
        errors.append(f"{run_id}: run_meta is not bound to the base sidecar")
    if errors:
        return None, errors
    return {
        "run_id": run_id,
        "pcu_id": run["pcu_id"],
        "arm_id": run["arm_id"],
        "seed": int(run["seed"]),
        "route_entry_id": int(run["route_entry_id"]),
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }, []


def _score_deadline_witness(
    run: dict[str, Any],
    *,
    project_root: Path,
    results_dir: Path,
    sidecar: dict[str, Any],
    target_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    from .scoring import score_one

    return score_one(
        run,
        project_root=project_root,
        results_dir=results_dir,
        sidecar=sidecar,
        target_index=target_index,
        write_result=False,
    )


def _build_deadline_calibration(
    project_root: Path,
    results_dir: Path,
    *,
    seeds: Iterable[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Build a complete candidate without mutating results or suite data."""

    from .scoring import _targets_for_suite

    root = Path(project_root)
    results = Path(results_dir)
    seed_set = {int(seed) for seed in seeds}
    if seed_set != {1, 2, 3}:
        raise ValueError("deadline calibration requires exactly oracle seeds 1,2,3")
    build_dir = root / "build/suite"
    ledger_path = build_dir / "oracle_expected_runs.json"
    sidecar_path = build_dir / "sidecar.private.json"
    manifest_path = build_dir / "manifest.json"
    provenance_path = build_dir / "provenance.json"
    runs = _read_runs(ledger_path)
    sidecar = load_json(sidecar_path)
    target_index = _targets_for_suite(root, sidecar)
    expected_bundle = _deadline_expected_input_hashes(root)
    base_sidecar_sha256 = sha256_file(sidecar_path)
    arm_deadlines = {
        (str(pcu["id"]), str(arm["id"])): float(arm["deadline_seconds"])
        for pcu in sidecar.get("pcus", [])
        for arm in pcu.get("arms", [])
    }
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    errors: list[str] = []
    for run in runs:
        if run.get("track") != "oracle" or int(run.get("paraphrase", -1)) != 0:
            errors.append(f"unexpected oracle calibration identity: {run.get('run_id')}")
            continue
        key = (str(run["pcu_id"]), str(run["arm_id"]))
        seed = int(run["seed"])
        if seed in grouped.setdefault(key, {}):
            errors.append(f"duplicate oracle calibration seed: {key[0]}/{key[1]}/s{seed}")
        grouped[key][seed] = run
    if len(grouped) != 256 or len(runs) != 768 or len(arm_deadlines) != 256:
        errors.append(
            "oracle calibration suite must contain exactly 256 arms/768 runs, "
            f"got {len(arm_deadlines)}/{len(runs)}"
        )

    completion_by_run: dict[str, float] = {}
    witness_by_run: dict[str, dict[str, Any]] = {}
    for run in runs:
        provenance, provenance_errors = _deadline_witness_provenance(
            run,
            results_dir=results,
            expected_bundle=expected_bundle,
            base_sidecar_sha256=base_sidecar_sha256,
        )
        errors.extend(provenance_errors)
        if provenance is None:
            continue
        rescored = _score_deadline_witness(
            run,
            project_root=root,
            results_dir=results,
            sidecar=sidecar,
            target_index=target_index,
        )
        completion = rescored.get("completion_time") if isinstance(rescored, dict) else None
        valid = (
            isinstance(rescored, dict)
            and rescored.get("status") == "complete"
            and rescored.get("success") is True
            and isinstance(completion, (int, float))
            and not isinstance(completion, bool)
            and math.isfinite(float(completion))
            and float(completion) >= 0
        )
        if not valid:
            errors.append(f"{run['run_id']}: base-sidecar pure trace rescore did not pass")
            continue
        completion_by_run[run["run_id"]] = float(completion)
        provenance["completion_seconds"] = float(completion)
        witness_by_run[run["run_id"]] = provenance

    entries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        by_seed = grouped[key]
        if set(by_seed) != seed_set:
            errors.append(f"{key[0]}/{key[1]} does not contain exactly seeds 1,2,3")
            continue
        times = {
            str(seed): completion_by_run[by_seed[seed]["run_id"]]
            for seed in sorted(seed_set)
            if by_seed[seed]["run_id"] in completion_by_run
        }
        if len(times) != 3:
            continue
        maximum = max(times.values())
        calibrated = int(math.ceil(1.5 * maximum + 5.0))
        previous = arm_deadlines.get(key)
        if previous is None:
            errors.append(f"{key[0]}/{key[1]} missing base arm deadline")
            continue
        entries.append({
            "pcu_id": key[0],
            "arm_id": key[1],
            "seed_completion_seconds": times,
            "max_completion_seconds": maximum,
            "deadline_seconds": calibrated,
            "previous_deadline_seconds": previous,
        })
    valid_runs = len(completion_by_run)
    passed = not errors and len(entries) == 256 and valid_runs == 768
    source = {
        "oracle_ledger": str(ledger_path),
        "oracle_ledger_sha256": sha256_file(ledger_path),
        "base_sidecar": str(sidecar_path),
        "base_sidecar_sha256": base_sidecar_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_sha256": sha256_file(provenance_path),
        "results_dir": str(results.resolve()),
        "expected_input_hashes": expected_bundle,
    }
    retained_entries = entries if passed else []
    retained_witnesses = [witness_by_run[run["run_id"]] for run in runs] if passed else []
    calibration_id = stable_hash({
        "formula": DEADLINE_FORMULA,
        "source": source,
        "entries": retained_entries,
        "witnesses": retained_witnesses,
    }) if passed else None
    return {
        "schema_version": "language2drive.deadline_calibration.v1",
        "generated_at": _utc_now(),
        "status": "passed" if passed else "blocked_pending_oracle_witnesses",
        "formula": DEADLINE_FORMULA,
        "calibration_id": calibration_id,
        "mutates_current_suite": False,
        "required_runs": 768,
        "valid_runs": valid_runs,
        "expected_arms": 256,
        "calibrated_arms": len(retained_entries),
        "entries": retained_entries,
        "witnesses": retained_witnesses,
        "errors": errors,
        "source": source,
    }


def calibrate_deadlines(
    project_root: Path,
    results_dir: Path,
    output: Path,
    *,
    seeds: Iterable[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Write only a complete deadline-freeze candidate; blocked scans are read-only."""

    value = _build_deadline_calibration(project_root, results_dir, seeds=seeds)
    if value.get("status") == "passed":
        atomic_json(output, value)
    else:
        # A stale blocked artifact from pre-v1 fail-closed behavior must not
        # masquerade as the current invocation's candidate.  Passed evidence
        # is never removed here: it remains auditable and freeze will fully
        # revalidate it before any suite mutation.
        try:
            previous = load_json(output)
        except FileNotFoundError:
            previous = None
        if isinstance(previous, dict) and previous.get("status") != "passed":
            Path(output).unlink(missing_ok=True)
            _fsync_directory(Path(output).parent)
    return value


def _commit_deadline_artifacts(
    project_root: Path,
    derived: dict[str, Any],
    overlay: dict[str, Any],
) -> None:
    """Commit sidecar first and overlay last, with rollback on any failure."""

    root = Path(project_root)
    _, derived_path, overlay_path = _deadline_artifact_paths(root)
    staging_root = root / ".runtime/.tmp"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="deadline-freeze-", dir=staging_root))
    staged_derived = staging / derived_path.name
    staged_overlay = staging / overlay_path.name
    atomic_json(staged_derived, derived, mode=0o600)
    atomic_json(staged_overlay, overlay, mode=0o600)
    backup_derived = staging / "previous-sidecar.private.json"
    backup_overlay = staging / "previous-deadlines.private.json"
    had_derived = derived_path.is_file()
    had_overlay = overlay_path.is_file()
    installed_derived = False
    installed_overlay = False
    try:
        if had_derived:
            shutil.copy2(derived_path, backup_derived)
        if had_overlay:
            shutil.copy2(overlay_path, backup_overlay)
        os.replace(staged_derived, derived_path)
        installed_derived = True
        _fsync_directory(derived_path.parent)
        # Keep any previous overlay in place until this final atomic rename.
        # During the narrow sidecar-first window, readers see a hash mismatch
        # and fail closed; they never fall back to the base policy.  The
        # overlay rename is therefore the single activation commit point.
        os.replace(staged_overlay, overlay_path)
        installed_overlay = True
        _fsync_directory(overlay_path.parent)
    except BaseException:
        # Restore both halves best-effort before re-raising the original
        # exception.  Readers fail closed throughout any mismatch window.
        try:
            if installed_overlay:
                if had_overlay:
                    os.replace(backup_overlay, overlay_path)
                else:
                    overlay_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            if installed_derived:
                if had_derived:
                    os.replace(backup_derived, derived_path)
                else:
                    derived_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            _fsync_directory(derived_path.parent)
        except OSError:
            pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def freeze_deadlines(
    project_root: Path,
    calibration_path: Path,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Atomically freeze a fully revalidated 768-witness calibration."""

    from .scoring import _targets_for_suite

    root = Path(project_root)
    existing = deadline_freeze_status(root)
    if existing["valid"] and not replace:
        return {**existing, "status": "already_frozen", "changed": False}
    calibration = load_json(calibration_path)
    if (
        calibration.get("schema_version") != "language2drive.deadline_calibration.v1"
        or calibration.get("status") != "passed"
        or calibration.get("valid_runs") != 768
        or calibration.get("calibrated_arms") != 256
    ):
        raise ValueError("deadline freeze requires a complete passed 768-witness calibration")
    results_dir = Path(str(calibration.get("source", {}).get("results_dir", "")))
    revalidated = _build_deadline_calibration(root, results_dir)
    if revalidated.get("status") != "passed":
        raise ValueError("deadline calibration is no longer complete/current; suite was not modified")
    if calibration.get("calibration_id") != revalidated.get("calibration_id"):
        raise ValueError("deadline calibration content/provenance changed; suite was not modified")

    base_path, derived_path, overlay_path = _deadline_artifact_paths(root)
    base = load_json(base_path)
    entries = revalidated["entries"]
    derived = _derive_deadline_sidecar(base, entries)
    target_index = _targets_for_suite(root, derived)
    ledger_runs = _read_runs(root / "build/suite/oracle_expected_runs.json")
    rescored_rows: list[dict[str, Any]] = []
    for run in ledger_runs:
        rescored = _score_deadline_witness(
            run,
            project_root=root,
            results_dir=results_dir,
            sidecar=derived,
            target_index=target_index,
        )
        if not isinstance(rescored, dict) or rescored.get("status") != "complete" or rescored.get("success") is not True:
            raise ValueError(
                f"{run['run_id']}: trace does not pass under calibrated deadline; suite was not modified"
            )
        rescored_rows.append({
            "run_id": run["run_id"],
            "completion_seconds": rescored.get("completion_time"),
            "deadline_seconds": rescored.get("deadline_seconds"),
            "success": True,
        })

    staging_root = root / ".runtime/.tmp"
    staging_root.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(prefix="deadline-sidecar-hash-", dir=staging_root)
    os.close(descriptor)
    staged_path = Path(staged_name)
    try:
        atomic_json(staged_path, derived, mode=0o600)
        derived_hash = sha256_file(staged_path)
    finally:
        staged_path.unlink(missing_ok=True)
    source = {
        "base_sidecar_sha256": sha256_file(base_path),
        "oracle_ledger_sha256": sha256_file(root / "build/suite/oracle_expected_runs.json"),
        "manifest_sha256": sha256_file(root / "build/suite/manifest.json"),
        "provenance_sha256": sha256_file(root / "build/suite/provenance.json"),
        "expected_input_hashes": revalidated["source"]["expected_input_hashes"],
        "calibration": str(Path(calibration_path).resolve()),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_id": revalidated["calibration_id"],
        "results_dir": str(results_dir.resolve()),
    }
    new_deadline_rescore = {
        "policy": "pure ContractEngine trace rescore; no result files mutated during freeze",
        "valid_runs": len(rescored_rows),
        "all_passed": len(rescored_rows) == 768,
        "rows_sha256": stable_hash(rescored_rows),
    }
    overlay = {
        "schema_version": "language2drive.deadline_freeze.v1",
        "created_at": _utc_now(),
        "status": "frozen",
        "formula": DEADLINE_FORMULA,
        "required_runs": 768,
        "valid_runs": 768,
        "expected_arms": 256,
        "calibrated_arms": 256,
        "deadline_change_policy": "exact_formula_then_all_trace_rescore",
        "base_witness_sidecar_retained": str(base_path),
        "source": source,
        "entries": entries,
        "witnesses": revalidated["witnesses"],
        "derived_sidecar": {
            "path": str(derived_path),
            "sha256": derived_hash,
            "only_mutation": "256 arm and 768 route deadline_seconds fields",
        },
        "new_deadline_rescore": new_deadline_rescore,
    }
    overlay["freeze_id"] = stable_hash({
        "formula": overlay["formula"],
        "source": source,
        "entries": entries,
        "witnesses": overlay["witnesses"],
        "derived_sidecar_sha256": derived_hash,
        "new_deadline_rescore": new_deadline_rescore,
    })
    _commit_deadline_artifacts(root, derived, overlay)
    committed = deadline_freeze_status(root)
    if not committed["valid"]:
        raise RuntimeError("deadline freeze commit validation failed: " + "; ".join(committed["errors"]))
    return {
        **committed,
        "changed": True,
        "base_sidecar_sha256": sha256_file(base_path),
        "derived_sidecar_sha256": sha256_file(derived_path),
        "overlay_sha256": sha256_file(overlay_path),
    }


def _point_in_polygon(point: tuple[float, float], polygon: Any) -> bool:
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    j = len(vertices) - 1
    for i, vertex in enumerate(vertices):
        xi, yi = float(vertex[0]), float(vertex[1])
        xj, yj = float(vertices[j][0]), float(vertices[j][1])
        if (yi > y) != (yj > y) and x <= (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _path_hits_target(path: list[dict[str, Any]], target: dict[str, Any]) -> bool | None:
    if isinstance(target.get("polygon"), list):
        return any(_point_in_polygon((float(point["x"]), float(point["y"])), target["polygon"]) for point in path)
    center = target.get("center")
    radius = target.get("radius_m")
    if isinstance(center, dict) and isinstance(radius, (int, float)):
        return any(
            math.hypot(float(point["x"]) - float(center["x"]), float(point["y"]) - float(center["y"]))
            <= float(radius)
            for point in path
        )
    return None


def _gate_ids(expression: Any, bindings: dict[str, str] | None = None) -> set[str]:
    bindings = {} if bindings is None else bindings
    output: set[str] = set()
    if isinstance(expression, dict):
        binding = expression.get("bind_reference")
        if isinstance(binding, dict) and isinstance(binding.get("reference"), str) and isinstance(binding.get("value"), str):
            bindings[binding["reference"]] = binding["value"]
        gate = expression.get("enter_gate")
        if isinstance(gate, dict):
            identifier = gate.get("id")
            if isinstance(identifier, str):
                output.add(bindings.get(identifier.lstrip("$"), identifier))
            elif isinstance(identifier, dict) and isinstance(identifier.get("$ref"), str):
                resolved = bindings.get(identifier["$ref"])
                if resolved:
                    output.add(resolved)
        for value in expression.values():
            output.update(_gate_ids(value, bindings))
    elif isinstance(expression, list):
        for value in expression:
            output.update(_gate_ids(value, bindings))
    return output


def _target_geometry_index(sidecar: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("id"), str) and value.get("type") in {"gate", "region"}:
                if isinstance(value.get("polygon"), list) or isinstance(value.get("center"), dict):
                    output[value["id"]] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(sidecar.get("pcus", []))
    visit(sidecar.get("geometry_registry", {}))
    return output


def structural_qualification(project_root: Path, output: Path) -> dict[str, Any]:
    """Check static executability/provenance without claiming a CARLA witness."""

    build_dir = project_root / "build/suite"
    manifest_path = build_dir / "manifest.json"
    sidecar_path = build_dir / "sidecar.private.json"
    manifest = load_json(manifest_path)
    sidecar = load_json(sidecar_path)
    public_routes = manifest.get("routes", [])
    private_routes = sidecar.get("routes", [])
    geometry_index = _target_geometry_index(sidecar)
    errors: list[str] = []
    if len(public_routes) != 768 or len(private_routes) != 768:
        errors.append(f"route cardinality must be 768/768, got {len(public_routes)}/{len(private_routes)}")
    public_ids = [row.get("route_entry_id") for row in public_routes]
    private_ids = [row.get("route_entry_id") for row in private_routes]
    if len(set(public_ids)) != len(public_ids) or len(set(private_ids)) != len(private_ids):
        errors.append("public/private route_entry_id values must each be unique")
    if set(public_ids) != set(private_ids):
        errors.append("public/private route_entry_id sets differ")

    arm_rows = []
    target_checked = target_passed = 0
    opposite_checked = opposite_clear = 0
    for pcu in sidecar.get("pcus", []):
        starts = []
        pcu_arms = pcu.get("arms", [])
        for arm_index, arm in enumerate(pcu_arms):
            arm_errors = []
            path = arm.get("oracle_path")
            target_type = arm.get("target", {}).get("type")
            minimum_points = 1 if target_type == "hold" else 2
            if not isinstance(path, list) or len(path) < minimum_points or not all(_finite_point(point) for point in path):
                arm_errors.append("invalid_oracle_path")
                path = []
            if path:
                starts.append(path[0])
            ratio = arm.get("route_length_ratio")
            if ratio is not None and not 0.8 <= float(ratio) <= 1.25:
                arm_errors.append("route_length_ratio_outside_0.8_1.25")
            if arm.get("contract") == arm.get("opposite_contract"):
                arm_errors.append("own_and_opposite_contract_identical")
            # A legal lane-level oracle may intentionally end at the parking
            # aisle; the final off-lane low-speed parking maneuver is governed
            # by the footprint contract and requires a dynamic controller
            # witness.  Treating aisle centerline non-intersection as a static
            # topology failure would be a false negative.
            geometric_types = {"gate", "region"}
            terminal_state_contract = (
                target_type in geometric_types and "dwell" in json.dumps(arm.get("contract", {}), sort_keys=True)
            )
            target_hit = (
                _path_hits_target(path, arm.get("target", {}))
                if path and target_type in geometric_types
                else None
            )
            opposite_hit: bool | None = None
            own_gate_checks: dict[str, bool] = {}
            opposite_gate_checks: dict[str, bool] = {}
            if target_hit is not None:
                target_checked += 1
                target_passed += bool(target_hit)
                if not target_hit:
                    arm_errors.append("legal_path_does_not_intersect_geometric_target")
            if len(pcu_arms) == 2 and path and not terminal_state_contract:
                opposite_target = pcu_arms[1 - arm_index].get("target", {})
                if opposite_target.get("type") in geometric_types:
                    opposite_hit = _path_hits_target(path, opposite_target)
                    opposite_checked += 1
                    opposite_clear += opposite_hit is False
                    if opposite_hit:
                        arm_errors.append("legal_path_intersects_opposite_geometric_target")
            if path:
                own_gate_ids = _gate_ids(arm.get("contract", {}))
                opposite_gate_ids = _gate_ids(arm.get("opposite_contract", {}))
                for identifier in sorted(own_gate_ids):
                    if identifier not in geometry_index:
                        arm_errors.append(f"missing_own_contract_gate_geometry:{identifier}")
                        continue
                    hit = bool(_path_hits_target(path, geometry_index[identifier]))
                    own_gate_checks[identifier] = hit
                    if not hit:
                        arm_errors.append(f"legal_path_misses_own_contract_gate:{identifier}")
                # If own/opposite contain the same gates in a different strict
                # order, set intersection alone cannot determine the verdict;
                # those cases remain for ContractEngine/dynamic qualification.
                for identifier in sorted(opposite_gate_ids - own_gate_ids):
                    if identifier not in geometry_index:
                        arm_errors.append(f"missing_opposite_contract_gate_geometry:{identifier}")
                        continue
                    hit = bool(_path_hits_target(path, geometry_index[identifier]))
                    opposite_gate_checks[identifier] = hit
                    if hit:
                        arm_errors.append(f"legal_path_hits_opposite_contract_gate:{identifier}")
            arm_rows.append({
                "pcu_id": pcu.get("id"),
                "arm_id": arm.get("id"),
                "passed": not arm_errors,
                "errors": arm_errors,
                "geometric_target_check": target_hit,
                "opposite_geometric_target_check": opposite_hit,
                "own_contract_gate_checks": own_gate_checks,
                "opposite_contract_gate_checks": opposite_gate_checks,
            })
        if len(starts) == 2 and math.hypot(
            float(starts[0]["x"]) - float(starts[1]["x"]),
            float(starts[0]["y"]) - float(starts[1]["y"]),
        ) > 0.25:
            errors.append(f"{pcu.get('id')}: arms do not share the same start")

    failed_arms = [row for row in arm_rows if not row["passed"]]
    if failed_arms:
        errors.append(f"{len(failed_arms)} arms failed structural checks")
    value = {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "qualification_name": "oracle_path_structural_qualification",
        "is_dynamic_carla_rollout": False,
        "disclaimer": (
            "This checks finite directed oracle-path artifacts and target intersection only. "
            "It is not one of the required 768 dynamic privileged-oracle CARLA witness rollouts."
        ),
        "status": "passed" if not errors else "failed",
        "pcu_count": len(sidecar.get("pcus", [])),
        "arm_count": len(arm_rows),
        "route_entry_count": len(private_routes),
        "arms_passed": len(arm_rows) - len(failed_arms),
        "arms_failed": len(failed_arms),
        "geometric_targets_checked": target_checked,
        "geometric_targets_passed": target_passed,
        "opposite_geometric_targets_checked": opposite_checked,
        "opposite_geometric_targets_clear": opposite_clear,
        "errors": errors,
        "arms": arm_rows,
        "dynamic_oracle_status": "pending_execution",
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "sidecar": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
        },
    }
    atomic_json(output, value)
    return value
