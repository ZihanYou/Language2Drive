"""Exact installed-CARLA runtime asset inventory authority.

The route XML is deliberately not an installation manifest: it may contain
routes for maps that are intentionally excluded from a live run.  Instead we
freeze the exact regular-file inventory of the currently installed runtime.
Every plan resolution re-enumerates that scope and rehashes each file.  A
successful resolution may be passed directly into plan construction, but is
never cached across independent resolutions.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .io import atomic_json, sha256_file, stable_hash


SCHEMA_VERSION = "language2drive.carla_runtime_asset_manifest.v1"
METHOD = "exact_scoped_installed_carla_runtime_regular_files_sha256_v1"
PROVENANCE_SCHEMA_VERSION = (
    "language2drive.carla_runtime_asset_manifest_provenance.v1"
)
MAP_WITNESS_SCHEMA_VERSION = "language2drive.carla_available_maps_witness.v1"
INVENTORY_ROOTS = (
    "CarlaUE4/Content",
    "CarlaUE4/Config",
    "CarlaUE4/Plugins/Carla/Content",
    "CarlaUE4/Plugins/CarlaTools/Content",
    "CarlaUE4/Plugins/Converters/MDL/Content",
    "CarlaUE4/Plugins/Streetmap/Content",
    "CarlaUE4/Plugins/Streetmap/Config",
    "CarlaUE4/Saved/Config/Linux",
    "Engine",
)
REQUIRED_FILES = (
    "VERSION",
    "CarlaUE4.sh",
    "CarlaUE4/AssetRegistry.bin",
    "CarlaUE4/CarlaUE4.sh",
    "CarlaUE4/CarlaUE4.uproject",
    "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping",
    "CarlaUE4/Config/DefaultGame.ini",
    "CarlaUE4/Content/Carla/Config/AdditionalMaps.Package.json",
    "CarlaUE4/Plugins/Carla/Carla.uplugin",
    "CarlaUE4/Plugins/CarlaTools/CarlaTools.uplugin",
    "CarlaUE4/Plugins/Converters/MDL/MDL.uplugin",
    "CarlaUE4/Plugins/Converters/SimReady/SimReady.uplugin",
    "CarlaUE4/Plugins/Streetmap/StreetMap.uplugin",
    "PythonAPI/carla/dist/carla-0.9.16-cp311-cp311-linux_x86_64.whl",
)
REQUIRED_ABSENT_PATHS = (
    "ContentVersions.txt",
    "Util/ContentVersions.txt",
    "CarlaUE4/ContentVersions.txt",
    "CarlaUE4/Content/ContentVersions.txt",
)
ROOT_KEYS = {
    "schema_version", "method", "resolved_carla_root", "scope",
    "directories", "directories_sha256", "directory_count", "files",
    "files_sha256", "file_count", "total_size_bytes", "inventory_sha256",
}
SCOPE_KEYS = {"inventory_roots", "required_files", "required_absent_paths"}
FILE_KEYS = {"path", "size_bytes", "link_count", "sha256"}
PROVENANCE_KEYS = {
    "schema_version", "manifest_path", "manifest_sha256",
    "resolved_carla_root", "generator_path", "generator_sha256",
    "generated_at", "cold_generation_elapsed_seconds",
    "available_maps_witness", "available_maps_witness_sha256",
    "boot_id", "freeze_monotonic_ticks", "clock_ticks_per_second",
    "qualification_boundary",
}
MAP_WITNESS_KEYS = {
    "schema_version", "method", "resolved_carla_root", "captured_at",
    "client_version", "server_version", "available_maps",
    "available_maps_sha256",
}
HEX = set("0123456789abcdef")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _boot_clock_authority() -> tuple[str, int, int]:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        monotonic_ticks = int(time.clock_gettime(time.CLOCK_BOOTTIME) * ticks_per_second)
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(f"cannot capture Linux boot-clock authority: {error}") from error
    if not boot_id or ticks_per_second <= 0 or monotonic_ticks < 0:
        raise ValueError("Linux boot-clock authority is invalid")
    return boot_id, monotonic_ticks, ticks_per_second


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized POSIX relative path")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{label} is not canonical")
    return normalized


def _resolved_regular_root(carla_root: Path) -> Path:
    logical = Path(carla_root)
    resolved = logical.resolve(strict=True)
    metadata = resolved.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or resolved.is_symlink():
        raise ValueError("resolved CARLA runtime root must be a real directory")
    return resolved


def _checked_directory(root: Path, relative: str) -> Path:
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"CARLA runtime inventory directory is not a real directory: {relative}")
    return path


def _raise_walk_error(error: OSError) -> None:
    raise ValueError(f"cannot enumerate CARLA runtime inventory: {error}") from error


def _enumerate_inventory(
    resolved_root: Path,
    *,
    inventory_roots: tuple[str, ...],
    required_files: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    directories: set[str] = set()
    files: set[str] = set(required_files)
    for relative_root in inventory_roots:
        directory = _checked_directory(resolved_root, relative_root)
        for current_text, names, filenames in os.walk(
            directory, followlinks=False, onerror=_raise_walk_error,
        ):
            current = Path(current_text)
            relative_current = current.relative_to(resolved_root).as_posix()
            directories.add(relative_current)
            names.sort()
            filenames.sort()
            for name in names:
                child = current / name
                metadata = child.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(
                        "CARLA runtime inventory contains a non-directory entry "
                        f"where a directory is required: {child.relative_to(resolved_root)}"
                    )
            for name in filenames:
                path = current / name
                metadata = path.lstat()
                relative = path.relative_to(resolved_root).as_posix()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(
                        f"CARLA runtime inventory contains a non-regular file: {relative}"
                    )
                files.add(relative)
    for relative in required_files:
        _relative_path(relative, "required file")
        parent = PurePosixPath(relative).parent
        if str(parent) != ".":
            _checked_directory(resolved_root, parent.as_posix())
        metadata = (resolved_root / relative).lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"required CARLA runtime asset is not a regular file: {relative}")
    return sorted(directories), sorted(files)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_nlink,
    )


def _hash_regular_file(
    path: Path,
) -> tuple[int, int, str, tuple[int, int, int, int, int, int]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open CARLA runtime asset without following links: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"CARLA runtime asset is not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = _file_identity(before)
    identity_after = _file_identity(after)
    if identity_before != identity_after:
        raise ValueError(f"CARLA runtime asset changed while hashing: {path}")
    return before.st_size, before.st_nlink, digest.hexdigest(), identity_after


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open regular file without following links: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"expected a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns, before.st_nlink,
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns, after.st_nlink,
    )
    if identity_before != identity_after:
        raise ValueError(f"file changed while reading: {path}")
    return b"".join(chunks)


def _load_strict_json_bytes(raw: bytes, path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error


def _load_regular_json(path: Path) -> tuple[Any, str]:
    raw = _read_regular_bytes(Path(path))
    return _load_strict_json_bytes(raw, Path(path)), hashlib.sha256(raw).hexdigest()


def validate_available_maps_witness(value: Any, resolved_root: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != MAP_WITNESS_KEYS:
        raise ValueError("CARLA available-maps witness shape differs")
    if value["schema_version"] != MAP_WITNESS_SCHEMA_VERSION:
        raise ValueError("CARLA available-maps witness schema differs")
    if value["method"] != "carla_client_get_available_maps_v1":
        raise ValueError("CARLA available-maps witness method differs")
    if value["resolved_carla_root"] != str(resolved_root):
        raise ValueError("CARLA available-maps witness root differs")
    for key in ("captured_at", "client_version", "server_version"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"CARLA available-maps witness {key} must be nonempty")
    maps = value["available_maps"]
    if (
        not isinstance(maps, list)
        or not all(isinstance(item, str) and item for item in maps)
        or maps != sorted(set(maps))
    ):
        raise ValueError("CARLA available maps must be sorted, unique strings")
    if value["available_maps_sha256"] != stable_hash(maps):
        raise ValueError("CARLA available-maps witness hash differs")
    return value


def build_available_maps_witness(carla_root: Path, client: Any) -> dict[str, Any]:
    """Capture a read-only server inventory; this never starts/stops CARLA."""

    resolved = _resolved_regular_root(carla_root)
    raw_maps = client.get_available_maps()
    if (
        not isinstance(raw_maps, list)
        or not all(isinstance(item, str) and item for item in raw_maps)
        or len(set(raw_maps)) != len(raw_maps)
    ):
        raise ValueError("CARLA get_available_maps returned invalid or duplicate values")
    maps = sorted(raw_maps)
    value = {
        "schema_version": MAP_WITNESS_SCHEMA_VERSION,
        "method": "carla_client_get_available_maps_v1",
        "resolved_carla_root": str(resolved),
        "captured_at": _utc_now(),
        "client_version": str(client.get_client_version()),
        "server_version": str(client.get_server_version()),
        "available_maps": maps,
        "available_maps_sha256": stable_hash(maps),
    }
    return validate_available_maps_witness(value, resolved)


def _scope(
    inventory_roots: tuple[str, ...],
    required_files: tuple[str, ...],
    required_absent_paths: tuple[str, ...],
) -> dict[str, list[str]]:
    return {
        "inventory_roots": list(inventory_roots),
        "required_files": list(required_files),
        "required_absent_paths": list(required_absent_paths),
    }


def build_manifest(
    carla_root: Path,
    *,
    inventory_roots: tuple[str, ...] = INVENTORY_ROOTS,
    required_files: tuple[str, ...] = REQUIRED_FILES,
    required_absent_paths: tuple[str, ...] = REQUIRED_ABSENT_PATHS,
) -> dict[str, Any]:
    """Build a canonical manifest from a stable, no-symlink inventory."""

    resolved = _resolved_regular_root(carla_root)
    for label, paths in (
        ("inventory root", inventory_roots),
        ("required file", required_files),
        ("required absent path", required_absent_paths),
    ):
        if len(set(paths)) != len(paths):
            raise ValueError(f"duplicate {label}")
        for index, value in enumerate(paths):
            _relative_path(value, f"{label}[{index}]")
    for relative in required_absent_paths:
        try:
            (resolved / relative).lstat()
        except FileNotFoundError:
            continue
        raise ValueError(f"required-absent CARLA runtime path exists: {relative}")
    directories, relative_files = _enumerate_inventory(
        resolved, inventory_roots=inventory_roots, required_files=required_files,
    )
    rows: list[dict[str, Any]] = []
    for relative in relative_files:
        size, link_count, digest, _identity = _hash_regular_file(resolved / relative)
        rows.append({
            "path": relative,
            "size_bytes": size,
            "link_count": link_count,
            "sha256": digest,
        })
    scope = _scope(inventory_roots, required_files, required_absent_paths)
    directories_sha = stable_hash(directories)
    files_sha = stable_hash(rows)
    total_size = sum(row["size_bytes"] for row in rows)
    inventory_sha = stable_hash({
        "resolved_carla_root": str(resolved),
        "scope": scope,
        "directories_sha256": directories_sha,
        "files_sha256": files_sha,
        "file_count": len(rows),
        "total_size_bytes": total_size,
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "resolved_carla_root": str(resolved),
        "scope": scope,
        "directories": directories,
        "directories_sha256": directories_sha,
        "directory_count": len(directories),
        "files": rows,
        "files_sha256": files_sha,
        "file_count": len(rows),
        "total_size_bytes": total_size,
        "inventory_sha256": inventory_sha,
    }


def _validate_manifest_shape(
    manifest: Any,
    *,
    resolved_root: Path,
    inventory_roots: tuple[str, ...],
    required_files: tuple[str, ...],
    required_absent_paths: tuple[str, ...],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not isinstance(manifest, dict) or set(manifest) != ROOT_KEYS:
        raise ValueError("CARLA runtime manifest root shape differs")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["method"] != METHOD:
        raise ValueError("CARLA runtime manifest schema or method differs")
    if manifest["resolved_carla_root"] != str(resolved_root):
        raise ValueError("CARLA runtime manifest resolved root differs")
    scope = manifest["scope"]
    expected_scope = _scope(inventory_roots, required_files, required_absent_paths)
    if not isinstance(scope, dict) or set(scope) != SCOPE_KEYS or scope != expected_scope:
        raise ValueError("CARLA runtime manifest scope differs")
    directories = manifest["directories"]
    if not isinstance(directories, list):
        raise ValueError("CARLA runtime manifest directories must be a list")
    normalized_directories = [
        _relative_path(value, f"manifest directories[{index}]")
        for index, value in enumerate(directories)
    ]
    if normalized_directories != sorted(set(normalized_directories)):
        raise ValueError("CARLA runtime manifest directories are not sorted and unique")
    if _strict_int(manifest["directory_count"], "directory_count") != len(directories):
        raise ValueError("CARLA runtime manifest directory count differs")
    if manifest["directories_sha256"] != stable_hash(directories):
        raise ValueError("CARLA runtime manifest directory hash differs")
    rows = manifest["files"]
    if not isinstance(rows, list):
        raise ValueError("CARLA runtime manifest files must be a list")
    normalized_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != FILE_KEYS:
            raise ValueError(f"CARLA runtime manifest file row {index} shape differs")
        normalized_rows.append({
            "path": _relative_path(row["path"], f"manifest files[{index}].path"),
            "size_bytes": _strict_int(row["size_bytes"], f"files[{index}].size_bytes"),
            "link_count": _strict_int(row["link_count"], f"files[{index}].link_count", minimum=1),
            "sha256": _sha(row["sha256"], f"files[{index}].sha256"),
        })
    if normalized_rows != rows:
        raise ValueError("CARLA runtime manifest file rows are not canonical")
    paths = [row["path"] for row in rows]
    if paths != sorted(set(paths)):
        raise ValueError("CARLA runtime manifest file rows are not sorted and unique")
    if _strict_int(manifest["file_count"], "file_count") != len(rows):
        raise ValueError("CARLA runtime manifest file count differs")
    total_size = sum(row["size_bytes"] for row in rows)
    if _strict_int(manifest["total_size_bytes"], "total_size_bytes") != total_size:
        raise ValueError("CARLA runtime manifest total size differs")
    if manifest["files_sha256"] != stable_hash(rows):
        raise ValueError("CARLA runtime manifest file hash differs")
    expected_inventory = stable_hash({
        "resolved_carla_root": str(resolved_root),
        "scope": scope,
        "directories_sha256": manifest["directories_sha256"],
        "files_sha256": manifest["files_sha256"],
        "file_count": len(rows),
        "total_size_bytes": total_size,
    })
    if manifest["inventory_sha256"] != expected_inventory:
        raise ValueError("CARLA runtime manifest inventory hash differs")
    return normalized_directories, normalized_rows


def validate_manifest(
    manifest_path: Path,
    carla_root: Path,
    *,
    inventory_roots: tuple[str, ...] = INVENTORY_ROOTS,
    required_files: tuple[str, ...] = REQUIRED_FILES,
    required_absent_paths: tuple[str, ...] = REQUIRED_ABSENT_PATHS,
) -> dict[str, Any]:
    """Re-enumerate and rehash every frozen runtime file, failing closed."""

    manifest_file = Path(manifest_path)
    manifest_bytes = _read_regular_bytes(manifest_file)
    manifest = _load_strict_json_bytes(manifest_bytes, manifest_file)
    resolved = _resolved_regular_root(carla_root)
    expected_directories, rows = _validate_manifest_shape(
        manifest,
        resolved_root=resolved,
        inventory_roots=inventory_roots,
        required_files=required_files,
        required_absent_paths=required_absent_paths,
    )
    for relative in required_absent_paths:
        try:
            (resolved / relative).lstat()
        except FileNotFoundError:
            continue
        raise ValueError(f"required-absent CARLA runtime path exists: {relative}")
    actual_directories, actual_files = _enumerate_inventory(
        resolved, inventory_roots=inventory_roots, required_files=required_files,
    )
    if actual_directories != expected_directories:
        raise ValueError("CARLA runtime directory inventory has missing or extra entries")
    expected_files = [row["path"] for row in rows]
    if actual_files != expected_files:
        raise ValueError("CARLA runtime file inventory has missing or extra entries")
    local_identities: dict[str, tuple[int, int, int, int, int, int]] = {}
    for row in rows:
        size, link_count, digest, identity = _hash_regular_file(resolved / row["path"])
        if (size, link_count, digest) != (
            row["size_bytes"], row["link_count"], row["sha256"],
        ):
            raise ValueError(f"CARLA runtime asset differs from manifest: {row['path']}")
        local_identities[row["path"]] = identity
    final_directories, final_files = _enumerate_inventory(
        resolved, inventory_roots=inventory_roots, required_files=required_files,
    )
    if final_directories != expected_directories or final_files != expected_files:
        raise ValueError("CARLA runtime inventory changed while validating")
    for row in rows:
        path = resolved / row["path"]
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or _file_identity(metadata) != local_identities[row["path"]]
        ):
            raise ValueError(f"CARLA runtime asset changed while validating: {row['path']}")
    for relative in required_absent_paths:
        try:
            (resolved / relative).lstat()
        except FileNotFoundError:
            continue
        raise ValueError(f"required-absent CARLA runtime path appeared: {relative}")
    file_rows = {row["path"]: row for row in rows}
    shipping = file_rows[
        "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    ]["sha256"]
    return {
        "resolved_carla_root": str(resolved),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "inventory_sha256": manifest["inventory_sha256"],
        "files_sha256": manifest["files_sha256"],
        "file_count": manifest["file_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "shipping_binary_sha256": shipping,
    }


def validate_server_readonly_permissions(
    manifest_path: Path,
    carla_root: Path,
    *,
    logical_root: Path,
    server_uid: int,
    server_gids: set[int],
) -> dict[str, Any]:
    """Prove the CARLA server account cannot mutate its frozen runtime tree.

    Hashing before/after a job is insufficient when the server account can
    replace a map and restore it between checks.  This gate covers every
    frozen directory/file and both logical and resolved path ancestors.  The
    writable caches remain outside the CARLA tree and are attested separately.
    """

    if type(server_uid) is not int or server_uid <= 0:
        raise ValueError("CARLA server UID must be a positive non-root integer")
    if (
        not isinstance(server_gids, set)
        or not server_gids
        or any(type(value) is not int or value < 0 for value in server_gids)
    ):
        raise ValueError("CARLA server group authority is invalid")
    manifest_file = Path(manifest_path)
    manifest = _load_strict_json_bytes(
        _read_regular_bytes(manifest_file), manifest_file,
    )
    resolved = _resolved_regular_root(carla_root)
    directories, rows = _validate_manifest_shape(
        manifest,
        resolved_root=resolved,
        inventory_roots=INVENTORY_ROOTS,
        required_files=REQUIRED_FILES,
        required_absent_paths=REQUIRED_ABSENT_PATHS,
    )

    def writable(metadata: os.stat_result) -> bool:
        mode = stat.S_IMODE(metadata.st_mode)
        if metadata.st_uid == server_uid:
            return bool(mode & stat.S_IWUSR)
        if metadata.st_gid in server_gids:
            return bool(mode & stat.S_IWGRP)
        return bool(mode & stat.S_IWOTH)

    entries: dict[str, dict[str, Any]] = {}

    def add(path: Path, label: str, expected_type: str) -> None:
        metadata = path.lstat()
        if expected_type == "directory" and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"CARLA permission authority directory differs: {path}")
        if expected_type == "file" and not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"CARLA permission authority file differs: {path}")
        if expected_type == "symlink" and not stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"CARLA permission authority symlink differs: {path}")
        if expected_type != "symlink" and writable(metadata):
            raise ValueError(f"CARLA server account can mutate frozen path: {path}")
        row = {
            "label": label,
            "path": str(path),
            "type": expected_type,
            "mode": stat.S_IMODE(metadata.st_mode),
            "uid": int(metadata.st_uid),
            "gid": int(metadata.st_gid),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
        }
        if expected_type == "symlink":
            row["target"] = os.readlink(path)
        entries[f"{label}:{path}"] = row

    # Resolved ancestors control replacement of the frozen root itself.
    resolved_ancestors = list(reversed((resolved, *resolved.parents)))
    for path in resolved_ancestors:
        add(path, "resolved_ancestor", "directory")

    # Logical path ancestors bind the repository's software indirection.  A
    # symlink is protected by its parent directory; its exact target is also
    # recorded so a root-owned retarget cannot silently preserve the claim.
    logical = Path(logical_root)
    current = Path(logical.anchor)
    for part in logical.parts[1:]:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            add(current, "logical_ancestor", "symlink")
            break
        add(current, "logical_ancestor", "directory")

    for relative in directories:
        add(resolved / relative, "manifest_directory", "directory")
    for row in rows:
        add(resolved / row["path"], "manifest_file", "file")
    canonical = [entries[key] for key in sorted(entries)]
    return {
        "schema_version": "language2drive.carla_server_readonly_permissions.v1",
        "logical_root": str(logical),
        "resolved_root": str(resolved),
        "server_uid": server_uid,
        "server_gids": sorted(server_gids),
        "entry_count": len(canonical),
        "entries_sha256": stable_hash(canonical),
    }


def freeze_manifest(
    carla_root: Path,
    manifest_path: Path,
    provenance_path: Path,
    *,
    generator_path: Path,
    available_maps_witness_path: Path,
    inventory_roots: tuple[str, ...] = INVENTORY_ROOTS,
    required_files: tuple[str, ...] = REQUIRED_FILES,
    required_absent_paths: tuple[str, ...] = REQUIRED_ABSENT_PATHS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically freeze a canonical manifest plus non-authority telemetry."""

    started = time.monotonic()
    manifest = build_manifest(
        carla_root,
        inventory_roots=inventory_roots,
        required_files=required_files,
        required_absent_paths=required_absent_paths,
    )
    atomic_json(manifest_path, manifest)
    # Revalidation is intentionally cold at this API boundary and proves the
    # bytes just written describe the still-current runtime.
    validated = validate_manifest(
        manifest_path,
        carla_root,
        inventory_roots=inventory_roots,
        required_files=required_files,
        required_absent_paths=required_absent_paths,
    )
    generator = Path(generator_path).resolve(strict=True)
    witness_raw, _witness_file_sha = _load_regular_json(available_maps_witness_path)
    witness = validate_available_maps_witness(
        witness_raw, Path(validated["resolved_carla_root"]),
    )
    boot_id, freeze_monotonic_ticks, clock_ticks_per_second = (
        _boot_clock_authority()
    )
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": validated["manifest_sha256"],
        "resolved_carla_root": validated["resolved_carla_root"],
        "generator_path": str(generator),
        "generator_sha256": sha256_file(generator),
        "generated_at": _utc_now(),
        "cold_generation_elapsed_seconds": round(time.monotonic() - started, 6),
        "available_maps_witness": witness,
        "available_maps_witness_sha256": stable_hash(witness),
        "boot_id": boot_id,
        "freeze_monotonic_ticks": freeze_monotonic_ticks,
        "clock_ticks_per_second": clock_ticks_per_second,
        "qualification_boundary": (
            "generated_at and elapsed are telemetry; canonical manifest, fresh "
            "per-resolution rehashes, and same-boot process start after "
            "freeze_monotonic_ticks are qualification authority"
        ),
    }
    if set(provenance) != PROVENANCE_KEYS:
        raise AssertionError("runtime manifest provenance shape differs")
    atomic_json(provenance_path, provenance)
    return manifest, provenance


def validate_provenance(
    provenance_path: Path,
    *,
    manifest_path: Path,
    carla_root: Path,
    generator_path: Path,
) -> dict[str, Any]:
    """Validate the manifest-generation record and embedded live map witness."""

    value, provenance_sha = _load_regular_json(provenance_path)
    if not isinstance(value, dict) or set(value) != PROVENANCE_KEYS:
        raise ValueError("CARLA runtime manifest provenance shape differs")
    if value["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("CARLA runtime manifest provenance schema differs")
    resolved = _resolved_regular_root(carla_root)
    if value["resolved_carla_root"] != str(resolved):
        raise ValueError("CARLA runtime manifest provenance root differs")
    manifest = Path(manifest_path).resolve(strict=True)
    if value["manifest_path"] != str(manifest):
        raise ValueError("CARLA runtime manifest provenance path differs")
    if value["manifest_sha256"] != sha256_file(manifest):
        raise ValueError("CARLA runtime manifest provenance hash differs")
    generator = Path(generator_path).resolve(strict=True)
    if value["generator_path"] != str(generator):
        raise ValueError("CARLA runtime manifest generator path differs")
    if value["generator_sha256"] != sha256_file(generator):
        raise ValueError("CARLA runtime manifest generator hash differs")
    if not isinstance(value["generated_at"], str) or not value["generated_at"]:
        raise ValueError("CARLA runtime manifest generated_at must be nonempty")
    elapsed = value["cold_generation_elapsed_seconds"]
    if (
        not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
        or not float(elapsed) >= 0 or not float(elapsed) < float("inf")
    ):
        raise ValueError("CARLA runtime manifest elapsed telemetry is invalid")
    current_boot_id, _current_ticks, current_hz = _boot_clock_authority()
    if value["boot_id"] != current_boot_id:
        raise ValueError("CARLA runtime provenance belongs to another boot")
    freeze_ticks = _strict_int(
        value["freeze_monotonic_ticks"], "freeze_monotonic_ticks",
    )
    clock_hz = _strict_int(
        value["clock_ticks_per_second"], "clock_ticks_per_second", minimum=1,
    )
    if clock_hz != current_hz or freeze_ticks > _current_ticks:
        raise ValueError("CARLA runtime provenance boot clock differs")
    witness = validate_available_maps_witness(
        value["available_maps_witness"], resolved,
    )
    if value["available_maps_witness_sha256"] != stable_hash(witness):
        # The embedded object is authority.  Its canonical hash avoids relying
        # on an external witness path after provenance has been frozen.
        raise ValueError("embedded available-maps witness hash differs")
    expected_boundary = (
        "generated_at and elapsed are telemetry; canonical manifest, fresh "
        "per-resolution rehashes, and same-boot process start after "
        "freeze_monotonic_ticks are qualification authority"
    )
    if value["qualification_boundary"] != expected_boundary:
        raise ValueError("CARLA runtime provenance qualification boundary differs")
    return {
        "provenance_sha256": provenance_sha,
        "available_maps_sha256": witness["available_maps_sha256"],
        "client_version": witness["client_version"],
        "server_version": witness["server_version"],
        "available_map_count": len(witness["available_maps"]),
        "boot_id": value["boot_id"],
        "freeze_monotonic_ticks": freeze_ticks,
        "clock_ticks_per_second": clock_hz,
    }
