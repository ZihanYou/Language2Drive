from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from .frozen_paths import ProvenancePath as Path
from typing import Any, Iterable, Iterator


def topology_spawn_token(town: str, x: Any, y: Any) -> str:
    """Return the suite's canonical one-metre physical-spawn cell token."""

    coordinates = []
    for name, value in (("x", x), ("y", y)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value))):
            raise ValueError(f"spawn {name} must be a finite JSON number")
        coordinates.append(round(float(value)))
    if not isinstance(town, str) or not town:
        raise ValueError("spawn town must be a nonempty string")
    return f"spawn:{town}:{coordinates[0]}:{coordinates[1]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any, *, pretty: bool) -> str:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options)


def stable_hash(value: Any) -> str:
    """Hash canonical, standards-compliant JSON (NaN/Infinity are rejected)."""

    return hashlib.sha256(_json_text(value, pretty=False).encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist a rename where the platform supports directory fsync."""

    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    path = Path(path)
    # Evaluator-only sidecars are intentionally not public artifacts.  Keep the
    # safe default even if a caller forgets to pass an explicit mode.
    selected_mode = 0o600 if ".private." in path.name else 0o644
    atomic_text(path, _json_text(value, pretty=True) + "\n", mode=selected_mode if mode is None else mode)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, mode: int = 0o644) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise TypeError(f"JSONL row {index} must be an object")
                stream.write(_json_text(row, pretty=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield value
