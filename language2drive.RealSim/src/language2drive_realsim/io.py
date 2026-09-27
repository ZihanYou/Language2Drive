"""Strict deterministic JSON helpers used for frozen evidence."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError("non-standard JSON constant: %s" % value)


def _unique_object(pairs: Any) -> Any:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(
            stream,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )


def atomic_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    """Publish complete JSON; optionally refuse an existing path atomically.

    Exclusive publication links a fully written same-directory temporary file.
    It refuses existing files and symlinks, including ones created concurrently.
    Each file is atomic; multiple calls are not a multi-file transaction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, str(path))
        else:
            os.link(temporary, str(path))
            os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_jsonl(path: Path) -> Any:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except ValueError as error:
                raise ValueError("%s:%d: %s" % (path, line_number, error)) from error
            if not isinstance(row, dict):
                raise ValueError("%s:%d must be an object" % (path, line_number))
            rows.append(row)
    return rows
