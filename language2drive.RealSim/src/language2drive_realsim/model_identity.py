"""Small, reproducible checkpoint identity without hashing model shards."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .io import file_hash, stable_hash


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

_QWEN_DRIVE_MODEL_ID = "Qwen/Qwen-Drive-1.0-4B"
_QWEN_DRIVE_CHECKPOINT_REVISION = "28484089a7cc8c335cf5089fb0745cf7c49b6eaa"
_QWEN_DRIVE_CODE_REVISION = "28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0"
_QWEN_DRIVE_CHECKPOINT_DIRECTORY = (
    "qwen-drive-1.0-4b-official-hf-" + _QWEN_DRIVE_CHECKPOINT_REVISION
)
_QWEN_DRIVE_CODE_DIRECTORY = "qwen-drive-1.0-4b-official-" + _QWEN_DRIVE_CODE_REVISION
_QWEN_DRIVE_SOURCE_RECEIPT_SHA256 = (
    "c7918c6e1da30c65a4295807b7083143d593f1cc8baf1746aacba32c1fbc9263"
)
_QWEN_DRIVE_FILES = {
    "chat_template.jinja": 7756,
    "chat_template.json": 5143,
    "config.json": 6091,
    "generation_config.json": 244,
    "merges.txt": 3353259,
    "model.safetensors": 9078630512,
    "preprocessor_config.json": 390,
    "tokenizer.json": 12807970,
    "tokenizer_config.json": 13815,
    "vocab.json": 6722759,
    "planner-rl/config.json": 913,
    "planner-rl/model.safetensors": 2079739550,
}

_AUTOVLA_MODEL_ID = "Zewei-Zhou/AutoVLA"
_AUTOVLA_CHECKPOINT_REVISION = "a7d7ba3ed7529b248d2694c2defa31b35208340f"
_AUTOVLA_CODE_REVISION = "ba34eed74ce6729e7986592d0e66cbaca397b4fa"
_AUTOVLA_BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_AUTOVLA_BASE_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
_AUTOVLA_CHECKPOINT_FILE = "AutoVLA_PDMS_89.ckpt"
_AUTOVLA_CHECKPOINT_BYTES = 16_292_664_780
_AUTOVLA_CHECKPOINT_RECEIPT_SHA256 = (
    "f5f6fdaddf41167986d850487727e45b13b3d3b3fe591094b1af65781de0ea6d"
)
_AUTOVLA_BASE_RECEIPT_SHA256 = (
    "57760dbacda71bfbce5569b66389e9389c3851bebf63bf40243b73095681f936"
)
_AUTOVLA_BASE_FILES = {
    ".gitattributes": 1519,
    "LICENSE": 7387,
    "README.md": 18276,
    "chat_template.json": 1050,
    "config.json": 1373,
    "generation_config.json": 216,
    "merges.txt": 1671839,
    "model-00001-of-00002.safetensors": 3982649232,
    "model-00002-of-00002.safetensors": 3526688744,
    "model.safetensors.index.json": 65448,
    "preprocessor_config.json": 350,
    "tokenizer.json": 7031645,
    "tokenizer_config.json": 5702,
    "vocab.json": 2776833,
}

_OPENDRIVEVLA_MODEL_ID = "OpenDriveVLA/OpenDriveVLA-0.5B"
_OPENDRIVEVLA_CHECKPOINT_REVISION = "5b219caae79ae65b7068fc9cf442220b75f07dff"
_OPENDRIVEVLA_CODE_REVISION = "10e8095bc618d508cb70cca37b6956ac4db6e9f3"
_OPENDRIVEVLA_FILES = {
    "added_tokens.json": 1049,
    "config.json": 1857,
    "generation_config.json": 247,
    "merges.txt": 1671853,
    "model.safetensors": 1466080342,
    "special_tokens_map.json": 2813,
    "tokenizer.json": 7034860,
    "tokenizer_config.json": 8007,
    "vocab.json": 2776833,
}

_LMDRIVE_MODEL_ID = "OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0"
_LMDRIVE_CHECKPOINT_REVISION = "081ca1b028c73ceb83918a0b45a5e7c5322b26a8"
_LMDRIVE_VISION_REVISION = "ce055800ef3a458bad5b7fff18ed52034edcea45"
_LMDRIVE_LLAVA_REVISION = "4481d270cc22fd5c4d1bb5df129622006ccd9234"
_LMDRIVE_CODE_REVISION = "308d8da0b4ff3463b1649fe483a9b2c56b108189"
_LMDRIVE_FILES = {
    "bert-base-uncased/config.json": 570,
    "bert-base-uncased/model.safetensors": 440449768,
    "bert-base-uncased/tokenizer.json": 466062,
    "bert-base-uncased/tokenizer_config.json": 48,
    "bert-base-uncased/vocab.txt": 231508,
    "llava-v1.5-7b/config.json": 1161,
    "llava-v1.5-7b/generation_config.json": 124,
    "llava-v1.5-7b/pytorch_model-00001-of-00002.bin": 9976634558,
    "llava-v1.5-7b/pytorch_model-00002-of-00002.bin": 3542276251,
    "llava-v1.5-7b/pytorch_model.bin.index.json": 27068,
    "llava-v1.5-7b/special_tokens_map.json": 438,
    "llava-v1.5-7b/tokenizer.model": 499723,
    "llava-v1.5-7b/tokenizer_config.json": 749,
    "lmdrive/llava-v1.5-checkpoint.pth": 946473021,
    "vision/vision-encoder-r50.pth.tar": 376584921,
}


def _bind_identity(result: Dict[str, Any]) -> str:
    """Bind content identity while keeping the local mount path relocatable."""

    return stable_hash({key: value for key, value in result.items() if key != "path"})


def _huggingface_snapshot_revision(path: Path) -> Optional[str]:
    """Read the local-dir metadata revision written by huggingface_hub."""

    metadata_path = path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    try:
        first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return first_line if _REVISION_RE.fullmatch(first_line) else None


def _effective_named_sibling(requested: Path, expected_name: str) -> Path:
    """Resolve a known shared-workspace mount alias without accepting arbitrary assets."""

    if requested.name != expected_name or requested.exists():
        return requested
    local = Path(__file__).resolve().parents[2] / "model_zoo" / expected_name
    return local if local.exists() else requested


def _git_tracked_revision_and_clean(path: Path) -> tuple[Optional[str], bool]:
    """Inspect tracked source only, without changing Git's global safe-directory list."""

    git_dir = path / ".git"
    common = ["git", "--git-dir=" + str(git_dir), "--work-tree=" + str(path)]
    try:
        revision = subprocess.check_output(
            common + ["rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        clean = all(
            subprocess.run(
                common + arguments,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
            for arguments in (["diff", "--quiet"], ["diff", "--cached", "--quiet"])
        )
    except (OSError, subprocess.SubprocessError):
        return None, False
    return revision if _REVISION_RE.fullmatch(revision) else None, clean


def _exact_sized_files(root: Path, expected: Dict[str, int]) -> list[str]:
    return [
        name
        for name, size in expected.items()
        if not (root / name).is_file() or (root / name).stat().st_size != size
    ]


def _qwen_drive_identity(path: Path) -> Dict[str, Any]:
    effective_checkpoint = _effective_named_sibling(path, _QWEN_DRIVE_CHECKPOINT_DIRECTORY)
    requested_code = path.parent / _QWEN_DRIVE_CODE_DIRECTORY
    effective_code = _effective_named_sibling(requested_code, _QWEN_DRIVE_CODE_DIRECTORY)
    source_path = effective_checkpoint / "OFFICIAL_SOURCE.json"
    missing = _exact_sized_files(effective_checkpoint, _QWEN_DRIVE_FILES)
    revision, tracked_clean = _git_tracked_revision_and_clean(effective_code)
    source_valid = False
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_valid = (
            file_hash(source_path) == _QWEN_DRIVE_SOURCE_RECEIPT_SHA256
            and source.get("schema_version") == "l2d.official_model_source.v0.1"
            and source.get("model_id") == _QWEN_DRIVE_MODEL_ID
            and source.get("revision") == _QWEN_DRIVE_CHECKPOINT_REVISION
            and source.get("official_code_revision") == _QWEN_DRIVE_CODE_REVISION
            and source.get("planner") == "planner-rl"
            and source.get("files") == _QWEN_DRIVE_FILES
            and source.get("large_weight_hash_computed") is False
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        source_valid = False
    if not source_valid:
        missing.append("OFFICIAL_SOURCE.json:frozen_receipt_mismatch")
    if revision != _QWEN_DRIVE_CODE_REVISION or not tracked_clean:
        missing.append(_QWEN_DRIVE_CODE_DIRECTORY + ":tracked_source_mismatch")
    result: Dict[str, Any] = {
        # Keep the established Qwen identity schema used by the frozen remote
        # preparation while independently rebuilding it from local evidence.
        "schema_version": "l2d.qwen_drive.official_identity.v0.1",
        "model_id": _QWEN_DRIVE_MODEL_ID,
        "model_name": "Qwen-Drive-1.0-4B",
        "path": str(path.resolve()),
        "snapshot_revision": _QWEN_DRIVE_CHECKPOINT_REVISION if source_valid else None,
        "official_code_path": str(requested_code.resolve()),
        "official_code_revision": _QWEN_DRIVE_CODE_REVISION if revision == _QWEN_DRIVE_CODE_REVISION else revision,
        "planner": "planner-rl",
        "present": not _exact_sized_files(effective_checkpoint, _QWEN_DRIVE_FILES),
        "complete": not missing,
        "required_file_count": len(_QWEN_DRIVE_FILES),
        "size_bytes": sum(
            (effective_checkpoint / name).stat().st_size
            for name in _QWEN_DRIVE_FILES
            if (effective_checkpoint / name).is_file()
        ),
        "missing_required_files": missing,
        "large_weight_hash_computed": False,
    }
    result["identity_binding"] = _bind_identity(result)
    return result


def _autovla_identity(path: Path) -> Dict[str, Any]:
    checkpoint_path = path / _AUTOVLA_CHECKPOINT_FILE
    source_path = path / "OFFICIAL_SOURCE.json"
    code_root = path.parent / "autovla-official-ba34eed"
    base_root = path.parent / ("autovla-official-qwen-" + _AUTOVLA_BASE_REVISION)
    base_source_path = base_root / "OFFICIAL_SOURCE.json"
    missing = _exact_sized_files(
        path, {_AUTOVLA_CHECKPOINT_FILE: _AUTOVLA_CHECKPOINT_BYTES}
    )
    missing.extend("base:" + item for item in _exact_sized_files(base_root, _AUTOVLA_BASE_FILES))
    revision, tracked_clean = _git_tracked_revision_and_clean(code_root)
    checkpoint_receipt_valid = base_receipt_valid = False
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        checkpoint_receipt_valid = (
            file_hash(source_path) == _AUTOVLA_CHECKPOINT_RECEIPT_SHA256
            and source.get("schema_version") == "l2d.official_checkpoint.v1"
            and source.get("model") == "AutoVLA"
            and source.get("repository") == _AUTOVLA_MODEL_ID
            and source.get("revision") == _AUTOVLA_CHECKPOINT_REVISION
            and source.get("official_code_revision") == _AUTOVLA_CODE_REVISION
            and source.get("base_model_id") == _AUTOVLA_BASE_MODEL_ID
            and source.get("base_model_revision") == _AUTOVLA_BASE_REVISION
            and source.get("files") == {
                _AUTOVLA_CHECKPOINT_FILE: _AUTOVLA_CHECKPOINT_BYTES
            }
            and source.get("large_weight_hash_computed") is False
            and source.get("project_trained_checkpoint_used") is False
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        checkpoint_receipt_valid = False
    try:
        base_source = json.loads(base_source_path.read_text(encoding="utf-8"))
        base_receipt_valid = (
            file_hash(base_source_path) == _AUTOVLA_BASE_RECEIPT_SHA256
            and base_source.get("schema_version") == "l2d.official_upstream_asset.v1"
            and base_source.get("repository") == _AUTOVLA_BASE_MODEL_ID
            and base_source.get("revision") == _AUTOVLA_BASE_REVISION
            and base_source.get("verified_files") == _AUTOVLA_BASE_FILES
            and base_source.get("large_weight_hashes_computed") is False
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        base_receipt_valid = False
    if not checkpoint_receipt_valid:
        missing.append("OFFICIAL_SOURCE.json:frozen_receipt_mismatch")
    if not base_receipt_valid:
        missing.append("base:OFFICIAL_SOURCE.json:frozen_receipt_mismatch")
    if revision != _AUTOVLA_CODE_REVISION or not tracked_clean:
        missing.append("autovla-official-ba34eed:tracked_source_mismatch")
    result: Dict[str, Any] = {
        "schema_version": "l2d.autovla.official_identity.v0.1",
        "model_id": _AUTOVLA_MODEL_ID,
        "model_name": "AutoVLA",
        "path": str(path.resolve()),
        "snapshot_revision": _AUTOVLA_CHECKPOINT_REVISION if checkpoint_receipt_valid else None,
        "official_code_revision": revision,
        "official_code_tracked_clean": tracked_clean,
        "base_model_id": _AUTOVLA_BASE_MODEL_ID,
        "base_model_revision": _AUTOVLA_BASE_REVISION if base_receipt_valid else None,
        "present": checkpoint_path.is_file(),
        "complete": not missing,
        "shard_count": int(checkpoint_path.is_file()),
        "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else 0,
        "indexed_parameter_count": None,
        "indexed_tensor_bytes": None,
        "index_sha256": None,
        "config_sha256": None,
        "missing_shards": [] if checkpoint_path.is_file() else [_AUTOVLA_CHECKPOINT_FILE],
        "missing_required_files": missing,
        "source_receipt_sha256": file_hash(source_path) if source_path.is_file() else None,
        "base_source_receipt_sha256": (
            file_hash(base_source_path) if base_source_path.is_file() else None
        ),
        "large_weight_hash_computed": False,
    }
    result["identity_binding"] = _bind_identity(result)
    return result


def _opendrivevla_identity(path: Path) -> Dict[str, Any]:
    """Bind the pinned official snapshot by immutable revisions and exact sizes.

    The checkpoint was transferred from the user-authorized remote host.  We
    deliberately do not stream-hash the 1.46 GB shard; strict Transformers
    loading is recorded separately by the execution campaign.
    """

    code_root = path.parent.parent / "opendrivevla-official-10e8095b"
    missing = _exact_sized_files(path, _OPENDRIVEVLA_FILES)
    revision, tracked_clean = _git_tracked_revision_and_clean(code_root)
    if revision != _OPENDRIVEVLA_CODE_REVISION or not tracked_clean:
        missing.append("opendrivevla-official-10e8095b:tracked_source_mismatch")
    result: Dict[str, Any] = {
        "schema_version": "l2d.opendrivevla.official_identity.v0.1",
        "model_id": _OPENDRIVEVLA_MODEL_ID,
        "model_name": "OpenDriveVLA",
        "path": str(path.resolve()),
        "snapshot_revision": _OPENDRIVEVLA_CHECKPOINT_REVISION,
        "official_code_revision": revision,
        "official_code_tracked_clean": tracked_clean,
        "source_provenance": "user_authorized_remote_copy",
        "present": not _exact_sized_files(path, _OPENDRIVEVLA_FILES),
        "complete": not missing,
        "required_file_count": len(_OPENDRIVEVLA_FILES),
        "shard_count": int((path / "model.safetensors").is_file()),
        "size_bytes": sum(
            (path / name).stat().st_size
            for name in _OPENDRIVEVLA_FILES
            if (path / name).is_file()
        ),
        "indexed_parameter_count": None,
        "indexed_tensor_bytes": None,
        "index_sha256": None,
        "config_sha256": file_hash(path / "config.json") if (path / "config.json").is_file() else None,
        "missing_shards": [] if (path / "model.safetensors").is_file() else ["model.safetensors"],
        "missing_required_files": missing,
        "large_weight_hash_computed": False,
    }
    result["identity_binding"] = _bind_identity(result)
    return result


def _lmdrive_identity(path: Path) -> Dict[str, Any]:
    """Bind the three pinned LMDrive weight layers without hashing weights."""

    code_root = path.parent / "lmdrive-official-308d8da"
    missing = _exact_sized_files(path, _LMDRIVE_FILES)
    revision, tracked_clean = _git_tracked_revision_and_clean(code_root)
    if revision != _LMDRIVE_CODE_REVISION or not tracked_clean:
        missing.append("lmdrive-official-308d8da:tracked_source_mismatch")
    result: Dict[str, Any] = {
        "schema_version": "l2d.lmdrive.synthetic_null_lidar_identity.v0.1",
        "model_id": _LMDRIVE_MODEL_ID,
        "model_name": "LMDrive synthetic-null-LiDAR",
        "variant": "synthetic-null-LiDAR; not official-input-equivalent",
        "path": str(path.resolve()),
        "snapshot_revision": _LMDRIVE_CHECKPOINT_REVISION,
        "vision_snapshot_revision": _LMDRIVE_VISION_REVISION,
        "llava_base_snapshot_revision": _LMDRIVE_LLAVA_REVISION,
        "official_code_revision": revision,
        "official_code_tracked_clean": tracked_clean,
        "present": not _exact_sized_files(path, _LMDRIVE_FILES),
        "complete": not missing,
        "required_file_count": len(_LMDRIVE_FILES),
        "shard_count": 4,
        "size_bytes": sum(
            (path / name).stat().st_size
            for name in _LMDRIVE_FILES
            if (path / name).is_file()
        ),
        "missing_required_files": missing,
        "large_weight_hash_computed": False,
    }
    result["identity_binding"] = _bind_identity(result)
    return result


def checkpoint_identity(path: Path, model_id: Optional[str] = None) -> Dict[str, Any]:
    """Describe an indexed safetensors snapshot using only small metadata files.

    Shard names and byte sizes are bound into the identity, but shard contents
    are not re-hashed. Hugging Face's immutable snapshot revision plus the
    index/config hashes provide the content-addressed source identity while
    keeping startup independent of a 70+ GB sequential read.
    """

    path = Path(path)
    if model_id == "Language2Drive/SimLingo-Posttrained-epoch004":
        from .simlingo_posttrained_identity import checkpoint_identity as local_identity
        return local_identity(path)
    if model_id == _QWEN_DRIVE_MODEL_ID:
        return _qwen_drive_identity(path)
    if model_id == _AUTOVLA_MODEL_ID:
        return _autovla_identity(path)
    if model_id == _OPENDRIVEVLA_MODEL_ID:
        return _opendrivevla_identity(path)
    if model_id == _LMDRIVE_MODEL_ID:
        return _lmdrive_identity(path)
    if model_id == "RenzKa/simlingo":
        checkpoint = path
        model_root = checkpoint.parents[2] if checkpoint.is_file() else path
        source_path = model_root / "OFFICIAL_SOURCE.json"
        config_path = model_root / ".hydra" / "config.yaml"
        result: Dict[str, Any] = {
            "model_id": model_id,
            "path": str(checkpoint.resolve()),
            "snapshot_revision": None,
            "present": checkpoint.is_file(),
            "complete": False,
            "shard_count": int(checkpoint.is_file()),
            "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
            "indexed_parameter_count": None,
            "indexed_tensor_bytes": None,
            "index_sha256": None,
            "config_sha256": file_hash(config_path) if config_path.is_file() else None,
            "missing_shards": [] if checkpoint.is_file() else [str(checkpoint)],
            "missing_required_files": [],
        }
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
            expected_files = source["files"]
            expected_checkpoint = model_root / "checkpoints/epoch=013.ckpt/pytorch_model.pt"
            missing = [
                name
                for name, size in expected_files.items()
                if not (model_root / name).is_file()
                or (model_root / name).stat().st_size != size
            ]
            result.update(
                snapshot_revision=source["revision"],
                complete=(
                    source.get("model_id") == model_id
                    and checkpoint.resolve() == expected_checkpoint.resolve()
                    and not missing
                    and source.get("large_weight_hash_computed") is False
                ),
                missing_required_files=missing,
                official_code_revision=source.get("official_code_revision"),
                base_model_id=source.get("base_model_id"),
                base_model_revision=source.get("base_model_revision"),
                source_receipt_sha256=file_hash(source_path),
                large_weight_hash_computed=False,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            result["missing_required_files"] = [str(source_path)]
        result["identity_binding"] = _bind_identity(result)
        return result

    required_files = {
        "nvidia/Alpamayo2-Super": (
            "chat_template.jinja",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
        ),
    }.get(model_id, ())
    missing_required_files = [
        name for name in required_files if not (path / name).is_file()
    ]
    result: Dict[str, Any] = {
        "model_id": model_id,
        "path": str(path.resolve()),
        "snapshot_revision": _huggingface_snapshot_revision(path),
        "present": False,
        "complete": False,
        "shard_count": 0,
        "size_bytes": 0,
        "indexed_parameter_count": None,
        "indexed_tensor_bytes": None,
        "index_sha256": None,
        "config_sha256": None,
        "missing_shards": [],
        "missing_required_files": missing_required_files,
    }
    index_path = path / "model.safetensors.index.json"
    config_path = path / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        result["identity_binding"] = _bind_identity(result)
        return result

    result["present"] = True
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
        missing = [shard for shard in shards if not (path / shard).is_file()]
        present = [path / shard for shard in shards if (path / shard).is_file()]
        metadata = index.get("metadata", {})
        result.update(
            {
                "complete": bool(shards) and not missing and not missing_required_files,
                "shard_count": len(shards),
                "size_bytes": sum(shard.stat().st_size for shard in present),
                "indexed_parameter_count": metadata.get("total_parameters"),
                "indexed_tensor_bytes": metadata.get("total_size"),
                "index_sha256": file_hash(index_path),
                "config_sha256": file_hash(config_path),
                "missing_shards": missing,
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    result["identity_binding"] = _bind_identity(result)
    return result
