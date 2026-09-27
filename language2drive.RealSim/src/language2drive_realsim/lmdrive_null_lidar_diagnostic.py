"""CPU-only contract for an opt-in LMDrive null-LiDAR diagnostic.

This module does not load LMDrive, transform images, or call a service.  It
validates the raw inputs required by a future adapter and constructs the one
explicitly disclosed synthetic input proposed for the unchanged LiDAR tower.
It is not an official camera-only LMDrive implementation and is never formal.
Camera provenance flags and distinct object identities are caller assertions
and structural checks, not proof of genuine rendered pixels or non-copying.
Artifact paths, sizes and the supplied revision do not certify weight identity;
passing preflight is not a successful model load or forward execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping


PROFILE = "lmdrive-official-weights-synthetic-null-lidar-20260917"
EXPECTED_CODE_REVISION = "308d8da0b4ff3463b1649fe483a9b2c56b108189"
DIAGNOSTIC_ONLY = True
FORMAL_EVALUATION = False
UPDATES_METRICS = False

_CAMERAS = {
    "rgb_front": (1.3, 0.0, 2.3, 0.0, 1200, 900, 100.0),
    "rgb_left": (1.3, 0.0, 2.3, -60.0, 400, 300, 100.0),
    "rgb_right": (1.3, 0.0, 2.3, 60.0, 400, 300, 100.0),
    "rgb_rear": (-1.3, 0.0, 2.3, 180.0, 400, 300, 100.0),
}
_CODE_FILES = (
    "leaderboard/team_code/lmdriver_agent.py",
    "leaderboard/team_code/lmdriver_config.py",
    "vision_encoder/timm/models/memfuser.py",
    "vision_encoder/timm/models/pointpillar.py",
    "LAVIS/lavis/models/drive_models/drive.py",
)
_LMDRIVE_NAME = "llava-v1.5-checkpoint.pth"
_LMDRIVE_BYTES = 946_473_021
_VISION_NAME = "vision-encoder-r50.pth.tar"
_VISION_BYTES = 376_584_921
_BASE_WEIGHT_NAMES = (
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
)
_BASE_WEIGHT_BYTES = 13_518_910_809
_BASE_SMALL_FILES = ("config.json", "tokenizer.model", "tokenizer_config.json")


class LMDriveDiagnosticContractError(ValueError):
    """The diagnostic input would violate the disclosed model contract."""


@dataclass(frozen=True)
class CameraFrame:
    """A caller-described view; rendering provenance is not verified here."""

    role: str
    sensor_id: str
    pixels: Any
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float
    width: int
    height: int
    fov_deg: float
    real_rendered_view: bool = True
    derived_from: str | None = None


@dataclass(frozen=True)
class SyntheticNullLidar:
    """Immutable bytes for one disclosed in-range zero point plus padding."""

    point_bytes: bytes
    shape: tuple[int, int] = (40_000, 4)
    dtype: str = "float32"
    num_points: int = 1
    interpretation: str = "synthetic_null_lidar_sentinel_not_camera_only"


@dataclass(frozen=True)
class LMDriveDiagnosticRawInput:
    """Validated raw adapter input; official image transforms are still due."""

    profile: str
    cameras: tuple[CameraFrame, ...]
    speed_m_s: float
    instruction: str
    lidar: SyntheticNullLidar
    center_view_policy: str = "resize_rgb_front_to_800x600_then_official_center_crop"
    target_point_supplied: bool = False
    diagnostic_only: bool = True
    formal_evaluation: bool = False
    updates_metrics: bool = False


@dataclass(frozen=True)
class ArtifactPaths:
    code_root: Path
    lmdrive_checkpoint: Path
    vision_checkpoint: Path
    llava_base: Path


@dataclass(frozen=True)
class PreflightReport:
    profile: str
    artifact_metadata_ready: bool
    input_ready: bool
    loaded_model_contract_attested: bool
    ready_for_single_request_smoke: bool
    errors: tuple[str, ...]
    identity_scope: str = "revision_path_and_size_only_no_weight_hash"
    diagnostic_only: bool = True
    formal_evaluation: bool = False
    updates_metrics: bool = False


def _close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-6)


def _pixel_contract(pixels: Any) -> tuple[tuple[int, ...], str]:
    shape = getattr(pixels, "shape", None)
    dtype = getattr(pixels, "dtype", None)
    if shape is None or dtype is None:
        raise LMDriveDiagnosticContractError("camera pixels must expose shape and dtype")
    return tuple(int(v) for v in shape), str(dtype)


def validate_camera_frames(frames: Mapping[str, CameraFrame]) -> tuple[CameraFrame, ...]:
    """Check the declared four-view rig; do not certify pixel provenance."""

    if not isinstance(frames, Mapping):
        raise TypeError("frames must be a mapping")
    if set(frames) != set(_CAMERAS):
        missing = sorted(set(_CAMERAS) - set(frames))
        extra = sorted(set(frames) - set(_CAMERAS))
        raise LMDriveDiagnosticContractError(
            f"exact real four-view rig required; missing={missing}, extra={extra}"
        )

    sensor_ids: set[str] = set()
    pixel_ids: set[int] = set()
    ordered = []
    for role in _CAMERAS:
        frame = frames[role]
        if not isinstance(frame, CameraFrame) or frame.role != role:
            raise LMDriveDiagnosticContractError(f"{role} camera role changed")
        if not frame.real_rendered_view or frame.derived_from is not None:
            raise LMDriveDiagnosticContractError(
                f"{role} must be an independent real rendered view"
            )
        if not frame.sensor_id or frame.sensor_id in sensor_ids:
            raise LMDriveDiagnosticContractError("camera sensor ids must be unique")
        sensor_ids.add(frame.sensor_id)
        if id(frame.pixels) in pixel_ids:
            raise LMDriveDiagnosticContractError("camera pixel objects must be distinct")
        pixel_ids.add(id(frame.pixels))

        x_m, y_m, z_m, yaw_deg, width, height, fov_deg = _CAMERAS[role]
        numeric = (
            (frame.x_m, x_m, "x"),
            (frame.y_m, y_m, "y"),
            (frame.z_m, z_m, "z"),
            (frame.yaw_deg, yaw_deg, "yaw"),
            (frame.fov_deg, fov_deg, "fov"),
        )
        for actual, expected, field in numeric:
            if not _close(actual, expected):
                raise LMDriveDiagnosticContractError(f"{role} {field} changed")
        if (frame.width, frame.height) != (width, height):
            raise LMDriveDiagnosticContractError(f"{role} resolution changed")
        shape, dtype = _pixel_contract(frame.pixels)
        if shape != (height, width, 3) or dtype != "uint8":
            raise LMDriveDiagnosticContractError(
                f"{role} pixels must be HWC uint8 at declared resolution"
            )
        ordered.append(frame)
    return tuple(ordered)


def build_synthetic_null_lidar() -> SyntheticNullLidar:
    """Build the candidate sentinel without importing NumPy, Torch, or LMDrive."""

    # 40,000 rows x 4 little/big-endian invariant IEEE-zero float32 values.
    return SyntheticNullLidar(point_bytes=bytes(40_000 * 4 * 4))


def build_diagnostic_raw_input(
    frames: Mapping[str, CameraFrame], *, speed_m_s: float, instruction: str
) -> LMDriveDiagnosticRawInput:
    """Validate and package raw data for a future, separately sealed adapter."""

    ordered = validate_camera_frames(frames)
    if isinstance(speed_m_s, bool) or not isinstance(speed_m_s, (int, float)):
        raise TypeError("speed_m_s must be a finite number")
    speed = float(speed_m_s)
    if not math.isfinite(speed) or speed < 0.0:
        raise LMDriveDiagnosticContractError("speed_m_s must be finite and non-negative")
    if not isinstance(instruction, str):
        raise TypeError("instruction must be text")
    instruction = instruction.strip()
    if not instruction or "\x00" in instruction:
        raise LMDriveDiagnosticContractError("instruction must be non-empty trusted text")
    return LMDriveDiagnosticRawInput(
        profile=PROFILE,
        cameras=ordered,
        speed_m_s=speed,
        instruction=instruction,
        lidar=build_synthetic_null_lidar(),
    )


def attest_loaded_model_contract(model: Any) -> tuple[str, ...]:
    """Inspect, but never mutate, the two route-leakage-critical model flags."""

    errors = []
    if getattr(model, "has_gru_decoder", None) is not False:
        errors.append("loaded model must attest has_gru_decoder=False")
    visual = getattr(model, "visual_encoder", None)
    if visual is None:
        errors.append("loaded model has no visual_encoder")
    else:
        if getattr(visual, "return_feature", None) is not True:
            errors.append("visual_encoder must attest return_feature=True")
        if getattr(visual, "lidar_backbone", None) is None:
            errors.append("unchanged Memfuser LiDAR branch is absent")
    return tuple(errors)


def _regular_file_size(path: Path) -> int:
    if path.is_symlink() or not path.is_file():
        raise LMDriveDiagnosticContractError(f"not a regular non-symlink file: {path}")
    return path.stat().st_size


def _artifact_errors(paths: ArtifactPaths, code_revision: str) -> tuple[str, ...]:
    errors = []
    if code_revision != EXPECTED_CODE_REVISION:
        errors.append("official LMDrive code revision changed")
    for relative in _CODE_FILES:
        path = paths.code_root / relative
        try:
            _regular_file_size(path)
        except LMDriveDiagnosticContractError as exc:
            errors.append(str(exc))

    singles = (
        (paths.lmdrive_checkpoint, _LMDRIVE_NAME, _LMDRIVE_BYTES),
        (paths.vision_checkpoint, _VISION_NAME, _VISION_BYTES),
    )
    for path, name, expected_size in singles:
        if path.name != name:
            errors.append(f"official artifact filename changed: {path}")
            continue
        try:
            size = _regular_file_size(path)
        except LMDriveDiagnosticContractError as exc:
            errors.append(str(exc))
        else:
            if size != expected_size:
                errors.append(f"official artifact size changed: {path}")

    if paths.llava_base.is_symlink() or not paths.llava_base.is_dir():
        errors.append("LLaVA base must be a local non-symlink directory")
    else:
        total = 0
        for name in _BASE_WEIGHT_NAMES:
            try:
                total += _regular_file_size(paths.llava_base / name)
            except LMDriveDiagnosticContractError as exc:
                errors.append(str(exc))
        if total != _BASE_WEIGHT_BYTES:
            errors.append("LLaVA base weight size total changed")
        for name in _BASE_SMALL_FILES:
            try:
                _regular_file_size(paths.llava_base / name)
            except LMDriveDiagnosticContractError as exc:
                errors.append(str(exc))
    return tuple(errors)


def inspect_preflight(
    paths: ArtifactPaths,
    frames: Mapping[str, CameraFrame],
    *,
    code_revision: str,
    loaded_model: Any | None = None,
) -> PreflightReport:
    """Metadata-only readiness check; it never reads checkpoint contents."""

    artifact_errors = _artifact_errors(paths, code_revision)
    try:
        validate_camera_frames(frames)
    except (TypeError, LMDriveDiagnosticContractError) as exc:
        input_errors = (str(exc),)
    else:
        input_errors = ()
    if loaded_model is None:
        model_errors = ("loaded model contract has not been attested",)
    else:
        model_errors = attest_loaded_model_contract(loaded_model)
    errors = artifact_errors + input_errors + model_errors
    return PreflightReport(
        profile=PROFILE,
        artifact_metadata_ready=not artifact_errors,
        input_ready=not input_errors,
        loaded_model_contract_attested=loaded_model is not None and not model_errors,
        ready_for_single_request_smoke=not errors,
        errors=errors,
    )
