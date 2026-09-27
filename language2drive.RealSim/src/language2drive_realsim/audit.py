"""Repository, runtime, and upstream AlpaSim capability audit."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .constants import ALPASIM_COMMIT
from .model_identity import checkpoint_identity


def _select_checkpoint_identity(
    *,
    environment_variable: str,
    candidates: list[Path],
    model_id: str,
) -> Dict[str, Any]:
    """Select the first complete local checkpoint without hiding an override.

    An explicit environment override remains authoritative, including when it
    is incomplete.  For defaults, prefer the first complete candidate and
    retain the first observed/incomplete identity when none is complete.  This
    lets the standalone RealSim checkout use its verified ``model_zoo`` copy
    while preserving the historical reference-tree fallback.
    """

    override = os.environ.get(environment_variable)
    if override:
        return checkpoint_identity(Path(override), model_id)

    identities = [checkpoint_identity(path, model_id) for path in candidates]
    for identity in identities:
        if identity["complete"]:
            return identity
    for identity in identities:
        if identity["present"]:
            return identity
    return identities[0]


def _run(command: list, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.check_output(command, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _available(command: list[str], cwd: Optional[Path] = None) -> bool:
    """Return whether a command completes successfully without trusting its text."""

    try:
        subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=15,
        )
        return True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _gpu_inventory() -> list[Dict[str, Any]]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_free_mib": int(fields[3]),
                }
            )
        except ValueError:
            continue
    return rows


def _venv_inventory(
    python_path: Path, project_root: Path, reference_root: Path
) -> Dict[str, Any]:
    """Describe the dedicated AlpaSim venv without importing its packages."""

    config_path = python_path.parent.parent / "pyvenv.cfg"
    config: Dict[str, str] = {}
    if config_path.is_file():
        try:
            for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = raw_line.partition("=")
                if separator:
                    config[key.strip().lower()] = value.strip()
        except OSError:
            config = {}
    include_system = config.get("include-system-site-packages", "").lower()
    resolved_python = python_path.resolve(strict=False)
    venv_root = python_path.parent.parent.resolve(strict=False)

    def is_below(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root.resolve())
            return True
        except ValueError:
            return False

    return {
        "python_launcher": str(python_path.absolute()),
        "resolved_base_interpreter": str(resolved_python),
        "venv_root": str(venv_root),
        "pyvenv_config": str(config_path.resolve(strict=False)),
        "present": python_path.is_file(),
        "include_system_site_packages": (
            include_system == "true" if include_system in {"true", "false"} else None
        ),
        "venv_inside_realsim_project": is_below(venv_root, project_root),
        "venv_inside_language2drive_v1": is_below(venv_root, reference_root),
        "base_interpreter_inside_language2drive_v1": is_below(
            resolved_python, reference_root
        ),
    }


def inspect_environment(project_root: Path) -> Dict[str, Any]:
    project_root = Path(project_root)
    upstream = project_root / "vendor" / "alpasim"
    alpasim_sha = _run(["git", "rev-parse", "HEAD"], upstream) if upstream.exists() else None
    reference = project_root.parent / "language2drive.v1"
    l2d_sha = _run(["git", "rev-parse", "HEAD"], reference) if reference.exists() else None
    scene_files = list(upstream.glob("data/nre-artifacts/**/*.usdz")) if upstream.exists() else []
    real_scene_files = []
    pointer_scene_files = []
    for path in scene_files:
        if path.stat().st_size < 1024:
            pointer_scene_files.append(str(path))
        else:
            real_scene_files.append(str(path))
    # The workspace accepts 3.11/3.12, but the pinned stock driver package is
    # narrower (`>=3.12,<3.13`), so live end-to-end readiness requires 3.12.
    python_ok = sys.version_info[:2] == (3, 12)
    token_environment = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    token_files = (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "stored_tokens",
        Path.home() / ".huggingface" / "token",
    )
    cached_token = any(path.is_file() and path.stat().st_size > 0 for path in token_files)
    docker_cli_available = shutil.which("docker") is not None
    # Some Docker CLI builds return zero for a formatted ``docker info`` even
    # when the daemon socket rejects the caller.  A non-empty server version is
    # therefore part of the reachability proof; exit status alone is not.
    docker_server_version = (
        _run(["docker", "info", "--format", "{{.ServerVersion}}"])
        if docker_cli_available
        else None
    )
    docker_daemon_reachable = bool(
        docker_server_version
        and docker_server_version.lower() not in {"null", "<no value>"}
    )
    docker_compose_available = docker_cli_available and _available(["docker", "compose", "version"])
    docker_buildx_available = docker_cli_available and _available(["docker", "buildx", "version"])
    nvidia_container_runtime_available = shutil.which("nvidia-container-runtime") is not None
    nvidia_container_cli_usable = bool(
        shutil.which("nvidia-container-cli")
        and _available(["nvidia-container-cli", "info"])
    )
    closed_loop_runtime_ready = bool(
        docker_daemon_reachable
        and docker_compose_available
        and docker_buildx_available
        and nvidia_container_runtime_available
        and nvidia_container_cli_usable
        and shutil.which("nvidia-smi")
    )
    gpu_inventory = _gpu_inventory()
    # The pinned AlpaSim documentation specifies about 40 GiB of VRAM for
    # standard Alpamayo 1.5 inference.  This is a host-execution gate, not a
    # claim about the smaller checkpoint's on-disk size.
    driver_min_vram_mib = 40 * 1024
    driver_memory_capacity_ready = any(
        row["memory_total_mib"] >= driver_min_vram_mib for row in gpu_inventory
    )
    driver_memory_headroom_ready = any(
        row["memory_free_mib"] >= driver_min_vram_mib for row in gpu_inventory
    )
    # The released A2 checkpoint is roughly 71.6 GB on disk and its standard
    # loader places the full BF16 model on one CUDA device. Treat an 80 GiB
    # device with at least 72 GiB currently free as the conservative pre-smoke
    # gate; an actual load smoke remains authoritative.
    alpamayo2_min_device_vram_mib = 80 * 1024
    alpamayo2_min_free_vram_mib = 72 * 1024
    alpamayo2_memory_capacity_ready = any(
        row["memory_total_mib"] >= alpamayo2_min_device_vram_mib
        for row in gpu_inventory
    )
    alpamayo2_memory_headroom_ready = any(
        row["memory_free_mib"] >= alpamayo2_min_free_vram_mib
        for row in gpu_inventory
    )
    alpasim_python = upstream / ".venv" / "bin" / "python"
    alpasim_python_environment = _venv_inventory(
        alpasim_python, project_root, reference
    )
    alpasim_environment_isolated = bool(
        alpasim_python_environment["present"]
        and alpasim_python_environment["include_system_site_packages"] is False
        and alpasim_python_environment["venv_inside_realsim_project"]
        and not alpasim_python_environment["venv_inside_language2drive_v1"]
        and not alpasim_python_environment[
            "base_interpreter_inside_language2drive_v1"
        ]
    )
    alpasim_environment_installed = alpasim_python.is_file() and _available(
        [
            str(alpasim_python),
            "-c",
            (
                "import importlib.util as u,sys;"
                "sys.exit(0 if u.find_spec('alpasim_wizard') and "
                "u.find_spec('alpasim_driver') else 1)"
            ),
        ]
    )
    realsim_plugin_registered = alpasim_python.is_file() and _available(
        [
            str(alpasim_python),
            "-c",
            (
                "from importlib.metadata import entry_points;"
                "models={e.name:e.value for e in entry_points(group='alpasim.models')};"
                "configs={e.name:e.value for e in entry_points(group='alpasim.configs')};"
                "assert models.get('l2d_alpamayo1_5','').endswith("
                "':LanguageConditionedAlpamayo15Model');"
                "assert models.get('l2d_alpamayo2_super','').endswith("
                "':LanguageConditionedAlpamayo2Model');"
                "assert models.get('l2d_privileged_oracle','').endswith("
                "':PrivilegedOraclePathModel');"
                "assert configs.get('l2d_realsim')=='language2drive_realsim.configs'"
            ),
        ]
    )
    checkpoint = _select_checkpoint_identity(
        environment_variable="L2D_ALPAMAYO_CHECKPOINT",
        candidates=[
            project_root / "model_zoo" / "alpamayo15-verified-7aba8293",
            reference / "model_zoo" / "alpamayo1.5" / "checkpoints",
        ],
        model_id="nvidia/Alpamayo-1.5-10B",
    )
    alpamayo2_checkpoint_path = Path(
        os.environ.get(
            "L2D_ALPAMAYO2_CHECKPOINT",
            str(project_root / "model_zoo" / "alpamayo2-super"),
        )
    )
    alpamayo2_checkpoint = checkpoint_identity(
        alpamayo2_checkpoint_path, "nvidia/Alpamayo2-Super"
    )
    base_execution_ready = bool(
        upstream.exists()
        and alpasim_sha == ALPASIM_COMMIT
        and python_ok
        and shutil.which("uv")
        and closed_loop_runtime_ready
        and alpasim_environment_isolated
        and alpasim_environment_installed
        and realsim_plugin_registered
        and real_scene_files
    )
    # Oracle qualification is deliberately independent of the optional
    # benchmark-model checkpoint.  The privileged oracle is CPU-only; the GPU
    # requirement here belongs to the NuRec renderer/runtime.  Keeping these
    # readiness domains separate prevents a missing Alpamayo checkpoint from
    # incorrectly blocking benchmark-infrastructure qualification.
    oracle_execution_ready = base_execution_ready
    alpamayo1_5_execution_ready = bool(
        base_execution_ready
        and driver_memory_headroom_ready
        and (checkpoint["complete"] or token_environment or cached_token)
    )
    alpamayo2_super_execution_ready = bool(
        base_execution_ready
        and alpamayo2_memory_capacity_ready
        and alpamayo2_memory_headroom_ready
        and (
            alpamayo2_checkpoint["complete"]
            or token_environment
            or cached_token
        )
    )
    benchmark_model_execution_ready = alpamayo1_5_execution_ready or alpamayo2_super_execution_ready
    scene_acquisition_ready = bool(
        upstream.exists()
        and alpasim_sha == ALPASIM_COMMIT
        and python_ok
        and alpasim_environment_isolated
        and alpasim_environment_installed
        and (token_environment or cached_token)
    )
    return {
        "schema_version": "l2d.realsim.inspect.v0.1",
        "project_root": str(project_root.resolve()),
        "language2drive_reference_sha": l2d_sha,
        "alpasim_checkout_present": upstream.exists(),
        "alpasim_git_sha": alpasim_sha,
        "alpasim_git_sha_pinned": alpasim_sha == ALPASIM_COMMIT,
        "python_version": sys.version.split()[0],
        "python_supported_by_alpasim": python_ok,
        "uv_available": shutil.which("uv") is not None,
        "docker_available": docker_cli_available,
        "docker_server_version": docker_server_version,
        "docker_daemon_reachable": docker_daemon_reachable,
        "docker_compose_available": docker_compose_available,
        "docker_buildx_available": docker_buildx_available,
        "nvidia_container_runtime_available": nvidia_container_runtime_available,
        "nvidia_container_cli_usable": nvidia_container_cli_usable,
        "closed_loop_runtime_ready": closed_loop_runtime_ready,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "gpu_inventory": gpu_inventory,
        "alpamayo1_5_standard_driver_min_vram_mib": driver_min_vram_mib,
        "alpamayo1_5_driver_memory_capacity_ready": driver_memory_capacity_ready,
        "alpamayo1_5_driver_memory_headroom_ready": driver_memory_headroom_ready,
        "alpamayo2_super_min_device_vram_mib": alpamayo2_min_device_vram_mib,
        "alpamayo2_super_min_free_vram_mib": alpamayo2_min_free_vram_mib,
        "alpamayo2_super_driver_memory_capacity_ready": (
            alpamayo2_memory_capacity_ready
        ),
        "alpamayo2_super_driver_memory_headroom_ready": (
            alpamayo2_memory_headroom_ready
        ),
        "alpasim_python_environment": alpasim_python_environment,
        "alpasim_environment_isolated_from_language2drive_v1": alpasim_environment_isolated,
        "alpasim_environment_installed": alpasim_environment_installed,
        "realsim_alpasim_plugin_registered": realsim_plugin_registered,
        "alpamayo1_5_checkpoint": checkpoint,
        "alpamayo2_super_checkpoint": alpamayo2_checkpoint,
        "hf_token_environment_present": token_environment,
        "hf_cached_token_present": cached_token,
        "hf_token_present": token_environment or cached_token,
        "scene_acquisition_ready": scene_acquisition_ready,
        "oracle_execution_ready": oracle_execution_ready,
        "benchmark_model_execution_ready": benchmark_model_execution_ready,
        "alpamayo1_5_execution_ready": alpamayo1_5_execution_ready,
        "alpamayo2_super_execution_ready": alpamayo2_super_execution_ready,
        "real_nurec_artifact_count": len(real_scene_files),
        "git_lfs_pointer_artifact_count": len(pointer_scene_files),
        # Backward-compatible release gate consumed by ``finalize``.  Formal
        # release qualification runs through the privileged oracle; model
        # smoke remains optional when no checkpoint is available.
        "release_execution_ready": oracle_execution_ready,
        "observed_api": {
            "camera_images": "EgodriverService.submit_image_observation",
            "ego_pose_speed_acceleration": "EgodriverService.submit_egomotion_observation",
            "route": "EgodriverService.submit_route",
            "free_form_language_field": False,
            "route_derived_command": True,
            "neutral_command_adapter": "LanguageConditionedEgoDriverService sets UNKNOWN and requires use_waypoint_commands=false",
            "eval_state": [
                "ego/actor trajectories",
                "vector map",
                "routes",
                "camera frames",
                "collision",
                "offroad",
                "black-image",
            ],
        },
    }
