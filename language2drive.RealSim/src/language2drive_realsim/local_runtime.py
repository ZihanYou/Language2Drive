"""Prepare a loopback-only hybrid deployment from official Wizard output.

Only NRE needs its application container. Other services use the pinned project
venv. This exploratory runtime does not create qualified PCUs or witnesses.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import uuid

from .feasibility import prepare_feasibility, prepare_additional_feasibility
from .io import atomic_json


def prepare_local_runtime(root, wizard_dir, output_dir, gpu, proposal_key=None,
                          runtime_port=6008, driver_port=50052, metrics_port=6000, *, review_batch=None,
                          agent_candidate=None):
    import yaml

    root, wizard_dir, output_dir = [Path(p).resolve() for p in (root, wizard_dir, output_dir)]
    if type(gpu) is not int or gpu < 0:
        raise ValueError("gpu must be an explicit non-negative host GPU index")
    ports = (runtime_port, driver_port, metrics_port, 6005, 6006, 6007)
    if any(type(port) is not int or not 1 <= port <= 65535 for port in ports) or len(set(ports)) != len(ports):
        raise ValueError("service ports must be valid and distinct")
    if not output_dir.is_relative_to(root / ".runtime"):
        raise ValueError("output must be inside the project's .runtime directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty; preserve previous run evidence")
    if agent_candidate is not None:
        from .agent_curation import load_agent_candidate
        if review_batch is not None:
            raise ValueError("agent authority and a human review batch are distinct modes")
        queue = {"candidates": [load_agent_candidate(root, agent_candidate)]}
    else:
        queue = (prepare_feasibility(root) if review_batch is None else
                 prepare_additional_feasibility(root, review_batch))
    candidates = [row for row in queue["candidates"] if not row["input_errors"]
                  and (proposal_key is None or row["proposal_key"] == proposal_key)]
    if not candidates:
        raise ValueError("no currently reviewed, input-ready candidate matches")
    candidate = candidates[0]

    def read(name):
        return yaml.safe_load((wizard_dir / name).read_text())

    user = read("generated-user-config-0.yaml")
    camera_timing = None
    if agent_candidate is not None:
        from .camera_timing import camera_timing_preflight
        camera_timing = camera_timing_preflight(candidate, [c["logical_id"] for c in user["simulation_config"]["cameras"]])
    network = read("generated-network-config.yaml")
    driver = read("driver-config.yaml")
    if driver["model"]["model_type"] != "l2d_privileged_oracle":
        raise ValueError("Wizard must use driver=l2d_privileged_oracle")
    driver["host"], driver["port"] = "127.0.0.1", driver_port
    driver["output_dir"] = str(output_dir / "oracle-driver")
    metadata = read("run_metadata.yaml")
    metadata["source_wizard_run_uuid"] = metadata["run_uuid"]
    metadata["run_uuid"] = str(uuid.uuid4())
    metadata["run_name"] = "LR-" + metadata["run_uuid"]
    metadata["run_time"] = datetime.now(timezone.utc).isoformat()
    metadata["run_dir"] = str(output_dir)
    renderer = deepcopy(read("docker-compose.yaml")["services"]["renderer-0"])
    if renderer["image"] != "nvcr.io/nvidia/nre/nre-ga:26.04":
        raise ValueError("unexpected renderer image; review pinned Wizard image first")
    artifact = Path(candidate["scene_execution_proposal_not_frozen"]["artifact_path"])
    user["scene_provider"]["usdz"]["data_dir"] = str(artifact.parent)
    user["scene_provider"]["usdz"]["artifact_cache_size"] = 2
    user["simulation_config"]["force_gt_frame_cache"]["cache_dir"] = str(output_dir / "frame-cache")
    user["scenes"] = [{"scene_id": candidate["scene_id_private"],
                       "start_time_offset_us": candidate["scene_execution_proposal_not_frozen"]["start_time_offset_us"]}]
    user["nr_workers"] = 1
    user["endpoints"]["sensorsim_cache_size"] = "2"
    for name in ("renderer", "physics", "controller", "driver", "trafficsim"):
        user["endpoints"][name]["n_concurrent_rollouts"] = 1
    if any(user["endpoints"][name]["skip"] for name in ("renderer", "physics", "controller")):
        raise ValueError("real renderer, physics and controller must not be skipped")
    if not user["endpoints"]["driver"]["skip"] or network["driver"]["endpoints"]:
        raise ValueError("Wizard must use driver_source=external_dynamic")
    # The pinned daemon writes startup driver versions into every ASL. Dynamic
    # overrides leave that authority as <skip>, even when inference executes.
    # Register our dedicated host driver before startup to retain real identity.
    user["endpoints"]["driver"]["skip"] = False
    network["driver"]["endpoints"] = [{"address": f"127.0.0.1:{driver_port}", "managed": False}]
    for name, port in {"renderer": 6005, "physics": 6006, "controller": 6007}.items():
        network[name]["endpoints"] = [{"address": f"127.0.0.1:{port}", "managed": True}]
    user["prometheus"]["url"] = "http://127.0.0.1:6001"
    user["prometheus"]["worker_ports"] = [metrics_port]
    renderer.pop("networks", None)
    renderer["ports"] = ["127.0.0.1:6005:6005"]
    renderer["deploy"]["resources"]["reservations"]["devices"] = [
        {"driver": "nvidia", "device_ids": [str(gpu)], "capabilities": ["gpu"]}]
    command = renderer["command"]
    if len(command) != 2 or command[0] != "-c":
        raise ValueError("unexpected Wizard renderer command format")
    script = command[1]
    if len(re.findall(r"--port=6005\b", script)) != 1:
        raise ValueError("Wizard must use baseport=6000 for this loopback deployment")
    for flag, value in (("cache-size", "2"), ("max-workers", "1")):
        script, count = re.subn(r"--" + flag + r"=\d+\b", "--" + flag + "=" + value, script)
        if count != 1:
            raise ValueError("missing or duplicate renderer flag: " + flag)
    script, count = re.subn(r"--artifact-glob=\S+", lambda _: "--artifact-glob=" + shlex.quote("/mnt/nre-data/" + artifact.name), script)
    if count != 1:
        raise ValueError("missing or duplicate renderer artifact glob")
    renderer["command"] = ["-c", script]
    mounts = []
    for mount in renderer["volumes"]:
        source, destination, *mode = mount.split(":")
        if destination == "/mnt/nre-data":
            source = str(artifact.parent)
        mounts.append(source + ":" + destination + ":ro")
    renderer["volumes"] = mounts
    executable = str(root / "vendor/alpasim/.venv/bin/python")
    commands = {
        "renderer": ["docker", "compose", "-f", str(output_dir / "renderer-compose.yaml"), "up", "renderer-0"],
        "physics": [executable, "-m", "alpasim_physics.server", "--host", "127.0.0.1", "--port", "6006",
                    "--artifact-glob", str(artifact), "--use-ground-mesh=true", "--cache-size", "2"],
        "controller": [executable, "-m", "alpasim_controller.server", "--host", "127.0.0.1", "--port", "6007",
                       "--config", str(output_dir / "controller-config.yaml"), "--log_dir", str(output_dir)],
        "runtime": [executable, "-m", "language2drive_realsim.host_runtime", "--user-config", str(output_dir / "user-config.yaml"),
                    "--network-config", str(output_dir / "network-config.yaml"), "--eval-config", str(output_dir / "eval-config.yaml"),
                    "--log-dir", str(output_dir), "--serve", "--listen-address", f"127.0.0.1:{runtime_port}"],
    }
    manifest = {"schema_version": "l2d.realsim.local_runtime.private.v0.1",
                "scope": "exploratory_runtime_not_oracle_witness", "qualified_pcu_count": 0,
                "candidate": candidate, "gpu_index": gpu, "commands": commands,
                "runtime_address": f"127.0.0.1:{runtime_port}", "driver_address": f"127.0.0.1:{driver_port}",
                "service_environment": {"physics": {"CUDA_VISIBLE_DEVICES": str(gpu), "WARP_CACHE_PATH": str(output_dir / "warp-cache")},
                                        "controller": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                        "runtime": {"CUDA_VISIBLE_DEVICES": ""}},
                "not_started": True,
                "requirements_remaining": ["GPU allocation approval and capacity check", "NRE application smoke",
                                           "private oracle plans and sidecar", "dual-arm real closed-loop validation"]}
    if review_batch is not None:
        manifest["review_batch"] = review_batch
        manifest["review_input_paths"] = queue["review_input_paths"]
    if agent_candidate is not None:
        manifest["camera_timing_preflight"] = camera_timing
        manifest["curation_authority"] = candidate["curation_authority"]
        manifest["agent_candidate_source"] = str(Path(agent_candidate).resolve())
        manifest["final_human_review_required"] = True
    configs = {"user-config.yaml": user, "network-config.yaml": network,
               "renderer-compose.yaml": {"services": {"renderer-0": renderer}},
               "controller-config.yaml": read("controller-config.yaml"),
               "eval-config.yaml": read("eval-config.yaml"), "driver-config.yaml": driver,
               "run_metadata.yaml": metadata}
    configs["eval-config.yaml"]["num_processes"] = 1
    configs["eval-config.yaml"]["video"]["render_video"] = False
    manifest["driver_binding"] = "registered_static_for_asl_authority"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        (output_dir / name).write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(output_dir / "launch.private.json", manifest)
    return manifest
