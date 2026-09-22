"""SimLingo native closed-loop agent adapted to Language2Drive public context."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import carla
import numpy as np
import torch
from leaderboard.autoagents import autonomous_agent

MODEL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODEL_ROOT.parents[1]
for _path in (
    MODEL_ROOT.parent / "language2drive_adapters" / "vendor",
    MODEL_ROOT / "vendor",
    MODEL_ROOT / "code",
    REPO_ROOT,
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from team_code.language2drive_context import (  # noqa: E402
    L2D_TRACK_STANDARDIZED,
    PublicContextState,
    normalize_l2d_track,
)


def get_entry_point():
    return "SimLingoLanguage2DriveAgent"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Command:
    def __init__(self, value: int):
        self.value = value

    def __eq__(self, other):
        return self.value == getattr(other, "value", other)


def _command(instruction: str) -> tuple[int, float]:
    text = instruction.lower()
    lane = "lane" in text
    if "left" in text or "左" in text:
        return (5 if lane else 1), 2.8
    if "right" in text or "右" in text:
        return (6 if lane else 2), -2.8
    return (3 if "straight" in text else 4), 0.0


class _PublicTargetPlanner:
    def __init__(self, owner):
        self.owner = owner
        self.delegate = None

    @staticmethod
    def convert_gps_to_carla(_gps):
        return np.zeros(3, dtype=np.float64)

    def run_step(self, position):
        code, lateral = _command(self.owner.route_instruction)
        yaw = 0.0
        if self.delegate is not None and getattr(self.delegate, "filter_initialized", False):
            yaw = float(self.delegate.ukf.x[2])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        origin = np.asarray(position[:2], dtype=np.float64)
        result = []
        for distance in (6.0, 12.0, 18.0):
            target = origin + rotation.dot(np.array([distance, lateral]))
            result.append((np.r_[target, 0.0], _Command(code)))
        return result


class SimLingoLanguage2DriveAgent(autonomous_agent.AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self._l2d_carla_host = carla_host
        self._l2d_carla_port = carla_port

    def set_language2drive_context(self, public_context):
        state = getattr(self, "_l2d_context_state", None)
        if state is None:
            state = PublicContextState()
            self._l2d_context_state = state
        context = state.bind(public_context)
        self.evaluation_track = context.track
        self.route_instruction = context.instruction
        return context.to_dict()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        if getattr(self, "evaluation_track", None) == L2D_TRACK_STANDARDIZED:
            raise RuntimeError("set_global_plan() is forbidden on Language2Drive Standardized Track")
        self._global_plan = global_plan_gps
        self._global_plan_world_coord = global_plan_world_coord

    def setup(self, path_to_conf_file, route_index=None):  # noqa: ARG002
        context = getattr(getattr(self, "_l2d_context_state", None), "context", None)
        self.evaluation_track = context.track if context else normalize_l2d_track(os.getenv("L2D_TRACK"))
        self.route_instruction = context.instruction if context else ""
        if self.evaluation_track == L2D_TRACK_STANDARDIZED and context is None:
            raise RuntimeError("standardized track requires set_language2drive_context()")

        adapter_path = Path(str(path_to_conf_file).split("+", 1)[0]).resolve()
        cfg = json.loads(adapter_path.read_text(encoding="utf-8"))
        port_gpu_map = cfg.get("parallel_port_gpu_map", {})
        try:
            gpu_rank = int(port_gpu_map[str(int(self._l2d_carla_port))])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"SimLingo has no declared GPU for CARLA port {self._l2d_carla_port}"
            ) from error
        if gpu_rank < 0 or gpu_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"SimLingo GPU {gpu_rank} is unavailable; visible count={torch.cuda.device_count()}"
            )
        # Upstream constructs ``torch.device('cuda')`` without an index.  Set
        # the current device before importing/constructing the upstream model
        # so seven evaluator processes do not all allocate on GPU 0.
        torch.cuda.set_device(gpu_rank)
        self.gpu_rank = gpu_rank
        # Keep the model-local path instead of resolving the checkpoint symlink.
        # The official SimLingo agent locates its Hydra config three parents above
        # this path (``simlingo/.hydra/config.yaml``).
        checkpoint = (adapter_path.parent / cfg["checkpoint"]).absolute()
        backbone = (adapter_path.parent / cfg["internvl_backbone"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SimLingo checkpoint missing; run download.sh: {checkpoint}")
        if not (backbone / "model.safetensors").is_file():
            raise FileNotFoundError(f"InternVL2-1B backbone missing: {backbone}")

        team = MODEL_ROOT / "code" / "team_code"
        import team_code
        for name in ("transfuser_utils", "config_simlingo", "nav_planner", "simlingo_utils"):
            module = _load(f"team_code.{name}", team / f"{name}.py")
            setattr(team_code, name, module)
        _load("scenario_logger", team / "scenario_logger.py")
        upstream = _load("_l2d_simlingo_upstream_agent", team / "agent_simlingo.py")

        original_load = upstream.OmegaConf.load

        def local_config(path):
            loaded = original_load(path)
            loaded.model.vision_model.variant = str(backbone)
            loaded.model.language_model.variant = str(backbone)
            return loaded

        upstream.OmegaConf.load = local_config
        upstream.to_absolute_path = lambda _path: str(backbone)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ["SAVE_PATH"] = str(
            MODEL_ROOT / "runtime_logs" / f"gpu-{gpu_rank}-port-{self._l2d_carla_port}"
        )

        self.delegate = upstream.LingoAgent(
            self._l2d_carla_host, self._l2d_carla_port
        )
        # Upstream uses ROUTES only for log-directory naming.  Hide the real
        # evaluator route identity on the standardized track.
        original_routes = os.environ.get("ROUTES")
        if self.evaluation_track == L2D_TRACK_STANDARDIZED:
            os.environ["ROUTES"] = "data/benchmarks/language2drive/public.xml"
        try:
            # In the official agent, a non-None ``route_index`` turns
            # ``save_path`` into a pathlib.Path and later concatenates it with
            # strings.  Use its supported ``checkpoint+save_root`` form while
            # keeping route_index unset.
            self.delegate.setup(f"{checkpoint}+l2d", None)
        finally:
            if original_routes is None:
                os.environ.pop("ROUTES", None)
            else:
                os.environ["ROUTES"] = original_routes
        self.delegate.route_instruction = self.route_instruction
        self.delegate.custom_prompt = self.route_instruction
        self.delegate.user_flag = 2
        self.delegate.save_path_metric = None
        self.delegate.get_metric_info = lambda: {}
        planner = _PublicTargetPlanner(self)
        planner.delegate = self.delegate

        def safe_init(delegate):
            delegate._route_planner = planner
            delegate.initialized = True
            delegate.metric_info = {}

        self.delegate._init = types.MethodType(safe_init, self.delegate)
        self.controller_source = "simlingo:setup"

    def sensors(self):
        return self.delegate.sensors()

    def run_step(self, input_data, timestamp, sensors=None):
        self.delegate.custom_prompt = self.route_instruction
        self.delegate.user_flag = 2
        control = self.delegate.run_step(input_data, timestamp, sensors)
        self.controller_source = "simlingo:native_model_controller"
        return control

    def destroy(self, results=None):  # noqa: ARG002
        delegate = getattr(self, "delegate", None)
        if delegate is not None:
            # Upstream destroy assumes setup reached model construction.  Make
            # cleanup safe for partial setup and avoid its optional logging-only
            # config fields, which are not needed by this adapter.
            for attr in ("model", "processor", "config"):
                if hasattr(delegate, attr):
                    delattr(delegate, attr)
            self.delegate = None
