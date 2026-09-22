"""Route-private-safe wrappers around released LMDrive agents."""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import carla
import numpy as np
from leaderboard.autoagents import autonomous_agent

from team_code.language2drive_context import (
    L2D_TRACK_STANDARDIZED,
    PublicContextState,
    normalize_l2d_track,
)


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_light_stubs() -> None:
    if "easydict" not in sys.modules:
        easy = types.ModuleType("easydict")

        class EasyDict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__

        easy.EasyDict = EasyDict
        sys.modules["easydict"] = easy
    if "pygame" not in sys.modules:
        pygame = types.ModuleType("pygame")
        pygame.HWSURFACE = 0
        pygame.DOUBLEBUF = 0
        pygame.init = lambda: None
        pygame.quit = lambda: None
        pygame.font = types.SimpleNamespace(init=lambda: None)
        pygame.time = types.SimpleNamespace(Clock=lambda: object())

        class _Display:
            def blit(self, *_args, **_kwargs):
                return None

        pygame.display = types.SimpleNamespace(
            set_mode=lambda *_args, **_kwargs: _Display(),
            set_caption=lambda *_args: None,
            flip=lambda: None,
        )
        pygame.event = types.SimpleNamespace(get=lambda: [])
        pygame.surfarray = types.SimpleNamespace(make_surface=lambda value: value)
        sys.modules["pygame"] = pygame


def _load_native_lavis(lavis_root: Path, bert_tokenizer: Path) -> None:
    """Load only the driving model, bypassing LAVIS' eager model zoo imports.

    Both released agents import ``lavis.common.registry`` and then look up the
    registered ``vicuna_drive`` class.  Importing the top-level upstream LAVIS
    package also imports unrelated diffusion models whose private Transformers
    dependencies are incompatible with the benchmark environment.  A package
    shell keeps normal submodule imports working while avoiding that eager
    import tree.
    """

    for name in tuple(sys.modules):
        if name == "lavis" or name.startswith("lavis."):
            del sys.modules[name]

    lavis = types.ModuleType("lavis")
    lavis.__path__ = [str(lavis_root)]
    lavis.__package__ = "lavis"
    sys.modules["lavis"] = lavis

    models_root = lavis_root / "models"
    models = types.ModuleType("lavis.models")
    models.__path__ = [str(models_root)]
    models.__package__ = "lavis.models"
    sys.modules["lavis.models"] = models

    base_model = importlib.import_module("lavis.models.base_model")
    models.BaseModel = base_model.BaseModel

    registry_module = importlib.import_module("lavis.common.registry")
    registry = registry_module.registry
    registry.register_path("library_root", str(lavis_root))
    registry.register_path("repo_root", str(lavis_root.parent))
    registry.register_path("cache_root", str(lavis_root.parent / ".cache"))
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])

    drive_module = importlib.import_module("lavis.models.drive_models.drive")

    def init_tokenizer(cls, truncation_side="right"):  # noqa: ARG001
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(
            str(bert_tokenizer),
            truncation_side=truncation_side,
            local_files_only=True,
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    drive_module.Blip2Base.init_tokenizer = classmethod(init_tokenizer)

    qformer_module = importlib.import_module(drive_module.Blip2Base.__module__)

    def init_qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        import torch

        encoder_config = qformer_module.BertConfig.from_pretrained(
            str(bert_tokenizer), local_files_only=True
        )
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        qformer = qformer_module.BertLMHeadModel.from_pretrained(
            str(bert_tokenizer),
            config=encoder_config,
            local_files_only=True,
        )
        query_tokens = torch.nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(
            mean=0.0, std=encoder_config.initializer_range
        )
        return qformer, query_tokens

    drive_module.Blip2Base.init_Qformer = classmethod(init_qformer)


class _NoDisplay:
    def __init__(self):
        self._surface = None

    def run_interface(self, _input):
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def _quit(self):
        return None


class _Command:
    def __init__(self, value: int):
        self.value = value

    def __eq__(self, other):
        return self.value == getattr(other, "value", other)


def _instruction_command(text: str) -> tuple[int, float]:
    value = text.lower()
    lane = "lane" in value
    if "left" in value or "左" in value:
        return (5 if lane else 1), 2.8
    if "right" in value or "右" in value:
        return (6 if lane else 2), -2.8
    return (3 if "straight" in value else 4), 0.0


class _PublicInstructionPlanner:
    def __init__(self, owner):
        self.owner = owner

    def command2instruct(self, *_args):
        return self.owner.route_instruction

    def pos2notice(self, *_args):
        return ""

    def traffic_notice(self, *_args):
        return ""

    def command2mislead(self, *_args):
        return ""


class _InstructionTargetPlanner:
    """Produces only an instruction-conditioned target, never a route target."""

    mean = np.zeros(2, dtype=np.float64)
    scale = np.ones(2, dtype=np.float64)
    route: list[Any] = []

    def __init__(self, owner):
        self.owner = owner
        self.compass = 0.0

    def run_step(self, pos):
        command, lateral = _instruction_command(self.owner.route_instruction)
        theta = self.compass + math.pi / 2.0
        rotation = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        local = np.array([12.0, lateral], dtype=np.float64)
        target = np.asarray(pos, dtype=np.float64) + rotation.dot(local)
        return target, _Command(command)


def _load_native(kind: str, model_root: Path):
    _install_light_stubs()
    if kind == "lmdrive":
        root = model_root / "code"
        team = root / "leaderboard" / "team_code"
        extra = (root / "LAVIS", root / "vision_encoder")
        agent_path = team / "lmdriver_agent.py"
        class_name = "LMDriveAgent"
        helpers = ("planner", "pid_controller")
    else:
        raise ValueError(f"unknown native delegate {kind}")
    for path in extra:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    _load_native_lavis(
        extra[0] / "lavis", model_root / "checkpoints" / "bert-base-uncased"
    )
    import team_code

    for helper in helpers:
        module = _load_file(f"team_code.{helper}", team / f"{helper}.py")
        setattr(team_code, helper, module)
    module = _load_file(f"_l2d_{kind}_upstream_agent", agent_path)
    module.SAVE_PATH = None
    module.DisplayInterface = _NoDisplay
    return getattr(module, class_name)


class Language2DriveNativeDelegateAgent(autonomous_agent.AutonomousAgent):
    """Delegates perception/planning to released native CARLA model code."""

    native_kind = ""
    model_root = Path(".")

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self._l2d_carla_host = carla_host
        self._l2d_carla_port = carla_port

    def set_language2drive_context(self, public_context: dict[str, Any]) -> dict[str, Any]:
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
            raise RuntimeError(
                "set_global_plan() is forbidden on Language2Drive Standardized Track"
            )
        self._global_plan = global_plan_gps
        self._global_plan_world_coord = global_plan_world_coord

    def setup(self, path_to_conf_file: str, route_index=None) -> None:  # noqa: ARG002
        context = getattr(getattr(self, "_l2d_context_state", None), "context", None)
        self.evaluation_track = (
            context.track if context is not None else normalize_l2d_track(os.getenv("L2D_TRACK"))
        )
        self.route_instruction = context.instruction if context is not None else ""
        if self.evaluation_track == L2D_TRACK_STANDARDIZED and context is None:
            raise RuntimeError("standardized track requires set_language2drive_context()")
        config_path = Path(str(path_to_conf_file).split("+", 1)[0]).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        vendor = self.model_root / "vendor"
        shared_vendor = self.model_root.parent / "language2drive_adapters" / "vendor"
        for path in (shared_vendor, vendor):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        cls = _load_native(self.native_kind, self.model_root)
        self.delegate = cls(self._l2d_carla_host, self._l2d_carla_port)
        self.delegate.setup(str(config_path))
        self.delegate.route_instruction = self.route_instruction
        self.delegate.evaluation_track = self.evaluation_track
        self.controller_source = f"{self.native_kind}:setup"

        owner = self
        route_planner = _InstructionTargetPlanner(owner)
        instruction_planner = _PublicInstructionPlanner(owner)

        def safe_init(delegate):
            delegate._route_planner = route_planner
            delegate._instruction_planner = instruction_planner
            delegate.initialized = True
            delegate.town_id = "Language2Drive"
            delegate.scenario_cofing_name = ""

        def safe_position(delegate, tick_data):
            route_planner.compass = float(tick_data.get("compass", 0.0))
            return np.zeros(2, dtype=np.float64)

        self.delegate._init = types.MethodType(safe_init, self.delegate)
        self.delegate._get_position = types.MethodType(safe_position, self.delegate)

    def sensors(self):
        return self.delegate.sensors()

    def run_step(self, input_data, timestamp, sensors=None):
        self.delegate.route_instruction = self.route_instruction
        control = self.delegate.run_step(input_data, timestamp)
        self.controller_source = f"{self.native_kind}:native_model_controller"
        return control

    def destroy(self, results=None):  # noqa: ARG002
        delegate = getattr(self, "delegate", None)
        if delegate is not None:
            try:
                # Both upstream agents assume setup reached ``self.net``.
                # Benchmark cleanup must also work after a partial setup failure.
                if hasattr(delegate, "net"):
                    delegate.destroy()
                else:
                    display = getattr(delegate, "_hic", None)
                    if display is not None and hasattr(display, "_quit"):
                        display._quit()
            finally:
                self.delegate = None
