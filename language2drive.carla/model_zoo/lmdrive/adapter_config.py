"""Absolute, model-local paths consumed by the released LMDrive agent."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class GlobalConfig:
    turn_KP, turn_KI, turn_KD, turn_n = 1.25, 0.75, 0.3, 40
    speed_KP, speed_KI, speed_KD, speed_n = 5.0, 0.5, 1.0, 40
    max_throttle, brake_speed, brake_ratio, clip_delta = 0.75, 0.1, 1.1, 0.35
    llm_model = str(ROOT / "checkpoints" / "llava-v1.5-7b")
    preception_model = "memfuser_baseline_e1d3_return_feature"
    preception_model_ckpt = str(ROOT / "checkpoints" / "vision_encoder.pth.tar")
    lmdrive_ckpt = str(ROOT / "checkpoints" / "lmdrive_llava.pth")
    agent_use_notice = False
    sample_rate = 2

    def __init__(self, **kwargs):
        missing = [path for path in (self.llm_model, self.preception_model_ckpt, self.lmdrive_ckpt) if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("LMDrive assets missing; run download.sh: " + ", ".join(missing))
        for key, value in kwargs.items():
            setattr(self, key, value)
