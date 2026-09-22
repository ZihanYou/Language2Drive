"""
Load the Emu3.5 VisionTokenizer (IBQ) model from a local checkpoint.
"""

import os
import torch
from omegaconf import OmegaConf

from .ibq import IBQ


def load_vision_tokenizer(model_root: str, device: str = "cuda:0") -> IBQ:
    """
    Load VQ model from local config.yaml + model.ckpt under
    ``model_root/Emu3.5-VisionTokenizer/``.
    """
    vq_path = os.path.join(model_root, "Emu3.5-VisionTokenizer")
    cfg_path = os.path.join(vq_path, "config.yaml")
    ckpt_path = os.path.join(vq_path, "model.ckpt")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = OmegaConf.load(cfg_path)
    tokenizer = IBQ(**cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    tokenizer.load_state_dict(ckpt)
    tokenizer.eval().to(device)
    return tokenizer
