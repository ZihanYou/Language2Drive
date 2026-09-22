"""
Bundled Emu3.5 VisionTokenizer (IBQ) for decoding visual tokens back to images.
Adapted from BAAI/Emu3.5 official code to be fully self-contained.
"""

from .ibq import IBQ
from .loader import load_vision_tokenizer

__all__ = ["IBQ", "load_vision_tokenizer"]
