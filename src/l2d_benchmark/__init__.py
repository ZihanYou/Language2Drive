"""Public SDK. The legacy CARLA package keeps its original namespace."""
from .adapter import Request, OutputSpec, load_adapter
__version__ = "0.1.0"
__all__ = ["Request", "OutputSpec", "load_adapter"]
