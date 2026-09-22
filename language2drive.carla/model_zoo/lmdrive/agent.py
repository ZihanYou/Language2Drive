"""LMDrive official CARLA agent wrapped for the standardized L2D contract."""
from pathlib import Path
import sys

MODEL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ROOT.parent))
from language2drive_adapters.native_delegate import Language2DriveNativeDelegateAgent  # noqa: E402


def get_entry_point():
    return "LMDriveLanguage2DriveAgent"


class LMDriveLanguage2DriveAgent(Language2DriveNativeDelegateAgent):
    native_kind = "lmdrive"
    model_root = MODEL_ROOT
