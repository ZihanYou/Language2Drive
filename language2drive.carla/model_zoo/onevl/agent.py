"""OneVL latent-CoT trajectory bridge for Language2Drive."""
from pathlib import Path
import sys

MODEL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ROOT.parent))
from language2drive_adapters import Language2DriveTrajectoryAgent  # noqa: E402


def get_entry_point():
    return "OneVLLanguage2DriveAgent"


class OneVLLanguage2DriveAgent(Language2DriveTrajectoryAgent):
    model_root = MODEL_ROOT
