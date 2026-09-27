"""Portable paths shared by construction and evaluation tools."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('L2D_NUSCENES_ROOT', str(PROJECT_ROOT / 'data/nuscenes'))).expanduser().resolve()
ARTIFACT_ROOT = Path(os.environ.get('L2D_ARTIFACT_ROOT', str(PROJECT_ROOT / 'artifacts/nuscenes_test'))).expanduser().resolve()
RELEASE_ROOT = Path(os.environ.get('L2D_RELEASE_DIR', str(PROJECT_ROOT / 'releases/nuscenes-test-v1.0.0'))).expanduser().resolve()
