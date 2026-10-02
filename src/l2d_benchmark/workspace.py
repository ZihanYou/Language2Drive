"""Source checkout discovery and explicit extraction of wheel-bundled backends."""
from pathlib import Path
import hashlib
import json
import shutil

TRACKS = {
    "carla": "language2drive.carla",
    "realsim": "language2drive.RealSim",
    "nuscenes": "language2drive.OpenloopTrack.nScenes",
    "navsim": "language2drive.OpenloopTrack.navsim",
    "wod": "language2drive.OpenloopTrack.WOD-E2E",
}

def discover(raw=None):
    if raw:
        root = Path(raw).expanduser().resolve()
        if not all((root / name).is_dir() for name in TRACKS.values()):
            raise ValueError("Workspace must contain the five original track directories; run language2drive init --workspace PATH")
        return root
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for root in (start, *start.parents):
            if all((root / name).is_dir() for name in TRACKS.values()):
                return root
    raise ValueError("No workspace found. Run language2drive init --workspace ./Language2Drive, then pass --workspace ./Language2Drive")


def initialize(destination):
    source = Path(__file__).parent / "resources" / "backends"
    manifest = source / "manifest.json"
    if not manifest.is_file():
        # Editable installations already have a workspace.
        return discover(destination)
    root = Path(destination).expanduser().resolve()
    inventory = json.loads(manifest.read_text(encoding="utf-8"))
    for name, digest in inventory.items():
        target = root / name
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError("Refusing symlink in workspace destination")
        if target.exists() and (not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest):
            raise ValueError(f"Existing workspace file differs: {name}; choose a fresh destination")
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Wheel resource checksum mismatch: {name}")
    for name in inventory:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source / name, target)
    return root
