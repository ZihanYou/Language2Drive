#!/usr/bin/env python3
"""Build a self-contained wheel from a clean staging tree; never vendor weights.

Backend code retains original licenses. The wheel preserves legacy paths by
extracting them with `language2drive init`, rather than installing colliding
legacy module names into the model environment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
TRACK_DIRS = ['language2drive.carla', 'language2drive.RealSim', 'language2drive.OpenloopTrack.nScenes', 'language2drive.OpenloopTrack.navsim', 'language2drive.OpenloopTrack.WOD-E2E']


def include(path):
    parts = path.parts
    if parts[0] not in TRACK_DIRS:
        return False
    if any(p.startswith('.') for p in parts):
        return False
    if any(p in ('__pycache__', '.venv', '.runtime-env', 'checkpoints', 'weights', 'node_modules') for p in parts):
        return False
    if parts[0] == 'language2drive.carla' and 'model_zoo' in parts and 'code' in parts:
        # External model source is not needed for the unified adapter.
        return path.name.startswith(('LICENSE', 'NOTICE'))
    if path.suffix in ('.pt', '.pth', '.ckpt', '.safetensors', '.whl', '.mp4', '.zip', '.gz', '.zst'):
        return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'dist')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    # Include the one newly added runtime template before its first commit too.
    tracked += ['language2drive.RealSim/src/language2drive_realsim/configs/driver/l2d_unified.yaml']
    # Evaluation workspaces need guides and adapter examples, not website media.
    # Keep the release independent of the size of the public case-video library.
    starters = ['README.md', 'LICENSE'] + [
        str(path.relative_to(ROOT))
        for folder, suffixes in (
            ('examples', {'.py', '.json', '.yaml', '.yml', '.md'}),
            ('docs', {'.md'}),
        )
        for path in (ROOT / folder).rglob('*')
        if path.is_file() and path.suffix in suffixes
        and not any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(ROOT).parts)
    ]
    tracked += starters
    starters += ['scripts/validate_gpu_campaign.py']
    tracked += ['scripts/validate_gpu_campaign.py', 'language2drive.OpenloopTrack.navsim/data/RELEASE_CHECKSUMS.json']
    with tempfile.TemporaryDirectory(prefix='l2d-wheel-') as temp:
        stage = Path(temp)
        for name in ('pyproject.toml', 'README.md', 'LICENSE'):
            shutil.copy2(ROOT / name, stage / name)
        shutil.copytree(ROOT / 'src', stage / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
        resources = stage / 'src/l2d_benchmark/resources/backends'
        manifest = {}
        for name in sorted(set(tracked)):
            rel = Path(name)
            source = ROOT / rel
            if not name or (not include(rel) and name not in starters) or not source.is_file():
                continue
            if source.is_symlink():
                raise ValueError(f'Backend symlink cannot be bundled: {name}')
            target = resources / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        resources.mkdir(parents=True, exist_ok=True)
        (resources / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
        subprocess.run([sys.executable, '-m', 'build', '--wheel', '--outdir', str(output), str(stage)], check=True)
    for wheel in output.glob('language2drive-*.whl'):
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        wheel.with_suffix(wheel.suffix + '.sha256').write_text(f'{digest}  {wheel.name}\n')
        print(f'{wheel.name}: {wheel.stat().st_size / 1024**2:.2f} MiB; {len(manifest)} verified backend files')


if __name__ == '__main__':
    main()
