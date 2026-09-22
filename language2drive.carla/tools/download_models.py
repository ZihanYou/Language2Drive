#!/usr/bin/env python3
"""Download selected model assets from their original Hugging Face publishers."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    spec = json.loads((ROOT / 'model_zoo/WEIGHTS.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(spec))
    parser.add_argument('--endpoint', default=os.getenv('HF_ENDPOINT', 'https://hf-mirror.com'))
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(spec[args.model], indent=2))
        return
    from huggingface_hub import snapshot_download
    for item in spec[args.model]:
        target = ROOT / 'model_zoo' / args.model / item['directory']
        options = dict(repo_id=item['repo_id'], revision=item['revision'], local_dir=target,
                       allow_patterns=item.get('allow'), max_workers=4)
        snapshot_download(**options, endpoint=args.endpoint)
    base = ROOT / 'model_zoo' / args.model
    if args.model == 'onevl':
        subprocess.run([sys.executable, str(base / 'prune_inference_index.py'),
                        str(base / 'checkpoints/OneVL_NAVSIM')], check=True)
    if args.model == 'lmdrive':
        for name, target in [('lmdrive_llava.pth', 'lmdrive_adapter/llava-v1.5-checkpoint.pth'),
                             ('vision_encoder.pth.tar', 'vision_encoder/vision-encoder-r50.pth.tar')]:
            path = base / 'checkpoints' / name
            if not path.exists():
                path.symlink_to(target)
    print('Model files downloaded. Run model_zoo/preflight.py --model ' + args.model)

if __name__ == '__main__':
    main()
