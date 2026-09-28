"""Download an external checkpoint from the Hugging Face mirror."""
import argparse
import os
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--revision', required=True, help='Pin an upstream commit')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo-type', choices=('model', 'dataset'), default='model')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=args.repo, repo_type=args.repo_type,
                      revision=args.revision, local_dir=args.output,
                      endpoint='https://hf-mirror.com', max_workers=args.workers)

if __name__ == '__main__':
    main()
