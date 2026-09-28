#!/usr/bin/env python3
"""Download the official OpenScene test sensors and frozen model components.

Run from anywhere. Downloads resume safely and are stored only below this track.
External repositories and checkpoints retain their original licenses.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TRACK = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

CHECKPOINTS = {
    "qwen_drive": ("Qwen/Qwen-Drive-1.0-4B", "qwen-drive-1.0-4b",
                   "28484089a7cc8c335cf5089fb0745cf7c49b6eaa", ["planner-sft/*", "perception/*"]),
    "qwen_base": ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen2.5-vl-3b-instruct",
                  "66285546d2b821cf421d4f5eb2576359d3770cd3", []),
    "autovla": ("Zewei-Zhou/AutoVLA", "autovla",
                "a7d7ba3ed7529b248d2694c2defa31b35208340f", []),
    "drivema": ("zwc2003/DriveMA-2B", "drivema-2b",
                "d5f4905d2032da943a7ae8f22d5b2b8efbe4b0f7", []),
    "recog_vlm": ("owl10/ReCogDrive-VLM-2B", "recogdrive-vlm-2b",
                  "16873acca08e3c04ab229b3d973f39aeba9db68d", []),
    "recog_rl": ("owl10/ReCogDrive-2B-RL", "recogdrive-2b-rl",
                 "7346ede73f03ed577811024f1a5d16fb063a47b9", []),
    "onevl": ("xiaomi-research/OneVL_NAVSIM", "onevl-navsim",
              "c1e893e9a03b8312f9328d2b51267836a9f7bf8a", []),
}

REPOSITORIES = {
    "qwen_drive": ("QwenLM/Qwen-Drive-1.0", "qwen_drive",
                   "28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0"),
    "autovla": ("ucla-mobility/AutoVLA", "autovla",
                 "ba34eed74ce6729e7986592d0e66cbaca397b4fa"),
    "drivema": ("Tsinghua-MARS-Lab/DriveMA", "drivema",
                 "de20e6c64878bc3413d26d69362c3be9b3a91533"),
    "recogdrive": ("xiaomi-research/recogdrive", "recogdrive",
                   "6b8d8f5e01346c71094651c81dcaf66405dbc04e"),
    "onevl": ("xiaomi-research/onevl", "onevl",
              "f4e6f05572b2f66611259bcd5563529833f30469"),
}


def direct_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.lower().endswith("proxy")}
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    return env


def run(args: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(args), flush=True)
    if not dry_run:
        subprocess.run(args, cwd=TRACK, env=env, check=True)


def checkout_repositories(selected: list[str], env: dict[str, str], dry_run: bool,
                          git_mirror: str) -> None:
    dest_root = TRACK / "model_eval/repos"
    dest_root.mkdir(parents=True, exist_ok=True)
    for key in selected:
        name, dest_name, revision = REPOSITORIES[key]
        dest = dest_root / dest_name
        if not (dest / ".git").is_dir():
            url = f"{git_mirror.rstrip('/')}/https://github.com/{name}.git" if git_mirror else f"https://github.com/{name}.git"
            run(["git", "clone", url, str(dest)], env=env, dry_run=dry_run)
        if not dry_run:
            current = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                              env=env, text=True).strip()
            if current == revision:
                print(f"verified {key}: {revision}", flush=True)
                continue
        run(["git", "-C", str(dest), "fetch", "origin", revision], env=env, dry_run=dry_run)
        run(["git", "-C", str(dest), "checkout", "--detach", revision], env=env, dry_run=dry_run)
        if not dry_run:
            current = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                              env=env, text=True).strip()
            if current != revision:
                raise RuntimeError(f"repository revision mismatch for {key}: {current}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="download camera, LiDAR, model repositories and checkpoints")
    parser.add_argument("--sensors", action="store_true",
                        help="download all 32 test sensor shards per selected modality")
    parser.add_argument("--cameras-only", action="store_true",
                        help="skip LiDAR when downloading sensors (the five baseline adapters use cameras)")
    parser.add_argument("--repos", action="store_true", help="download pinned official model code")
    parser.add_argument("--weights", action="store_true", help="download pinned official checkpoints")
    parser.add_argument("--models", nargs="+", choices=sorted(CHECKPOINTS),
                        help="limit checkpoint downloads to these components")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--git-mirror", default="https://gh-proxy.com",
                        help="GitHub mirror prefix; pass an empty string for direct GitHub")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not (args.all or args.sensors or args.repos or args.weights or args.models):
        parser.error("select --all, --sensors, --repos, --weights, or --models")
    env = direct_environment()
    selected_checkpoints = args.models or list(CHECKPOINTS)
    if args.all or args.sensors:
        cmd = [PYTHON, str(TRACK / "scripts/download_test_modelscope.py"),
               "--root", str(TRACK / "assets"), "--workers", str(args.workers),
               "--start", "0", "--stop", "32"]
        if args.cameras_only:
            cmd.extend(["--modalities", "camera"])
        run(cmd, env=env, dry_run=args.dry_run)
    if args.all or args.repos:
        checkout_repositories(list(REPOSITORIES), env, args.dry_run, args.git_mirror)
    if args.all or args.weights or args.models:
        if args.dry_run:
            for key in selected_checkpoints:
                repo, dest, revision, ignores = CHECKPOINTS[key]
                cmd = [PYTHON, str(TRACK / "model_eval/download_weight.py"),
                       repo, str(TRACK / "model_eval/weights" / dest),
                       "--revision", revision, "--workers", str(args.workers)]
                for pattern in ignores:
                    cmd.extend(["--ignore", pattern])
                run(cmd, env=env, dry_run=True)
            return
        try:
            import huggingface_hub  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Install huggingface-hub first: python -m pip install huggingface-hub==0.36.0") from exc
        for key in selected_checkpoints:
            repo, dest, revision, ignores = CHECKPOINTS[key]
            cmd = [PYTHON, str(TRACK / "model_eval/download_weight.py"),
                   repo, str(TRACK / "model_eval/weights" / dest),
                   "--revision", revision, "--workers", str(args.workers)]
            for pattern in ignores:
                cmd.extend(["--ignore", pattern])
            run(cmd, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
