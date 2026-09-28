"""Resume and verify OpenScene v1.1 test sensors from the ModelScope mirror.

All requests disable proxy use. Each archive is checked against its mirrored
Content-Length, SHA-256 hashed, extracted, and only then removed. A per-shard
marker makes the command safe to restart after interruption.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

BASE = "https://modelscope.cn/datasets/OpenDriveLab/OpenScene/resolve/master/openscene-v1.1"
EXTRACT_LIMIT = threading.Semaphore(2)
PRINT_LOCK = threading.Lock()


def direct_env():
    return {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}


def url_for(modality, index):
    name = f"openscene_sensor_test_{modality}_{index}.tgz"
    return f"{BASE}/openscene_sensor_test_{modality}/{name}", name


def remote_info(url, env):
    for attempt in range(10):
        result = subprocess.run(
            ["curl", "--noproxy", "*", "-sSIL", "--fail", "--max-time", "60", url],
            capture_output=True, text=True, env=env, check=False,
        )
        if result.returncode == 0:
            break
        time.sleep(min(3 * (attempt + 1), 15))
    else:
        raise RuntimeError("mirror HEAD failed repeatedly: " + url)
    lengths = [int(line.split(":", 1)[1].strip()) for line in result.stdout.splitlines()
               if line.lower().startswith("content-length:")]
    if not lengths:
        raise RuntimeError("mirror did not provide Content-Length: " + url)
    hashes = [line.split(":", 1)[1].strip().strip('"') for line in result.stdout.splitlines()
              if line.lower().startswith("x-linked-etag:")]
    return lengths[-1], hashes[-1] if hashes else None


def sha256(path, limit=None):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = limit
        while remaining is None or remaining > 0:
            chunk = handle.read(min(8 * 1024 * 1024, remaining) if remaining is not None else 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def repair_oversized_archive(archive, expected, expected_sha):
    """Keep an intact mirrored prefix; discard a corrupted resume otherwise."""
    if not archive.exists() or archive.stat().st_size <= expected:
        return
    if expected_sha and len(expected_sha) == 64 and sha256(archive, expected) == expected_sha:
        with archive.open("r+b") as handle:
            handle.truncate(expected)
    else:
        archive.unlink()


def download_with_verified_ranges(url, archive, expected, env):
    """Append bytes only when the CDN proves the requested range was served."""
    suffix = archive.with_name(archive.name + ".range")
    headers = archive.with_name(archive.name + ".headers")
    for attempt in range(20):
        offset = archive.stat().st_size if archive.exists() else 0
        if offset == expected:
            return
        suffix.unlink(missing_ok=True)
        headers.unlink(missing_ok=True)
        result = subprocess.run(
            ["curl", "--noproxy", "*", "-sS", "-L", "--fail", "--retry", "3",
             "--retry-delay", "3", "--connect-timeout", "30",
             "--range", f"{offset}-", "--dump-header", str(headers),
             "-o", str(suffix), url], env=env, check=False,
        )
        raw = headers.read_text(errors="replace") if headers.exists() else ""
        responses = [block for block in raw.replace("\r\n", "\n").split("\n\n")
                     if block.startswith("HTTP/")]
        final = responses[-1] if responses else ""
        status = final.splitlines()[0].split()[1] if final else ""
        content_range = next((line.split(":", 1)[1].strip().lower()
                              for line in final.splitlines()
                              if line.lower().startswith("content-range:")), "")
        valid_range = content_range.startswith(f"bytes {offset}-") and content_range.endswith(f"/{expected}")
        size = suffix.stat().st_size if suffix.exists() else 0
        if offset and (status != "206" or not valid_range):
            # Some CDNs ignore Range and return the whole object. Never append it.
            if status == "200" and result.returncode == 0 and size == expected:
                suffix.replace(archive)
                return
            if status == "200":
                archive.unlink()
            time.sleep(min(attempt + 1, 10))
            continue
        if status not in ("200", "206") or (status == "206" and not valid_range):
            time.sleep(min(attempt + 1, 10))
            continue
        if size <= 0 or size > expected - offset:
            time.sleep(min(attempt + 1, 10))
            continue
        with archive.open("ab") as target, suffix.open("rb") as source:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        if archive.stat().st_size == expected:
            suffix.unlink(missing_ok=True)
            headers.unlink(missing_ok=True)
            return
    raise RuntimeError(f"download incomplete for {archive.name}: "
                       f"{archive.stat().st_size if archive.exists() else 0} != {expected}")


def download_one(modality, index, root, scratch, state, env):
    url, name = url_for(modality, index)
    marker = state / (name + ".json")
    if marker.is_file():
        return {"name": name, "status": "already_complete"}
    expected, expected_sha = remote_info(url, env)
    archive = scratch / name
    repair_oversized_archive(archive, expected, expected_sha)
    download_with_verified_ranges(url, archive, expected, env)
    actual = archive.stat().st_size
    if actual != expected:
        raise RuntimeError(f"size mismatch for {name}: {actual} != {expected}")
    digest = sha256(archive)
    if expected_sha and len(expected_sha) == 64 and digest != expected_sha:
        archive.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {name}: {digest} != {expected_sha}; corrupt archive removed")
    with EXTRACT_LIMIT:
        if shutil.disk_usage(root).free < 30 * 1024**3:
            raise RuntimeError("less than 30 GiB free; extraction stopped")
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(root)], check=True)
    record = {"name": name, "url": url, "bytes": actual, "sha256": digest,
              "mirror_sha256": expected_sha,
              "mirror": "ModelScope", "proxy": "disabled", "extracted": True}
    marker.write_text(json.dumps(record, indent=2))
    archive.unlink()
    return {"name": name, "status": "complete", "bytes": actual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--modalities", nargs="+", choices=["camera", "lidar"], default=["camera", "lidar"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    scratch = root / "navsim_test_downloads"
    state = scratch / "completed"
    scratch.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    env = direct_env()
    tasks = [(mode, index) for mode in args.modalities for index in range(args.start, args.stop)]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, mode, index, root, scratch, state, env): (mode, index)
                   for mode, index in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                with PRINT_LOCK:
                    print(json.dumps(result), flush=True)
            except Exception as exc:
                mode, index = futures[future]
                failures.append((mode, index, str(exc)))
                with PRINT_LOCK:
                    print(json.dumps({"name": f"{mode}_{index}", "status": "failed", "reason": str(exc)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
