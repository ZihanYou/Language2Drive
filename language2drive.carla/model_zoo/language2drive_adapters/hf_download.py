"""Small resumable Hugging Face downloader used by model-local shell scripts."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote


def _curl_command(token: str | None, *, quiet: bool) -> tuple[list[str], str | None]:
    command = [
        "/usr/bin/curl",
        "--fail",
        "--location",
        "--show-error",
        "--retry",
        "5",
        "--connect-timeout",
        "30",
    ]
    if quiet:
        command.append("--silent")
    else:
        command.append("--progress-bar")
    config = None
    if token:
        # Feed credentials on stdin so they are neither copied to disk nor
        # visible in the process command line.
        command.extend(["--config", "-"])
        config = f'header = "Authorization: Bearer {token}"\n'
    return command, config


def _curl_env() -> dict[str, str]:
    environment = os.environ.copy()
    # A login shell may activate conda after tmux creates the pane.  The system
    # curl must not load ABI-incompatible libcurl/libssl from that environment.
    environment.pop("LD_LIBRARY_PATH", None)
    return environment


def _fast_lfs_download(
    url: str,
    output: Path,
    expected_size: int,
    expected_sha256: str | None,
    token: str | None,
) -> bool:
    # The Rust transfer client opens parallel direct range requests.  It is
    # substantially faster than the Xet client on this host and all proxy
    # variables have already been removed by ``main`` for hf-mirror.com.
    if os.getenv("L2D_ENABLE_HF_TRANSFER", "1") != "1":
        return False
    try:
        import hf_transfer
    except ImportError:
        return False

    fast_partial = output.with_name(output.name + ".hf_transfer.part")
    fast_partial.unlink(missing_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else None
    os.environ.setdefault("TOKIO_WORKER_THREADS", "128")
    try:
        hf_transfer.download(
            url,
            str(fast_partial),
            max_files=128,
            chunk_size=32 * 1024 * 1024,
            parallel_failures=32,
            max_retries=10,
            headers=headers,
        )
        if fast_partial.stat().st_size != expected_size:
            raise RuntimeError(
                f"size mismatch: {fast_partial.stat().st_size} != {expected_size}"
            )
        if expected_sha256:
            with fast_partial.open("rb") as stream:
                actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"sha256 mismatch: {actual_sha256} != {expected_sha256}"
                )
        output.with_name(output.name + ".part").unlink(missing_ok=True)
        fast_partial.replace(output)
        return True
    except Exception as exc:
        fast_partial.unlink(missing_ok=True)
        print(
            f"parallel hf_transfer failed ({type(exc).__name__}: {exc}); "
            "falling back to resumable curl",
            file=sys.stderr,
            flush=True,
        )
        return False


def _curl_snapshot(
    repo_id: str,
    local_dir: Path,
    allow: list[str],
    ignore: list[str],
    token: str | None,
    endpoint: str,
) -> Path:
    """Download a model snapshot with curl when requests/proxy TLS is unstable."""

    api = f"{endpoint}/api/models/{quote(repo_id, safe='/')}?blobs=true"
    response = None
    for api_attempt in range(1, 21):
        command, config = _curl_command(token, quiet=True)
        response = subprocess.run(
            [*command, api],
            input=config,
            text=True,
            capture_output=True,
            env=_curl_env(),
        )
        if response.returncode == 0:
            break
        if api_attempt < 20:
            delay = min(api_attempt * 2, 10)
            print(
                f"curl API attempt {api_attempt}/20 failed "
                f"(exit {response.returncode}); retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    assert response is not None
    response.check_returncode()
    metadata = json.loads(response.stdout)
    revision = metadata.get("sha") or "main"
    siblings = metadata.get("siblings") or []
    if not siblings:
        raise RuntimeError(f"Hugging Face API returned no files for {repo_id}")

    selected = []
    for sibling in siblings:
        name = sibling.get("rfilename")
        if not name:
            continue
        if allow and not any(fnmatch.fnmatch(name, pattern) for pattern in allow):
            continue
        if ignore and any(fnmatch.fnmatch(name, pattern) for pattern in ignore):
            continue
        selected.append(sibling)
    if not selected:
        raise RuntimeError(f"filters selected no files from {repo_id}")

    for index, sibling in enumerate(selected, 1):
        name = sibling["rfilename"]
        lfs = sibling.get("lfs") or {}
        expected_size = lfs.get("size")
        output = local_dir / name
        partial = output.with_name(output.name + ".part")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file() and (
            expected_size is None or output.stat().st_size == expected_size
        ):
            expected_sha256 = lfs.get("sha256")
            if expected_sha256:
                with output.open("rb") as stream:
                    actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual_sha256 == expected_sha256:
                    print(f"[{index}/{len(selected)}] reuse verified {name}", flush=True)
                    continue
                print(
                    f"[{index}/{len(selected)}] checksum mismatch; replace {name}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"[{index}/{len(selected)}] reuse {name}", flush=True)
                continue
        if partial.is_file() and expected_size is not None and partial.stat().st_size == expected_size:
            expected_sha256 = lfs.get("sha256")
            if expected_sha256:
                with partial.open("rb") as stream:
                    actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        f"cached partial sha256 mismatch for {name}: "
                        f"{actual_sha256} != {expected_sha256}"
                    )
            partial.replace(output)
            print(f"[{index}/{len(selected)}] completed cached partial {name}", flush=True)
            continue

        url = (
            f"{endpoint}/{quote(repo_id, safe='/')}/resolve/"
            f"{quote(revision, safe='')}/{quote(name, safe='/')}"
        )
        print(f"[{index}/{len(selected)}] download {name}", flush=True)
        if expected_size is not None and expected_size >= 16 * 1024 * 1024:
            if _fast_lfs_download(
                url,
                output,
                expected_size,
                lfs.get("sha256"),
                token,
            ):
                continue
        result = None
        for transfer_attempt in range(1, 21):
            command, config = _curl_command(token, quiet=False)
            result = subprocess.run(
                [*command, "--continue-at", "-", "--output", str(partial), url],
                input=config,
                text=True,
                env=_curl_env(),
            )
            if result.returncode == 0:
                break
            if result.returncode == 33:
                # This endpoint rejected Range for the current partial.
                partial.unlink(missing_ok=True)
            if transfer_attempt < 20:
                delay = min(transfer_attempt * 2, 10)
                print(
                    f"curl transfer attempt {transfer_attempt}/20 failed "
                    f"(exit {result.returncode}); retrying in {delay}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        assert result is not None
        result.check_returncode()
        if expected_size is not None and partial.stat().st_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {name}: {partial.stat().st_size} != {expected_size}"
            )
        expected_sha256 = lfs.get("sha256")
        if expected_sha256:
            with partial.open("rb") as stream:
                actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"sha256 mismatch for {name}: {actual_sha256} != {expected_sha256}"
                )
        partial.replace(output)
    return local_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--ignore", action="append", default=[])
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)

    # Keep downloader packages and all HF/Xet cache state inside model_zoo.
    # The conda environment intentionally remains untouched, and the system
    # overlay has much less free space than /root/autodl-tmp.
    vendor = Path(__file__).resolve().parent / "vendor"
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
    token = os.getenv("HF_TOKEN")
    if not token:
        original_home = Path(
            os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        )
        token_path = original_home / "token"
        try:
            token = token_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            token = None
    cache_root = args.local_dir / ".cache" / "huggingface"
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_XET_CACHE"] = str(cache_root / "xet")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    endpoint = os.getenv("L2D_HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    if endpoint == "https://hf-mirror.com":
        # The machine's SSH-forwarded HTTP proxy is unreliable for the
        # Hugging Face domains, while this mirror and its CDN are reachable
        # directly.  The change is process-local and never mutates the shell.
        for name in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(name, None)

    path = None
    last_error = None
    if os.getenv("L2D_PREFER_HF_TRANSFER", "1") == "1":
        try:
            path = _curl_snapshot(
                args.repo_id,
                args.local_dir,
                args.allow,
                args.ignore,
                token,
                endpoint,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"parallel mirror download failed ({type(exc).__name__}: {exc}); "
                "falling back to huggingface_hub",
                file=sys.stderr,
                flush=True,
            )

    if path is None:
        from huggingface_hub import snapshot_download

        for attempt in range(1, 3):
            try:
                path = snapshot_download(
                    repo_id=args.repo_id,
                    local_dir=args.local_dir,
                    allow_patterns=args.allow or None,
                    ignore_patterns=args.ignore or None,
                    max_workers=2,
                    token=token,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    break
                delay = attempt * 5
                print(
                    f"download attempt {attempt}/2 failed: "
                    f"{type(exc).__name__}: {exc}; retrying in {delay}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
    if path is None:
        print(
            f"huggingface_hub failed ({type(last_error).__name__}: {last_error}); "
            "falling back to resumable curl",
            file=sys.stderr,
            flush=True,
        )
        path = _curl_snapshot(
            args.repo_id,
            args.local_dir,
            args.allow,
            args.ignore,
            token,
            endpoint,
        )
    assert path is not None
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
