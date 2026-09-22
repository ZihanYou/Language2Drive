#!/usr/bin/env python3
"""Download one immutable file with resume, fallback, and exact verification."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit


PROXY_VARIABLES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)


def _clear_proxy_environment() -> None:
    """Keep the machine's SSH forwarding proxy out of this child process."""

    for name in PROXY_VARIABLES:
        os.environ.pop(name, None)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_verified(
    path: Path,
    expected_size: int,
    expected_sha256: str,
    *,
    report_mismatch: bool,
) -> bool:
    if not path.is_file():
        return False
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        if report_mismatch:
            print(
                f"size mismatch for {path}: {actual_size} != {expected_size}",
                file=sys.stderr,
                flush=True,
            )
        return False
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        if report_mismatch:
            print(
                f"sha256 mismatch for {path}: "
                f"{actual_sha256} != {expected_sha256}",
                file=sys.stderr,
                flush=True,
            )
        return False
    return True


def _curl_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # Avoid conda's libcurl/libssl being loaded by the system curl binary.
    environment.pop("LD_LIBRARY_PATH", None)
    for name in PROXY_VARIABLES:
        environment.pop(name, None)
    return environment


def _promote_partial(
    partial: Path,
    target: Path,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    if not _is_verified(
        partial,
        expected_size,
        expected_sha256,
        report_mismatch=False,
    ):
        return False
    partial.replace(target)
    print(f"verified {target}", flush=True)
    return True


def _curl_download(
    url: str,
    target: Path,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    partial = target.with_name(target.name + ".part")
    if partial.is_file():
        partial_size = partial.stat().st_size
        if partial_size == expected_size:
            if _promote_partial(
                partial,
                target,
                expected_size,
                expected_sha256,
            ):
                return True
            print(
                f"discarding checksum-invalid complete partial {partial}",
                file=sys.stderr,
                flush=True,
            )
            partial.unlink()
        elif partial_size > expected_size:
            print(
                f"discarding oversized partial {partial} ({partial_size} bytes)",
                file=sys.stderr,
                flush=True,
            )
            partial.unlink()

    for attempt in range(1, 11):
        command = [
            "/usr/bin/curl",
            "--fail",
            "--location",
            "--show-error",
            "--retry",
            "5",
            "--connect-timeout",
            "30",
            "--progress-bar",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
        result = subprocess.run(command, env=_curl_environment())
        if result.returncode == 0:
            if _promote_partial(
                partial,
                target,
                expected_size,
                expected_sha256,
            ):
                return True
            print(
                f"downloaded content from {url} failed exact verification",
                file=sys.stderr,
                flush=True,
            )
            partial.unlink(missing_ok=True)
            return False

        if result.returncode == 33:
            # This server rejected Range for the current partial. Retrying
            # without it is safe because the final file is still untouched.
            partial.unlink(missing_ok=True)
        if attempt < 10:
            delay = min(attempt * 2, 10)
            print(
                f"curl attempt {attempt}/10 failed (exit {result.returncode}); "
                f"retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="preferred immutable source URL")
    parser.add_argument("target", type=Path)
    parser.add_argument("expected_size", type=int)
    parser.add_argument("expected_sha256")
    parser.add_argument(
        "--fallback-url",
        action="append",
        default=[],
        help="alternate URL containing the exact same bytes (repeatable)",
    )
    args = parser.parse_args()
    if args.expected_size <= 0:
        parser.error("expected_size must be positive")
    args.expected_sha256 = args.expected_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_sha256):
        parser.error("expected_sha256 must contain exactly 64 hexadecimal digits")
    return args


def main() -> int:
    args = _parse_args()
    _clear_proxy_environment()
    args.target.parent.mkdir(parents=True, exist_ok=True)

    if _is_verified(
        args.target,
        args.expected_size,
        args.expected_sha256,
        report_mismatch=True,
    ):
        print(f"reuse verified {args.target}", flush=True)
        return 0

    # The shared vendor is intentionally model-local; importing it does not
    # alter simlingo-py311. Prefer parallel ranges only for the primary mirror.
    vendor = Path(__file__).resolve().parent / "vendor"
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
    from hf_download import _fast_lfs_download

    standard_partial = args.target.with_name(args.target.name + ".part")
    primary_host = (urlsplit(args.url).hostname or "").lower()
    hf_parallel_source = primary_host in {
        "hf-mirror.com",
        "huggingface.co",
    }
    if not standard_partial.exists() and hf_parallel_source and _fast_lfs_download(
        args.url,
        args.target,
        args.expected_size,
        args.expected_sha256,
        os.getenv("HF_TOKEN"),
    ):
        if not _is_verified(
            args.target,
            args.expected_size,
            args.expected_sha256,
            report_mismatch=True,
        ):
            raise RuntimeError("parallel downloader produced an invalid final file")
        print(f"verified {args.target}", flush=True)
        return 0

    urls = [args.url, *args.fallback_url]
    for index, url in enumerate(urls, 1):
        print(f"source {index}/{len(urls)}: {url}", flush=True)
        if _curl_download(
            url,
            args.target,
            args.expected_size,
            args.expected_sha256,
        ):
            return 0

    raise RuntimeError(
        f"all sources failed for {args.target}; a resumable .part may remain"
    )


if __name__ == "__main__":
    raise SystemExit(main())
