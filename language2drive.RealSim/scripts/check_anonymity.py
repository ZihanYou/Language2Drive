"""Offline review-package hygiene scan; reports locations, never secret values."""
from pathlib import Path
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'personal_absolute_path': re.compile(r'/(?:home|Users)/[^/\s]+/|/(?:root|NEW_EDS)/'),
    'private_network_endpoint': re.compile(r'\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b'),
    'access_token': re.compile(r'\bhf_[A-Za-z0-9]{20,}\b|\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'credential_url': re.compile(r'\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+:[^\s/@]+@'),
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'literal_credential': re.compile(r'''(?i)\b(?:password|passwd|api_key|hf_token|secret_key)["']?\s*[:=]\s*["'][^"'\s]{5,}["']'''),
}
FORBIDDEN = {'.git', '.runtime', '__pycache__', '.pytest_cache', '.venv', 'node_modules',
             'vendor', 'model_zoo', 'outputs', 'build', 'dist'}


def scan(root=ROOT):
    findings = []
    checked = 0
    candidates = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            if path.is_symlink() or name in FORBIDDEN or name.endswith('.egg-info'):
                findings.append({'file': str(path.relative_to(root)), 'rule': 'nonrelease_directory'})
                dirs.remove(name)
        candidates.extend(parent / name for name in files)
    for path in sorted(candidates):
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append({'file': str(relative), 'rule': 'symlink'})
            continue
        if path.is_dir():
            continue
        if any(p in FORBIDDEN for p in relative.parts) or '.private.' in path.name:
            findings.append({'file': str(relative), 'rule': 'nonrelease_file'})
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except (UnicodeError, OSError):
            findings.append({'file': str(relative), 'rule': 'nontext_or_unreadable'})
            continue
        checked += 1
        for name, pattern in RULES.items():
            scan_content = content
            if str(relative) == 'config/alpasim-runtime.uv.lock' and name == 'private_network_endpoint':
                # cuRAND's four-component package version is not an IP endpoint.
                # Keep URL hosts and every other lock field subject to the scan.
                curand_version = '.'.join(('10', '3', '9', '90'))
                scan_content = re.sub(r'(?m)^version = "' + re.escape(curand_version) + r'"$', 'version = "CURAND_VERSION"', content)
                scan_content = scan_content.replace(f'nvidia_curand_cu12-{curand_version}-', 'nvidia_curand_cu12-CURAND_VERSION-')
            if pattern.search(scan_content):
                findings.append({'file': str(relative), 'rule': name})
    return {'checked_text_files': checked, 'findings': findings,
            'passed': not findings, 'scope': 'direct_marker_and_payload_hygiene_not_absolute_anonymity'}


if __name__ == '__main__':
    report = scan()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
