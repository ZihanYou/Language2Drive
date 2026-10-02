"""Extract registry pins verbatim from a frozen uv export for hash-checked downloads.

Direct URLs, Git revisions and workspace packages remain under uv sync authority.
No package version, marker or artifact hash is changed by this transport step.
"""
import argparse
from pathlib import Path
import re


def registry_requirements(source):
    blocks, current = [], []
    for line in source.splitlines(keepends=True):
        if line.strip() and not line[0].isspace():
            if current:
                blocks.append(''.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(''.join(current))
    selected = []
    for block in blocks:
        if not re.match(r'^[A-Za-z0-9_.-]+==', block):
            continue
        if '--hash=sha256:' not in block:
            raise ValueError('Registry requirement has no frozen artifact hashes: ' + block.splitlines()[0])
        selected.append(block)
    if not selected:
        raise ValueError('Frozen export contains no hash-bound registry requirements')
    return ''.join(selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(registry_requirements(args.source.read_text()))
