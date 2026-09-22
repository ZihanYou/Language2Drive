"""Resolve immutable source-provenance identifiers inside the release checkout.

The source snapshot includes hashed capture records. Their original path strings
are part of the content identity and must remain unchanged. ProvenancePath keeps
that identity when formatted, while filesystem access uses the packaged copy.
Live runtime authority uses pathlib.Path directly and is frozen on each host.
"""
from __future__ import annotations
import os
from pathlib import Path, PosixPath

ROOT = Path(__file__).resolve().parents[2]
_PREFIXES = (
    ('/root/autodl-tmp/lwq/language2drive.v1/.runtime/', 'assets/provenance/runtime/'),
    ('/root/miniconda3/envs/simlingo-py311/lib/python3.11/site-packages/carla/', 'assets/provenance/python/carla/'),
    ('/root/autodl-tmp/lwq/language2drive.v1/', ''),
    ('/root/autodl-tmp/simlingo_fzh/tools/create_data/', 'assets/annotations/'),
    ('/root/autodl-tmp/simlingo_fzh/software/road_mesh/', 'road_mesh/'),
    ('/root/autodl-tmp/lwq/simlingo/software/road_mesh/', 'road_mesh/'),
    ('/root/autodl-tmp/simlingo_fzh/software/Carla-Language2Drive/', 'carla_runtime/'),
    ('/root/autodl-tmp/lwq/simlingo/software/Carla-Language2Drive/', 'carla_runtime/'),
)

def packaged_path(value: str | os.PathLike) -> Path:
    """Map a recorded identity to its release-local bytes; never fall back."""
    text = str(value)
    if text in {prefix + 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping' for prefix, destination in _PREFIXES if destination == 'carla_runtime/'}:
        return ROOT / 'assets/provenance/carla/CarlaUE4-Linux-Shipping'
    for prefix, destination in _PREFIXES:
        if text.startswith(prefix):
            relative = Path(text[len(prefix):])
            if '..' in relative.parts:
                raise ValueError('invalid source provenance path')
            return ROOT / destination / relative
    return Path(text)

class ProvenancePath(PosixPath):
    def __fspath__(self) -> str:
        return str(packaged_path(str(self)))

    def resolve(self, strict: bool = False) -> 'ProvenancePath':
        target = packaged_path(str(self))
        if str(target) != str(self):
            if strict:
                target.resolve(strict=True)
            return self
        return type(self)(super().resolve(strict=strict))
