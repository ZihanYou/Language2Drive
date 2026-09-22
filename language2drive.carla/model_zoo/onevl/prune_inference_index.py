#!/usr/bin/env python3
"""Keep only OneVL weights used by its default trajectory generation path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    index_path = root / "model.safetensors.index.json"
    upstream_path = root / "model.safetensors.index.upstream.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map", {})

    # The released checkpoint also bundles two full auxiliary decoders used
    # only by infer_onevl.py's optional --decoder-explain modes.  Default
    # trajectory generation loads Qwen3VLForConditionalGeneration and uses
    # only the student model plus lm_head, exactly as the official script does.
    inference_map = {
        name: shard
        for name, shard in weight_map.items()
        if name == "lm_head.weight" or name.startswith("model.")
    }
    if not inference_map:
        raise RuntimeError(f"no OneVL inference weights found in {index_path}")
    optional_prefixes = (
        "_latent_cot_aux_decoder.",
        "_latent_cot_latent_proj.",
        "_latent_cot_visual_aux_decoder.",
        "_latent_cot_visual_latent_proj.",
    )
    unexpected_excluded = sorted(
        name
        for name in weight_map
        if name not in inference_map and not name.startswith(optional_prefixes)
    )
    if unexpected_excluded:
        raise RuntimeError(
            "refusing to prune unknown OneVL weight keys: "
            + ", ".join(unexpected_excluded[:20])
        )
    shards = sorted(set(inference_map.values()))
    expected = [
        "model-00001-of-00007.safetensors",
        "model-00002-of-00007.safetensors",
    ]
    if shards != expected:
        raise RuntimeError(f"unexpected OneVL inference shard set: {shards}")
    missing = [str(root / name) for name in shards if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("missing OneVL inference shards: " + ", ".join(missing))

    # Retain the upstream index for provenance while exposing a standard local
    # index that Transformers can load without requesting unused aux shards.
    if any(name.startswith("_latent_cot_") for name in weight_map):
        upstream_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    pruned = {
        "metadata": {"total_size": sum((root / name).stat().st_size for name in shards)},
        "weight_map": inference_map,
    }
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(pruned, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(index_path)
    print(
        f"OneVL inference index: {len(inference_map)} tensors in "
        f"{len(shards)} shards"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
