#!/usr/bin/env python3
"""
Compare original input images with decoded visual-token images.

Reads inference result JSON (from OneVL), extracts:
  - Original input images from messages[].content (type=image)
  - Predicted visual tokens from visual_decoder_explain
    (<|image start|>...<|image end|> blocks)

Decodes visual tokens back to images via the Emu3.5 VisionTokenizer (IBQ)
and saves both side-by-side for human comparison.

Usage:
  source /path/to/venv/bin/activate
  cd /path/to/onevl_opensource
  python scripts/visualize_predict_image_tokens.py \\
    --model_root /path/to/emu35_model_root \\
    --predict_json output/ar1_explain/ar1_results_explain.json \\
    --out_dir output/ar1_explain_compare \\
    -n 10 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Visual-token parsing (self-contained, no external dependency)
# ---------------------------------------------------------------------------

BOI_TOKEN = "<|image start|>"
EOI_TOKEN = "<|image end|>"
EOL_TOKEN = "<|extra_200|>"
IMG_TOKEN = "<|image token|>"
VISUAL_TOKEN_RE = re.compile(r"<\|visual token (\d+)\|>")
IMAGE_BLOCK_RE = re.compile(r"<\|image start\|>.*?<\|image end\|>", re.DOTALL)


def extract_image_blocks(visual_decoder_explain: str) -> list[str]:
    """Return all <|image start|>...<|image end|> blocks."""
    return IMAGE_BLOCK_RE.findall(visual_decoder_explain or "")


def parse_token_block(content: str) -> np.ndarray:
    """
    Parse a single image-token block into an (H, W) int64 numpy array.
    """
    content = content.strip()
    if not (content.startswith(BOI_TOKEN) and content.endswith(EOI_TOKEN)):
        raise ValueError("Not a valid image token block (missing BOI/EOI)")
    inner = content[len(BOI_TOKEN):-len(EOI_TOKEN)]
    if IMG_TOKEN in inner:
        inner = inner.split(IMG_TOKEN, 1)[1]
    rows_str = inner.split(EOL_TOKEN)
    grid: list[list[int]] = []
    for r in rows_str:
        ids = VISUAL_TOKEN_RE.findall(r)
        if ids:
            grid.append([int(x) for x in ids])
    if not grid:
        raise ValueError("No visual tokens found in block")
    W = len(grid[0])
    for i in range(1, len(grid)):
        row = grid[i]
        if len(row) > W:
            grid[i] = row[:W]
        elif len(row) < W:
            prev = grid[i - 1]
            pad_token = prev[-1] if prev else 0
            grid[i] = row + [pad_token] * (W - len(row))
    return np.array(grid, dtype=np.int64)


def tokens_to_image(token_grid: np.ndarray, vq_model, embed_dim: int = 256,
                    device: str = "cuda:0") -> Image.Image:
    """Decode an (H, W) token grid to a PIL Image using the VQ model."""
    tg = torch.from_numpy(token_grid).long().to(device)
    h, w = tg.shape
    decoded = vq_model.decode_code(tg[None], shape=(1, h, w, embed_dim)).float()
    decoded = decoded[0].permute(1, 2, 0)
    arr = ((decoded + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def get_input_images_from_messages(messages: list) -> list[str]:
    """Collect image paths from messages[].content where type=image."""
    paths: list[str] = []
    for msg in messages or []:
        for part in msg.get("content") or []:
            if part.get("type") == "image" and part.get("image"):
                paths.append(part["image"])
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description="Save original images + decoded visual-token images for comparison"
    )
    parser.add_argument(
        "--predict_json", type=str,
        default=os.path.join(project_root, "output", "ar1_explain", "ar1_results_explain.json"),
        help="Inference result JSON path (JSON array)",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=os.path.join(project_root, "output", "ar1_explain_visualize"),
    )
    parser.add_argument("-n", type=int, default=20000, help="Max samples to process")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model_root", type=str,
        default="emu35",
        help="Emu3.5 model root (contains Emu3.5-VisionTokenizer/)",
    )
    args = parser.parse_args()

    sys.path.insert(0, project_root)
    from vq_decoder import load_vision_tokenizer

    with open(args.predict_json, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    samples = data[: args.n]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading VisionTokenizer from {args.model_root} ...")
    vq_model = load_vision_tokenizer(args.model_root, device=args.device)
    embed_dim = getattr(vq_model.quantize, "embed_dim",
                        getattr(vq_model.quantize, "e_dim", 256))

    manifest: list[dict] = []
    for rank, item in enumerate(samples):
        messages = item.get("messages") or []
        input_paths = get_input_images_from_messages(messages)
        visual_str = item.get("visual_decoder_explain") or ""

        blocks = extract_image_blocks(visual_str)
        sub = os.path.join(args.out_dir, f"sample_{rank:04d}")
        os.makedirs(sub, exist_ok=True)

        meta: dict = {
            "sample_rank": rank,
            "input_images": input_paths,
            "num_decoded_blocks": len(blocks),
            "decoder_explain": item.get("decoder_explain", ""),
        }

        for j, src in enumerate(input_paths):
            ext = os.path.splitext(src)[1] or ".jpg"
            dst_name = f"input_{j:02d}{ext}"
            dst = os.path.join(sub, dst_name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                meta[f"input_{j:02d}_missing"] = True
                with open(dst + ".missing.txt", "w") as f:
                    f.write(src)

        for b, block in enumerate(blocks):
            try:
                grid = parse_token_block(block)
                img = tokens_to_image(grid, vq_model, embed_dim=embed_dim,
                                      device=args.device)
                out_png = os.path.join(sub, f"decoded_from_tokens_{b:02d}.png")
                img.save(out_png)
            except Exception as e:
                meta[f"decode_error_{b}"] = str(e)

        with open(os.path.join(sub, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        manifest.append(meta)

        if (rank + 1) % 10 == 0 or rank == 0:
            print(f"  [{rank + 1}/{len(samples)}] decoded {len(blocks)} blocks, "
                  f"{len(input_paths)} input images")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "predict_json": os.path.abspath(args.predict_json),
                "n_processed": len(samples),
                "out_dir": os.path.abspath(args.out_dir),
                "samples": manifest,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"Done. {len(samples)} samples processed -> {args.out_dir}")


if __name__ == "__main__":
    main()
