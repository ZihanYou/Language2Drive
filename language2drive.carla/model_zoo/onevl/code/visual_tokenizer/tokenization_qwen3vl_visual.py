"""
Fast-loading Qwen3-VL tokenizer with 131k visual tokens.

Visual tokens live in model.vocab (fast BPE hash-map load) rather than
added_tokens (slow Aho-Corasick build).  A regex pre-split in the Python
wrapper ensures encode/call with visual token text produces single IDs.

Strategy: replace each <|visual token XXXXXX|> with a NUL byte (\x00)
before sending to the Rust backend, then swap the NUL-byte token ID (188)
with the real visual-token ID in the output.
"""

import re
from typing import List, Optional, Union

from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

_VISUAL_RE = re.compile(r"<\|visual token (\d{6})\|>")
_VISUAL_TOKEN_START_ID = 151674
_PLACEHOLDER_CHAR = "\x00"
_PLACEHOLDER_TOKEN_ID = 188


class Qwen3VLVisualTokenizerFast(Qwen2TokenizerFast):

    # ---------- public encode() ----------
    def encode(self, text, text_pair=None, add_special_tokens=True, **kwargs):
        if isinstance(text, str) and _VISUAL_RE.search(text):
            replaced, vids = _replace_visual(text)
            pair_replaced, pair_vids = None, []
            if text_pair is not None and isinstance(text_pair, str):
                pair_replaced, pair_vids = _replace_visual(text_pair)
            ids = super().encode(
                replaced,
                text_pair=pair_replaced if pair_replaced is not None else text_pair,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )
            _swap_ids(ids, vids + pair_vids)
            return ids
        return super().encode(text, text_pair, add_special_tokens=add_special_tokens, **kwargs)

    # ---------- batch path (powers __call__) ----------
    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        has_visual = any(
            _text_has_visual(item) for item in batch_text_or_text_pairs
        )
        if not has_visual:
            return super()._batch_encode_plus(batch_text_or_text_pairs, **kwargs)

        replaced_batch = []
        all_vids: list[list[int]] = []

        for item in batch_text_or_text_pairs:
            if isinstance(item, (tuple, list)):
                text, pair = item[0], (item[1] if len(item) > 1 else None)
            else:
                text, pair = item, None

            vids: list[int] = []
            if isinstance(text, str) and _VISUAL_RE.search(text):
                text, tvids = _replace_visual(text)
                vids.extend(tvids)
            if pair is not None and isinstance(pair, str) and _VISUAL_RE.search(pair):
                pair, pvids = _replace_visual(pair)
                vids.extend(pvids)

            replaced_batch.append((text, pair) if pair is not None else text)
            all_vids.append(vids)

        result = super()._batch_encode_plus(replaced_batch, **kwargs)

        for i, vids in enumerate(all_vids):
            if not vids:
                continue
            ids = result["input_ids"][i]
            tensor_type = None
            if hasattr(ids, "tolist"):
                tensor_type = type(ids)
                device = ids.device if hasattr(ids, "device") else None
                dtype = ids.dtype
                ids = ids.tolist()
            _swap_ids(ids, vids)
            if tensor_type is not None:
                import torch
                t = torch.tensor(ids, dtype=dtype)
                if device is not None:
                    t = t.to(device)
                result["input_ids"][i] = t
            else:
                result["input_ids"][i] = ids

        return result


def _text_has_visual(item) -> bool:
    t = item[0] if isinstance(item, (tuple, list)) else item
    return isinstance(t, str) and _VISUAL_RE.search(t) is not None


def _replace_visual(text: str):
    """Replace visual tokens with NUL bytes, return (new_text, ordered_visual_ids)."""
    vids: list[int] = []

    def _repl(m):
        vids.append(_VISUAL_TOKEN_START_ID + int(m.group(1)))
        return _PLACEHOLDER_CHAR

    new_text = _VISUAL_RE.sub(_repl, text)
    return new_text, vids


def _swap_ids(ids: list, vids: list[int]):
    """In-place replace placeholder token IDs with real visual-token IDs."""
    vi = 0
    for j in range(len(ids)):
        if ids[j] == _PLACEHOLDER_TOKEN_ID and vi < len(vids):
            ids[j] = vids[vi]
            vi += 1
