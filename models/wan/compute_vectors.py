#!/usr/bin/env python3
"""Compute a Wan text-encoder steering vector via difference-of-means.

Usage
-----
python -m models.wan.compute_vectors \\
    --pairs_file path/to/concept.jsonl \\
    --out_dir outputs/concept
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from steering import (
    DTYPE_MAP,
    compute_difference_of_means,
    find_all_spans,
    save_steering_outputs,
    split_style_terms,
    validate_max_pairs,
    validate_path_exists,
)

Span = Tuple[int, int]


def _prompt_clean(text: str) -> str:
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_wan_text_encoder(
    model_id: str,
    tokenizer_subfolder: str,
    text_encoder_subfolder: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[AutoTokenizer, UMT5EncoderModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=tokenizer_subfolder, use_fast=True)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder=text_encoder_subfolder,
        torch_dtype=dtype,
    ).to(device)
    text_encoder.eval()
    if hasattr(text_encoder, "shared") and hasattr(text_encoder, "encoder"):
        encoder = getattr(text_encoder, "encoder")
        if hasattr(encoder, "embed_tokens"):
            with torch.no_grad():
                encoder.embed_tokens.weight.data.copy_(text_encoder.shared.weight.data)
    return tokenizer, text_encoder


def _encode_prompt(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    text: str,
    device: str,
    max_sequence_length: int,
) -> tuple[torch.Tensor, Optional[List[Span]], str, List[str]]:
    cleaned_text = _prompt_clean(text)
    tokenizer_kwargs = dict(
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    try:
        encoding = tokenizer([cleaned_text], return_offsets_mapping=True, **tokenizer_kwargs)
        has_offsets = True
    except (TypeError, ValueError, NotImplementedError):
        encoding = tokenizer([cleaned_text], **tokenizer_kwargs)
        has_offsets = False

    input_ids = encoding.input_ids.to(device)
    mask = encoding.attention_mask.to(device)
    seq_len = int(mask[0].sum().item())

    with torch.no_grad():
        hidden = text_encoder(input_ids, mask).last_hidden_state

    hidden = hidden[0, :seq_len].to(dtype=torch.float32).cpu()
    if not torch.isfinite(hidden).all():
        raise ValueError("Non-finite values in text encoder hidden states.")

    offsets: Optional[List[Span]]
    if has_offsets:
        offsets = encoding["offset_mapping"][0][:seq_len].tolist()
    else:
        offsets = None

    token_pieces = tokenizer.convert_ids_to_tokens(encoding.input_ids[0][:seq_len].tolist())
    return hidden, offsets, cleaned_text, token_pieces


def _get_style_token_indices(
    text: str,
    style: str,
    offsets: Optional[List[Span]],
    token_pieces: Sequence[str],
) -> List[int]:
    terms = split_style_terms(style)
    if not terms:
        return []

    matched: set[int] = set()
    parts_set = {t.lower() for t in terms}

    if offsets is None:
        for i, piece in enumerate(token_pieces):
            normalized = (piece or "").strip().lstrip("▁").lower()
            if normalized in parts_set:
                matched.add(i)
        return sorted(matched)

    for term in terms:
        for span_start, span_end in find_all_spans(text, term):
            for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:
                    continue
                if tok_start < span_end and tok_end > span_start:
                    matched.add(tok_idx)

    return sorted(matched)


def collect_style_representations(
    pairs_file: Path,
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    device: str,
    max_sequence_length: int,
    max_pairs: int,
) -> tuple[List[np.ndarray], List[int]]:
    style_vectors: List[np.ndarray] = []
    style_labels: List[int] = []
    processed = 0

    with pairs_file.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Collecting Wan text-encoder representations"):
            if max_pairs != -1 and processed >= max_pairs:
                break
            example = json.loads(line)
            pos_text, neg_text = example["pos"], example["neg"]
            pos_style, neg_style = example.get("pos_style"), example.get("neg_style")
            if not (pos_style and neg_style):
                raise ValueError(
                    f"Each example must include pos_style/neg_style; "
                    f"got pos_style={pos_style}, neg_style={neg_style}"
                )

            pos_hidden, pos_offsets, pos_clean, pos_tokens = _encode_prompt(
                tokenizer, text_encoder, pos_text, device, max_sequence_length
            )
            neg_hidden, neg_offsets, neg_clean, neg_tokens = _encode_prompt(
                tokenizer, text_encoder, neg_text, device, max_sequence_length
            )

            pos_idx = _get_style_token_indices(pos_clean, pos_style, pos_offsets, pos_tokens)
            neg_idx = _get_style_token_indices(neg_clean, neg_style, neg_offsets, neg_tokens)
            if not (pos_idx and neg_idx):
                raise ValueError(
                    f"Failed to align style spans to token offsets. "
                    f"pos_style={pos_style}, neg_style={neg_style}"
                )

            pos_vec = torch.stack([pos_hidden[i] for i in pos_idx]).mean(0)
            neg_vec = torch.stack([neg_hidden[i] for i in neg_idx]).mean(0)

            style_vectors.append(pos_vec.numpy().astype(np.float32))
            style_labels.append(1)
            style_vectors.append(neg_vec.numpy().astype(np.float32))
            style_labels.append(0)
            processed += 1

    return style_vectors, style_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute a Wan steering vector from contrastive prompt pairs.")
    parser.add_argument("--pairs_file", type=validate_path_exists, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--tokenizer_subfolder", type=str, default="tokenizer")
    parser.add_argument("--text_encoder_subfolder", type=str, default="text_encoder")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=sorted(DTYPE_MAP.keys()), default="float32")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--max_pairs", type=validate_max_pairs, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer, text_encoder = load_wan_text_encoder(
        model_id=args.model_id,
        tokenizer_subfolder=args.tokenizer_subfolder,
        text_encoder_subfolder=args.text_encoder_subfolder,
        device=args.device,
        dtype=DTYPE_MAP[args.dtype],
    )
    vectors, labels = collect_style_representations(
        pairs_file=args.pairs_file,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=args.device,
        max_sequence_length=args.max_sequence_length,
        max_pairs=args.max_pairs,
    )
    steering, max_projection, min_projection = compute_difference_of_means(vectors, labels)
    save_steering_outputs(Path(args.out_dir), steering, max_projection, min_projection)


if __name__ == "__main__":
    main()
