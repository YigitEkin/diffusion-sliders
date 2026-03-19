#!/usr/bin/env python3
"""Compute a Flux 2 steering vector from Mistral-3 prompt embeddings.

Usage
-----
python -m models.flux2.compute_vectors \\
    --pairs_file path/to/cartoon.jsonl \\
    --out_dir models/flux2/assets/my_concept
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from steering import (
    DTYPE_MAP,
    compute_difference_of_means,
    save_steering_outputs,
    validate_max_pairs,
    validate_path_exists,
)
from ._utils import (
    MAX_SEQUENCE_LENGTH,
    SYSTEM_MESSAGE,
    TEXT_ENCODER_OUT_LAYERS,
    encode_flux2_prompt_embeds,
    get_style_token_positions,
    load_flux2_text_stack,
    pool_positions,
    prepare_flux2_text_inputs,
)


# ---------------------------------------------------------------------------
# Style representation collection
# ---------------------------------------------------------------------------


def collect_style_representations(args: argparse.Namespace) -> tuple[List[np.ndarray], List[int]]:
    processor, text_encoder = load_flux2_text_stack(
        model_name_or_path=args.model,
        device=args.device,
        dtype_name=args.dtype,
    )
    special_token_ids = getattr(processor.tokenizer, "all_special_ids", [])
    image_token_id = getattr(text_encoder.config, "image_token_id", None)

    style_vectors: List[np.ndarray] = []
    style_labels: List[int] = []
    processed = 0

    with args.pairs_file.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Collecting Flux 2 prompt spans"):
            if args.max_pairs != -1 and processed >= args.max_pairs:
                break

            example = json.loads(line)
            pos_text, neg_text = example["pos"], example["neg"]
            pos_style, neg_style = example.get("pos_style"), example.get("neg_style")
            if not pos_style or not neg_style:
                raise ValueError(
                    f"Each example must include pos_style and neg_style; "
                    f"got pos_style={pos_style}, neg_style={neg_style}"
                )

            pos_inputs = prepare_flux2_text_inputs(
                processor=processor, prompt=pos_text,
                max_sequence_length=args.max_sequence_length, system_message=args.system_message,
            )
            neg_inputs = prepare_flux2_text_inputs(
                processor=processor, prompt=neg_text,
                max_sequence_length=args.max_sequence_length, system_message=args.system_message,
            )

            pos_positions = get_style_token_positions(
                formatted_prompt=pos_inputs["formatted_prompt"], user_prompt=pos_text, style=pos_style,
                plain_offsets=pos_inputs["plain_offsets"],
                plain_to_input_positions=pos_inputs["plain_to_input_positions"],
                actual_input_ids=pos_inputs["input_ids"][0].tolist(),
                special_token_ids=special_token_ids, image_token_id=image_token_id,
            )
            neg_positions = get_style_token_positions(
                formatted_prompt=neg_inputs["formatted_prompt"], user_prompt=neg_text, style=neg_style,
                plain_offsets=neg_inputs["plain_offsets"],
                plain_to_input_positions=neg_inputs["plain_to_input_positions"],
                actual_input_ids=neg_inputs["input_ids"][0].tolist(),
                special_token_ids=special_token_ids, image_token_id=image_token_id,
            )
            if not pos_positions or not neg_positions:
                raise ValueError(
                    f"Failed to align style spans to Flux 2 text tokens. "
                    f"pos_style={pos_style}, neg_style={neg_style}"
                )

            pos_embeds = encode_flux2_prompt_embeds(
                text_encoder=text_encoder, processor=processor, prompt=pos_text, device=args.device,
                max_sequence_length=args.max_sequence_length, system_message=args.system_message,
                hidden_states_layers=args.text_encoder_out_layers,
            )
            neg_embeds = encode_flux2_prompt_embeds(
                text_encoder=text_encoder, processor=processor, prompt=neg_text, device=args.device,
                max_sequence_length=args.max_sequence_length, system_message=args.system_message,
                hidden_states_layers=args.text_encoder_out_layers,
            )

            pos_vector = pool_positions(pos_embeds, pos_positions)
            neg_vector = pool_positions(neg_embeds, neg_positions)
            if pos_vector is None or neg_vector is None:
                raise ValueError(
                    f"Failed to pool Flux 2 embeddings for style spans. "
                    f"pos_style={pos_style}, neg_style={neg_style}"
                )

            style_vectors.append(pos_vector.numpy().astype(np.float32))
            style_labels.append(1)
            style_vectors.append(neg_vector.numpy().astype(np.float32))
            style_labels.append(0)
            processed += 1

    if not style_vectors:
        raise ValueError("No valid style vectors were collected from the provided pairs file.")
    return style_vectors, style_labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Flux 2 steering vectors from contrastive prompt pairs."
    )
    parser.add_argument("--pairs_file", type=validate_path_exists, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.2-dev")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--max_pairs", type=validate_max_pairs, default=-1)
    parser.add_argument("--max_sequence_length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--system_message", type=str, default=SYSTEM_MESSAGE)
    parser.add_argument("--text_encoder_out_layers", type=int, nargs="+", default=list(TEXT_ENCODER_OUT_LAYERS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vectors, labels = collect_style_representations(args)
    steering, max_projection, min_projection = compute_difference_of_means(vectors, labels)
    save_steering_outputs(args.out_dir, steering, max_projection, min_projection)


if __name__ == "__main__":
    main()
