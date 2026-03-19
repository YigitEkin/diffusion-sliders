#!/usr/bin/env python3
"""Compute a Qwen2.5-VL steering vector via difference-of-means on style-token spans.

Qwen2.5-VL is a VLM, so style-token pooling cannot be done on raw prompt text alone.
This script:
1. Builds the same chat-style prompt template used for Qwen image edit prompting.
2. Finds style spans only inside the user content region (not the system template).
3. Maps those style-token indices onto the processor-expanded sequence
   (each `<|image_pad|>` expands from 1 token to many after processor encoding).
4. Pools last-layer hidden states for the style tokens and computes a DoM steering vector.

Usage
-----
python -m models.qwen.compute_vectors \\
    --pairs_file path/to/cartoon.jsonl \\
    --out_dir models/qwen/assets/steering_vectors/my_concept
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from .qwen_2_5 import Qwen2_5_VLForConditionalGeneration
from steering import (
    DTYPE_MAP,
    compute_difference_of_means,
    find_all_spans,
    save_steering_outputs,
    split_style_terms,
    validate_max_pairs,
    validate_path_exists,
)

warnings.filterwarnings("ignore")

LayerSpan = Tuple[int, int]

DEFAULT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain "
    "how the user's text instruction should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
DEFAULT_IMAGE_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


# ---------------------------------------------------------------------------
# VLM prompt construction
# ---------------------------------------------------------------------------


def build_vlm_prompt(
    user_prompt: str,
    prompt_template: str,
    image_prompt_template: str,
    num_images: int,
) -> Tuple[str, str]:
    """Return ``(full_prompt, user_content)`` with image placeholders prepended."""
    image_prefix = "".join(image_prompt_template.format(i + 1) for i in range(num_images))
    user_content = image_prefix + user_prompt
    return prompt_template.format(user_content), user_content


def get_style_token_indices_in_user_content(
    full_prompt: str,
    user_content: str,
    user_prompt: str,
    style: str,
    offsets: Iterable[LayerSpan],
) -> List[int]:
    """Return token indices that overlap style spans, constrained to the user-content region."""
    terms = split_style_terms(style)
    if not terms:
        return []

    user_start = full_prompt.find(user_content)
    if user_start < 0:
        raise ValueError("Failed to locate user content in the full prompt template.")

    if user_content.endswith(user_prompt):
        prompt_global_start = user_start + len(user_content) - len(user_prompt)
    else:
        prompt_global_start = user_start
        user_prompt = user_content

    style_spans_global: List[LayerSpan] = [
        (prompt_global_start + start, prompt_global_start + end)
        for term in terms
        for start, end in find_all_spans(user_prompt, term)
    ]
    if not style_spans_global:
        return []

    token_indices = set()
    for token_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if any(tok_start < span_end and tok_end > span_start for span_start, span_end in style_spans_global):
            token_indices.add(token_idx)
    return sorted(token_indices)


# ---------------------------------------------------------------------------
# Image-token expansion mapping
# ---------------------------------------------------------------------------


def compute_image_expansion_lengths(
    image_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> List[int]:
    """Compute how many tokens each `<|image_pad|>` placeholder expands to."""
    if image_grid_thw is None:
        return []
    return [int(v) for v in (image_grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()]


def map_unexpanded_to_expanded_indices(
    unexpanded_indices: Sequence[int],
    unexpanded_input_ids: Sequence[int],
    expanded_input_ids: Sequence[int],
    image_token_id: int,
    expansion_lengths: Sequence[int],
) -> List[int]:
    """Map token indices from the raw tokenizer sequence to the processor-expanded sequence."""
    placeholder_positions = [i for i, tid in enumerate(unexpanded_input_ids) if tid == image_token_id]
    if not placeholder_positions:
        return list(unexpanded_indices)
    if not expansion_lengths:
        expansion_lengths = [1] * len(placeholder_positions)
    if len(expansion_lengths) != len(placeholder_positions):
        raise ValueError(
            f"Mismatch: {len(placeholder_positions)} placeholders vs {len(expansion_lengths)} expansions."
        )

    deltas_by_pos = {pos: max(int(length), 1) - 1 for pos, length in zip(placeholder_positions, expansion_lengths)}
    cumulative_delta = [0] * (len(unexpanded_input_ids) + 1)
    running = 0
    for i in range(len(unexpanded_input_ids)):
        cumulative_delta[i] = running
        running += deltas_by_pos.get(i, 0)
    cumulative_delta[len(unexpanded_input_ids)] = running

    mapped: List[int] = []
    for idx in unexpanded_indices:
        mapped_idx = idx + cumulative_delta[idx]
        if mapped_idx >= len(expanded_input_ids):
            raise ValueError(
                f"Mapped index {mapped_idx} out of bounds for expanded sequence length {len(expanded_input_ids)}."
            )
        mapped.append(mapped_idx)
    return mapped


# ---------------------------------------------------------------------------
# Model loading and encoding
# ---------------------------------------------------------------------------


def parse_image_paths(image_paths: str) -> List[Image.Image]:
    if not image_paths:
        return []
    images = []
    for raw_path in image_paths.split(","):
        path = Path(raw_path.strip())
        if not path.exists():
            raise ValueError(f"Image path does not exist: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def load_qwen_vl(
    model_name_or_path: str,
    device: str,
    dtype_name: str,
) -> Tuple[AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration]:
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype `{dtype_name}`. Choose from: {list(DTYPE_MAP.keys())}")
    dtype = DTYPE_MAP[dtype_name]
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        print("Requested low-precision dtype on CPU; overriding to float32.")
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for offset mapping.")
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    ).to(device).eval()
    return tokenizer, processor, model


def encode_and_pool_style(
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    device: str,
    text: str,
    style: str,
    prompt_template: str,
    image_prompt_template: str,
    images: Sequence[Image.Image],
) -> Optional[torch.Tensor]:
    """Encode *text* with Qwen2.5-VL and mean-pool the last-layer hidden states at style tokens."""
    full_prompt, user_content = build_vlm_prompt(
        user_prompt=text,
        prompt_template=prompt_template,
        image_prompt_template=image_prompt_template,
        num_images=len(images),
    )
    unexpanded = tokenizer(full_prompt, return_offsets_mapping=True, add_special_tokens=True)
    unexpanded_ids = unexpanded["input_ids"]
    offsets = unexpanded["offset_mapping"]

    unexpanded_style_indices = get_style_token_indices_in_user_content(
        full_prompt=full_prompt,
        user_content=user_content,
        user_prompt=text,
        style=style,
        offsets=offsets,
    )
    if not unexpanded_style_indices:
        return None

    images_for_processor: Union[None, Image.Image, List[Image.Image]]
    if len(images) == 0:
        images_for_processor = None
    elif len(images) == 1:
        images_for_processor = images[0]
    else:
        images_for_processor = list(images)

    model_inputs = processor(
        text=[full_prompt],
        images=images_for_processor,
        padding=True,
        return_tensors="pt",
    ).to(device)

    forward_kwargs = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "output_hidden_states": True,
        "output_attentions": False,
    }
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
        if key in model_inputs:
            forward_kwargs[key] = model_inputs[key]

    with torch.no_grad():
        outputs = model(**forward_kwargs)

    last_hidden = outputs.hidden_states[-1]
    expanded_ids = model_inputs.input_ids[0].tolist()
    image_grid_thw = model_inputs.get("image_grid_thw", None)
    expansion_lengths = compute_image_expansion_lengths(
        image_grid_thw=image_grid_thw,
        spatial_merge_size=model.model.visual.spatial_merge_size,
    )

    expected_len = len(unexpanded_ids) + sum(max(length, 1) - 1 for length in expansion_lengths)
    if expected_len != len(expanded_ids):
        raise ValueError(
            f"Token alignment failed: expected {expected_len}, got {len(expanded_ids)}."
        )

    expanded_style_indices = map_unexpanded_to_expanded_indices(
        unexpanded_indices=unexpanded_style_indices,
        unexpanded_input_ids=unexpanded_ids,
        expanded_input_ids=expanded_ids,
        image_token_id=model.config.image_token_id,
        expansion_lengths=expansion_lengths,
    )
    vectors = [last_hidden[0, idx].detach().cpu() for idx in expanded_style_indices]
    if not vectors:
        return None
    return torch.stack(vectors).mean(0).float()


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------


def collect_style_representations(
    pairs_file: Path,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    device: str,
    max_pairs: int,
    prompt_template: str,
    image_prompt_template: str,
    images: Sequence[Image.Image],
) -> Tuple[List[np.ndarray], List[int]]:
    style_vectors: List[np.ndarray] = []
    style_labels: List[int] = []
    limit = None if max_pairs == -1 else max_pairs
    processed = 0

    with pairs_file.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=limit, desc="Collecting Qwen2.5-VL style representations"):
            if limit is not None and processed >= limit:
                break
            example = json.loads(line)
            pos_text, neg_text = example["pos"], example["neg"]
            pos_style, neg_style = example.get("pos_style"), example.get("neg_style")
            if not (pos_style and neg_style):
                raise ValueError(
                    f"Both pos_style and neg_style are required; "
                    f"got pos_style={pos_style}, neg_style={neg_style}."
                )
            common = dict(
                tokenizer=tokenizer, processor=processor, model=model, device=device,
                prompt_template=prompt_template, image_prompt_template=image_prompt_template,
                images=images,
            )
            pos_vec = encode_and_pool_style(text=pos_text, style=pos_style, **common)
            neg_vec = encode_and_pool_style(text=neg_text, style=neg_style, **common)
            if pos_vec is None or neg_vec is None:
                raise ValueError(
                    f"Unable to find/pool style token vectors for pair:\n"
                    f"  pos_style={pos_style!r}, pos={pos_text!r}\n"
                    f"  neg_style={neg_style!r}, neg={neg_text!r}"
                )
            style_vectors.extend([pos_vec.numpy().astype(np.float32), neg_vec.numpy().astype(np.float32)])
            style_labels.extend([1, 0])
            processed += 1

    return style_vectors, style_labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a Qwen2.5-VL steering vector from contrastive prompt pairs."
    )
    parser.add_argument("--pairs_file", type=validate_path_exists, required=True,
                        help="Path to JSONL with {pos, neg, pos_style, neg_style}.")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Output directory for steering vector files.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--max_pairs", type=validate_max_pairs, default=-1,
                        help="Number of pairs to process (-1 = all).")
    parser.add_argument("--image_paths", type=str, default="",
                        help="Optional comma-separated image paths for vision placeholders.")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE,
                        help="Prompt template with exactly one '{}' slot for user content.")
    parser.add_argument("--image_prompt_template", type=str, default=DEFAULT_IMAGE_PROMPT_TEMPLATE,
                        help="Per-image prefix template with one '{}' slot for image index.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.prompt_template.count("{}") != 1:
        raise ValueError("--prompt_template must include exactly one '{}' slot.")
    if args.image_prompt_template.count("{}") != 1:
        raise ValueError("--image_prompt_template must include exactly one '{}' slot.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = parse_image_paths(args.image_paths)
    pair_msg = args.max_pairs if args.max_pairs != -1 else "all"
    image_msg = f" with {len(images)} image(s)" if images else " without images"
    print(f"Using {args.model} on {args.device} ({args.dtype}), "
          f"processing {pair_msg} pairs{image_msg} from {args.pairs_file}.")

    tokenizer, processor, model = load_qwen_vl(args.model, args.device, args.dtype)
    vectors, labels = collect_style_representations(
        pairs_file=args.pairs_file,
        tokenizer=tokenizer,
        processor=processor,
        model=model,
        device=args.device,
        max_pairs=args.max_pairs,
        prompt_template=args.prompt_template,
        image_prompt_template=args.image_prompt_template,
        images=images,
    )
    if not vectors:
        raise ValueError("No valid vectors were collected; check dataset annotations and settings.")

    steering, max_projection, min_projection = compute_difference_of_means(vectors, labels)
    save_steering_outputs(out_dir, steering, max_projection, min_projection)


if __name__ == "__main__":
    main()
