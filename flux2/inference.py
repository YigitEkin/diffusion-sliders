#!/usr/bin/env python3
"""Flux 2 steering inference — generates a grid across a range of strengths.

Usage
-----
python -m models.flux2.inference \\
    --input_image path/to/image.png \\
    --steering_vector models/flux2/assets/cartoon/steering_last_layer.npy \\
    --prompt "make the scene cartoon" \\
    --tokens_to_edit cartoon \\
    --strengths -10 -8 -6 -4 -2 0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import load_image
from PIL import Image

from steering import DTYPE_MAP, load_steering_vector
from ._utils import (
    MAX_SEQUENCE_LENGTH,
    SYSTEM_MESSAGE,
    TEXT_ENCODER_OUT_LAYERS,
    TURBO_SIGMAS,
    apply_steering,
    build_pipeline,
    encode_flux2_prompt_embeds,
    find_indices_to_edit,
)



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def make_grid(images: list[Image.Image], cols: int = 4, pad: int = 16) -> Image.Image:
    if not images:
        raise ValueError("No images provided for grid creation.")
    w, h = images[0].size
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), (255, 255, 255))
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        grid.paste(image, (col * (w + pad), row * (h + pad)))
    return grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flux 2 steering inference.")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--steering_vector", type=str, required=True,
                        help="Path to steering_last_layer.npy")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tokens_to_edit", type=str, nargs="+", required=True)
    parser.add_argument("--strengths", type=float, nargs="+", required=True,
                        help="Steering strengths to sweep (e.g. -10 -8 -6 -4 -2 0)")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_flux2_steering"))
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--lora", action="store_true", default=True,
                        help="Load Turbo LoRA for faster inference.")
    parser.add_argument("--distributed", action="store_true", default=True,
                        help="Use distributed device mapping across available GPUs.")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Defaults to 2.5 with LoRA, 4.0 without.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--max_sequence_length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--text_encoder_out_layers", type=int, nargs="+",
                        default=list(TEXT_ENCODER_OUT_LAYERS))
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    guidance_scale = args.guidance_scale if args.guidance_scale is not None else (2.5 if args.lora else 4.0)
    num_inference_steps = 8 if args.lora else 50

    pipe = build_pipeline(torch_dtype=DTYPE_MAP[args.dtype], use_lora=args.lora, use_distributed=args.distributed)
    condition_image = load_image(args.input_image).convert("RGB")
    steering_vector = load_steering_vector(args.steering_vector, device="cpu")

    idx_to_edit = find_indices_to_edit(
        pipe=pipe,
        prompt=args.prompt,
        tokens_to_edit=args.tokens_to_edit,
        max_sequence_length=args.max_sequence_length,
    )
    print(f"Editing token indices: {idx_to_edit}")

    text_encoder_device = pipe._get_module_execution_device(pipe.text_encoder)
    base_prompt_embeds = encode_flux2_prompt_embeds(
        text_encoder=pipe.text_encoder,
        processor=pipe.tokenizer,
        prompt=args.prompt,
        device=text_encoder_device,
        max_sequence_length=args.max_sequence_length,
        system_message=pipe.system_message,
        hidden_states_layers=args.text_encoder_out_layers,
    )

    generated_images: list[Image.Image] = []
    for strength in args.strengths:
        prompt_embeds = apply_steering(
            base_prompt_embeds=base_prompt_embeds,
            idx_to_edit=idx_to_edit,
            steering_vector=steering_vector,
            factor=strength,
        )
        output = pipe(
            image=condition_image,
            prompt=None,
            prompt_embeds=prompt_embeds,
            sigmas=TURBO_SIGMAS if args.lora else None,
            height=args.height,
            width=args.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=args.max_sequence_length,
            text_encoder_out_layers=args.text_encoder_out_layers,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
        image = output.images[0]
        out_path = args.out_dir / f"steer_{strength:.3f}.png"
        image.save(out_path)
        generated_images.append(image)
        print(f"Saved: {out_path}")

    grid = make_grid(generated_images, cols=min(4, len(generated_images)))
    grid_path = args.out_dir / "grid.png"
    grid.save(grid_path)
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
