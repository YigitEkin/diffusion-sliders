#!/usr/bin/env python3
"""Wan steering inference — generates a grid of first frames across a range of strengths.

Usage
-----
python -m models.wan.inference \\
    --prompt "Two cats boxing on a stage." \\
    --tokens_to_edit boxing cats \\
    --steering_vector outputs/concept/steering_last_layer.npy \\
    --strengths 0.0 1.0 2.0 3.0 4.0 \\
    --out_dir outputs/concept/inference
"""

from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from steering import DTYPE_MAP, load_steering_vector
from .pipeline import DEFAULT_MODEL_ID, DEFAULT_NEGATIVE_PROMPT, WanSteeringCallback, build_pipeline


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _make_grid(images: list[Image.Image], cols: int = 4, pad: int = 16) -> Image.Image:
    if not images:
        raise ValueError("No images provided for grid creation.")
    w, h = images[0].size
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), (255, 255, 255))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * (w + pad), r * (h + pad)))
    return grid


def _first_frame(video: list) -> Image.Image:
    frame = video[0]
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wan steering inference.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tokens_to_edit", type=str, nargs="+", required=True)
    parser.add_argument("--steering_vector", type=str, required=True,
                        help="Path to steering_last_layer.npy")
    parser.add_argument("--strengths", type=float, nargs="+", required=True,
                        help="Steering strengths to sweep (e.g. 0 1 2 3 4)")
    parser.add_argument("--out_dir", type=str, default="outputs_wan_steering")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=sorted(DTYPE_MAP.keys()), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--flow_shift", type=float, default=5.0,
                        help="5.0 for 720p, 3.0 for 480p.")
    parser.add_argument("--schedule_type", type=str, choices=["constant", "linear"], default="constant")
    parser.add_argument("--fps", type=int, default=16)
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _seed_everything(args.seed)

    pipe = build_pipeline(
        model_id=args.model_id,
        torch_dtype=DTYPE_MAP[args.dtype],
        device=args.device,
        flow_shift=args.flow_shift,
    )
    steering_vector = load_steering_vector(args.steering_vector, device=args.device)

    first_frames: list[Image.Image] = []
    for strength in args.strengths:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        callback = WanSteeringCallback(
            pipeline=pipe,
            prompt=args.prompt,
            tokens_to_edit=args.tokens_to_edit,
            steering_vector=steering_vector,
            factor=float(strength),
            schedule_type=args.schedule_type,
        )
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["prompt_embeds", "negative_prompt_embeds"],
        ).frames[0]

        out_path = os.path.join(args.out_dir, f"steer_{float(strength):.3f}.mp4")
        export_to_video(video, out_path, fps=args.fps)
        print(f"Saved: {out_path}")
        first_frames.append(_first_frame(video))

    grid = _make_grid(first_frames, cols=min(4, len(first_frames)))
    grid_path = os.path.join(args.out_dir, "grid.png")
    grid.save(grid_path)
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
