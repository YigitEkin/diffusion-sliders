#!/usr/bin/env python3
"""Wan end-to-end steering workflow.

Runs all steps in a single process (one model load) with skip logic so
re-runs only execute missing steps:

  1. [concurrent API] Generate contrastive dataset + select steering tokens
  2. Compute steering vectors  (Wan text encoder only; freed after)
  3. Load full pipeline        (kept in memory for step 4)
  4. Inference: grid across the steering range

Usage
-----
python -m models.wan.run \\
    --concept cartoon \\
    --prompt "Two anthropomorphic cats boxing on a stage." \\
    --out_dir outputs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from dataset.generate import generate_dataset
from dataset.select_tokens import select_tokens
from steering import DTYPE_MAP, compute_difference_of_means, load_steering_vector, save_steering_outputs
from .compute_vectors import collect_style_representations, load_wan_text_encoder
from .pipeline import DEFAULT_MODEL_ID, DEFAULT_NEGATIVE_PROMPT, WanSteeringCallback, build_pipeline


def _make_grid(images: list[Image.Image], cols: int = 4, pad: int = 16) -> Image.Image:
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
    parser = argparse.ArgumentParser(description="Wan end-to-end steering workflow.")
    parser.add_argument("--concept", required=True, help="Concept name, e.g. 'cartoon'.")
    parser.add_argument("--prompt", required=True, help="Generation prompt.")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"),
                        help="Root output directory (default: outputs).")
    # Dataset generation
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Contrastive pairs to generate (default: 100).")
    parser.add_argument("--openai_model", type=str, default="gpt-4o",
                        help="OpenAI model for dataset generation (default: gpt-4o).")
    # Token selection
    parser.add_argument("--token_model", type=str, default="Qwen/Qwen3-8B",
                        help="Local HuggingFace model for token selection (default: Qwen/Qwen3-8B).")
    # Shared credentials (only needed if dataset generation is required)
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key for dataset generation. Defaults to OPENAI_API_KEY env var.")
    # Compute-vectors
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="Wan model ID for text encoder and pipeline.")
    parser.add_argument("--tokenizer_subfolder", type=str, default="tokenizer")
    parser.add_argument("--text_encoder_subfolder", type=str, default="text_encoder")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--encoder_dtype", type=str, default="float32", choices=list(DTYPE_MAP.keys()),
                        help="dtype for the text encoder (default: float32).")
    parser.add_argument("--max_pairs", type=int, default=-1,
                        help="Max contrastive pairs for vector computation (-1 = all).")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    # Inference
    parser.add_argument("--num_outputs", type=int, default=5,
                        help="Number of output videos across the steering range (default: 5).")
    parser.add_argument("--strength_min", type=float, default=0.0,
                        help="Minimum steering strength (default: 0.0).")
    parser.add_argument("--strength_max", type=float, default=5.0,
                        help="Maximum steering strength (default: 5.0).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--flow_shift", type=float, default=5.0,
                        help="5.0 for 720p, 3.0 for 480p.")
    parser.add_argument("--schedule_type", type=str, choices=["constant", "linear"], default="constant")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    concept_dir = args.out_dir / args.concept
    concept_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = concept_dir / f"{args.concept}.jsonl"
    tokens_path = concept_dir / "tokens_to_edit.json"
    inference_dir = concept_dir / "inference"

    # -----------------------------------------------------------------------
    # Step 1: Dataset generation + token selection
    # -----------------------------------------------------------------------
    needs_dataset = not jsonl_path.exists()
    needs_tokens = not tokens_path.exists()

    if needs_dataset:
        from openai import OpenAI
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: set OPENAI_API_KEY or pass --api_key.", file=sys.stderr)
            sys.exit(1)
        openai_client = OpenAI(api_key=api_key)
        records = generate_dataset(args.concept, args.num_examples, args.openai_model, openai_client)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Dataset saved → {jsonl_path} ({len(records)} pairs)")

    if needs_tokens:
        print(f"Selecting tokens with {args.token_model}…")
        tokens = select_tokens(args.prompt, args.concept, args.token_model, args.device)
        tokens_path.write_text(json.dumps(tokens))

    tokens_to_edit: list[str] = json.loads(tokens_path.read_text())
    print(f"Tokens to edit {'(new)' if needs_tokens else '(cached)'}: {tokens_to_edit}")

    # -----------------------------------------------------------------------
    # Step 2: Compute steering vectors (Wan text encoder only; freed after)
    # -----------------------------------------------------------------------
    vectors_done = (concept_dir / "steering_last_layer.npy").exists()
    if not vectors_done:
        print("Computing steering vectors…")
        tokenizer, text_encoder = load_wan_text_encoder(
            model_id=args.model_id,
            tokenizer_subfolder=args.tokenizer_subfolder,
            text_encoder_subfolder=args.text_encoder_subfolder,
            device=args.device,
            dtype=DTYPE_MAP[args.encoder_dtype],
        )
        vectors, labels = collect_style_representations(
            pairs_file=jsonl_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=args.device,
            max_sequence_length=args.max_sequence_length,
            max_pairs=args.max_pairs,
        )
        steering_vec, max_proj, min_proj = compute_difference_of_means(vectors, labels)
        save_steering_outputs(concept_dir, steering_vec, max_proj, min_proj)
        del text_encoder, tokenizer
        torch.cuda.empty_cache()
        print(f"Steering vectors saved → {concept_dir}")
    else:
        print("Steering vectors already computed, skipping.")

    # -----------------------------------------------------------------------
    # Step 3: Load full pipeline
    # -----------------------------------------------------------------------
    print("Loading pipeline…")
    pipe = build_pipeline(
        model_id=args.model_id,
        torch_dtype=DTYPE_MAP[args.dtype],
        device=args.device,
        flow_shift=args.flow_shift,
    )
    steering_vector = load_steering_vector(concept_dir / "steering_last_layer.npy", device=args.device)

    # -----------------------------------------------------------------------
    # Step 4: Inference across the steering range
    # -----------------------------------------------------------------------
    inference_dir.mkdir(parents=True, exist_ok=True)
    strengths = np.linspace(args.strength_min, args.strength_max, args.num_outputs).tolist()
    print(f"Generating {args.num_outputs} videos from {args.strength_min:.4f} to {args.strength_max:.4f}…")

    first_frames: list[Image.Image] = []
    for strength in strengths:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        callback = WanSteeringCallback(
            pipeline=pipe,
            prompt=args.prompt,
            tokens_to_edit=tokens_to_edit,
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

        out_path = inference_dir / f"steer_{float(strength):.3f}.mp4"
        export_to_video(video, str(out_path), fps=args.fps)
        print(f"Saved: {out_path}")
        first_frames.append(_first_frame(video))

    grid = _make_grid(first_frames, cols=min(4, len(first_frames)))
    grid.save(inference_dir / "grid.png")
    print(f"Grid saved → {inference_dir / 'grid.png'}")


if __name__ == "__main__":
    main()
