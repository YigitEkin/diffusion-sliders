#!/usr/bin/env python3
"""Qwen Image Edit end-to-end steering workflow.

Runs all steps in a single process (one model load) with skip logic so
re-runs only execute missing steps:

  1. [concurrent API] Generate contrastive dataset + select steering tokens
  2. Compute steering vectors  (Qwen2.5-VL text encoder only; freed after)
  3. Load full pipeline        (kept in memory for steps 4 and 5)
  4. Elastic-band search
  5. Inference: grid across the valid steering range

Usage
-----
python -m models.qwen.run \\
    --concept cartoon \\
    --prompt "make the scene cartoon" \\
    --input_image path/to/image.png
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
from PIL import Image

from dataset.generate import generate_dataset
from dataset.select_tokens import select_tokens
from steering import DTYPE_MAP, compute_difference_of_means, load_steering_vector, save_steering_outputs
from steering.elastic_band import (
    canonical_strength,
    elastic_band_search,
    find_effective_minimum,
    load_min_projection_value,
    summarize_valid_range,
)
from ._utils import build_pipeline
from .compute_vectors import (
    DEFAULT_IMAGE_PROMPT_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    collect_style_representations,
    load_qwen_vl,
)
from steering.elastic_band import ElasticBandConfig
from .elastic_band import ElasticBandQwenRunner

def _make_grid(images: list[Image.Image], cols: int = 4, pad: int = 16) -> Image.Image:
    w, h = images[0].size
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), (255, 255, 255))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * (w + pad), r * (h + pad)))
    return grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen Image Edit end-to-end steering workflow.")
    parser.add_argument("--concept", required=True, help="Concept name, e.g. 'cartoon'.")
    parser.add_argument("--prompt", required=True, help="Editing prompt, e.g. 'make the scene cartoon'.")
    parser.add_argument("--input_image", required=True, help="Path to the input conditioning image.")
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
    # Inference
    parser.add_argument("--num_outputs", type=int, default=8,
                        help="Number of output images across the valid range (default: 8).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--lora", action="store_true", default=True,
                        help="Load Lightning LoRA for faster inference.")
    parser.add_argument("--no_lora", dest="lora", action="store_false")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()),
                        help="dtype for the full pipeline (default: bfloat16).")
    # Compute-vectors
    parser.add_argument("--vl_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Qwen2.5-VL model for steering vector computation.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vl_dtype", type=str, default="float32", choices=list(DTYPE_MAP.keys()),
                        help="dtype for the VL text encoder (default: float32).")
    parser.add_argument("--max_pairs", type=int, default=-1,
                        help="Max contrastive pairs to use for vector computation (-1 = all).")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to elastic-band YAML config. Auto-selected from concept type if omitted "
                             "(configs/qwen_local.yaml for Local Edit, configs/qwen_global.yaml otherwise).")
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
    concept_type_path = concept_dir / "concept_type.json"
    eb_dir = concept_dir / "elastic_band"
    eb_summary_path = eb_dir / "summary.json"
    inference_dir = concept_dir / "inference"

    # -----------------------------------------------------------------------
    # Step 1: Dataset generation + token selection (concurrent API calls)
    # -----------------------------------------------------------------------
    needs_dataset = not jsonl_path.exists()
    needs_tokens = not tokens_path.exists() or not concept_type_path.exists()

    # -----------------------------------------------------------------------
    # Step 1a: Dataset generation (requires OpenAI API key)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Step 1b: Token selection (local Qwen3-8B, no API key needed)
    # -----------------------------------------------------------------------
    if needs_tokens:
        print(f"Selecting tokens with {args.token_model}…")
        tokens, concept_type = select_tokens(args.prompt, args.concept, args.token_model, args.device)
        tokens_path.write_text(json.dumps(tokens))
        concept_type_path.write_text(json.dumps(concept_type))

    tokens_to_edit: list[str] = json.loads(tokens_path.read_text())
    concept_type: str = json.loads(concept_type_path.read_text())
    print(f"Tokens to edit {'(new)' if needs_tokens else '(cached)'}: {tokens_to_edit}")
    print(f"Concept type {'(new)' if needs_tokens else '(cached)'}: {concept_type}")

    # Auto-select config based on concept type unless the user overrides it.
    if args.config is not None:
        config_path = args.config
    else:
        config_path = Path("configs/qwen_local.yaml" if concept_type == "Local Edit" else "configs/qwen_global.yaml")
        print(f"Auto-selected config: {config_path}")
    elastic_band_config = ElasticBandConfig.from_yaml(config_path)

    # -----------------------------------------------------------------------
    # Step 2: Compute steering vectors (Qwen2.5-VL text encoder only)
    # -----------------------------------------------------------------------
    vectors_done = (concept_dir / "steering_last_layer.npy").exists()
    if not vectors_done:
        print("Computing steering vectors…")
        tokenizer, processor, vl_model = load_qwen_vl(
            model_name_or_path=args.vl_model,
            device=args.device,
            dtype_name=args.vl_dtype,
        )
        vectors, labels = collect_style_representations(
            pairs_file=jsonl_path,
            tokenizer=tokenizer,
            processor=processor,
            model=vl_model,
            device=args.device,
            max_pairs=args.max_pairs,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            image_prompt_template=DEFAULT_IMAGE_PROMPT_TEMPLATE,
            images=[],
        )
        steering_vec, max_proj, min_proj = compute_difference_of_means(vectors, labels)
        save_steering_outputs(concept_dir, steering_vec, max_proj, min_proj)
        # Free the VL model before loading the full pipeline
        del vl_model, processor, tokenizer
        torch.cuda.empty_cache()
        print(f"Steering vectors saved → {concept_dir}")
    else:
        print("Steering vectors already computed, skipping.")

    # -----------------------------------------------------------------------
    # Step 3: Load full pipeline (kept in memory for steps 4 and 5)
    # -----------------------------------------------------------------------
    print("Loading pipeline…")
    pipe = build_pipeline(DTYPE_MAP[args.dtype], args.lora).to(args.device)
    steering_vector = load_steering_vector(concept_dir / "steering_last_layer.npy", device=args.device)
    input_image = Image.open(args.input_image).convert("RGB")

    # -----------------------------------------------------------------------
    # Step 4: Elastic-band search
    # -----------------------------------------------------------------------
    if not eb_summary_path.exists():
        eb_dir.mkdir(parents=True, exist_ok=True)
        print("Running elastic-band search…")
        runner = ElasticBandQwenRunner(
            pipe=pipe,
            prompt=args.prompt,
            tokens_to_edit=tokens_to_edit,
            input_image=input_image,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            dtype=DTYPE_MAP[args.dtype],
        )
        stored_min = load_min_projection_value(concept_dir)
        initialization = find_effective_minimum(
            runner=runner,
            concept_dir=eb_dir,
            concept_name=args.concept,
            steering_vector=steering_vector,
            initial_min=stored_min,
            config=elastic_band_config,
        )
        elastic_result = elastic_band_search(
            runner=runner,
            concept_dir=eb_dir,
            concept_name=args.concept,
            steering_vector=steering_vector,
            a_min=initialization["search_minimum_value"],
            a_max=0.0,
            config=elastic_band_config,
        )
        valid_range = summarize_valid_range(elastic_result["valid_control_points"])
        eb_summary = {
            "concept": args.concept,
            "prompt": args.prompt,
            "tokens_to_edit": tokens_to_edit,
            "seed": args.seed,
            "initialization": initialization,
            "elastic_band": elastic_band_config.to_dict(),
            "valid_range": valid_range,
            "all_generated_strengths": sorted(runner.image_cache.keys()),
            "elastic_search_result": elastic_result,
        }
        eb_summary_path.write_text(json.dumps(eb_summary, indent=2))
        print(f"Elastic band complete. Valid range: {valid_range}")
    else:
        print("Elastic-band search already done, loading summary…")
        eb_summary = json.loads(eb_summary_path.read_text())
        valid_range = eb_summary["valid_range"]
        runner = ElasticBandQwenRunner(
            pipe=pipe,
            prompt=args.prompt,
            tokens_to_edit=tokens_to_edit,
            input_image=input_image,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            dtype=DTYPE_MAP[args.dtype],
        )
        print(f"Valid range (cached): {valid_range}")

    # -----------------------------------------------------------------------
    # Step 5: Inference across the valid range
    # -----------------------------------------------------------------------
    inference_dir.mkdir(parents=True, exist_ok=True)
    a_min = valid_range["minimum_valid_value"]
    a_max = valid_range["maximum_valid_value"]
    strengths = np.linspace(a_min, a_max, args.num_outputs).tolist()
    print(f"Generating {args.num_outputs} images from {a_min:.4f} to {a_max:.4f}…")

    runner.clear_caches()
    runner.generate_images(inference_dir, args.concept, steering_vector, strengths)

    images = [runner.image_cache[canonical_strength(s)] for s in strengths]
    grid = _make_grid(images, cols=min(4, len(images)))
    grid.save(inference_dir / "grid.png")
    print(f"Grid saved → {inference_dir / 'grid.png'}")


if __name__ == "__main__":
    main()
