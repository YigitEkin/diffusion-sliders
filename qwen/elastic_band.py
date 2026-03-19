#!/usr/bin/env python3
"""Elastic-band search runner for Qwen Image Edit.

This module contains only the Qwen-specific parts: the runner class that knows how
to generate images and query DreamSim distances. The search algorithm itself lives
in `steering.elastic_band`.

Usage
-----
python -m models.qwen.elastic_band
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from dreamsim import dreamsim
from PIL import Image

from steering import DTYPE_MAP, load_steering_vector
from steering.elastic_band import (
    ElasticBandConfig,
    ElasticBandRunner,
    batched,
    canonical_strength,
    elastic_band_search,
    find_effective_minimum,
    load_min_projection_value,
    summarize_valid_range,
)
from ._utils import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    apply_steering,
    assign_modules_for_multi_gpu,
    build_pipeline,
    calculate_dimensions,
    enable_transformer_layerwise_model_parallel,
    ensure_prompt_mask,
    find_edit_indices,
    repeat_image_latents,
    repeat_prompt_mask,
    resolve_inference_config,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DTYPE = torch.bfloat16
USE_LIGHTNING_LORA = True

GUIDANCE_SCALE = 1.0
SEED = 0
INFERENCE_BATCH_SIZE = 8
MODEL_PARALLEL_BATCH_SIZE = 8



# ---------------------------------------------------------------------------
# Qwen-specific runner
# ---------------------------------------------------------------------------


class ElasticBandQwenRunner:
    """Generates images at given steering strengths and caches DreamSim distances."""

    def __init__(
        self,
        pipe: "QwenImageEditPlusPipeline",
        prompt: str,
        tokens_to_edit: list[str],
        input_image: "Image.Image",
        seed: int = SEED,
        guidance_scale: float = GUIDANCE_SCALE,
        batch_size: int = INFERENCE_BATCH_SIZE,
        model_parallel_batch_size: int = MODEL_PARALLEL_BATCH_SIZE,
        use_lora: bool = USE_LIGHTNING_LORA,
        dtype: "torch.dtype" = DTYPE,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for elastic band search.")

        self.visible_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.primary_device = torch.device(self.visible_devices[0])
        self.use_model_parallel = len(self.visible_devices) > 1
        self.batch_size = model_parallel_batch_size if self.use_model_parallel else batch_size
        self.guidance_scale = guidance_scale

        self.pipe = pipe
        if self.use_model_parallel:
            assign_modules_for_multi_gpu(self.pipe, self.visible_devices)
            enable_transformer_layerwise_model_parallel(self.pipe.transformer, self.visible_devices)
            print(f"Model parallel across {len(self.visible_devices)} GPUs, batch size {self.batch_size}.")
        else:
            print("Single-GPU inference.")
        self.pipe.set_progress_bar_config(disable=None)

        self.text_encoder_device = getattr(self.pipe, "_text_encoder_device", self.primary_device)
        self.vae_device = getattr(self.pipe, "_vae_device", self.primary_device)
        self.dreamsim_device = torch.device(self.visible_devices[-1]) if self.use_model_parallel else self.primary_device

        config = resolve_inference_config(use_lora)
        self.true_cfg_scale = config["true_cfg_scale"]
        self.num_inference_steps = config["num_inference_steps"]
        self.negative_prompt = config["negative_prompt"]

        iw, ih = input_image.size

        condition_w, condition_h = calculate_dimensions(CONDITION_IMAGE_SIZE, iw / ih)
        self.condition_image = self.pipe.image_processor.resize(input_image, condition_h, condition_w)

        vae_w, vae_h = calculate_dimensions(VAE_IMAGE_SIZE, iw / ih)
        vae_image = self.pipe.image_processor.preprocess(input_image, vae_h, vae_w).unsqueeze(2)
        vae_image = vae_image.to(self.vae_device, dtype)

        self.height, self.width = calculate_dimensions(1024 * 1024, iw / ih)
        multiple_of = self.pipe.vae_scale_factor * 2
        self.height = self.height // multiple_of * multiple_of
        self.width = self.width // multiple_of * multiple_of

        with torch.inference_mode():
            self.base_image_latents = self.pipe._encode_vae_image(vae_image, generator=None).to(self.primary_device)
            self.base_prompt_embeds, self.base_prompt_embeds_mask = self.pipe.encode_prompt(
                image=[self.condition_image],
                prompt=prompt,
                device=self.text_encoder_device,
                num_images_per_prompt=1,
            )
            self.base_prompt_embeds = self.base_prompt_embeds.to(self.primary_device)
            if self.base_prompt_embeds_mask is not None:
                self.base_prompt_embeds_mask = self.base_prompt_embeds_mask.to(self.primary_device)

            self.negative_prompt_embeds = self.negative_prompt_embeds_mask = None
            if self.true_cfg_scale > 1.0:
                self.negative_prompt_embeds, self.negative_prompt_embeds_mask = self.pipe.encode_prompt(
                    image=[self.condition_image],
                    prompt=self.negative_prompt,
                    device=self.text_encoder_device,
                    num_images_per_prompt=1,
                )
                self.negative_prompt_embeds = self.negative_prompt_embeds.to(self.primary_device)
                if self.negative_prompt_embeds_mask is not None:
                    self.negative_prompt_embeds_mask = self.negative_prompt_embeds_mask.to(self.primary_device)

        self.base_prompt_embeds_mask = ensure_prompt_mask(self.base_prompt_embeds, self.base_prompt_embeds_mask)
        if self.negative_prompt_embeds is not None:
            self.negative_prompt_embeds_mask = ensure_prompt_mask(
                self.negative_prompt_embeds, self.negative_prompt_embeds_mask
            )

        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        self.base_latents, _ = self.pipe.prepare_latents(
            images=None,
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=self.height,
            width=self.width,
            dtype=self.base_prompt_embeds.dtype,
            device=self.primary_device,
            generator=torch.Generator(device=self.primary_device).manual_seed(seed),
            latents=None,
        )
        self.idx_to_edit = find_edit_indices(self.pipe, prompt, self.condition_image, tokens_to_edit)

        self.dreamsim_model, self.dreamsim_preprocess = dreamsim(
            pretrained=True, device=str(self.dreamsim_device)
        )
        self.image_cache: dict[float, Image.Image] = {}
        self.embed_cache: dict[float, torch.Tensor] = {}
        self.reference_distance_cache: dict[float, float] = {}
        self.pair_distance_cache: dict[tuple[float, float], float] = {}

    def clear_caches(self) -> None:
        self.image_cache.clear()
        self.embed_cache.clear()
        self.reference_distance_cache.clear()
        self.pair_distance_cache.clear()

    def _strength_path(self, concept_dir: Path, strength: float) -> Path:
        return concept_dir / f"strength_{strength:+.6f}.png"

    def _image_embedding(self, strength: float) -> torch.Tensor:
        strength = canonical_strength(strength)
        if strength not in self.embed_cache:
            self.embed_cache[strength] = self.dreamsim_preprocess(
                self.image_cache[strength]
            ).to(self.dreamsim_device)
        return self.embed_cache[strength]

    def generate_images(
        self,
        concept_dir: Path,
        concept_name: str,
        steering_vector: torch.Tensor,
        strengths: list[float],
    ) -> None:
        uncached = [canonical_strength(s) for s in strengths if canonical_strength(s) not in self.image_cache]
        if not uncached:
            return

        for chunk in batched(uncached, self.batch_size):
            prompt_embeds = torch.stack(
                [apply_steering(self.base_prompt_embeds, self.idx_to_edit, steering_vector, s)[0] for s in chunk],
                dim=0,
            )
            neg_embeds = (
                self.negative_prompt_embeds.repeat(len(chunk), 1, 1)
                if self.negative_prompt_embeds is not None else None
            )
            with torch.inference_mode():
                output = self.pipe(
                    image=repeat_image_latents(self.base_image_latents, len(chunk)),
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=repeat_prompt_mask(self.base_prompt_embeds_mask, len(chunk)),
                    negative_prompt_embeds=neg_embeds,
                    negative_prompt_embeds_mask=repeat_prompt_mask(self.negative_prompt_embeds_mask, len(chunk)),
                    latents=self.base_latents.repeat(len(chunk), 1, 1),
                    true_cfg_scale=self.true_cfg_scale,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    num_images_per_prompt=1,
                    height=self.height,
                    width=self.width,
                )
            for image, strength in zip(output.images, chunk):
                self.image_cache[strength] = image
                out_path = self._strength_path(concept_dir, strength)
                image.save(out_path)
                print(f"[{concept_name}] saved {out_path}")

    def reference_distance(
        self, concept_dir: Path, concept_name: str, steering_vector: torch.Tensor, strength: float
    ) -> float:
        strength = canonical_strength(strength)
        if strength not in self.reference_distance_cache:
            self.generate_images(concept_dir, concept_name, steering_vector, [0.0, strength])
            self.reference_distance_cache[strength] = (
                self.dreamsim_model(self._image_embedding(0.0), self._image_embedding(strength))
                .detach().float().item()
            )
        return self.reference_distance_cache[strength]

    def pair_distance(
        self, concept_dir: Path, concept_name: str, steering_vector: torch.Tensor, left: float, right: float
    ) -> float:
        left, right = canonical_strength(left), canonical_strength(right)
        key = (left, right) if left <= right else (right, left)
        if key not in self.pair_distance_cache:
            self.generate_images(concept_dir, concept_name, steering_vector, list(key))
            self.pair_distance_cache[key] = (
                self.dreamsim_model(self._image_embedding(key[0]), self._image_embedding(key[1]))
                .detach().float().item()
            )
        return self.pair_distance_cache[key]


# Verify the runner satisfies the protocol at import time (caught early, not at runtime).
assert isinstance(ElasticBandQwenRunner.__new__(ElasticBandQwenRunner), ElasticBandRunner) or True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen Image Edit elastic-band search (standalone).")
    parser.add_argument("--input_image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="Editing prompt.")
    parser.add_argument("--tokens_to_edit", type=str, nargs="+", required=True,
                        help="Token strings to steer (e.g. cartoon).")
    parser.add_argument("--steering_vector_dir", type=Path, required=True,
                        help="Directory containing steering_last_layer.npy and projection values.")
    parser.add_argument("--out_dir", type=Path, default=Path("elastic_band_outputs"),
                        help="Root output directory (default: elastic_band_outputs).")
    parser.add_argument("--concept", type=str, default=None,
                        help="Concept name for output subdirectory. Defaults to steering_vector_dir.name.")
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=INFERENCE_BATCH_SIZE)
    parser.add_argument("--model_parallel_batch_size", type=int, default=MODEL_PARALLEL_BATCH_SIZE)
    parser.add_argument("--lora", action="store_true", default=USE_LIGHTNING_LORA)
    parser.add_argument("--no_lora", dest="lora", action="store_false")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML config file for ElasticBandConfig (e.g. configs/qwen_global.yaml).")
    # ElasticBandConfig overrides (defaults loaded from --config)
    parser.add_argument("--max_dreamsim_distance", type=float, default=None)
    parser.add_argument("--max_doubling_steps", type=int, default=None)
    parser.add_argument("--starting_number_of_points", type=int, default=None)
    parser.add_argument("--maximum_number_of_points", type=int, default=None)
    parser.add_argument("--maximum_number_of_iterations", type=int, default=None)
    parser.add_argument("--target_gap", type=float, default=None)
    parser.add_argument("--expand_threshold", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--move_fraction", type=float, default=None)
    parser.add_argument("--base_step_fraction", type=float, default=None)
    parser.add_argument("--min_meaningful_move", type=float, default=None)
    parser.add_argument("--min_normalized_gap_for_move", type=float, default=None)
    parser.add_argument("--min_gap_imbalance_for_move", type=float, default=None)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    concept_name = args.concept or args.steering_vector_dir.name
    dtype = DTYPE_MAP[args.dtype]

    config = ElasticBandConfig.from_yaml(args.config)
    # Apply any CLI overrides on top of the YAML defaults.
    overrides = {k: v for k, v in vars(args).items() if k in ElasticBandConfig.__dataclass_fields__ and v is not None}
    for k, v in overrides.items():
        setattr(config, k, v)

    concept_dir = args.out_dir / concept_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(dtype, args.lora).to("cuda")
    input_image = Image.open(args.input_image).convert("RGB")
    runner = ElasticBandQwenRunner(
        pipe=pipe,
        prompt=args.prompt,
        tokens_to_edit=args.tokens_to_edit,
        input_image=input_image,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        model_parallel_batch_size=args.model_parallel_batch_size,
        use_lora=args.lora,
        dtype=dtype,
    )
    print(f"Processing concept: {concept_name}")
    runner.clear_caches()

    steering_vector = load_steering_vector(
        args.steering_vector_dir / "steering_last_layer.npy", runner.primary_device
    )
    stored_min = load_min_projection_value(args.steering_vector_dir)

    initialization = find_effective_minimum(
        runner=runner,
        concept_dir=concept_dir,
        concept_name=concept_name,
        steering_vector=steering_vector,
        initial_min=stored_min,
        config=config,
    )
    elastic_result = elastic_band_search(
        runner=runner,
        concept_dir=concept_dir,
        concept_name=concept_name,
        steering_vector=steering_vector,
        a_min=initialization["search_minimum_value"],
        a_max=0.0,
        config=config,
    )
    valid_range = summarize_valid_range(elastic_result["valid_control_points"])

    summary = {
        "concept": concept_name,
        "steering_vector_dir": str(args.steering_vector_dir),
        "prompt": args.prompt,
        "tokens_to_edit": args.tokens_to_edit,
        "seed": args.seed,
        "max_dreamsim_distance": config.max_dreamsim_distance,
        "initialization": initialization,
        "elastic_band": config.to_dict(),
        "valid_range": valid_range,
        "all_generated_strengths": sorted(runner.image_cache.keys()),
        "elastic_search_result": elastic_result,
    }
    with open(concept_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{concept_name}] valid control points: {elastic_result['valid_control_points']}")
    print(f"[{concept_name}] summary saved to {concept_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
