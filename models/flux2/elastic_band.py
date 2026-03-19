#!/usr/bin/env python3
"""Elastic-band search runner for Flux 2.

Contains only the Flux 2-specific runner class. The search algorithm lives in
`steering.elastic_band`.

Usage
-----
python -m models.flux2.elastic_band
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from dreamsim import dreamsim
from diffusers.utils import load_image
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
# Defaults
# ---------------------------------------------------------------------------

DTYPE = torch.bfloat16
USE_DISTRIBUTED = True
USE_LORA = True

HEIGHT = 1024
WIDTH = 1024
GUIDANCE_SCALE = 2.5 if USE_LORA else 4.0
SEED = 42
INFERENCE_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Flux 2-specific runner
# ---------------------------------------------------------------------------


class ElasticBandFlux2Runner:
    """Generates images at given steering strengths and caches DreamSim distances."""

    def __init__(
        self,
        pipe: "Flux2Pipeline",
        prompt: str,
        tokens_to_edit: list[str],
        condition_image: "Image.Image",
        seed: int = SEED,
        batch_size: int = INFERENCE_BATCH_SIZE,
        use_lora: bool = USE_LORA,
        height: int = HEIGHT,
        width: int = WIDTH,
        guidance_scale: float = GUIDANCE_SCALE,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        text_encoder_out_layers: tuple = TEXT_ENCODER_OUT_LAYERS,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for elastic band search.")

        self.visible_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.primary_device = torch.device(self.visible_devices[0])
        self.dreamsim_device = torch.device(self.visible_devices[-1])
        self.batch_size = batch_size
        self.seed = seed
        self.use_lora = use_lora
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.max_sequence_length = max_sequence_length
        self.text_encoder_out_layers = text_encoder_out_layers

        self.pipe = pipe
        self.pipe.set_progress_bar_config(disable=None)
        self.text_encoder_device = self.pipe._get_module_execution_device(self.pipe.text_encoder)

        self.condition_image = condition_image
        self.idx_to_edit = find_indices_to_edit(
            pipe=self.pipe,
            prompt=prompt,
            tokens_to_edit=tokens_to_edit,
            max_sequence_length=max_sequence_length,
        )
        self.base_prompt_embeds = encode_flux2_prompt_embeds(
            text_encoder=self.pipe.text_encoder,
            processor=self.pipe.tokenizer,
            prompt=prompt,
            device=self.text_encoder_device,
            max_sequence_length=max_sequence_length,
            system_message=self.pipe.system_message,
            hidden_states_layers=text_encoder_out_layers,
        )

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
            prompt_embeds = torch.cat(
                [
                    apply_steering(self.base_prompt_embeds, self.idx_to_edit, steering_vector, s)
                    for s in chunk
                ],
                dim=0,
            )
            generators = [torch.Generator(device="cuda").manual_seed(self.seed) for _ in chunk]
            with torch.inference_mode():
                output = self.pipe(
                    image=self.condition_image,
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    sigmas=TURBO_SIGMAS if self.use_lora else None,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=8 if self.use_lora else 50,
                    guidance_scale=self.guidance_scale,
                    max_sequence_length=self.max_sequence_length,
                    text_encoder_out_layers=self.text_encoder_out_layers,
                    generator=generators,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flux 2 elastic-band search (standalone).")
    parser.add_argument("--input_image", required=True, help="Path to the input conditioning image.")
    parser.add_argument("--prompt", required=True, help="Editing prompt.")
    parser.add_argument("--tokens_to_edit", type=str, nargs="+", required=True,
                        help="Token strings to steer (e.g. cartoon).")
    parser.add_argument("--steering_vector_dir", type=Path, required=True,
                        help="Directory containing steering_last_layer.npy and projection values.")
    parser.add_argument("--out_dir", type=Path, default=Path("elastic_band_outputs"),
                        help="Root output directory (default: elastic_band_outputs).")
    parser.add_argument("--concept", type=str, default=None,
                        help="Concept name for output subdirectory. Defaults to steering_vector_dir.name.")
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Defaults to 2.5 with LoRA, 4.0 without.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=INFERENCE_BATCH_SIZE)
    parser.add_argument("--lora", action="store_true", default=USE_LORA)
    parser.add_argument("--no_lora", dest="lora", action="store_false")
    parser.add_argument("--distributed", action="store_true", default=USE_DISTRIBUTED)
    parser.add_argument("--no_distributed", dest="distributed", action="store_false")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--max_sequence_length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--text_encoder_out_layers", type=int, nargs="+",
                        default=list(TEXT_ENCODER_OUT_LAYERS))
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML config file for ElasticBandConfig (e.g. configs/flux2_global.yaml).")
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

    guidance_scale = args.guidance_scale if args.guidance_scale is not None else (2.5 if args.lora else 4.0)
    concept_name = args.concept or args.steering_vector_dir.name
    dtype = DTYPE_MAP[args.dtype]

    config = ElasticBandConfig.from_yaml(args.config)
    # Apply any CLI overrides on top of the YAML defaults.
    overrides = {k: v for k, v in vars(args).items() if k in ElasticBandConfig.__dataclass_fields__ and v is not None}
    for k, v in overrides.items():
        setattr(config, k, v)

    concept_dir = args.out_dir / concept_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(dtype, args.lora, args.distributed)
    condition_image = load_image(args.input_image).convert("RGB")
    runner = ElasticBandFlux2Runner(
        pipe=pipe,
        prompt=args.prompt,
        tokens_to_edit=args.tokens_to_edit,
        condition_image=condition_image,
        seed=args.seed,
        batch_size=args.batch_size,
        use_lora=args.lora,
        height=args.height,
        width=args.width,
        guidance_scale=guidance_scale,
        max_sequence_length=args.max_sequence_length,
        text_encoder_out_layers=tuple(args.text_encoder_out_layers),
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
