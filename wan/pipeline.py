"""Wan pipeline builder and steering callback."""

from __future__ import annotations

import html
import re
from typing import List, Literal

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)


def build_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    flow_shift: float = 5.0,
) -> WanPipeline:
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch_dtype)
    pipe.scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift,
    )
    pipe.to(device)
    return pipe


def _prompt_clean(text: str) -> str:
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_piece(piece: str) -> str:
    return (piece or "").strip().lstrip("▁").lower()


def _expand_terms(terms: list[str]) -> set[str]:
    expanded: set[str] = set()
    for term in terms:
        for part in re.split(r"[_\s]+", term.strip()):
            normalized = _normalize_piece(part)
            if normalized:
                expanded.add(normalized)
    return expanded


def _subsequence_matches(sequence: List[str], pattern: List[str]) -> List[int]:
    if not sequence or not pattern or len(pattern) > len(sequence):
        return []
    matched: List[int] = []
    m = len(pattern)
    for start in range(len(sequence) - m + 1):
        if sequence[start : start + m] == pattern:
            matched.extend(range(start, start + m))
    return matched


class WanSteeringCallback:
    """Callback that applies a steering vector to prompt embeddings at each denoising step."""

    def __init__(
        self,
        pipeline: WanPipeline,
        prompt: str,
        tokens_to_edit: List[str],
        steering_vector: torch.Tensor,
        factor: float,
        schedule_type: Literal["linear", "constant"] = "constant",
    ):
        self.pipeline = pipeline
        self.prompt = prompt
        self.tokens_to_edit = tokens_to_edit
        self.steering_vector = steering_vector
        self.factor = factor
        self.schedule_type = schedule_type
        self._idx_to_edit: List[int] | None = None
        self._original_prompt_embeds: torch.Tensor | None = None

    def _get_token_indices(self) -> List[int]:
        token_pieces = self.pipeline.tokenizer.tokenize(_prompt_clean(self.prompt))
        normalized_tokens = [_normalize_piece(tok) for tok in token_pieces]
        terms = _expand_terms(self.tokens_to_edit)

        indices: List[int] = []
        for raw_term in self.tokens_to_edit:
            term_clean = _prompt_clean(raw_term)
            if not term_clean:
                continue
            term_pieces = self.pipeline.tokenizer.tokenize(term_clean)
            term_norm = [_normalize_piece(tok) for tok in term_pieces if _normalize_piece(tok)]
            indices.extend(_subsequence_matches(normalized_tokens, term_norm))

        if not indices:
            indices = [i for i, tok in enumerate(normalized_tokens) if tok in terms]

        indices = sorted(set(indices))
        if not indices:
            raise ValueError(
                f"No tokenizer pieces matched tokens_to_edit={self.tokens_to_edit}. "
                f"Prompt pieces: {normalized_tokens}"
            )
        return indices

    def _get_step_coefficient(self, step_index: int, total_timesteps: int) -> float:
        if self.schedule_type == "linear":
            return (step_index / max(total_timesteps, 1)) * self.factor
        return self.factor

    @torch.inference_mode()
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        if self._original_prompt_embeds is None:
            self._original_prompt_embeds = callback_kwargs["prompt_embeds"].clone()
        prompt_embeds = self._original_prompt_embeds.clone()
        if self._idx_to_edit is None:
            self._idx_to_edit = self._get_token_indices()

        if prompt_embeds.ndim != 3:
            raise ValueError(f"Expected prompt_embeds [B, T, C], got {tuple(prompt_embeds.shape)}")

        step_coeff = self._get_step_coefficient(
            step_index, int(getattr(pipe, "num_timesteps", step_index + 1))
        )
        steering = self.steering_vector.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        for batch_idx in range(prompt_embeds.shape[0]):
            for token_idx in self._idx_to_edit:
                if token_idx < prompt_embeds.shape[1]:
                    prompt_embeds[batch_idx, token_idx, :] += step_coeff * steering

        callback_kwargs["prompt_embeds"] = prompt_embeds
        return callback_kwargs
