"""Shared Qwen-specific utilities used by both inference and elastic-band search.

All generic steering utilities (compute_difference_of_means, find_all_spans, etc.)
live in the top-level `steering` package. This module holds only the parts that
are tightly coupled to the QwenImageEdit pipeline and transformer architecture.
"""

from __future__ import annotations

import math
import re
from types import MethodType
from typing import Optional

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import QwenImageTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask
from diffusers.utils import deprecate
from PIL import Image

from .pipeline import CONDITION_IMAGE_SIZE, QwenImageEditPlusPipeline, VAE_IMAGE_SIZE, calculate_dimensions

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHT_LOCAL = "../../ckpts/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LORA_WEIGHT_HF = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------


def resolve_inference_config(use_lora: bool) -> dict:
    return {
        "negative_prompt": " ",
        "true_cfg_scale": 1.0 if use_lora else 4.0,
        "num_inference_steps": 8 if use_lora else 50,
    }


def build_pipeline(torch_dtype: torch.dtype, use_lora: bool) -> QwenImageEditPlusPipeline:
    if use_lora:
        model = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            transformer=model,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
        )
        import os
        if os.path.isfile(LORA_WEIGHT_LOCAL):
            pipe.load_lora_weights(LORA_WEIGHT_LOCAL)
        else:
            pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT_HF)
        pipe.fuse_lora()
        return pipe

    return QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)


# ---------------------------------------------------------------------------
# Token-index alignment (Qwen-specific: handles VLM image-token expansion)
# ---------------------------------------------------------------------------


def find_edit_indices(
    pipe: QwenImageEditPlusPipeline,
    prompt: str,
    condition_image: Image.Image,
    tokens_to_edit: list[str],
) -> list[int]:
    """Map *tokens_to_edit* in *prompt* to their indices in the encoder's embedding sequence.

    Accounts for the VLM image-token expansion (`<|image_pad|>` → many tokens) and
    the prompt template prefix dropped before the text encoder.
    """
    from steering import find_all_spans

    image_prefix = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    user_content = image_prefix + prompt
    templated_prompt = pipe.prompt_template_encode.format(user_content)

    user_start = templated_prompt.find(user_content)
    if user_start == -1:
        raise ValueError("Failed to locate user content inside the templated prompt.")
    user_prompt_start = user_start + len(image_prefix)

    target_terms = [
        part
        for token in tokens_to_edit
        for part in re.split(r"[_\s]+", token.strip())
        if part
    ]
    style_spans_global = [
        (user_prompt_start + start, user_prompt_start + end)
        for term in target_terms
        for start, end in find_all_spans(prompt, term)
    ]
    if not style_spans_global:
        raise ValueError(f"No edit spans found for tokens_to_edit={tokens_to_edit!r}")

    text_only = pipe.tokenizer(
        templated_prompt,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    unexpanded_ids = text_only["input_ids"]
    offsets = text_only["offset_mapping"]
    unexpanded_indices = [
        i
        for i, (tok_start, tok_end) in enumerate(offsets)
        if tok_start != tok_end
        and any(tok_start < span_end and tok_end > span_start for span_start, span_end in style_spans_global)
    ]
    if not unexpanded_indices:
        raise ValueError(
            f"No token indices matched tokens_to_edit={tokens_to_edit!r} in prompt={prompt!r}"
        )

    model_inputs = pipe.processor(
        text=[templated_prompt],
        images=[condition_image],
        padding=True,
        return_tensors="pt",
    )
    expanded_ids = model_inputs.input_ids[0].tolist()
    image_token_id = pipe.text_encoder.config.image_token_id
    placeholder_positions = [i for i, tid in enumerate(unexpanded_ids) if tid == image_token_id]

    image_grid_thw = model_inputs.get("image_grid_thw", None)
    if image_grid_thw is not None:
        spatial_merge = pipe.text_encoder.model.visual.spatial_merge_size
        expansion_lengths = [int(v) for v in (image_grid_thw.prod(-1) // (spatial_merge ** 2)).tolist()]
    else:
        expansion_lengths = []
    if not expansion_lengths:
        expansion_lengths = [1] * len(placeholder_positions)
    if len(expansion_lengths) != len(placeholder_positions):
        raise ValueError(
            f"Placeholder/expansion mismatch: {len(placeholder_positions)} placeholders vs "
            f"{len(expansion_lengths)} expansions."
        )

    exp_by_pos = {pos: max(length, 1) - 1 for pos, length in zip(placeholder_positions, expansion_lengths)}
    cumulative_delta = [0] * (len(unexpanded_ids) + 1)
    running = 0
    for i in range(len(unexpanded_ids)):
        cumulative_delta[i] = running
        running += exp_by_pos.get(i, 0)
    cumulative_delta[len(unexpanded_ids)] = running

    expanded_indices = [idx + cumulative_delta[idx] for idx in unexpanded_indices]
    for idx in expanded_indices:
        if idx >= len(expanded_ids):
            raise ValueError(f"Expanded index {idx} out of bounds for sequence length {len(expanded_ids)}.")

    valid_positions = torch.nonzero(model_inputs.attention_mask[0], as_tuple=False).squeeze(1).tolist()
    pos_to_rank = {pos: rank for rank, pos in enumerate(valid_positions)}
    drop_idx = pipe.prompt_template_encode_start_idx

    idx_to_edit = sorted(set(
        rank - drop_idx
        for pos in expanded_indices
        if (rank := pos_to_rank.get(pos)) is not None and rank - drop_idx >= 0
    ))
    if not idx_to_edit:
        raise ValueError(
            f"Matched tokens were removed by drop_idx={drop_idx}. "
            f"tokens_to_edit={tokens_to_edit!r}, prompt={prompt!r}"
        )

    valid_ids = model_inputs.input_ids[0][model_inputs.attention_mask[0].bool()]
    embed_tokens = pipe.tokenizer.convert_ids_to_tokens(valid_ids[drop_idx:].tolist())
    selected = [embed_tokens[i] for i in idx_to_edit if i < len(embed_tokens)]
    print("Editing token indices:", idx_to_edit)
    print("Editing tokens:", selected)
    return idx_to_edit


# ---------------------------------------------------------------------------
# Steering application
# ---------------------------------------------------------------------------


def apply_steering(
    base_prompt_embeds: torch.Tensor,
    idx_to_edit: list[int],
    steering_vector: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Add *factor* × *steering_vector* to the prompt embeddings at *idx_to_edit*."""
    prompt_embeds = base_prompt_embeds.clone()
    steering_vec = steering_vector.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    for idx in idx_to_edit:
        if idx < prompt_embeds.shape[1]:
            prompt_embeds[0, idx, :] += factor * steering_vec
    return prompt_embeds


def ensure_prompt_mask(
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if prompt_embeds_mask is not None:
        return prompt_embeds_mask
    return torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device)


def repeat_image_latents(image_latents: torch.Tensor, batch_size: int) -> torch.Tensor:
    if batch_size == 1:
        return image_latents
    return image_latents.repeat(*([batch_size] + [1] * (image_latents.ndim - 1)))


def repeat_prompt_mask(
    prompt_embeds_mask: Optional[torch.Tensor], batch_size: int
) -> Optional[torch.Tensor]:
    if prompt_embeds_mask is None:
        return None
    return prompt_embeds_mask.repeat(batch_size, 1)


# ---------------------------------------------------------------------------
# Multi-GPU setup
# ---------------------------------------------------------------------------


def move_data_to_device(data, device: torch.device):
    """Recursively move tensors (or nested structures of tensors) to *device*."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, tuple):
        return tuple(move_data_to_device(item, device) for item in data)
    if isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    if isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    return data


def assign_modules_for_multi_gpu(pipe: QwenImageEditPlusPipeline, devices: list[str]) -> None:
    if len(devices) <= 1:
        return
    text_encoder_device = torch.device(devices[min(1, len(devices) - 1)])
    vae_device = torch.device(devices[min(2, len(devices) - 1)])
    pipe.text_encoder.to(text_encoder_device)
    pipe.vae.to(vae_device)
    pipe._text_encoder_device = text_encoder_device
    pipe._vae_device = vae_device


def build_uneven_block_device_map(
    block_count: int, devices: list[torch.device]
) -> list[torch.device]:
    if len(devices) <= 1:
        return [devices[0]] * block_count
    if len(devices) == 2:
        first_share = max(1, block_count // 3)
        counts = [first_share, block_count - first_share]
    else:
        first_share = max(1, block_count // (2 * len(devices)))
        remaining = block_count - first_share
        base = remaining // (len(devices) - 1)
        extra = remaining % (len(devices) - 1)
        counts = [first_share] + [base + (1 if i < extra else 0) for i in range(len(devices) - 1)]
    block_devices: list[torch.device] = []
    for device, count in zip(devices, counts):
        block_devices.extend([device] * count)
    return block_devices[:block_count]


def enable_transformer_layerwise_model_parallel(
    transformer: QwenImageTransformer2DModel, devices: list[str]
) -> None:
    """Distribute transformer blocks across *devices* and monkey-patch the forward pass."""
    if len(devices) <= 1:
        return

    device_objs = [torch.device(d) for d in devices]
    block_count = len(transformer.transformer_blocks)
    block_device_map = build_uneven_block_device_map(block_count, device_objs)
    for block, device in zip(transformer.transformer_blocks, block_device_map):
        block.to(device)

    transformer._model_parallel_devices = device_objs
    transformer._model_parallel_block_devices = block_device_map
    for attr in ("img_in", "txt_norm", "txt_in", "time_text_embed", "pos_embed", "norm_out", "proj_out"):
        getattr(transformer, attr).to(device_objs[0])

    def model_parallel_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor = None,
        attention_kwargs: dict | None = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ):
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated. Use `encoder_hidden_states_mask` instead.",
                standard_warn=False,
            )

        first_device = self._model_parallel_devices[0]
        hidden_states = hidden_states.to(first_device)
        encoder_hidden_states = encoder_hidden_states.to(first_device)
        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(first_device)
        timestep = timestep.to(device=first_device, dtype=hidden_states.dtype)
        hidden_states = self.img_in(hidden_states)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * math.prod(s[0]) + [1] * sum(math.prod(s2) for s2 in s[1:]) for s in img_shapes],
                device=first_device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )
        if guidance is not None:
            guidance = guidance.to(device=first_device, dtype=hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )
        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=first_device)
        base_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}

        for index_block, block in enumerate(self.transformer_blocks):
            block_device = self._model_parallel_block_devices[index_block]
            hidden_states = hidden_states.to(block_device)
            encoder_hidden_states = encoder_hidden_states.to(block_device)
            block_temb = move_data_to_device(temb, block_device)
            block_image_rotary_emb = move_data_to_device(image_rotary_emb, block_device)
            block_modulate_index = move_data_to_device(modulate_index, block_device)
            block_attention_kwargs = base_attention_kwargs.copy()
            if encoder_hidden_states_mask is not None:
                block_mask = encoder_hidden_states_mask.to(block_device)
                batch_size, image_seq_len = hidden_states.shape[:2]
                image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=block_device)
                block_attention_kwargs["attention_mask"] = torch.cat([block_mask, image_mask], dim=1)
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=block_temb,
                image_rotary_emb=block_image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=block_modulate_index,
            )
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control].to(block_device)

        hidden_states = hidden_states.to(first_device)
        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states).to(first_device)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    transformer.forward = MethodType(model_parallel_forward, transformer)
