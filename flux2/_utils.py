"""Shared Flux 2 utilities used by inference, compute_vectors, and elastic_band.

All generic steering utilities (compute_difference_of_means, find_all_spans, etc.)
live in the top-level `steering` package. This module holds only the parts tightly
coupled to the Flux 2 pipeline and Mistral-3 text encoder.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

from .pipeline import SYSTEM_MESSAGE, Flux2Pipeline, format_input
from steering import DTYPE_MAP, find_all_spans, split_style_terms

MODEL_ID = "black-forest-labs/FLUX.2-dev"
LORA_REPO = "fal/FLUX.2-dev-Turbo"
LORA_WEIGHT = "flux.2-turbo-lora.safetensors"
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

TEXT_ENCODER_OUT_LAYERS = (10, 20, 30)
MAX_SEQUENCE_LENGTH = 512


# ---------------------------------------------------------------------------
# Text encoder loading
# ---------------------------------------------------------------------------


def load_flux2_text_stack(
    model_name_or_path: str,
    device: str,
    dtype_name: str,
) -> tuple[PixtralProcessor, Mistral3ForConditionalGeneration]:
    """Load the Flux 2 Mistral-3 text encoder and its PixtralProcessor."""
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype `{dtype_name}`. Choose from: {list(DTYPE_MAP.keys())}")
    dtype = DTYPE_MAP[dtype_name]
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    processor = PixtralProcessor.from_pretrained(model_name_or_path, subfolder="tokenizer")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).to(device)
    text_encoder.eval()
    return processor, text_encoder


# ---------------------------------------------------------------------------
# Prompt preparation and token alignment
# ---------------------------------------------------------------------------


def _format_prompt_text(processor: PixtralProcessor, prompt: str, system_message: str) -> str:
    rendered = processor.apply_chat_template(
        format_input(prompts=[prompt], system_message=system_message),
        add_generation_prompt=False,
        tokenize=False,
    )
    if isinstance(rendered, list):
        if len(rendered) != 1:
            raise ValueError(f"Expected a single formatted prompt, got {len(rendered)}.")
        rendered = rendered[0]
    return rendered


def _align_plain_tokens_to_inputs(
    actual_ids: Sequence[int],
    plain_ids: Sequence[int],
    valid_length: int,
) -> List[int]:
    positions: List[int] = []
    cursor = 0
    for plain_id in plain_ids:
        while cursor < valid_length and actual_ids[cursor] != plain_id:
            cursor += 1
        if cursor >= valid_length:
            raise ValueError("Failed to align plain-tokenized prompt with Flux 2 chat-template input ids.")
        positions.append(cursor)
        cursor += 1
    return positions


def prepare_flux2_text_inputs(
    processor: PixtralProcessor,
    prompt: str,
    max_sequence_length: int,
    system_message: str = SYSTEM_MESSAGE,
) -> dict:
    """Tokenize *prompt* via the Flux 2 chat template and return offset + alignment metadata."""
    messages = format_input(prompts=[prompt], system_message=system_message)
    formatted_prompt = _format_prompt_text(processor, prompt, system_message)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    tokenized = processor.tokenizer(
        formatted_prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
    )

    actual_ids = inputs["input_ids"][0].tolist()
    valid_length = int(inputs["attention_mask"][0].sum().item())
    aligned_positions = _align_plain_tokens_to_inputs(actual_ids, tokenized["input_ids"], valid_length)

    return {
        "formatted_prompt": formatted_prompt,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "plain_offsets": tokenized["offset_mapping"],
        "plain_to_input_positions": aligned_positions,
    }


def get_style_token_positions(
    formatted_prompt: str,
    user_prompt: str,
    style: str,
    plain_offsets: Sequence[tuple[int, int]],
    plain_to_input_positions: Sequence[int],
    actual_input_ids: Sequence[int],
    special_token_ids: Sequence[int],
    image_token_id: Optional[int] = None,
) -> List[int]:
    """Return the input-sequence positions of tokens overlapping the style span."""
    terms = split_style_terms(style)
    if not terms:
        return []

    user_start = formatted_prompt.rfind(user_prompt)
    if user_start < 0:
        raise ValueError("Failed to locate the user prompt inside the formatted Flux 2 chat template.")

    spans_global: List[tuple[int, int]] = [
        (user_start + start, user_start + end)
        for term in terms
        for start, end in find_all_spans(user_prompt, term)
    ]
    if not spans_global:
        return []

    special_ids = {int(tid) for tid in special_token_ids}
    matched_positions: set[int] = set()
    for plain_idx, (tok_start, tok_end) in enumerate(plain_offsets):
        if tok_start == tok_end:
            continue
        if not any(tok_start < span_end and tok_end > span_start for span_start, span_end in spans_global):
            continue
        input_pos = int(plain_to_input_positions[plain_idx])
        token_id = int(actual_input_ids[input_pos])
        if token_id in special_ids:
            continue
        if image_token_id is not None and token_id == int(image_token_id):
            continue
        matched_positions.add(input_pos)

    return sorted(matched_positions)


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------


@torch.inference_mode()
def encode_flux2_prompt_embeds(
    text_encoder: Mistral3ForConditionalGeneration,
    processor: PixtralProcessor,
    prompt: Union[str, List[str]],
    device: Union[str, torch.device],
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    system_message: str = SYSTEM_MESSAGE,
    hidden_states_layers: Sequence[int] = TEXT_ENCODER_OUT_LAYERS,
) -> torch.Tensor:
    """Encode *prompt* and return multi-layer hidden states concatenated along the hidden dim."""
    dtype = text_encoder.dtype
    prompts = [prompt] if isinstance(prompt, str) else prompt
    inputs = processor.apply_chat_template(
        format_input(prompts=prompts, system_message=system_message),
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    output = text_encoder(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        output_hidden_states=True,
        use_cache=False,
    )
    out = torch.stack([output.hidden_states[i] for i in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)
    batch_size, num_layers, seq_len, hidden_dim = out.shape
    return out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)


# ---------------------------------------------------------------------------
# Token-index lookup (used by both inference and elastic_band)
# ---------------------------------------------------------------------------


def find_indices_to_edit(
    pipe: Flux2Pipeline,
    prompt: str,
    tokens_to_edit: Sequence[str],
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    system_message: str = SYSTEM_MESSAGE,
) -> List[int]:
    """Return prompt-embedding positions for *tokens_to_edit* within *prompt*."""
    prepared = prepare_flux2_text_inputs(
        processor=pipe.tokenizer,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        system_message=system_message,
    )
    style = " ".join(
        part
        for token in tokens_to_edit
        for part in split_style_terms(token)
    )
    positions = get_style_token_positions(
        formatted_prompt=prepared["formatted_prompt"],
        user_prompt=prompt,
        style=style,
        plain_offsets=prepared["plain_offsets"],
        plain_to_input_positions=prepared["plain_to_input_positions"],
        actual_input_ids=prepared["input_ids"][0].tolist(),
        special_token_ids=getattr(pipe.tokenizer.tokenizer, "all_special_ids", []),
        image_token_id=getattr(pipe.text_encoder.config, "image_token_id", None),
    )
    if not positions:
        raise ValueError(
            f"No Flux 2 token positions matched tokens_to_edit={list(tokens_to_edit)!r} in prompt={prompt!r}"
        )
    return positions


# ---------------------------------------------------------------------------
# Steering application
# ---------------------------------------------------------------------------


def apply_steering(
    base_prompt_embeds: torch.Tensor,
    idx_to_edit: Sequence[int],
    steering_vector: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Add *factor* × *steering_vector* to *base_prompt_embeds* at *idx_to_edit* (all batch items)."""
    prompt_embeds = base_prompt_embeds.clone()
    steering_vec = steering_vector.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    if steering_vec.shape[-1] != prompt_embeds.shape[-1]:
        raise ValueError(
            f"Steering vector width {steering_vec.shape[-1]} does not match "
            f"prompt embeddings {prompt_embeds.shape[-1]}."
        )
    for batch_idx in range(prompt_embeds.shape[0]):
        for idx in idx_to_edit:
            if idx < prompt_embeds.shape[1]:
                prompt_embeds[batch_idx, idx, :] += factor * steering_vec
    return prompt_embeds


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------


def pool_positions(sequence_tensor: torch.Tensor, positions: Sequence[int]) -> Optional[torch.Tensor]:
    """Mean-pool *sequence_tensor[0]* at *positions* — strips the leading batch dim."""
    if not positions:
        return None
    return torch.stack(
        [sequence_tensor[0, idx].to(dtype=torch.float32).cpu() for idx in positions]
    ).mean(0)


def build_pipeline(
    torch_dtype: torch.dtype,
    use_lora: bool,
    use_distributed: bool,
    pipeline_device: str = "cuda",
) -> Flux2Pipeline:
    if use_distributed:
        pipe = Flux2Pipeline.from_pretrained_distributed(
            MODEL_ID,
            torch_dtype=torch_dtype,
            text_encoder_device_map="auto",
            transformer_device_map="auto",
            vae_device_map=None,
        )
    else:
        pipe = Flux2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype).to(pipeline_device)

    if use_lora:
        pipe.load_lora_weights_and_redistribute(LORA_REPO, weight_name=LORA_WEIGHT)
    return pipe
