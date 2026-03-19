#!/usr/bin/env bash
# Run the Flux 2 end-to-end steering pipeline.
#
# Required env vars:
#   CONCEPT      steering concept name, e.g. "cartoon"
#   PROMPT       editing prompt, e.g. "make the scene cartoon"
#   INPUT_IMAGE  path to the input conditioning image
#
# Optional env vars:
#   ELASTIC_BAND_CONFIG  path to the elastic-band YAML config (auto-selected if unset):
#                          configs/flux2_global.yaml  — global / style edits
#                          configs/flux2_local.yaml   — local / object edits
#
# Usage:
#   CONCEPT=cartoon PROMPT="make the scene cartoon" INPUT_IMAGE=photo.png bash run_flux2.sh

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
CONCEPT="${CONCEPT:?CONCEPT is required}"
PROMPT="${PROMPT:?PROMPT is required}"
INPUT_IMAGE="${INPUT_IMAGE:?INPUT_IMAGE is required}"
ELASTIC_BAND_CONFIG="${ELASTIC_BAND_CONFIG:-}"  # optional: auto-selected from concept type when empty

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-outputs/flux2}"
NUM_OUTPUTS="${NUM_OUTPUTS:-8}"

# ── Dataset generation ────────────────────────────────────────────────────────
NUM_EXAMPLES="${NUM_EXAMPLES:-100}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"

# ── Token selection ───────────────────────────────────────────────────────────
TOKEN_MODEL="${TOKEN_MODEL:-Qwen/Qwen3-8B}"

# ── Inference ─────────────────────────────────────────────────────────────────
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-}"  # leave empty for default (2.5 w/ LoRA, 4.0 without)
DTYPE="${DTYPE:-bfloat16}"

# ── Model / compute ───────────────────────────────────────────────────────────
MODEL="${MODEL:-black-forest-labs/FLUX.2-dev}"
DEVICE="${DEVICE:-cuda}"
MAX_PAIRS="${MAX_PAIRS:--1}"

# ── Build optional args ───────────────────────────────────────────────────────
EXTRA_ARGS=()
[[ -n "${GUIDANCE_SCALE}" ]] && EXTRA_ARGS+=(--guidance_scale "${GUIDANCE_SCALE}")
[[ -n "${ELASTIC_BAND_CONFIG}" ]] && EXTRA_ARGS+=(--config "${ELASTIC_BAND_CONFIG}")

python -m models.flux2.run \
    --concept        "${CONCEPT}" \
    --prompt         "${PROMPT}" \
    --input_image    "${INPUT_IMAGE}" \
    --out_dir        "${OUT_DIR}" \
    --num_outputs    "${NUM_OUTPUTS}" \
    --num_examples   "${NUM_EXAMPLES}" \
    --openai_model   "${OPENAI_MODEL}" \
    --token_model    "${TOKEN_MODEL}" \
    --seed           "${SEED}" \
    --height         "${HEIGHT}" \
    --width          "${WIDTH}" \
    --dtype          "${DTYPE}" \
    --model          "${MODEL}" \
    --device         "${DEVICE}" \
    --max_pairs      "${MAX_PAIRS}" \
    "${EXTRA_ARGS[@]}"
