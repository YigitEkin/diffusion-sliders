#!/usr/bin/env bash
# Run the Qwen Image Edit end-to-end steering pipeline.
#
# Required env vars:
#   CONCEPT      steering concept name, e.g. "cartoon"
#   PROMPT       editing prompt, e.g. "make the scene cartoon"
#   INPUT_IMAGE  path to the input image
#
# Optional env vars:
#   ELASTIC_BAND_CONFIG  path to the elastic-band YAML config (auto-selected if unset):
#                          configs/qwen_global.yaml  — global / style edits
#                          configs/qwen_local.yaml   — local / object edits
#
# Usage:
#   CONCEPT=cartoon PROMPT="without changing the background, scene layout and the persons identity, make the scene cartoon" INPUT_IMAGE=photo.png bash run_qwen.sh

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
CONCEPT="${CONCEPT:?CONCEPT is required}"
PROMPT="${PROMPT:?PROMPT is required}"
INPUT_IMAGE="${INPUT_IMAGE:?INPUT_IMAGE is required}"
ELASTIC_BAND_CONFIG="${ELASTIC_BAND_CONFIG:-}"  # optional: auto-selected from concept type when empty

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-outputs/qwen}"
NUM_OUTPUTS="${NUM_OUTPUTS:-8}"

# ── Dataset generation ────────────────────────────────────────────────────────
NUM_EXAMPLES="${NUM_EXAMPLES:-100}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"

# ── Token selection ───────────────────────────────────────────────────────────
TOKEN_MODEL="${TOKEN_MODEL:-Qwen/Qwen3-8B}"

# ── Inference ─────────────────────────────────────────────────────────────────
SEED="${SEED:-0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
DTYPE="${DTYPE:-bfloat16}"

# ── Model / compute ───────────────────────────────────────────────────────────
VL_MODEL="${VL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"
VL_DTYPE="${VL_DTYPE:-float32}"
MAX_PAIRS="${MAX_PAIRS:--1}"

# ── Build optional args ───────────────────────────────────────────────────────
EXTRA_ARGS=()
[[ -n "${ELASTIC_BAND_CONFIG}" ]] && EXTRA_ARGS+=(--config "${ELASTIC_BAND_CONFIG}")

python -m models.qwen.run \
    --concept        "${CONCEPT}" \
    --prompt         "${PROMPT}" \
    --input_image    "${INPUT_IMAGE}" \
    --out_dir        "${OUT_DIR}" \
    --num_outputs    "${NUM_OUTPUTS}" \
    --num_examples   "${NUM_EXAMPLES}" \
    --openai_model   "${OPENAI_MODEL}" \
    --token_model    "${TOKEN_MODEL}" \
    --seed           "${SEED}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --dtype          "${DTYPE}" \
    --vl_model       "${VL_MODEL}" \
    --device         "${DEVICE}" \
    --vl_dtype       "${VL_DTYPE}" \
    --max_pairs      "${MAX_PAIRS}" \
    "${EXTRA_ARGS[@]}"
