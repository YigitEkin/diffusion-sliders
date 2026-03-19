#!/usr/bin/env bash
# Run the Wan text-to-video end-to-end steering pipeline.
#
# Required env vars:
#   CONCEPT   steering concept name, e.g. "boxing"
#   PROMPT    generation prompt, e.g. "Two anthropomorphic cats boxing on a stage."
#
# Usage:
#   CONCEPT=beard PROMPT="Two anthropomorphic cats boxing on a stage." bash run_wan.sh

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
CONCEPT="${CONCEPT:?CONCEPT is required}"
PROMPT="${PROMPT:?PROMPT is required}"

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-outputs/wan}"
NUM_OUTPUTS="${NUM_OUTPUTS:-5}"

# ── Dataset generation ────────────────────────────────────────────────────────
NUM_EXAMPLES="${NUM_EXAMPLES:-100}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"

# ── Token selection ───────────────────────────────────────────────────────────
TOKEN_MODEL="${TOKEN_MODEL:-Qwen/Qwen3-8B}"

# ── Inference ─────────────────────────────────────────────────────────────────
SEED="${SEED:-42}"
STRENGTH_MIN="${STRENGTH_MIN:-0.0}"
STRENGTH_MAX="${STRENGTH_MAX:-5.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
FPS="${FPS:-16}"
SCHEDULE_TYPE="${SCHEDULE_TYPE:-constant}"

# ── Model / compute ───────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
ENCODER_DTYPE="${ENCODER_DTYPE:-float32}"
MAX_PAIRS="${MAX_PAIRS:--1}"

# ── Credentials ───────────────────────────────────────────────────────────────
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"

# ── Build optional args ───────────────────────────────────────────────────────
EXTRA_ARGS=()
[[ -n "${API_KEY}" ]] && EXTRA_ARGS+=(--api_key "${API_KEY}")

python -m models.wan.run \
    --concept              "${CONCEPT}" \
    --prompt               "${PROMPT}" \
    --out_dir              "${OUT_DIR}" \
    --num_outputs          "${NUM_OUTPUTS}" \
    --num_examples         "${NUM_EXAMPLES}" \
    --openai_model         "${OPENAI_MODEL}" \
    --token_model          "${TOKEN_MODEL}" \
    --seed                 "${SEED}" \
    --strength_min         "${STRENGTH_MIN}" \
    --strength_max         "${STRENGTH_MAX}" \
    --guidance_scale       "${GUIDANCE_SCALE}" \
    --height               "${HEIGHT}" \
    --width                "${WIDTH}" \
    --num_frames           "${NUM_FRAMES}" \
    --num_inference_steps  "${NUM_INFERENCE_STEPS}" \
    --flow_shift           "${FLOW_SHIFT}" \
    --fps                  "${FPS}" \
    --schedule_type        "${SCHEDULE_TYPE}" \
    --model_id             "${MODEL_ID}" \
    --device               "${DEVICE}" \
    --dtype                "${DTYPE}" \
    --encoder_dtype        "${ENCODER_DTYPE}" \
    --max_pairs            "${MAX_PAIRS}" \
    "${EXTRA_ARGS[@]}"
