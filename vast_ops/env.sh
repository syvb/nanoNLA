#!/bin/bash
# Shared config for the from-scratch length-penalty sweep on nanoNLA.
# Source this at the top of every stage script: `source "$(dirname "$0")/env.sh"`.
#
# Everything is parameterised through env vars with sane defaults so the same
# scripts run on a vast.ai box (persistent /workspace disk) or anywhere else.
# Override any of these by exporting before you call a stage, e.g.
#   NLA_WORKSPACE=/data/nla  PENALTIES="0.0 0.002"  bash vast_ops/05_rl_sweep.sh

set -euo pipefail

# ---- where everything lives (NOT /tmp — survives across stages) ----
export NLA_WORKSPACE="${NLA_WORKSPACE:-/workspace/nla}"
export NLA_REPO="${NLA_REPO:-$NLA_WORKSPACE/nanoNLA}"     # this repo, cloned on the box
export MILES_DIR="${MILES_DIR:-$NLA_WORKSPACE/miles}"      # radixark/miles checkout
export DATA="${DATA:-$NLA_WORKSPACE/data}"                 # warm-start parquets
export CKPTS="${CKPTS:-$NLA_WORKSPACE/ckpts}"              # SFT + RL checkpoints
export RESULTS="${RESULTS:-$NLA_REPO/experiment_results}" # persisted artifacts (committed)
export HF_HOME="${HF_HOME:-$NLA_WORKSPACE/hf}"             # model cache

# ---- model / NLA geometry (Qwen3-8B, layer 24 — nanoNLA default) ----
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export LAYER="${LAYER:-24}"
export CRITIC_NUM_LAYERS="${CRITIC_NUM_LAYERS:-25}"        # LAYER + 1 (truncated backbone)
export HF_DATASET="${HF_DATASET:-ceselder/qwen3-8b-nla-L24-finefineweb-100k}"

# ---- checkpoint name scheme ----
export AV_SFT_DIR="${AV_SFT_DIR:-$CKPTS/av_sft}"
export AR_SFT_DIR="${AR_SFT_DIR:-$CKPTS/ar_sft}"
export CRITIC_INIT="${CRITIC_INIT:-$CKPTS/critic_init}"
export RL_BASE="${RL_BASE:-$CKPTS/rl}"                     # per-penalty RL dirs hang off this

# HF-converted checkpoints the RL trainer + eval consume:
export AV_HF="${AV_HF:-$AV_SFT_DIR/hf}"
export AR_HF="${AR_HF:-$AR_SFT_DIR/hf}"

# ---- warm-start parquet names inside $DATA (the HF dataset's actual files) ----
# Each parquet has a sibling <name>.nla_meta.yaml sidecar (the NLA contract).
export AV_SFT_PARQUET="${AV_SFT_PARQUET:-$DATA/av_sft_shuf.parquet}"
export AR_SFT_PARQUET="${AR_SFT_PARQUET:-$DATA/ar_sft_shuf.parquet}"
export RL_PARQUET="${RL_PARQUET:-$DATA/rl_shuf.parquet}"
export VAL_PARQUET="${VAL_PARQUET:-$RL_PARQUET}"           # held-out eval reads past the RL cursor

# ---- training knobs (the validated "overnight" RL config; see qwen3_8b_run.md) ----
export AV_GPUS="${AV_GPUS:-2}"                             # SFT actor GPUs
export AR_GPUS="${AR_GPUS:-2}"
export RL_NUM_STEPS="${RL_NUM_STEPS:-250}"
export RL_BATCH_PROMPTS="${RL_BATCH_PROMPTS:-8}"
export RL_GROUP_SIZE="${RL_GROUP_SIZE:-4}"
export RL_MAX_NEW="${RL_MAX_NEW:-160}"
export RL_LR="${RL_LR:-1e-6}"
export RL_KL_BETA="${RL_KL_BETA:-0.01}"                    # KL vs frozen base (LoRA off) = the anti-hack anchor
export RL_LORA_R="${RL_LORA_R:-16}"
export RL_LORA_ALPHA="${RL_LORA_ALPHA:-32}"
export RL_MAX_ROWS="${RL_MAX_ROWS:-20000}"                # training cursor; eval reads past it

# ---- the sweep: penalties to try (0.0 = control RL, no penalty) ----
# base NLA (AV-SFT only, no RL) is evaluated separately for free in the sweep.
export PENALTIES="${PENALTIES:-0.0 0.001 0.002 0.004 0.006}"

# ---- held-out eval ----
export EVAL_N="${EVAL_N:-1000}"                           # samples per model to persist
export EVAL_SKIP_ROWS="${EVAL_SKIP_ROWS:-25000}"          # doc-disjoint from the RL training cursor
export EVAL_MAX_NEW="${EVAL_MAX_NEW:-150}"
export EVAL_TEMP="${EVAL_TEMP:-1.0}"

mkdir -p "$NLA_WORKSPACE" "$DATA" "$CKPTS" "$RESULTS" "$HF_HOME"

# Tag a penalty value into a filesystem-safe slug: 0.002 -> p0.002, 0.0 -> p0.0
penalty_slug() { echo "p${1}"; }

echo "[env] WORKSPACE=$NLA_WORKSPACE  MODEL=$MODEL L$LAYER  PENALTIES=[$PENALTIES]" >&2
