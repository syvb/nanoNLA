#!/bin/bash
# Shared config for the all-LoRA, Miles-free length-penalty sweep on nanoNLA.
# Source at the top of every stage: `source "$(dirname "$0")/env.sh"`.
#
# Whole pipeline needs only torch+transformers+peft+pyarrow + the nla package.
# No Miles, no SGLang, no flash-attn, no conda, no DCP conversion. Override any
# default by exporting before a stage.

set -euo pipefail

# ---- where everything lives (NOT /tmp — survives across stages) ----
export NLA_WORKSPACE="${NLA_WORKSPACE:-/workspace/nla}"
export NLA_REPO="${NLA_REPO:-$NLA_WORKSPACE/nanoNLA}"
export DATA="${DATA:-$NLA_WORKSPACE/data}"
export CKPTS="${CKPTS:-$NLA_WORKSPACE/ckpts}"
export RESULTS="${RESULTS:-$NLA_REPO/experiment_results}"
export HF_HOME="${HF_HOME:-$NLA_WORKSPACE/hf}"

# ---- model / NLA geometry (Qwen3-8B, layer 24 — nanoNLA default) ----
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export LAYER="${LAYER:-24}"
export HF_DATASET="${HF_DATASET:-ceselder/qwen3-8b-nla-L24-finefineweb-100k}"

# ---- warm-start data ----
# The published dataset is SLIM (no activation_vector — it's regenerated from
# detokenized_text_truncated with one forward pass). 01_fetch downloads the slim
# parquets; 01b_regen produces the _full parquets (activation_vector re-added),
# which is what SFT/RL/eval actually train on.
export AV_SLIM="${AV_SLIM:-$DATA/av_sft_shuf.parquet}"
export AR_SLIM="${AR_SLIM:-$DATA/ar_sft_shuf.parquet}"
export RL_SLIM="${RL_SLIM:-$DATA/rl_shuf.parquet}"
export AV_SFT_PARQUET="${AV_SFT_PARQUET:-$DATA/av_sft_full.parquet}"
export AR_SFT_PARQUET="${AR_SFT_PARQUET:-$DATA/ar_sft_full.parquet}"
export RL_PARQUET="${RL_PARQUET:-$DATA/rl_full.parquet}"
export VAL_PARQUET="${VAL_PARQUET:-$RL_PARQUET}"
# how many rows to regenerate per split (bounds the GPU cost; only what we train/eval on)
export REGEN_AV_ROWS="${REGEN_AV_ROWS:-50000}"
export REGEN_AR_ROWS="${REGEN_AR_ROWS:-50000}"
export REGEN_RL_ROWS="${REGEN_RL_ROWS:-30000}"   # covers RL train (0:20k) + eval (25k:26k)
export REGEN_BATCH="${REGEN_BATCH:-32}"

# ---- checkpoints (self-contained SFT saves merged HF directly — AV_HF/AR_HF ARE these dirs) ----
export AV_HF="${AV_HF:-$CKPTS/av_sft}"          # merged base+LoRA causal LM
export AR_HF="${AR_HF:-$CKPTS/ar_sft}"          # merged NLACriticModel
export CRITIC_INIT="${CRITIC_INIT:-$CKPTS/critic_init}"
export RL_BASE="${RL_BASE:-$CKPTS/rl}"          # per-penalty RL LoRA dirs hang off this

# ---- warm-start SFT (LoRA) ----
export SFT_STEPS="${SFT_STEPS:-1000}"
export SFT_AV_MICRO="${SFT_AV_MICRO:-8}"        # AV forward batch (8B, bf16, LoRA)
export SFT_AR_MICRO="${SFT_AR_MICRO:-16}"       # AR forward batch (5.9B truncated)
export SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"
export SFT_LR="${SFT_LR:-2e-5}"
export SFT_LORA_R="${SFT_LORA_R:-64}"           # higher rank: LoRA must learn injection-reading
export SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-128}"

# ---- RL (the validated "overnight" config) ----
export RL_NUM_STEPS="${RL_NUM_STEPS:-250}"
export RL_BATCH_PROMPTS="${RL_BATCH_PROMPTS:-8}"
export RL_GROUP_SIZE="${RL_GROUP_SIZE:-4}"
export RL_MAX_NEW="${RL_MAX_NEW:-160}"
export RL_LR="${RL_LR:-1e-5}"   # 1e-6 was too gentle to move the policy; 1e-5 is stable + effective
export RL_KL_BETA="${RL_KL_BETA:-0.01}"         # KL vs frozen base (LoRA off) = anti-hack anchor
export RL_LORA_R="${RL_LORA_R:-16}"
export RL_LORA_ALPHA="${RL_LORA_ALPHA:-32}"
export RL_MAX_ROWS="${RL_MAX_ROWS:-20000}"

# ---- the sweep (penalties, in the requested order) ----
# 0.0 = control RL (isolates the penalty's effect from RL itself); 0.03 = aggressive end.
export PENALTIES="${PENALTIES:-0.006 0.002 0.001 0.015 0.0 0.03}"

# ---- held-out eval ----
export EVAL_N="${EVAL_N:-1000}"
export EVAL_SKIP_ROWS="${EVAL_SKIP_ROWS:-25000}"   # doc-disjoint from the RL cursor
export EVAL_MAX_NEW="${EVAL_MAX_NEW:-$RL_MAX_NEW}"  # match the RL cap so lengths compare
export EVAL_TEMP="${EVAL_TEMP:-1.0}"

# ---- sweep parallelism: how many GPUs to fan the RL runs across ----
export SWEEP_GPUS="${SWEEP_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "${SWEEP_GPUS:-0}" -ge 1 ] 2>/dev/null || export SWEEP_GPUS=1

# ---- wandb / HF ----
export WANDB_PROJECT="${WANDB_PROJECT:-nla-lenpen}"
export HF_OWNER="${HF_OWNER:-syvb}"
export HF_PREFIX="${HF_PREFIX:-nanonla-qwen3-8b-L24}"   # uploaded repos: $HF_OWNER/$HF_PREFIX-{av,ar,rl-pXXX}

mkdir -p "$NLA_WORKSPACE" "$DATA" "$CKPTS" "$RESULTS" "$HF_HOME"

penalty_slug() { echo "p${1}"; }

echo "[env] WORKSPACE=$NLA_WORKSPACE MODEL=$MODEL L$LAYER GPUS=$SWEEP_GPUS PENALTIES=[$PENALTIES]" >&2
