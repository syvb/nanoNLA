#!/bin/bash
# Stage 2 — AV (Activation Verbalizer) warm-start SFT.
# Karvonen ADD norm-matched injection at layer 1 (the key knob). Trains the
# actor to verbalise an injected activation. ~0.5-1.0h on 2 GPUs (1000 steps,
# batch 32). Wraps configs/actor_sft.sh (Miles, --debug-train-only: no SGLang).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

[ -f "$HOME/.hf_token" ]  && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
[ -f "$HOME/.wandb_key" ] && export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < "$HOME/.wandb_key" | sed 's/^WANDB_API_KEY=//')}"
export WANDB_PROJECT="${WANDB_PROJECT:-nla-lenpen}"
export WANDB_NAME=av_sft
export PYTHONUNBUFFERED=1

# THE KEY KNOB: norm-matched additive injection at residual after layer 1.
export NLA_KARVONEN_INJECTION=1
export INSTRUCT_MODEL="$MODEL"
export AV_SFT_PARQUET="$AV_SFT_PARQUET"
export INJ_SCALE=raw                       # ignored under Karvonen mode, but actor_sft.sh requires it set
export SAVE_DIR="$AV_SFT_DIR"
mkdir -p "$SAVE_DIR"

echo "=== AV SFT START ($(date)) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

cd "$(python -c 'import miles, os; print(os.path.dirname(miles.__file__))')/.."

bash "$NLA_REPO/configs/actor_sft.sh" \
  --actor-num-gpus-per-node "$AV_GPUS" \
  --attn-implementation sdpa \
  --rollout-batch-size 32 --global-batch-size 32 --micro-batch-size 4 \
  --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
  --num-rollout 1000 --save-interval 500 \
  ${WANDB_API_KEY:+--use-wandb --wandb-project "$WANDB_PROJECT"}

echo "=== AV SFT END ($(date)) ==="
ls -d "$SAVE_DIR"/iter_* 2>/dev/null | tail -1 || true
