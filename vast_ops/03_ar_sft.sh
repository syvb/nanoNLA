#!/bin/bash
# Stage 3 — AR (Activation Reconstructor / critic) warm-start SFT.
# (a) build the truncated K+1-layer init (backbone + Linear(d,d) head), then
# (b) SFT it to predict the gold activation from the AV's explanation text.
# NLA_FREEZE_VALUE_HEAD=1 is required for stability (without it AR NaN'd by
# step ~29 — see docs/qwen3_8b_run.md). ~1.0-1.5h on 2 GPUs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

[ -f "$HOME/.hf_token" ]  && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
[ -f "$HOME/.wandb_key" ] && export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < "$HOME/.wandb_key" | sed 's/^WANDB_API_KEY=//')}"
export WANDB_PROJECT="${WANDB_PROJECT:-nla-lenpen}"
export PYTHONUNBUFFERED=1

# ---- (a) prepare critic init (one-time, ~10 min, 1 GPU) ----
if [ ! -f "$CRITIC_INIT/config.json" ]; then
  echo "=== prepare_critic_checkpoint ($(date)) ==="
  cd "$NLA_REPO"
  python -m nla.scripts.prepare_critic_checkpoint \
    --base-model "$MODEL" --num-layers "$LAYER" \
    --dataset-sidecar "$AR_SFT_PARQUET" \
    --output "$CRITIC_INIT"
else
  echo "=== critic init already present at $CRITIC_INIT ==="
fi

# ---- (b) AR SFT ----
export WANDB_NAME=ar_sft
export NLA_FREEZE_VALUE_HEAD=1
export AR_SFT_PARQUET="$AR_SFT_PARQUET"
export CRITIC_INIT_CKPT="$CRITIC_INIT"
export SAVE_DIR="$AR_SFT_DIR"
mkdir -p "$SAVE_DIR"

echo "=== AR SFT START ($(date)) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
cd "$(python -c 'import miles, os; print(os.path.dirname(miles.__file__))')/.."

# Paper's exact critic config: global_batch 256, micro 64 (averages 4x more
# samples/step, killing the gradient noise that destabilised 64/16 on Qwen3-8B).
bash "$NLA_REPO/configs/critic_sft.sh" \
  --actor-num-gpus-per-node "$AR_GPUS" \
  --attn-implementation sdpa \
  --rollout-batch-size 256 --global-batch-size 256 --micro-batch-size 64 \
  --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
  --num-rollout 1000 --save-interval 200 \
  ${WANDB_API_KEY:+--use-wandb --wandb-project "$WANDB_PROJECT"}

echo "=== AR SFT END ($(date)) ==="
ls -d "$SAVE_DIR"/iter_* 2>/dev/null | tail -1
