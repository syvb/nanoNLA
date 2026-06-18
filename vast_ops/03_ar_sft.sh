#!/bin/bash
# Stage 3 — AR (reconstructor/critic) warm-start, self-contained LoRA SFT.
# (a) build the truncated K+1-layer critic init (Miles-free model surgery), then
# (b) LoRA SFT the backbone (value head frozen identity) with normalized-MSE at
# the last token. Saves a merged NLACriticModel to $AR_HF. ~1 GPU.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
[ -f "$HOME/.hf_token" ]  && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
[ -f "$HOME/.wandb_key" ] && export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < "$HOME/.wandb_key")}"
export PYTHONUNBUFFERED=1
cd "$NLA_REPO"

# (a) critic init (one-time, fast, CPU/GPU model surgery — no Miles)
if [ ! -f "$CRITIC_INIT/config.json" ]; then
  echo "=== prepare_critic_checkpoint ($(date)) ==="
  python -m nla.scripts.prepare_critic_checkpoint \
    --base-model "$MODEL" --num-layers "$LAYER" \
    --dataset-sidecar "$AR_SFT_PARQUET" --output "$CRITIC_INIT"
else
  echo "=== critic init present: $CRITIC_INIT ==="
fi

# (b) AR LoRA SFT
WB=(--no-wandb); [ -n "${WANDB_API_KEY:-}" ] && WB=(--wandb-project "$WANDB_PROJECT" --wandb-name ar_sft)
echo "=== AR SFT (LoRA) START ($(date)) on GPU ${CUDA_VISIBLE_DEVICES:-0} ==="
python -m nla.train_sft_self_contained \
  --role ar --base-ckpt "$CRITIC_INIT" \
  --parquet "$AR_SFT_PARQUET" --sidecar "$AR_SFT_PARQUET" \
  --save-dir "$AR_HF" \
  --steps "$SFT_STEPS" --micro-batch "$SFT_AR_MICRO" --grad-accum "$SFT_GRAD_ACCUM" \
  --lr "$SFT_LR" --lora-r "$SFT_LORA_R" --lora-alpha "$SFT_LORA_ALPHA" \
  --seed 0 "${WB[@]}"
echo "=== AR SFT END ($(date)) -> $AR_HF ==="
ls "$AR_HF" 2>/dev/null | head || true
