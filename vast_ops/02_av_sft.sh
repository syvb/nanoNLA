#!/bin/bash
# Stage 2 — AV (verbalizer) warm-start, self-contained LoRA SFT.
# CE on the assistant explanation, activation injected at the ㊗ marker via the
# Karvonen layer-1 hook. Saves a merged HF checkpoint to $AV_HF. ~1 GPU.
# Runs on the GPU named by ${CUDA_VISIBLE_DEVICES:-0} (02 and 03 run in parallel).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
[ -f "$HOME/.hf_token" ]  && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
[ -f "$HOME/.wandb_key" ] && export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < "$HOME/.wandb_key")}"
export PYTHONUNBUFFERED=1

if [ -f "$AV_HF/config.json" ]; then echo "=== AV SFT already done ($AV_HF) — skip ==="; exit 0; fi
WB=(--no-wandb); [ -n "${WANDB_API_KEY:-}" ] && WB=(--wandb-project "$WANDB_PROJECT" --wandb-name av_sft)
echo "=== AV SFT (LoRA) START ($(date)) on GPU ${CUDA_VISIBLE_DEVICES:-0} ==="
cd "$NLA_REPO"
python -m nla.train_sft_self_contained \
  --role av --base-ckpt "$MODEL" \
  --parquet "$AV_SFT_PARQUET" --sidecar "$AV_SFT_PARQUET" \
  --save-dir "$AV_HF" \
  --steps "$SFT_STEPS" --micro-batch "$SFT_AV_MICRO" --grad-accum "$SFT_GRAD_ACCUM" \
  --lr "$SFT_LR" --lora-r "$SFT_LORA_R" --lora-alpha "$SFT_LORA_ALPHA" \
  --seed 0 "${WB[@]}"
echo "=== AV SFT END ($(date)) -> $AV_HF ==="
ls "$AV_HF" 2>/dev/null | head || true
