#!/bin/bash
# Stage 4 — convert both SFT checkpoints from FSDP/DCP to HF so the
# self-contained RL trainer and the eval can load them with from_pretrained.
#   AV: full HF causal LM      -> tools/convert_fsdp_to_hf.py
#   AR: NLACriticModel (backbone.* prefixes, K+1 layers) -> launch/convert_ar_dcp_to_hf.py
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
cd "$NLA_REPO"

latest_iter() { ls -1d "$1"/iter_* 2>/dev/null | sort | tail -1 || true; }

AV_ITER="$(latest_iter "$AV_SFT_DIR")"; test -n "$AV_ITER" || { echo "no AV iter in $AV_SFT_DIR"; exit 1; }
AR_ITER="$(latest_iter "$AR_SFT_DIR")"; test -n "$AR_ITER" || { echo "no AR iter in $AR_SFT_DIR"; exit 1; }
echo "=== converting AV $AV_ITER -> $AV_HF ==="
python tools/convert_fsdp_to_hf.py \
  --input-dir "$AV_ITER" --output-dir "$AV_HF" \
  --origin-hf-dir "$MODEL" --force

echo "=== converting AR $AR_ITER -> $AR_HF (num-layers $CRITIC_NUM_LAYERS) ==="
python launch/convert_ar_dcp_to_hf.py \
  --input-dir "$AR_ITER" --output-dir "$AR_HF" \
  --origin-hf-dir "$MODEL" --num-layers "$CRITIC_NUM_LAYERS"

# Carry the NLA sidecar onto the HF checkpoints so config.py resolves the
# model sidecar (authoritative for what THIS model was trained with).
cp -n "$AV_SFT_PARQUET.nla_meta.yaml" "$AV_HF/nla_meta.yaml" 2>/dev/null || true
cp -n "$AR_SFT_PARQUET.nla_meta.yaml" "$AR_HF/nla_meta.yaml" 2>/dev/null || true

echo "=== convert DONE ===  AV_HF=$AV_HF  AR_HF=$AR_HF"
ls "$AV_HF" "$AR_HF"
