#!/bin/bash
# Stage 6 — persist artifacts to HuggingFace (private). Models: AV, AR, RL LoRA
# adapters. Dataset: held-out samples + results markdown + plot.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
: "${HF_TOKEN:?need HF_TOKEN (or ~/.hf_token) to upload}"
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$NLA_REPO"
python vast_ops/upload_to_hf.py \
  --owner "$HF_OWNER" --prefix "$HF_PREFIX" \
  --av "$AV_HF" --ar "$AR_HF" --rl-base "$RL_BASE" \
  --results "$RESULTS" --penalties "$PENALTIES"
echo "=== upload DONE -> https://huggingface.co/$HF_OWNER ($HF_PREFIX-*) ==="
