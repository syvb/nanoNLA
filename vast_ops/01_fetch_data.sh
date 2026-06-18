#!/bin/bash
# Stage 1 — fetch the warm-start dataset (skips ~12h datagen + the Anthropic
# Batches API bill). Pulls the three parquets + their .nla_meta.yaml sidecars
# from the HF dataset into $DATA. ~795 MB, a few minutes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

[ -f "$HOME/.hf_token" ] && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"

echo "=== [data] downloading $HF_DATASET -> $DATA ==="
huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$DATA" \
  --include '*.parquet' '*.nla_meta.yaml'

echo "=== [data] present: ==="
ls -la "$DATA"/*.parquet "$DATA"/*.nla_meta.yaml
for f in "$AV_SFT_PARQUET" "$AR_SFT_PARQUET" "$RL_PARQUET"; do
  test -f "$f" || { echo "!! missing expected parquet: $f"; exit 1; }
  test -f "$f.nla_meta.yaml" || { echo "!! missing sidecar: $f.nla_meta.yaml"; exit 1; }
done
echo "=== [data] DONE ==="
