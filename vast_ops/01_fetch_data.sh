#!/bin/bash
# Stage 1 — fetch the SLIM warm-start dataset (prompts + Sonnet explanations +
# provenance, NO activation vectors — those are regenerated in 01b). Skips the
# ~12h datagen + the Anthropic Batches API bill. ~795 MB.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"

echo "=== [data] downloading $HF_DATASET -> $DATA ==="
huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$DATA" \
  --include '*.parquet' '*.nla_meta.yaml'

for f in "$AV_SLIM" "$AR_SLIM" "$RL_SLIM"; do
  test -f "$f" || { echo "!! missing slim parquet: $f"; exit 1; }
  test -f "$f.nla_meta.yaml" || { echo "!! missing sidecar: $f.nla_meta.yaml"; exit 1; }
done
echo "=== [data] slim parquets present ==="
ls -la "$DATA"/*_shuf.parquet
