#!/usr/bin/env bash
# Pipeline smoke test: generate N warm-start explanations with DeepSeek V4 Flash
# via OpenRouter, end-to-end through stage 2. Pure API + CPU — no GPU needed.
#
# Verifies: the OpenRouter provider works, stage 2 saves per-chunk (crash-safe),
# merges to a final parquet + sidecar, and degrades gracefully on row failures.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PY=.venv/bin/python
N="${N:-100}"
CHUNK="${CHUNK:-25}"          # small -> exercises multi-chunk save/resume
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
OUTDIR="${OUTDIR:-scripts/datagen_deepseek/out}"
IN="$OUTDIR/stage2_input.parquet"
OUT="$OUTDIR/explained.parquet"

export OPENROUTER_API_KEY="$(cat ~/.openrouter_key)"
export HF_TOKEN="$(cat ~/.hf_token)"

mkdir -p "$OUTDIR"

echo "=== [1/2] build $N-row stage-2 input from existing dataset ==="
$PY -m scripts.datagen_deepseek.build_test_input --n "$N" --output "$IN"

echo "=== [2/2] stage2 explain via OpenRouter ($MODEL), chunk=$CHUNK ==="
$PY -m nla.datagen.stage2_api_explain \
  --input "$IN" \
  --output "$OUT" \
  --provider-cls nla.datagen.providers.OpenRouterProvider \
  --provider-kwargs "{\"model\": \"$MODEL\", \"max_tokens\": 400, \"temperature\": 1.0, \"concurrency\": 16}" \
  --instruction-template "$(cat scripts/datagen_deepseek/new_prompt.txt)" \
  --chunk-size "$CHUNK"

echo "=== artifacts ==="
ls -la "$OUTDIR"
ls -la "$OUT.chunks" 2>/dev/null || true
