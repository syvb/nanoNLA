#!/bin/bash
# Stage 5 — the sweep. For each penalty: continue-from-SFT RL with that
# --length-penalty, then generate + score EVAL_N held-out samples. The base
# NLA (AV-SFT only, no RL) is evaluated once up front for the baseline column.
#
# Resumable: skips an RL run whose final LoRA already exists, and an eval whose
# jsonl already exists. /dev/shm embed dumps are cleaned between runs.
#
# Persisted to $RESULTS/heldout/<tag>.samples.jsonl (+ .summary.json) — enough
# to rebuild the length<->FVE tradeoff chart offline.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

[ -f "$HOME/.hf_token" ]  && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
[ -f "$HOME/.wandb_key" ] && export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < "$HOME/.wandb_key" | sed 's/^WANDB_API_KEY=//')}"
export WANDB_PROJECT="${WANDB_PROJECT:-nla-lenpen}"
export PYTHONUNBUFFERED=1
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"
mkdir -p "$NLA_EMBED_DUMP_DIR" "$RESULTS/heldout"
cd "$NLA_REPO"

test -d "$AV_HF" || { echo "missing AV_HF=$AV_HF (run 04_convert.sh)"; exit 1; }
test -d "$AR_HF" || { echo "missing AR_HF=$AR_HF (run 04_convert.sh)"; exit 1; }

eval_one() {  # tag  lora_dir_or_BASE
  local tag="$1" lora="$2" out="$RESULTS/heldout/$1.samples.jsonl"
  if [ -f "$out" ]; then echo "  eval $tag exists, skip"; return; fi
  local lora_flag=(); [ "$lora" != "BASE" ] && lora_flag=(--rl-lora "$lora")
  local base_flag=(); [ -n "${FVE_BASELINE:-}" ] && base_flag=(--fve-baseline "$FVE_BASELINE")
  python vast_ops/eval_heldout.py \
    --av-ckpt "$AV_HF" --ar-ckpt "$AR_HF" --sidecar "$RL_PARQUET" \
    --val-parquet "$VAL_PARQUET" --n-rows "$EVAL_N" --skip-rows "$EVAL_SKIP_ROWS" \
    --max-new "$EVAL_MAX_NEW" --temperature "$EVAL_TEMP" \
    --tag "$tag" --out "$out" "${lora_flag[@]}" "${base_flag[@]}"
}

# ---- base NLA (no RL); also pins the shared FVE baseline for every other model ----
echo "=== EVAL base (AV-SFT only) ==="
eval_one base BASE
export FVE_BASELINE="$(python3 -c "import json;print(json.load(open('$RESULTS/heldout/base.summary.json'))['fve_baseline'])")"
echo "=== shared FVE baseline = $FVE_BASELINE ==="

# ---- penalty sweep ----
for pen in $PENALTIES; do
  slug="$(penalty_slug "$pen")"
  save_dir="$RL_BASE/$slug"
  last_iter="$(ls -1d "$save_dir"/iter_* 2>/dev/null | sort | tail -1 || true)"
  if [ -z "$last_iter" ]; then
    echo "=== RL train  penalty=$pen  -> $save_dir ($(date)) ==="
    mkdir -p "$save_dir"
    wb=(--no-wandb)
    [ -n "${WANDB_API_KEY:-}" ] && wb=(--wandb-project "$WANDB_PROJECT" --wandb-name "rl_$slug")
    python -m nla.train_rl_self_contained \
      --av-ckpt "$AV_HF" --ar-ckpt "$AR_HF" \
      --rl-parquet "$RL_PARQUET" --sidecar "$RL_PARQUET" \
      --save-dir "$save_dir" \
      --num-steps "$RL_NUM_STEPS" --batch-prompts "$RL_BATCH_PROMPTS" \
      --group-size "$RL_GROUP_SIZE" --max-new-tokens "$RL_MAX_NEW" \
      --lr "$RL_LR" --kl-beta "$RL_KL_BETA" --clip-eps 0.2 \
      --lora-r "$RL_LORA_R" --lora-alpha "$RL_LORA_ALPHA" \
      --length-penalty "$pen" --max-rows "$RL_MAX_ROWS" \
      --save-every 50 --seed 0 "${wb[@]}"
    last_iter="$(ls -1d "$save_dir"/iter_* 2>/dev/null | sort | tail -1)"
    rm -rf "${NLA_EMBED_DUMP_DIR:?}/"* 2>/dev/null || true
  else
    echo "=== RL train penalty=$pen already done ($last_iter) ==="
  fi
  test -n "$last_iter" || { echo "!! no LoRA produced for penalty $pen"; exit 1; }
  echo "=== EVAL $slug  (LoRA $last_iter) ==="
  eval_one "$slug" "$last_iter"
done

# ---- analysis artifacts ----
echo "=== building comparison + results markdown ==="
python vast_ops/generate_comparison_md.py \
  --heldout-dir "$RESULTS/heldout" --base base \
  --out "$RESULTS/comparison_base_vs_penalty.md" --n-examples 100
python vast_ops/make_results.py \
  --heldout-dir "$RESULTS/heldout" --base base \
  --out "$RESULTS/RESULTS.md"

echo "=== SWEEP DONE ($(date)) ==="
ls -la "$RESULTS"/heldout/*.summary.json "$RESULTS"/*.md
