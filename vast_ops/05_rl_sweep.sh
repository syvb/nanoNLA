#!/bin/bash
# Stage 5 — base eval (FVE gate) + the penalty sweep, fanned across GPUs.
#
# 1. Eval the base NLA (AV-SFT only, no RL) on GPU 0 -> pins the shared FVE
#    baseline. If base FVE is junk, STOP HERE (cheap) before the sweep.
# 2. For each penalty: continue-from-SFT RL with that --length-penalty, then
#    generate+score EVAL_N held-out samples. Runs pinned to one GPU each,
#    $SWEEP_GPUS at a time (background + throttle).
# 3. Build comparison + RESULTS markdown.
#
# Resumable: skips an RL run whose final LoRA exists, and an eval whose jsonl
# exists. Per-penalty logs in $RESULTS/logs/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
[ -f "$HOME/.hf_token" ]  && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
[ -f "$HOME/.wandb_key" ] && export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < "$HOME/.wandb_key")}"
export PYTHONUNBUFFERED=1
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"
mkdir -p "$NLA_EMBED_DUMP_DIR" "$RESULTS/heldout" "$RESULTS/logs"
cd "$NLA_REPO"

test -d "$AV_HF" || { echo "missing AV_HF=$AV_HF (run 02_av_sft.sh)"; exit 1; }
test -d "$AR_HF" || { echo "missing AR_HF=$AR_HF (run 03_ar_sft.sh)"; exit 1; }

# ---- base NLA eval (GPU 0) — also pins the shared FVE baseline ----
if [ ! -f "$RESULTS/heldout/base.samples.jsonl" ]; then
  echo "=== EVAL base (AV-SFT only) on GPU 0 ==="
  CUDA_VISIBLE_DEVICES=0 python vast_ops/eval_heldout.py \
    --av-ckpt "$AV_HF" --ar-ckpt "$AR_HF" --sidecar "$RL_PARQUET" \
    --val-parquet "$VAL_PARQUET" --n-rows "$EVAL_N" --skip-rows "$EVAL_SKIP_ROWS" \
    --max-new "$EVAL_MAX_NEW" --temperature "$EVAL_TEMP" \
    --tag base --out "$RESULTS/heldout/base.samples.jsonl"
fi
FVE_BASELINE="$(python3 -c "import json;print(json.load(open('$RESULTS/heldout/base.samples.summary.json'))['fve_baseline'])")"
# FVE gate: require a scorable base NLA (valid FVE above floor AND real extraction).
python3 - "$RESULTS/heldout/base.samples.summary.json" <<'PY' || { echo "!! FVE GATE FAILED — warm-start looks broken. STOPPING before the sweep."; exit 2; }
import json, sys
s = json.load(open(sys.argv[1]))
fve, ext = s.get("fve"), s.get("extraction_rate", 0)
print(f"=== base FVE={fve}  extraction={ext:.0%}  (baseline mse={s['fve_baseline']:.4f}) ===")
ok = isinstance(fve, (int, float)) and fve > 0.15 and ext > 0.5
sys.exit(0 if ok else 1)
PY

# ---- one penalty: RL train + held-out eval, pinned to GPU $2 ----
run_penalty() {  # pen  gpu
  local pen="$1" gpu="$2"
  local slug; slug="$(penalty_slug "$pen")"
  local save_dir="$RL_BASE/$slug" out="$RESULTS/heldout/$slug.samples.jsonl"
  export CUDA_VISIBLE_DEVICES="$gpu"
  local last_iter; last_iter="$(ls -1d "$save_dir"/iter_* 2>/dev/null | sort | tail -1 || true)"
  if [ -z "$last_iter" ]; then
    echo "[$slug] RL train on GPU $gpu ($(date))"
    mkdir -p "$save_dir"
    local wb=(--no-wandb)
    [ -n "${WANDB_API_KEY:-}" ] && wb=(--wandb-project "$WANDB_PROJECT" --wandb-name "rl_$slug")
    python -m nla.train_rl_self_contained \
      --av-ckpt "$AV_HF" --ar-ckpt "$AR_HF" \
      --rl-parquet "$RL_PARQUET" --sidecar "$RL_PARQUET" --save-dir "$save_dir" \
      --num-steps "$RL_NUM_STEPS" --batch-prompts "$RL_BATCH_PROMPTS" \
      --group-size "$RL_GROUP_SIZE" --max-new-tokens "$RL_MAX_NEW" \
      --lr "$RL_LR" --kl-beta "$RL_KL_BETA" --clip-eps 0.2 \
      --lora-r "$RL_LORA_R" --lora-alpha "$RL_LORA_ALPHA" \
      --length-penalty "$pen" --max-rows "$RL_MAX_ROWS" --save-every 50 --seed 0 \
      --eval-every 20 --eval-n-prompts 16 --eval-skip-rows 20000 "${wb[@]}"
    last_iter="$(ls -1d "$save_dir"/iter_* 2>/dev/null | sort | tail -1 || true)"
  else
    echo "[$slug] RL already done ($last_iter)"
  fi
  [ -n "$last_iter" ] || { echo "[$slug] !! no LoRA produced"; return 1; }
  if [ ! -f "$out" ]; then
    echo "[$slug] EVAL on GPU $gpu (LoRA $last_iter)"
    python vast_ops/eval_heldout.py \
      --av-ckpt "$AV_HF" --ar-ckpt "$AR_HF" --sidecar "$RL_PARQUET" \
      --val-parquet "$VAL_PARQUET" --n-rows "$EVAL_N" --skip-rows "$EVAL_SKIP_ROWS" \
      --max-new "$EVAL_MAX_NEW" --temperature "$EVAL_TEMP" --fve-baseline "$FVE_BASELINE" \
      --tag "$slug" --out "$out" --rl-lora "$last_iter"
  fi
  rm -rf "${NLA_EMBED_DUMP_DIR:?}/"* 2>/dev/null || true
}

# ---- fan the penalties across $SWEEP_GPUS GPUs ----
echo "=== sweep: [$PENALTIES] across $SWEEP_GPUS GPU(s) ==="
i=0
pids=()
for pen in $PENALTIES; do
  gpu=$(( i % SWEEP_GPUS ))
  slug="$(penalty_slug "$pen")"
  ( run_penalty "$pen" "$gpu" ) >"$RESULTS/logs/$slug.log" 2>&1 &
  pids+=($!)
  i=$(( i + 1 ))
  # throttle: once SWEEP_GPUS jobs are in flight, wait for the oldest
  if [ "${#pids[@]}" -ge "$SWEEP_GPUS" ]; then
    wait "${pids[0]}" || echo "!! a penalty job failed (see $RESULTS/logs/) — continuing"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || echo "!! a penalty job failed — continuing"; done

# ---- verify every model produced a jsonl ----
missing=0
for pen in $PENALTIES; do
  slug="$(penalty_slug "$pen")"
  [ -f "$RESULTS/heldout/$slug.samples.jsonl" ] || { echo "!! MISSING $slug (see $RESULTS/logs/$slug.log)"; missing=1; }
done

# ---- analysis ----
echo "=== building comparison + results markdown ==="
python vast_ops/generate_comparison_md.py --heldout-dir "$RESULTS/heldout" --base base \
  --out "$RESULTS/comparison_base_vs_penalty.md" --n-examples 100
python vast_ops/make_results.py --heldout-dir "$RESULTS/heldout" --base base \
  --out "$RESULTS/RESULTS.md"
python vast_ops/plot_tradeoff.py --heldout-dir "$RESULTS/heldout" --out "$RESULTS/tradeoff.png" || echo "(plot skipped)"

echo "=== SWEEP DONE ($(date)) ==="
ls -la "$RESULTS"/heldout/*.summary.json "$RESULTS"/*.md 2>/dev/null || true
[ "$missing" -eq 0 ] || echo "WARNING: some penalties missing — see $RESULTS/logs/"
