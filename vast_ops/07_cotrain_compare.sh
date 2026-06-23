#!/bin/bash
# Co-train comparison: re-run λ=0.006 and λ=0.015 with --train-critic (AR co-trained
# alongside AV), then dual-eval each against (a) the frozen base AR (same ruler as
# the original frozen sweep -> isolates whether the AV itself improved) and
# (b) its own co-trained AR (the real co-trained NLA system FVE).
# One penalty per GPU, end-to-end. Resumable. Compare to frozen p0.006=.482 / p0.015=.450.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < "$HOME/.wandb_key" 2>/dev/null || true)}"
export PYTHONUNBUFFERED=1 NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"
mkdir -p "$NLA_EMBED_DUMP_DIR" "$RESULTS/heldout_ct" "$RESULTS/logs_ct"
cd "$NLA_REPO"

FVE_BASELINE=0.6703636050224304   # frozen sweep's base-eval baseline — REUSE for comparability
CT_BASE=/workspace/nla/ckpts/rl_ct

run_one() {  # pen gpu
  local pen="$1" gpu="$2" slug; slug="$(penalty_slug "$pen")"
  local save_dir="$CT_BASE/$slug"
  export CUDA_VISIBLE_DEVICES="$gpu"
  mkdir -p "$save_dir"
  local last_iter; last_iter="$(ls -1d "$save_dir"/iter_* 2>/dev/null | sort | tail -1 || true)"
  if [ -z "$last_iter" ]; then
    echo "[$slug] CO-TRAIN RL on GPU $gpu ($(date))"
    python -m nla.train_rl_self_contained \
      --av-ckpt "$AV_HF" --ar-ckpt "$AR_HF" \
      --rl-parquet "$RL_PARQUET" --sidecar "$RL_PARQUET" --save-dir "$save_dir" \
      --num-steps "$RL_NUM_STEPS" --batch-prompts "$RL_BATCH_PROMPTS" \
      --group-size "$RL_GROUP_SIZE" --max-new-tokens "$RL_MAX_NEW" \
      --lr "$RL_LR" --kl-beta "$RL_KL_BETA" --clip-eps 0.2 \
      --lora-r "$RL_LORA_R" --lora-alpha "$RL_LORA_ALPHA" \
      --length-penalty "$pen" --max-rows "$RL_MAX_ROWS" --save-every 50 --seed 0 \
      --train-critic --critic-lr 1e-5 --gradient-checkpointing \
      --eval-every 20 --eval-n-prompts 16 --eval-skip-rows 20000 \
      --wandb-project "$WANDB_PROJECT" --wandb-name "rl_${slug}_cotrain"
    last_iter="$(ls -1d "$save_dir"/iter_* 2>/dev/null | sort | tail -1 || true)"
  else echo "[$slug] RL already done ($last_iter)"; fi
  [ -n "$last_iter" ] || { echo "[$slug] !! no LoRA produced"; return 1; }
  [ -d "$save_dir/critic" ] || { echo "[$slug] !! no co-trained critic saved"; return 1; }

  # Eval A: frozen base AR ruler (compare directly to frozen sweep numbers)
  local outA="$RESULTS/heldout_ct/${slug}_ct_baseAR.samples.jsonl"
  if [ ! -f "$outA" ]; then
    echo "[$slug] EVAL base-AR ruler"
    python vast_ops/eval_heldout.py --av-ckpt "$AV_HF" --ar-ckpt "$AR_HF" \
      --sidecar "$RL_PARQUET" --val-parquet "$VAL_PARQUET" --n-rows "$EVAL_N" \
      --skip-rows "$EVAL_SKIP_ROWS" --max-new "$EVAL_MAX_NEW" --temperature "$EVAL_TEMP" \
      --fve-baseline "$FVE_BASELINE" --tag "${slug}_ct_baseAR" --out "$outA" --rl-lora "$last_iter"
  fi
  # Eval B: own co-trained AR ruler (real co-trained system FVE)
  local outB="$RESULTS/heldout_ct/${slug}_ct_ownAR.samples.jsonl"
  if [ ! -f "$outB" ]; then
    echo "[$slug] EVAL own co-trained-AR ruler"
    python vast_ops/eval_heldout.py --av-ckpt "$AV_HF" --ar-ckpt "$save_dir/critic" \
      --sidecar "$RL_PARQUET" --val-parquet "$VAL_PARQUET" --n-rows "$EVAL_N" \
      --skip-rows "$EVAL_SKIP_ROWS" --max-new "$EVAL_MAX_NEW" --temperature "$EVAL_TEMP" \
      --fve-baseline "$FVE_BASELINE" --tag "${slug}_ct_ownAR" --out "$outB" --rl-lora "$last_iter"
  fi
  rm -rf "${NLA_EMBED_DUMP_DIR:?}/"* 2>/dev/null || true
  echo "[$slug] DONE ($(date))"
}

run_one 0.006 0 > "$RESULTS/logs_ct/p0.006.log" 2>&1 &
P1=$!
run_one 0.015 1 > "$RESULTS/logs_ct/p0.015.log" 2>&1 &
P2=$!
wait $P1; echo "p0.006 exit=$?"
wait $P2; echo "p0.015 exit=$?"
echo "=== CO-TRAIN COMPARE DONE ($(date)) ==="
ls -la "$RESULTS"/heldout_ct/*.summary.json 2>/dev/null || echo "(no summaries!)"
