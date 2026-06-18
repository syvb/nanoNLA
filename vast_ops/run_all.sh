#!/bin/bash
# Master runner — chains the whole Miles-free pipeline, logging each stage.
# Designed for `nohup bash vast_ops/run_all.sh &` on the box. Resumable: every
# stage skips work it already finished.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
LOGS="$NLA_WORKSPACE/logs"; mkdir -p "$LOGS"

step() { echo "===== $* ($(date)) ====="; }

step "00 setup";       bash "$HERE/00_setup.sh"           2>&1 | tee "$LOGS/00_setup.log"
step "01 fetch data";  bash "$HERE/01_fetch_data.sh"      2>&1 | tee "$LOGS/01_data.log"
step "01b regen acts"; bash "$HERE/01b_regen_activations.sh" 2>&1 | tee "$LOGS/01b_regen.log"

# AV + AR SFT — parallel on two GPUs if available, else sequential on GPU 0.
if [ "${SWEEP_GPUS:-1}" -ge 2 ]; then
  step "02/03 AV+AR SFT in parallel (GPU 0 + 1)"
  CUDA_VISIBLE_DEVICES=0 bash "$HERE/02_av_sft.sh" >"$LOGS/02_av.log" 2>&1 &  pid_av=$!
  CUDA_VISIBLE_DEVICES=1 bash "$HERE/03_ar_sft.sh" >"$LOGS/03_ar.log" 2>&1 &  pid_ar=$!
  fail=0
  wait "$pid_av" || { echo "!! AV SFT failed (see $LOGS/02_av.log)"; fail=1; }
  wait "$pid_ar" || { echo "!! AR SFT failed (see $LOGS/03_ar.log)"; fail=1; }
  [ "$fail" -eq 0 ] || { echo "SFT failed — stopping"; exit 1; }
else
  step "02 AV SFT (GPU 0)"; CUDA_VISIBLE_DEVICES=0 bash "$HERE/02_av_sft.sh" 2>&1 | tee "$LOGS/02_av.log"
  step "03 AR SFT (GPU 0)"; CUDA_VISIBLE_DEVICES=0 bash "$HERE/03_ar_sft.sh" 2>&1 | tee "$LOGS/03_ar.log"
fi

step "05 base eval + sweep"; bash "$HERE/05_rl_sweep.sh" 2>&1 | tee "$LOGS/05_sweep.log"
step "06 upload";            bash "$HERE/06_upload.sh"   2>&1 | tee "$LOGS/06_upload.log"
step "ALL DONE"
echo "RESULTS at $RESULTS  |  HF: https://huggingface.co/$HF_OWNER ($HF_PREFIX-*)"
