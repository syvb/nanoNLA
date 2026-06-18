#!/bin/bash
# Stage 1b — regenerate the activation_vector column the slim dataset omits.
# For each split, slice to the rows we'll actually use, then re-run the base
# model over detokenized_text_truncated and take the layer-K final-token hidden
# state (tools/regenerate_activations.py). Produces the _full parquets.
# AV+AR regen run in parallel (GPU 0 + 1) when 2 GPUs are present; RL after.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
export HF_HOME PYTHONUNBUFFERED=1
cd "$NLA_REPO"

regen() {  # slim  full  nrows  gpu
  local slim="$1" full="$2" nrows="$3" gpu="$4"
  if [ -f "$full" ]; then echo "[regen] $full exists, skip"; return; fi
  local sliced="${full%.parquet}.slim$nrows.parquet"
  python - "$slim" "$sliced" "$nrows" <<'PY'
import sys, pyarrow.parquet as pq, pyarrow as pa
slim, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
pf = pq.ParquetFile(slim); n = min(n, pf.metadata.num_rows)
rows, got = [], 0
for b in pf.iter_batches(batch_size=16384):
    take = min(n - got, b.num_rows); rows.append(pa.Table.from_batches([b]).slice(0, take))
    got += take
    if got >= n: break
pq.write_table(pa.concat_tables(rows), out); print(f"sliced {got} -> {out}", flush=True)
PY
  echo "[regen] $full on GPU $gpu ($(date))"
  CUDA_VISIBLE_DEVICES="$gpu" python tools/regenerate_activations.py \
    --in "$sliced" --out "$full" --base-model "$MODEL" --batch-size "$REGEN_BATCH"
  cp -f "$slim.nla_meta.yaml" "$full.nla_meta.yaml"   # carry the NLA sidecar
  rm -f "$sliced"
}

if [ "${SWEEP_GPUS:-1}" -ge 2 ]; then
  regen "$AV_SLIM" "$AV_SFT_PARQUET" "$REGEN_AV_ROWS" 0 &  pid0=$!
  regen "$AR_SLIM" "$AR_SFT_PARQUET" "$REGEN_AR_ROWS" 1 &  pid1=$!
  wait "$pid0"; wait "$pid1"
  regen "$RL_SLIM" "$RL_PARQUET" "$REGEN_RL_ROWS" 0
else
  regen "$AV_SLIM" "$AV_SFT_PARQUET" "$REGEN_AV_ROWS" 0
  regen "$AR_SLIM" "$AR_SFT_PARQUET" "$REGEN_AR_ROWS" 0
  regen "$RL_SLIM" "$RL_PARQUET" "$REGEN_RL_ROWS" 0
fi
echo "=== [regen] DONE ==="
ls -la "$AV_SFT_PARQUET" "$AR_SFT_PARQUET" "$RL_PARQUET"
