#!/bin/bash
# One-shot setup for the reward/KL per-position decomposition on a FRESH
# stock-pytorch GPU box (H100 80GB comfortable; needs one 8B policy + the
# truncated critic resident, ~35 GB peak).
#
# Prereqs on the box BEFORE running this:
#   - /workspace/nanoNLA = this repo, rsynced
#   - /root/.hf_token    = HF token
#
# Downloads the merged AV-SFT (= the RL run's KL anchor), the p0.0 RL LoRA
# (vanilla arm of the length-penalty sweep), the frozen AR critic, and the
# slim rl parquet; then regenerates activation_vector for the held-out rows
# (rl_shuf rows SKIP..SKIP+N — the sweep's eval convention) with the base
# model, exactly like vast_ops/01b_regen_activations.sh.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
N="${N:-200}"
SKIP="${SKIP:-25000}"
pip install -q -e /workspace/nanoNLA "huggingface_hub>=0.34,<1.0" hf_transfer
mkdir -p /workspace/out /workspace/data
python - <<PY
from huggingface_hub import snapshot_download, hf_hub_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("syvb/nanonla-qwen3-8b-L24-av", local_dir="/workspace/av", token=tok, max_workers=16)
snapshot_download("syvb/nanonla-qwen3-8b-L24-ar", local_dir="/workspace/ar", token=tok, max_workers=16)
snapshot_download("syvb/nanonla-qwen3-8b-L24-rl-lora", allow_patterns="p0.0/*",
                  local_dir="/workspace/rl_lora", token=tok, max_workers=4)
for f in ("rl_shuf.parquet", "rl_shuf.parquet.nla_meta.yaml"):
    hf_hub_download("ceselder/qwen3-8b-nla-L24-finefineweb-100k", f,
                    repo_type="dataset", local_dir="/workspace/data", token=tok)
print("DOWNLOADS_DONE", flush=True)
PY
# slice the held-out rows, then regenerate their gold activations
python - "$SKIP" "$N" <<'PY'
import sys
import pyarrow as pa
import pyarrow.parquet as pq
skip, n = int(sys.argv[1]), int(sys.argv[2])
pf = pq.ParquetFile("/workspace/data/rl_shuf.parquet")
rows, seen = [], 0
for b in pf.iter_batches(batch_size=16384):
    t = pa.Table.from_batches([b])
    if seen + t.num_rows <= skip:
        seen += t.num_rows
        continue
    start = max(0, skip - seen)
    take = min(n - sum(r.num_rows for r in rows), t.num_rows - start)
    rows.append(t.slice(start, take))
    seen += t.num_rows
    if sum(r.num_rows for r in rows) >= n:
        break
out = pa.concat_tables(rows)
assert out.num_rows == n, out.num_rows
pq.write_table(out, "/workspace/data/rl_eval_slim.parquet")
print(f"sliced rows [{skip}, {skip+n}) -> rl_eval_slim.parquet", flush=True)
PY
cd /workspace/nanoNLA
python tools/regenerate_activations.py \
    --in /workspace/data/rl_eval_slim.parquet \
    --out /workspace/data/rl_eval_full.parquet \
    --base-model Qwen/Qwen3-8B --batch-size 32 --max-length 4096
cp -f /workspace/data/rl_shuf.parquet.nla_meta.yaml /workspace/data/rl_eval_full.parquet.nla_meta.yaml
echo "SETUP_DONE"
