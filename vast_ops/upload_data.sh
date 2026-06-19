#!/bin/bash
# Persist the regenerated FULL parquets (slim dataset + recomputed
# activation_vector) to a HF dataset so the ~3h of regen is reusable and
# survives box teardown. Uploads the sliced subsets we actually train/eval on.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < "$HOME/.hf_token")}"
: "${HF_TOKEN:?need HF_TOKEN}"
export HF_HUB_ENABLE_HF_TRANSFER=1
REPO="$HF_OWNER/$HF_PREFIX-data-full"
python - "$REPO" "$DATA" "$HF_DATASET" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, data, src = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
card = f"""---
license: apache-2.0
tags: [nla, activations, qwen3]
configs:
  - config_name: av_sft
    data_files: av_sft_full.parquet
  - config_name: ar_sft
    data_files: ar_sft_full.parquet
  - config_name: rl
    data_files: rl_full.parquet
---
# Qwen3-8B NLA — FULL parquets (activation_vector regenerated)

The slim splits of [{src}](https://huggingface.co/datasets/{src}) with the
`activation_vector` column recomputed (raw layer-24 residual at the final token
of `detokenized_text_truncated`). These are the sliced subsets used to train the
length-penalty NLA — `av_sft_full` / `ar_sft_full` (warm-start SFT) and
`rl_full` (RL + held-out eval). Use directly; no regeneration needed.
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
n = 0
for f in ("av_sft_full", "ar_sft_full", "rl_full"):
    for ext in (".parquet", ".parquet.nla_meta.yaml"):
        p = os.path.join(data, f + ext)
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f + ext, repo_id=repo, repo_type="dataset")
            print("uploaded", f + ext, flush=True); n += 1
        else:
            print("MISSING (skipped)", f + ext, flush=True)
print(f"DATA_UPLOAD_DONE: {n} files -> {repo}")
PY
echo "=== data upload -> https://huggingface.co/datasets/$REPO ==="
