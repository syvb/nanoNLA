#!/bin/bash
# Ordered-features NLA PoC — end-to-end on a single GPU box (Vast).
#
# slim HF dataset -> regen activations -> doc-level 25/25/50 split -> build
# av_sft/ar_sft/rl parquets -> warm-start SFT (AV, AR) -> generate-K ordered RL.
#
# Idempotent per stage (skips a stage whose output already exists), so a re-run
# resumes. Heavy logging with banners. Env overrides (defaults shown):
#   WORK=/workspace/nla_poc  REPO=$PWD  BASE_MODEL=Qwen/Qwen3-8B
#   AV_STEPS=300  AR_STEPS=300  RL_STEPS=250
#   HF_DATASET=syvb/nla-warmstart-explanations-finefineweb-sonnet46
set -euo pipefail

WORK=${WORK:-/workspace/nla_poc}
REPO=${REPO:-$PWD}
HF_DATASET=${HF_DATASET:-syvb/nla-warmstart-explanations-finefineweb-sonnet46}
HF_FILE=${HF_FILE:-data/train-00000-of-00001.parquet}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
AV_STEPS=${AV_STEPS:-300}
AR_STEPS=${AR_STEPS:-300}
RL_STEPS=${RL_STEPS:-120}
export HF_HOME=${HF_HOME:-/workspace/hf_home}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# wandb: auto-enable when a key is available (env or /root/.wandb_key), else off.
[ -z "${WANDB_API_KEY:-}" ] && [ -f /root/.wandb_key ] && export WANDB_API_KEY="$(cat /root/.wandb_key)"
WB_PROJECT=${WB_PROJECT:-nla-qwen3-8b}
wb() {  # $1 = run name -> emits wandb flags (or --no-wandb if no key)
  if [ -n "${WANDB_API_KEY:-}" ]; then echo "--wandb-project $WB_PROJECT --wandb-name $1"
  else echo "--no-wandb"; fi
}

# Push an artifact to HF immediately (best-effort) so a host failure mid-run
# doesn't lose completed work. Needs HF_TOKEN + COLLECTION_SLUG in the env.
HF_NS=${HF_NS:-syvb}
push_artifact() {  # $1=path  $2=repo-suffix  $3=type(model|dataset)
  if [ -n "${HF_TOKEN:-}" ] && [ -n "${COLLECTION_SLUG:-}" ]; then
    echo "[push] $1 -> $HF_NS/$2 (${3:-model})"
    python scripts/poc_ordered/push_ckpt.py --path "$1" --repo "$HF_NS/$2" --type "${3:-model}" || echo "[push] failed (continuing)"
  else
    echo "[push] skip $2 (set HF_TOKEN + COLLECTION_SLUG to enable)"
  fi
}

DATA=$WORK/data
CKPT=$WORK/ckpts
BUILD=$DATA/build
SPLIT=$DATA/split
mkdir -p "$DATA" "$CKPT" "$BUILD"
cd "$REPO"

banner() { echo; echo "=================== $* ==================="; date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# ----------------------------------------------------------------------------
banner "STAGE 1/8: download slim dataset ($HF_DATASET)"
SLIM=$DATA/slim.parquet
if [ ! -f "$SLIM" ]; then
  python - "$HF_DATASET" "$HF_FILE" "$SLIM" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
ds, f, out = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(ds, f, repo_type="dataset")
shutil.copy(p, out)
print("downloaded ->", out)
PY
else echo "  (skip) $SLIM exists"; fi

# ----------------------------------------------------------------------------
banner "STAGE 2/8: regenerate activations"
FULL=$DATA/base_with_activations.parquet
if [ ! -f "$FULL" ]; then
  python tools/regenerate_activations.py --in "$SLIM" --out "$FULL" \
    --base-model "$BASE_MODEL" --max-length 4096 --batch-size 16 \
    --chunk-size 512 --drop-mismatch
else echo "  (skip) $FULL exists"; fi

# ----------------------------------------------------------------------------
banner "STAGE 3/8: synthesize base sidecar"
if [ ! -f "$FULL.nla_meta.yaml" ]; then
  python scripts/poc_ordered/synth_base_sidecar.py --parquet "$FULL" --base-model "$BASE_MODEL"
else echo "  (skip) sidecar exists"; fi

# ----------------------------------------------------------------------------
banner "STAGE 4/8: document-level split 25/25/50"
if [ ! -f "$SPLIT/rl_raw.parquet" ]; then
  python -m nla.datagen.stage1_split --base "$FULL" \
    --av-sft-frac 0.25 --ar-sft-frac 0.25 --rl-frac 0.50 --seed 42 \
    --output-dir "$SPLIT"
else echo "  (skip) split exists"; fi

# ----------------------------------------------------------------------------
banner "STAGE 5/8: build training parquets (av_sft, ar_sft, rl)"
[ -f "$BUILD/av_sft_shuf.parquet" ] || python -m nla.datagen.stage3_build \
  --stage av_sft --input "$SPLIT/av_sft_raw.parquet" --output "$BUILD/av_sft_shuf.parquet"
[ -f "$BUILD/ar_sft_shuf.parquet" ] || python -m nla.datagen.stage3_build \
  --stage ar_sft --input "$SPLIT/ar_sft_raw.parquet" --output "$BUILD/ar_sft_shuf.parquet"
[ -f "$BUILD/rl_shuf.parquet" ]     || python -m nla.datagen.stage3_build \
  --stage rl     --input "$SPLIT/rl_raw.parquet"     --output "$BUILD/rl_shuf.parquet"
echo "row counts:"; python - "$BUILD" <<'PY'
import sys, pyarrow.parquet as pq
b = sys.argv[1]
for s in ("av_sft_shuf","ar_sft_shuf","rl_shuf"):
    print(f"  {s}: {pq.ParquetFile(b+'/'+s+'.parquet').metadata.num_rows} rows")
PY
push_artifact "$BUILD" nla-ordered-features-poc-data dataset

# ----------------------------------------------------------------------------
banner "STAGE 6/8: AV warm-start SFT ($AV_STEPS steps)"
AV_DIR=$CKPT/av_sft
if ! ls -d "$AV_DIR"/iter_* >/dev/null 2>&1; then
  python -m nla.train_sft --mode av --base-ckpt "$BASE_MODEL" \
    --parquet "$BUILD/av_sft_shuf.parquet" --sidecar "$BUILD/av_sft_shuf.parquet" \
    --save-dir "$AV_DIR" --num-steps "$AV_STEPS" --batch-size 64 \
    --use-lora --lora-r 128 --lora-alpha 16 --quant 4bit \
    --lr 3e-5 --gradient-checkpointing --save-every "$AV_STEPS" --seed 0 $(wb av_sft_slim)
else echo "  (skip) AV checkpoint exists"; fi
AV_CKPT=$(ls -d "$AV_DIR"/iter_* | sort | tail -1)
echo "AV_CKPT=$AV_CKPT"
push_artifact "$AV_CKPT" nla-ordered-features-av-sft model

# ----------------------------------------------------------------------------
banner "STAGE 7/8: AR warm-start SFT ($AR_STEPS steps)"
AR_DIR=$CKPT/ar_sft
if ! ls -d "$AR_DIR"/iter_* >/dev/null 2>&1; then
  python -m nla.train_sft --mode ar --base-ckpt "$BASE_MODEL" \
    --parquet "$BUILD/ar_sft_shuf.parquet" --sidecar "$BUILD/ar_sft_shuf.parquet" \
    --save-dir "$AR_DIR" --num-steps "$AR_STEPS" --batch-size 64 --ar-num-layers 25 \
    --use-lora --lora-r 128 --lora-alpha 16 --quant 4bit \
    --lr 3e-5 --gradient-checkpointing --save-every "$AR_STEPS" --seed 0 $(wb ar_sft_slim)
else echo "  (skip) AR checkpoint exists"; fi
AR_CKPT=$(ls -d "$AR_DIR"/iter_* | sort | tail -1)
echo "AR_CKPT=$AR_CKPT"
push_artifact "$AR_CKPT" nla-ordered-features-ar-sft model

# ----------------------------------------------------------------------------
# RL truncation mode. generate-K (RL_GENERATE_K=1) stops generation at K
# features — cheaper IN PRINCIPLE, but the per-token Python stopping criterion
# serializes generation (GPU-starved) and is currently SLOWER; until it has a
# GPU-native (token-id, no-decode) criterion, default to post-hoc per-group,
# which produces the same nested-dropout signal at normal generation speed.
if [ "${RL_GENERATE_K:-0}" = "1" ]; then
  RL_TRUNC="--rl-trunc-max-lines 10 --rl-trunc-generate"; RL_MODE="generate-K"
else
  RL_TRUNC="--rl-trunc-max-lines 10 --rl-trunc-mode per-group --rl-mask-loss-to-k"; RL_MODE="post-hoc/per-group+maskK"
fi
banner "STAGE 8/8: RL ordered-features ($RL_MODE, $RL_STEPS steps)"
RL_DIR=$CKPT/rl_ordered
python -m nla.train_rl_self_contained \
  --av-ckpt "$AV_CKPT" --ar-ckpt "$AR_CKPT" --base-ckpt "$BASE_MODEL" \
  --quant 4bit --device-map single \
  --rl-parquet "$BUILD/rl_shuf.parquet" --sidecar "$BUILD/rl_shuf.parquet" \
  --save-dir "$RL_DIR" \
  --num-steps "$RL_STEPS" --batch-prompts 8 --group-size 8 \
  --max-new-tokens 150 --temperature 1.0 --lr 1e-5 --kl-beta 0.01 --clip-eps 0.2 \
  --train-critic --critic-lr 5e-5 --logp-micro-batch 8 --critic-micro-batch 8 \
  --max-rows 3000 --eval-skip-rows 3000 --eval-every 10 --eval-n-prompts 20 \
  $RL_TRUNC \
  --save-every 50 --seed 0 $(wb rl_ordered_poc)

RL_CKPT=$(ls -d "$RL_DIR"/iter_* 2>/dev/null | sort | tail -1)
[ -n "$RL_CKPT" ] && push_artifact "$RL_CKPT" nla-ordered-features-rl model

banner "PIPELINE COMPLETE"
echo "AV : $AV_CKPT"
echo "AR : $AR_CKPT"
echo "RL : $RL_CKPT"
echo "build dir: $BUILD"
