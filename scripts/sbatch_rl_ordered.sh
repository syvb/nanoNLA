#!/bin/bash
#SBATCH --job-name=qwen3_rl_grpo_ordered
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --no-requeue
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out
# Ordered-features RL: identical to sbatch_rl_fixed.sh EXCEPT for the
# --rl-trunc-* flags. Generation STOPS after K features (K ~ Uniform[1,10], one
# K per prompt-group) via --rl-trunc-generate, so rollouts are cheaper and GRPO
# trains only on the scored features. The critic only ever sees that random
# prefix, so the AV is rewarded for front-loading the most reconstruction-
# relevant feature -> importance-ordered output. The stop is external (no EOS
# trained), so the AV still emits all features at inference.
# Warm-start SFT (AV + AR) is UNMODIFIED — reuse the iter_0001000 ckpts.
#
# PoC note: --num-steps 250 is past the fve saturation knee (~step 150) and
# enough to see the ordering emerge; bump to 500 to match the verified run.
set -euo pipefail
source /workspace-vast/celeste/.env
source /workspace-vast/celeste/envs/nla/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace-vast/celeste/nla-experiments
DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k
cd /workspace-vast/celeste/nla-experiments
python -m nla.train_rl_self_contained \
  --av-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_av_sft_lora_fixed/iter_0001000 \
  --ar-ckpt /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_ar_sft_lora_fixed/iter_0001000 \
  --base-ckpt Qwen/Qwen3-8B --quant 4bit \
  --rl-parquet $DATA/rl_shuf.parquet --sidecar $DATA/rl_shuf.parquet \
  --save-dir /workspace-vast/celeste/nla-ckpts/qwen3_8b_L24_rl_grpo_ordered \
  --num-steps 250 --batch-prompts 16 --group-size 16 \
  --max-new-tokens 150 --temperature 1.0 \
  --lr 1e-5 --kl-beta 0.01 --clip-eps 0.2 \
  --train-critic --critic-lr 5e-5 \
  --logp-micro-batch 2 --max-rows 30000 \
  --rl-trunc-max-lines 10 --rl-trunc-generate \
  --save-every 50 --eval-every 10 --eval-n-prompts 20 --eval-skip-rows 35000 \
  --max-grad-norm 1.0 \
  --wandb-project nla-qwen3-8b --wandb-name rl_grpo_ordered --seed 0
