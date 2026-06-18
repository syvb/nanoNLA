#!/bin/bash
# Stage 0 — bring up the training stack on a fresh GPU box.
#
# What this needs (and, deliberately, what it does NOT):
#   - Miles (radixark/miles @ the pinned commit) + the NLA integration patches
#     — required because AV/AR SFT go through Miles' train.py.
#   - flash-attn — Miles' ring_flash_attn assumes it. We use a PREBUILT wheel;
#     a source build OOMs the box (lesson from the main-repo run).
#   - a stock sglang wheel — only to satisfy `import sglang` inside Miles. We do
#     NOT apply the SGLang source patches and never start an SGLang server:
#     SFT runs with --debug-train-only, and RL is the self-contained HF-generate
#     trainer. This skips the slow/fragile part of docs/setup.md.
#   - peft==0.13.0 — newer peft needs torchao>=0.16; the env ships 0.9 (gotcha
#     documented in docs/qwen3_8b_run.md).
#
# Idempotent-ish: safe to re-run; clones are skipped if present.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

PIN="$(cut -d@ -f2 "$NLA_REPO/nla/miles_patches/UPSTREAM_PIN")"
echo "=== [setup] miles pin: $PIN ==="

# ---- 1. Miles ----
if [ ! -d "$MILES_DIR/.git" ]; then
  git clone https://github.com/radixark/miles.git "$MILES_DIR"
fi
cd "$MILES_DIR"
git fetch --all -q || true
git checkout "$PIN"
# Apply NLA integration patches (idempotent: skip if already applied).
for p in "$NLA_REPO"/nla/miles_patches/*.patch; do
  if git apply --reverse --check "$p" 2>/dev/null; then
    echo "  patch already applied: $(basename "$p")"
  else
    git apply "$p" && echo "  applied: $(basename "$p")"
  fi
done
pip install -e . -q

# ---- 2. flash-attn (prebuilt wheel — do NOT compile from source) ----
# Match the cu12/torch2.x/cp312 ABI of the box's torch. Adjust the wheel URL if
# the box ships a different torch/python (check: python -c 'import torch;print(torch.__version__)').
python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("flash_attn") else 1)
PY
if [ $? -ne 0 ]; then
  echo "=== [setup] installing prebuilt flash-attn wheel ==="
  pip install flash-attn --no-build-isolation -q || {
    echo "!! flash-attn pip build failed — install a prebuilt wheel matching the box's torch/cuda/python."
    echo "   e.g. flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    exit 1
  }
fi

# ---- 3. stock sglang (import-satisfier only; no source patches) ----
python -c "import sglang" 2>/dev/null || pip install "sglang[all]>=0.5.6" -q

# ---- 4. nanoNLA package + pinned peft ----
cd "$NLA_REPO"
pip install -e . -q
pip install "peft==0.13.0" -q

# ---- verify ----
echo "=== [setup] verify ==="
python -c "import miles, sglang, nla, peft, torch; print('ok torch', torch.__version__, 'peft', peft.__version__)"
echo "=== [setup] DONE ==="
