#!/bin/bash
# Stage 0 — bring up the (now trivial) stack. All-LoRA, Miles-free: the whole
# pipeline runs on torch + transformers + peft + pyarrow + the nla package.
# No conda, no SGLang, no flash-attn compile, no Miles. Assumes a stock PyTorch
# GPU image (torch + CUDA already present).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

echo "=== [setup] torch check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"

echo "=== [setup] deps ==="
# Pinned pair: transformers 4.53.x has Qwen3 AND still exports HybridCache (peft
# 0.15 imports it; transformers >=4.54 removed it). peft 0.15.2 matches.
pip install -q "transformers==4.53.2" "peft==0.15.2" "accelerate>=1.0" \
    pyarrow datasets safetensors huggingface_hub hf_transfer orjson pyyaml numpy wandb matplotlib

echo "=== [setup] nla package (editable, no extras) ==="
cd "$NLA_REPO"
pip install -e . -q --no-deps   # nla deps already covered above; --no-deps avoids pulling anything heavy

echo "=== [setup] verify (no miles/sglang needed) ==="
python -c "import nla, peft, transformers, pyarrow, torch; \
import nla.models, nla.injection, nla.config, nla.schema; \
print('ok: nla primitives import, peft', peft.__version__)"
echo "=== [setup] DONE ==="
