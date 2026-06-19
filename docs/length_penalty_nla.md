# Length-penalty NLA (Qwen3-8B, layer 24) — what it is & how to use it

A from-scratch **Natural Language Autoencoder** trained on Qwen3-8B's layer-24
residual stream, plus an RL **length-penalty** sweep. An NLA is a pair of models:

- **AV** (Activation Verbalizer): reads a residual-stream activation and writes a
  short natural-language explanation of it.
- **AR** (Activation Reconstructor / critic): reads that explanation text and
  predicts back the activation. Reconstruction quality = **FVE** (fraction of
  variance explained vs. a predict-the-mean baseline).

We then RL-tuned the AV with a per-token **length penalty**
(`reward -= λ · response_tokens`) to make explanations shorter, and swept λ.

## What we did

1. **Warm-start (all-LoRA, no Miles/SGLang):** trained the AV (LoRA, CE on the
   explanation with the activation injected at a marker token via a Karvonen
   layer-1 residual hook) and the AR (LoRA on a truncated 25-layer critic
   backbone, normalized-MSE at the last token). Reused the public warm-start
   dataset ([`ceselder/qwen3-8b-nla-L24-finefineweb-100k`](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k));
   activations are regenerated from `detokenized_text_truncated` with one forward pass.
2. **RL sweep** of the length penalty: λ ∈ {0.0 (control), 0.001, 0.002, 0.006,
   0.015, 0.03}, GRPO, LoRA r=16, LR 1e-5, KL-anchored to the SFT policy.
3. **Held-out eval:** 1000 prompts per model (same `idx` = same input across all
   models, for direct comparison), reporting length + FVE.

### The bug worth knowing about

A first pass showed length **flat** across all λ. Cause: batched `generate()`
right-pads every sample in a GRPO group to the group's longest length, so the
per-sample length the penalty acted on was **identical within each group** →
`−λ·length` was a constant that cancels in GRPO's group-relative advantage
`(r−μ)/σ` → **zero length gradient for any λ**. Fixed by trimming the rollout to
the true generated length (first EOS) and slicing the loss to match
(`nla/train_rl_self_contained.py`). After the fix, length responds cleanly to λ.

## Result (held-out, 1000 prompts/model)

| model | λ | mean tokens | FVE | note |
|---|--:|--:|--:|---|
| base (no RL) | — | 126 | 0.532 | warm-start only |
| control RL | 0.0 | 127 | 0.585 | RL, no penalty: same length, +FVE |
| | 0.001 | 92 | 0.592 | −27% length, FVE still up |
| | 0.002 | 74 | 0.570 | −41% |
| | 0.006 | 32 | 0.482 | −75% |
| | 0.015 | 25 | 0.450 | −80% |
| | 0.03 | 14 | 0.225 | −89%, FVE collapses (past the knee) |

Convex tradeoff: gentle λ shortens **for free** (RL compensates and even lifts
FVE); λ ≥ 0.006 trades FVE for brevity; λ = 0.03 is the cliff. The control
(λ=0, same length as base) isolates that the shortening comes from the *penalty*,
not RL itself.

## Artifacts on HuggingFace (all public)

| repo | what |
|---|---|
| [`syvb/nanonla-qwen3-8b-L24-av`](https://huggingface.co/syvb/nanonla-qwen3-8b-L24-av) | AV verbalizer (merged Qwen3-8B) |
| [`syvb/nanonla-qwen3-8b-L24-ar`](https://huggingface.co/syvb/nanonla-qwen3-8b-L24-ar) | AR critic (`NLACriticModel`, truncated 25-layer + value head) |
| [`syvb/nanonla-qwen3-8b-L24-rl-lora`](https://huggingface.co/syvb/nanonla-qwen3-8b-L24-rl-lora) | RL LoRA adapters, one subfolder per λ (`p0.0`, `p0.001`, `p0.002`, `p0.006`, `p0.015`, `p0.03`) |
| [`syvb/nanonla-qwen3-8b-L24-data-full`](https://huggingface.co/datasets/syvb/nanonla-qwen3-8b-L24-data-full) | warm-start parquets with regenerated `activation_vector` (configs: `av_sft`/`ar_sft`/`rl`) |
| [`syvb/nanonla-qwen3-8b-L24-results`](https://huggingface.co/datasets/syvb/nanonla-qwen3-8b-L24-results) | held-out completions jsonl + summaries + RESULTS/comparison/plot |
| [`syvb/nanonla-qwen3-8b-L24-completions`](https://huggingface.co/datasets/syvb/nanonla-qwen3-8b-L24-completions) | 7×1000 completions in one parquet (matched by `idx`), with FVE/NMSE/length/text |

## How to use the trained AV / AR

Needs the `nla` package from this repo (`pip install -e .`) plus `torch`,
`transformers`, `peft`, `datasets`. The primitives below mirror the proven
inference path in `vast_ops/eval_heldout.py`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_RE, normalize_activation, resolve_target_scale

DEV = "cuda"
AV_REPO = "syvb/nanonla-qwen3-8b-L24-av"
AR_REPO = "syvb/nanonla-qwen3-8b-L24-ar"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
cfg = load_nla_config(AV_REPO, tok)            # injection token IDs, prompt templates, scales

# ---------- AV: verbalize an activation ----------
av = AutoModelForCausalLM.from_pretrained(
    AV_REPO, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
# OPTIONAL: stack a length-penalty adapter for shorter explanations.
# subfolder ∈ {p0.0, p0.001, p0.002, p0.006, p0.015, p0.03}; omit for the base AV.
av = PeftModel.from_pretrained(av, "syvb/nanonla-qwen3-8b-L24-rl-lora", subfolder="p0.006")

# Karvonen injection hook: overwrite the residual after layer 1 at the marker token.
inj, Ln, Rn = cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id
state, vec = {"ids": None}, [None]
def _emb(m, a, kw, out):
    state["ids"] = (kw.get("input") if kw else None); state["ids"] = state["ids"] if state["ids"] is not None else (a[0] if a else None); return out
def _layer(m, a, out):
    resid = out[0] if isinstance(out, tuple) else out
    if state["ids"] is None or vec[0] is None or (state["ids"] == inj).sum() == 0: return out
    new = karvonen_inject_in_residual(state["ids"], resid, vec[0], inj, Ln, Rn)
    return (new, *out[1:]) if isinstance(out, tuple) else new
av.get_input_embeddings().register_forward_hook(_emb, with_kwargs=True)
_b = av.base_model if hasattr(av, "base_model") else av
while hasattr(_b, "model") and not hasattr(_b, "layers"): _b = _b.model
_b.layers[1].register_forward_hook(_layer)

# Get an activation + its prompt. (Any layer-24 residual works; here a held-out sample.)
row = load_dataset("syvb/nanonla-qwen3-8b-L24-data-full", "rl", split="train")[25000]
activation = torch.tensor(row["activation_vector"], dtype=torch.float32).unsqueeze(0).to(DEV)  # [1, d_model]
# The prompt is a chat message ending in "<concept><INJECT></concept>"; sub the marker char.
msgs = [{**m, "content": m["content"].replace("<INJECT>", cfg.injection_char)} for m in row["prompt"]]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEV)

vec[0] = activation                            # arm the hook for this generation
out = av.generate(ids, max_new_tokens=160, do_sample=True, temperature=1.0,
                  top_p=1.0, top_k=0, repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
vec[0] = None
text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
explanation = EXPLANATION_RE.search(text).group(1).strip()
print("explanation:", explanation)

# ---------- AR: reconstruct the activation from the explanation, score FVE ----------
ar = NLACriticModel.from_pretrained(AR_REPO, torch_dtype=torch.bfloat16).to(DEV).eval()
scale = resolve_target_scale(cfg.mse_scale, cfg.d_model)
crit_ids = tok(cfg.critic_prompt_template.format(explanation=explanation),
               add_special_tokens=True, return_tensors="pt").to(DEV)
with torch.no_grad():
    pred = ar(input_ids=crit_ids.input_ids).values[0, -1].float()       # last-token reconstruction
mse = torch.nn.functional.mse_loss(
    normalize_activation(pred[None], scale)[0],
    normalize_activation(activation[0][None].float(), scale)[0]).item()
print(f"reconstruction FVE = {1 - mse / 0.6704:.3f}")   # baseline mse_nrm = 0.6704 for this dataset
```

**To verbalize your OWN activation:** capture a layer-24 residual from Qwen3-8B
(the `nla.datagen.extractors.HFExtractor` / `tools/regenerate_activations.py`
take text → raw activation), pass it as `activation`, and reuse the prompt
template above (any user message ending in `<concept>㊗</concept>`). The AR loads
**only** with `NLACriticModel.from_pretrained` (it has a non-standard truncated
backbone + separate `value_head.safetensors`); a plain `AutoModel` would mis-load it.

## Reproduce / extend

The full pipeline is in `vast_ops/` (see `vast_ops/README.md`): `00_setup` →
`01_fetch_data` → `01b_regen_activations` → `02_av_sft` / `03_ar_sft` →
`05_rl_sweep` → `06_upload`. The length-penalty knob is `--length-penalty` in
`nla/train_rl_self_contained.py`; warm-start is `nla/train_sft_self_contained.py`.
wandb: project `octahedral-systems/nla-lenpen`.
