# Tips for running nanoNLA experiments (infra + gotchas)

Hard-won notes from running a from-scratch NLA + RL experiment on Qwen3-8B.
General-purpose — not specific to any one study.

## Environment / infra

- **You may not need Miles/SGLang.** Miles is only used by the *SFT warm-start*
  path in this repo; the RL trainer (`train_rl_self_contained.py`) is already
  standalone, and you can do SFT standalone too (`train_sft_self_contained.py`,
  all-LoRA). For small models that path needs only `torch + transformers + peft
  + pyarrow` on a stock PyTorch image — no conda, SGLang, flash-attn, or Miles.
  Skip the heavy stack unless you actually run SGLang rollouts.
- **Pin transformers + peft together.** `transformers==4.53.2` + `peft==0.15.2`
  is a known-good pair: 4.53 still exports `HybridCache` (peft imports it) *and*
  supports Qwen3. Newer transformers dropped `HybridCache` → `ImportError` from
  peft. Don't `pip install -U` blindly.
- **No flash-attn needed.** Use `attn_implementation="sdpa"` everywhere (SFT, RL,
  eval, extraction). Avoids a fragile/OOM-prone source build.
- **vast.ai teardown:** `echo y | vastai destroy instance <id>` — it prompts for
  confirmation, so a backgrounded/un-piped destroy silently *aborts*. Always
  verify afterwards (`vastai show instance <id>` errors when gone, or
  `show instances-v1` → `Total: 0`). The legacy `vastai show instances` is
  deprecated and errors out — a grep over its (failed) output can falsely read
  as "no instances". **Confirm teardown explicitly; an idle box bills silently.**
- **Freeing GPUs between runs:** `pkill` of a trainer doesn't always release VRAM
  immediately (orphaned children). Force it:
  `for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done`
  then re-check `memory.free` before relaunching, or the next job OOMs.
- **Run long jobs under `nohup`** and poll from outside (a loop that exits on a
  milestone file / process death). Don't hold an interactive session for hours.

## Data

- **The published warm-start datasets are SLIM** — no `activation_vector` column.
  Activations are regenerated from `detokenized_text_truncated` with one forward
  pass (`tools/regenerate_activations.py`). Budget GPU time: this is slower than
  expected (long source prefixes), and only regenerate the rows you'll use.
- **`prompt` ≠ source text.** The `prompt` column is the fixed verbalizer
  *instruction* (identical every row, e.g. "You are a meticulous AI
  researcher…"). The actual document being explained is
  `detokenized_text_truncated`, and the activation sits at its **final token**.
  Don't log the prompt as your "source text".
- **The sidecar (`nla_meta.yaml`) is the contract** — injection token IDs, prompt
  templates, `injection_scale`, `mse_scale`, `d_model`. Keep it beside every
  parquet and checkpoint; load it, don't hardcode.

## RL training (GRPO) — the load-bearing gotcha

- **Batched `generate()` right-pads every sample in a prompt's group to the
  group's longest length.** So any *per-sample, length-dependent* quantity
  (response length, anything you derive from it) is **identical within a group**
  → it's a constant offset that **cancels exactly** in GRPO's group-relative
  advantage `(r−μ)/σ` → **zero gradient**. We chased a length penalty for hours
  that did nothing for *any* λ or LR because of this. Fix: trim each rollout to
  its **true** generated length (up to the first EOS) before computing per-sample
  rewards/penalties **and** slice the loss to that true length (don't train on
  padding). Symptom of the bug: the quantity won't move and KL stays ~0.
- **LR is the difference between "policy frozen" and "diverged".** `1e-6` (the
  gentle "overnight" config) barely moves the policy (KL ~0.05); `1e-5` is a good
  default; `5e-5` spikes/diverges. Watch `kl_mean`: ~0 = nothing is happening;
  steadily climbing past ~1 = reward-hacking/divergence (check `mean_cjk` and
  extraction rate).
- **Per-step reward/FVE is extremely noisy** (bounces 0.2–0.7 with ~32
  samples/step). Don't conclude anything from a few steps — trust the held-out
  eval.
- **KL is anchored to the SFT policy for free** via `actor.disable_adapter()`
  (LoRA off) — no separate reference model to load.
- **Rollout sampler hygiene:** generate with `output_logits=True` (raw logits)
  and force `top_p=1.0, top_k=0, repetition_penalty=1.0`. Otherwise the sampler
  writes `-inf` into filtered logits → `old_logp = -inf` → `ratio = exp(Δ) = inf`
  → NaN loss. (Qwen3's `generation_config.json` sets these by default.)
- **CJK in generations is the loudest smoke test** that injection failed — grep
  for it.

## Eval

- **Use a fixed held-out slice** (same `skip_rows`/`n_rows`) for every model so
  rows are comparable by `idx`. It's how you get clean side-by-sides.
- **Held-out eval is single-sample generation → slow** (~1h per 1000 samples).
  Plan for it; it's often a bigger time sink than the RL.
- **Guard the degenerate case:** if extraction collapses (0 scorable samples),
  report `fve = null`, never a falsely-perfect `1.0`, and gate on extraction
  rate, not just FVE.
- **Match length semantics** between trainer and eval (count generated tokens the
  same way, incl/excl EOS), or your "length" numbers won't line up.
- **Add an FVE gate before the expensive sweep** — eval the base model first and
  stop (~cheaply) if the warm-start is junk.

## Orchestration / harness

- **Make every stage resumable** (skip if its output exists). You *will* restart.
- **Test the shell orchestrator's file reads, not just the Python.** A filename
  mismatch (`base.summary.json` vs `base.samples.summary.json`) under
  `set -euo pipefail` aborts the *whole run* at a command substitution, after the
  expensive part already ran. `bash -n` won't catch it.
- **`set -e` + `cmd | sort | tail` under `pipefail`** aborts when the pipe's
  first stage errors (e.g. `ls` on a missing dir). Append `|| true` to "best
  effort" lookups.
- **Back up raw logs before teardown** (or rely on wandb, which has the metrics).

## HuggingFace persistence

- **peft adapters won't upload as-is:** peft auto-writes a `README.md` whose
  `base_model:` is your *local* checkpoint path, which HF's card validator
  rejects. Upload with `ignore_patterns=["README.md"]`.
- **Dataset viewer `CastError`:** if a dataset repo mixes files with different
  schemas (e.g. per-sample `*.jsonl` + aggregate `*.summary.json`, or parquets
  with different columns), HF tries to merge them and fails. Add a `configs:`
  block to the dataset `README.md` to scope each config to specific files.
- **`*.png` (and other small non-LFS files) can be updated via the HF commit
  API** with base64-inline content — no `git-lfs`/`huggingface_hub` needed:
  `POST /api/datasets/{repo}/commit/main` with NDJSON
  `{"key":"file","value":{"path":...,"content":<b64>,"encoding":"base64"}}`.
  (Check `.gitattributes` — anything matching an LFS pattern needs the LFS flow.)
- **The AR critic loads only via `NLACriticModel.from_pretrained`**, not
  `AutoModel` — it's a truncated backbone + a separate `value_head.safetensors`.

## Cost/time intuition

- The big time sinks are **activation regeneration** and **single-sample held-out
  eval**, not the RL steps. Budget accordingly.
- **Aggressive runs finish faster** — shorter generations mean fewer tokens to
  decode per step, so high-penalty / short-output configs are cheaper wall-clock.
