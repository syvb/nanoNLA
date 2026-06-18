# vast_ops — from-scratch length-penalty sweep (nanoNLA)

Reproduces the length↔FVE tradeoff experiment on a **from-scratch** Qwen3-8B
NLA: train the AV + AR warm-start ourselves (reusing the public warm-start
dataset, skipping datagen), then sweep the RL **length penalty**
(`reward −= λ · response_tokens`, the `--length-penalty` knob added to
`nla/train_rl_self_contained.py`).

All stages are parameterised through `env.sh`; override any default by exporting
before a stage. Nothing is written to `/tmp` — everything lives under
`$NLA_WORKSPACE` (default `/workspace/nla`), and persisted artifacts land in
`experiment_results/` (committed).

## Pipeline

| stage | script | GPUs | ~time | what |
|---|---|--:|--:|---|
| 0 | `00_setup.sh` | — | ~1.5h | Miles@pin + patches, prebuilt flash-attn, stock sglang, peft==0.13 |
| 1 | `01_fetch_data.sh` | — | ~0.5h | pull `av_sft/ar_sft/rl` parquets (+sidecars) from HF |
| 2 | `02_av_sft.sh` | 2 | ~0.5–1h | AV SFT (Karvonen injection), 1000 steps |
| 3 | `03_ar_sft.sh` | 2 | ~1–1.5h | prepare critic init + AR SFT (frozen value head), 1000 steps |
| 4 | `04_convert.sh` | 1 | ~0.3h | DCP→HF for AV and AR |
| 5 | `05_rl_sweep.sh` | 1/run | ~3.25h/penalty | RL per penalty + 1000-sample held-out eval + markdown |

Estimated total ≈ **26 GPU-h → ~$66** at $2.5/H100-hr (≈ **$85–100** with an
infra-debug buffer). The shared phases (0–4) run once; only stage 5 fans out
per penalty and parallelizes across GPUs (1 GPU per RL run).

## Run it

```bash
export NLA_WORKSPACE=/workspace/nla
git clone git@github-nanonla:syvb/nanoNLA.git "$NLA_WORKSPACE/nanoNLA"
cd "$NLA_WORKSPACE/nanoNLA" && git checkout length-penalty
# keys: ~/.hf_token, ~/.wandb_key are picked up automatically if present
bash vast_ops/00_setup.sh
bash vast_ops/01_fetch_data.sh
bash vast_ops/02_av_sft.sh
bash vast_ops/03_ar_sft.sh
bash vast_ops/04_convert.sh
bash vast_ops/05_rl_sweep.sh           # sweeps $PENALTIES, evals, writes markdown
python vast_ops/plot_tradeoff.py --heldout-dir experiment_results/heldout \
    --out experiment_results/tradeoff.png
```

The sweep is **resumable**: a finished RL run (its final LoRA exists) or a
finished eval (its jsonl exists) is skipped on re-run.

## Knobs (`env.sh`)

- `PENALTIES` — the sweep (default `0.0 0.001 0.002 0.004 0.006`). `0.0` = control
  RL (no penalty); `base` (AV-SFT only, no RL) is evaluated for free.
- `RL_NUM_STEPS` (250), `RL_LR` (1e-6), `RL_LORA_R/ALPHA`, `RL_KL_BETA` (0.01) —
  the validated "overnight" config. KL is against the frozen base (LoRA disabled),
  which anchors the policy and prevents reward-hacking degeneration.
- `EVAL_N` (1000), `EVAL_SKIP_ROWS` (25000, doc-disjoint from the RL cursor).
- `AV_GPUS`/`AR_GPUS` (2) — SFT GPU count.

## Outputs (persisted under `experiment_results/`)

- `heldout/<tag>.samples.jsonl` — per-sample `n_tokens`, `mse`, `nmse`, `fve`,
  `explanation`, `source_text` for every model (`base`, `p0.0`, …). Enough to
  rebuild the tradeoff chart offline.
- `heldout/<tag>.summary.json` — aggregate FVE / NMSE / mean tokens per model.
- `comparison_base_vs_penalty.md` — 100 matched side-by-sides.
- `RESULTS.md` — per-model table + marginal FVE-per-token-saved tradeoff.
- `tradeoff.png` — length vs FVE curve.

## Notes / gotchas (carried from the main run + docs)

- **flash-attn**: install a *prebuilt wheel* matching the box's torch/cuda/python;
  a source build OOMs. `00_setup.sh` will error with the expected wheel name.
- **SGLang patches are NOT applied** — SFT is `--debug-train-only` (no server) and
  RL is self-contained (HF `generate`). Stock sglang just satisfies imports.
- **AR stability**: `NLA_FREEZE_VALUE_HEAD=1` (set by stage 3) — without it AR SFT
  NaNs by ~step 29.
- **Disk**: RL writes embedding dumps to `/dev/shm/nla`; the orchestrator cleans
  them between runs. Keep the AV/AR HF checkpoints (~16GB each); LoRA adapters are tiny.
