# vast_ops — all-LoRA, Miles-free length-penalty sweep (nanoNLA)

Reproduces the length↔FVE tradeoff on a **from-scratch** Qwen3-8B NLA, training
everything with **LoRA and no Miles / SGLang / flash-attn / conda**. The whole
pipeline runs on `torch + transformers + peft + pyarrow` + the `nla` package.

- **AV SFT** (`--role av`): LoRA on the base LM, CE on the explanation, activation
  injected at the `㊗` marker via the Karvonen layer-1 hook (the same hook RL/eval use).
- **AR SFT** (`--role ar`): LoRA on the truncated `NLACriticModel` backbone (value
  head frozen identity), normalized-MSE at the last token.
- **RL sweep**: the self-contained GRPO trainer with `--length-penalty`, KL-anchored
  to the frozen SFT policy (LoRA disabled).

Nothing is written to `/tmp`; everything lives under `$NLA_WORKSPACE`. Persisted
artifacts land in `experiment_results/` and on HuggingFace.

## Pipeline

| stage | script | GPUs | ~time | what |
|---|---|--:|--:|---|
| 0 | `00_setup.sh` | — | ~0.3h | `pip install torch/transformers/peft/...` + `nla` (no conda/Miles/SGLang) |
| 1 | `01_fetch_data.sh` | — | ~0.2h | HF warm-start parquets (+sidecars) |
| 2 | `02_av_sft.sh` | 1 | ~1h | AV LoRA SFT → merged HF at `$AV_HF` |
| 3 | `03_ar_sft.sh` | 1 | ~1h | critic init + AR LoRA SFT → merged at `$AR_HF` |
| 5 | `05_rl_sweep.sh` | N | ~7h | base FVE gate, then RL+eval per penalty fanned across GPUs |
| 6 | `06_upload.sh` | — | ~0.3h | push AV/AR/LoRA + results to HF (private) |

Stages 2 and 3 are independent → `run_all.sh` runs them in parallel on 2 GPUs.
Estimated **~18 GPU-h → ~$50–60** on 2×H100. A built-in **FVE gate** stops the
run (~$15 in) if the warm-start checkpoint is junk.

## Run it (one command)

```bash
export NLA_WORKSPACE=/workspace/nla
git clone git@github-nanonla:syvb/nanoNLA.git "$NLA_WORKSPACE/nanoNLA"
cd "$NLA_WORKSPACE/nanoNLA" && git checkout length-penalty
# ~/.hf_token, ~/.wandb_key are picked up automatically
nohup bash vast_ops/run_all.sh > "$NLA_WORKSPACE/run_all.log" 2>&1 &
```

`run_all.sh` chains 00→06 with per-stage logs in `$NLA_WORKSPACE/logs/`. It is
resumable — every stage skips finished work, so a killed/restarted run continues.

## Knobs (`env.sh`)

- `PENALTIES` — the sweep, in order (default `0.006 0.002 0.001 0.015`). `base`
  (AV-SFT only, no RL) is evaluated for free as the baseline column.
- `SWEEP_GPUS` — how many GPUs to fan the RL runs across (auto-detected).
- `SFT_STEPS` (1000), `SFT_LORA_R` (64) — warm-start LoRA. Higher rank gives the
  LoRA capacity to learn injection-reading.
- `RL_NUM_STEPS` (250), `RL_LR` (1e-6), `RL_KL_BETA` (0.01), `RL_LORA_R` (16).
- `EVAL_N` (1000), `EVAL_SKIP_ROWS` (25000, doc-disjoint from the RL cursor).
- `HF_OWNER`/`HF_PREFIX` — upload target (`syvb/nanonla-qwen3-8b-L24-*`).

## Outputs

Local `experiment_results/`:
- `heldout/<tag>.samples.jsonl` (+ `.summary.json`) — per-sample `n_tokens`, `mse`,
  `nmse`, `fve`, `explanation` for `base`, `p0.006`, `p0.002`, `p0.001`, `p0.015`.
- `comparison_base_vs_penalty.md` — 100 matched side-by-sides.
- `RESULTS.md` — per-model table + marginal FVE-per-token-saved tradeoff.
- `tradeoff.png` — length vs FVE curve.

HuggingFace (private, under `$HF_OWNER`):
- `$HF_PREFIX-av`, `$HF_PREFIX-ar` — merged checkpoints.
- `$HF_PREFIX-rl-lora` — RL LoRA adapters, a subfolder per penalty.
- `$HF_PREFIX-results` (dataset) — the held-out jsonl + markdown + plot.

## Notes

- **All wandb-tracked** under project `$WANDB_PROJECT` (default `nla-lenpen`):
  `sft_av`, `sft_ar`, and `rl_p<λ>` runs.
- **KL anchor**: RL KL is against the frozen SFT policy via `disable_adapter()`
  (LoRA off) — no separate ref model, prevents reward-hack drift.
- **Gradient checkpointing is intentionally off for AV** — it would re-run the
  forward in backward with `vectors_ref` already cleared, silently dropping the
  injection. Lower `SFT_AV_MICRO` if you OOM instead.
