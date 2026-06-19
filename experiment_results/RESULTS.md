# Length penalty vs reconstruction — from-scratch nanoNLA (Qwen3-8B, L24)

Held-out reconstruction quality vs the AV's explanation length, swept over the per-token length penalty `λ` (reward −= λ·response_tokens). FVE = fraction of variance explained (1 − NMSE); baseline = predict-the-mean (mse_nrm = 0.6704). Each model evaluated on 1000 held-out prompts.

## Per-model held-out
| model (λ) | mean tok | FVE | NMSE | FVE retained vs base | extraction |
|---|--:|--:|--:|--:|--:|
| base | 126 | 0.532 | 0.468 | 100% | 99% |
| p0.0 | 127 | 0.585 | 0.415 | 110% | 99% |
| p0.001 | 92 | 0.592 | 0.408 | 111% | 100% |
| p0.002 | 74 | 0.570 | 0.430 | 107% | 100% |
| p0.006 | 32 | 0.482 | 0.518 | 91% | 100% |
| p0.015 | 25 | 0.450 | 0.550 | 85% | 100% |
| p0.03 | 14 | 0.225 | 0.775 | 42% | 100% |

## Marginal tradeoff (each step = next-shorter model)
| from → to | Δtok (saved) | Δlen | ΔFVE (lost) | FVE per token saved |
|---|--:|--:|--:|--:|
| p0.0 → base | 1 | −1% | +0.053 | 0.0561 |
| base → p0.001 | 34 | −27% | -0.060 | -0.0018 |
| p0.001 → p0.002 | 18 | −19% | +0.022 | 0.0012 |
| p0.002 → p0.006 | 42 | −57% | +0.089 | 0.0021 |
| p0.006 → p0.015 | 7 | −21% | +0.031 | 0.0047 |
| p0.015 → p0.03 | 11 | −44% | +0.225 | 0.0201 |

## Endpoints
- Longest model **p0.0**: 127 tok, FVE 0.585
- Shortest model **p0.03**: 14 tok, FVE 0.225
- Overall: −113 tok (89%) costs +0.360 FVE → 0.0032 FVE per token saved on average.

## Notes
- `λ` is per **token** (GRPO normalizes advantages within each prompt group, so only the length spread matters — a single coefficient).
- `base` = AV-SFT + AR-SFT with **no RL**. The `p0.0` row (if present) is control RL with zero penalty — isolates the penalty's effect from RL itself.
- From-scratch nanoNLA sits at lower absolute FVE than continue-RL from a released checkpoint; the tradeoff *shape* is the result of interest.

## Artifacts (HuggingFace, public)

- Models: [`-av`](https://huggingface.co/syvb/nanonla-qwen3-8b-L24-av) (verbalizer), [`-ar`](https://huggingface.co/syvb/nanonla-qwen3-8b-L24-ar) (reconstructor/critic), [`-rl-lora`](https://huggingface.co/syvb/nanonla-qwen3-8b-L24-rl-lora) (RL LoRA adapters, one subfolder per λ)
- Datasets: [`-data-full`](https://huggingface.co/datasets/syvb/nanonla-qwen3-8b-L24-data-full) (warm-start parquets with regenerated activations), [`-results`](https://huggingface.co/datasets/syvb/nanonla-qwen3-8b-L24-results) (held-out jsonl + this markdown), [`-completions`](https://huggingface.co/datasets/syvb/nanonla-qwen3-8b-L24-completions) (1000 completions/model with FVE, for analysis)
- Code: branch `length-penalty` on `syvb/nanoNLA` (the `--length-penalty` knob + the padding-bug fix + `vast_ops/` harness)

## Method note (the bug that mattered)

A first pass showed length **flat** across all λ. Root cause: batched `generate()` right-pads every sample in a GRPO group to the group's longest length, so the per-sample length the penalty acted on was **identical within each group** → `−λ·length` was a constant that cancelled in the group-relative advantage `(r−μ)/σ` → zero length gradient for any λ. Fixed by trimming the rollout to the true generated length (first EOS) and slicing the loss to match. After the fix, length responds cleanly and monotonically to λ (the table above).
