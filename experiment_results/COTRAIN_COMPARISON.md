# Co-training the AR critic during RL — does it matter?

**Verdict: yes, modestly — and it's a Pareto gain at λ=0.006.** Co-training the AR
critic alongside the AV (`--train-critic`) leaves the deployed NLA **system**
(scored by its own co-trained AR) reconstructing *better than the frozen-critic
system at shorter explanation length*: +0.025 FVE (0.507 vs 0.482) at 26 vs 32
tokens for λ=0.006 (significant), same direction but not significant for λ=0.015
(+0.021, 22 vs 25 tokens). It is **not** a free lunch: scored by the *frozen* base
AR, the co-trained AV is significantly *worse* (−0.03 to −0.08 FVE) — so part of
the effect is the AV and critic **co-adapting** to each other, not the AV becoming
unconditionally better. Net: co-training is worth doing (the system improves and
gets terser), but the gain here is small, not a step change.

Re-ran two length-penalty points (λ=0.006, λ=0.015) with `--train-critic` (AR co-trained alongside the AV), against the original **frozen-critic** sweep. Everything else identical. All numbers: held-out n=1000 (rows 25k:26k, doc-disjoint from RL train), FVE on the shared predict-the-mean baseline 0.6704. ± = 95% CI on the mean.

Two rulers for the co-trained models:

- **base-AR**: scored by the *frozen* base AR — same ruler as the frozen sweep → isolates whether the **AV itself** improved (directly comparable to frozen).
- **own-AR**: scored by the model's *own co-trained* AR → the real co-trained **NLA system** FVE (AV+AR moved together).

| λ | model | mean tokens | FVE | ΔFVE vs frozen | sig? |
|---|-------|-------------|-----|----------------|------|
| 0.006 | frozen (base AR) | 31.9±0.2 | 0.482±0.016 | — | — |
| 0.006 | co-train (baseAR) | 26.0±0.2 | 0.451±0.017 | -0.031 (±0.023) | **sig** |
| 0.006 | co-train (ownAR) | 26.0±0.2 | 0.507±0.017 | +0.025 (±0.023) | **sig** |
| | | | | | |
| 0.015 | frozen (base AR) | 25.2±0.3 | 0.450±0.017 | — | — |
| 0.015 | co-train (baseAR) | 22.5±0.3 | 0.375±0.018 | -0.075 (±0.025) | **sig** |
| 0.015 | co-train (ownAR) | 22.4±0.2 | 0.471±0.017 | +0.021 (±0.024) | ns |
| | | | | | |

_Context: base NLA (no RL) = 125.6±0.8 tokens, FVE 0.532±0.018._


## Reading this

- **base-AR column** answers the user's question directly: with the same frozen ruler, did co-training the critic produce a better AV at the same length? A significant +ΔFVE here = co-training helps the policy.
- **own-AR column** is the co-trained system as it would actually be deployed (both halves trained). It can read higher partly because the AR adapted to this AV's style — legitimate as an absolute held-out number, but not a like-for-like isolation of the AV.

### Caveats (from pre-run review)

- The co-trained arm used `--gradient-checkpointing` (memory headroom for the co-trained 5.9B critic); the frozen arm did not. GC is a recompute/memory trade-off and is numerically neutral up to bf16 rounding, so it cannot account for an FVE delta — noted for full transparency.
- The wandb in-loop FVE *curves* are not comparable between arms (the co-trained arm's curve is scored by a moving critic). Only these final held-out snapshots are.
