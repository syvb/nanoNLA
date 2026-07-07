# RL training signal per token position: reconstruction vs KL (Qwen3-8B NLA)

How much does each RL loss component contribute at each token position of a
rollout from the Qwen3-8B nanoNLA? Companion analysis to the same
decomposition on the nla-doll v3 matryoshka NLA (branch
`v3-reward-kl-decomposition` in `kitft/natural_language_autoencoders`'s
`syvb` fork) — this NLA trains WITHOUT truncation, so the reward is only ever
computed on the full explanation.

In the length-penalty sweep that produced `syvb/nanonla-qwen3-8b-L24-rl-lora`
(`vast_ops/05_rl_sweep.sh` + `nla/train_rl_self_contained.py`, analysed arm =
**p0.0**, zero length penalty = the vanilla RL NLA):

- **Reconstruction** is sequence-level: `reward = −mse_nrm(critic(explanation),
  gold)` on the full `<explanation>…</explanation>` payload (frozen AR critic,
  `critic_predict` normalize-before-value-head path; failed extraction −2.0).
  GRPO broadcasts the group-normalized advantage over all response tokens, so
  its per-position structure is the counterfactual marginal reward
  **Δr(t) = r(t) − r(t−1)** of the first-t-token prefix.
- **KL** is genuinely per-token: `kl_beta · k3_t`, `k3 = exp(δ) − δ − 1`,
  `δ = log π_ref(x_t) − log π(x_t)` at the sampled token; ref = the merged
  AV-SFT with the RL LoRA **disabled** (exactly the training anchor),
  **kl_beta = 0.01**.

## Method (`sweep_reward_kl.py`, on-box)

For N=200 held-out rows (`rl_shuf` rows ≥ 25000, the sweep's eval convention;
`activation_vector` regenerated with `tools/regenerate_activations.py`):

1. batched rollouts at **T=1** (top_p=1, top_k=0), `max_new=160` (= RL_MAX_NEW),
   Karvonen layer-1 hook injection, policy = `syvb/nanonla-qwen3-8b-L24-av` +
   `rl-lora/p0.0`;
2. teacher-force each sequence with the adapter ON and OFF → per-position k3
   (the trained estimator), k1, and exact full-vocab KL;
3. critic-score **every response-token prefix**, extracting open-tolerantly
   (payload after `<explanation>`, up to `</explanation>` if present;
   contentless → the −2.0 floor) so the reward axis aligns 1:1 with the KL
   axis. Positions before the opening tag finishes sit at the floor — the
   marginal plots start after the median content start.

Caveats: prefix rewards are counterfactual (training never scored prefixes —
unlike the matryoshka NLA there is no exposure weighting to apply); the
raw-units comparison is the reward-vs-penalty *budget*, not the literal
gradient ratio (the reconstruction advantage is GRPO-group-normalized, the KL
term is not); end-of-training snapshot (KL accumulated over 250 steps).

## Run

```bash
# box: 1× H100 80GB, stock pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel image
rsync repo → /workspace/nanoNLA ; put HF token at /root/.hf_token
bash setup_box.sh                       # downloads + activation regen
cd /workspace/nanoNLA && python experiments/reward_kl_decomposition/sweep_reward_kl.py 200
# pull /workspace/out/rkl/* → results/ ; locally:
python plot_reward_kl.py
```

## Results (N=200 held-out rows, T=1, seed 0, 2026-07-07)

Figures: `results/fig_reward_kl_pertoken.png`, `results/fig_reward_kl_cumulative.png`.
Rollouts: median 125 tokens, 199/200 end with natural EOS, CJK 3/200.
Median content start t=9 (the `<explanation>\n` opening); before it the prefix
reward sits at the −2.0 floor by construction.

| t | r(t) | Δr(t) | k3(t) | β·k3(t) |
|---|---|---|---|---|
| 9 (content start) | −0.903 | (+1.10 vs floor) | 0.101 | 0.0010 |
| 15 | −0.770 | +0.007 | 0.053 | 0.0005 |
| 30 | −0.706 | +0.000 | 0.047 | 0.0005 |
| 50 | −0.580 | +0.018 | 0.067 | 0.0007 |
| 80 | −0.401 | +0.008 | 0.032 | 0.0003 |
| 110 | −0.299 | +0.000 | 0.030 | 0.0003 |
| 125 (median len) | −0.301 | −0.000 | 0.008 | 0.0001 |

**The mirror image of the truncation-trained (matryoshka) NLA:**

- **Reconstruction is NOT front-loaded.** Reward accrues steadily across the
  whole explanation: −0.84 right after the tag → −0.58 @50 → −0.40 @80 →
  −0.30 @110, with marginals staying meaningfully positive deep into the
  sequence (bumpy, feature-by-feature — visible as waves in Δr(t) as each
  new list item lands). The doll v3 matryoshka NLA earns ~90% of its reward
  in the first ~10 tokens; this NLA spreads it over ~110.
- **KL divergence from the SFT anchor is tiny everywhere**: mean k3 ≈ 0.035
  nats/token (median ~0.005 — most tokens barely moved; p99 ≈ 0.5), vs ~1.9
  mean for the doll v3 — **~55× less drift**. The largest per-position KL is
  at the first content token (k3 ≈ 0.10 @t=9): RL moved *what the
  explanation opens with* the most, a miniature of v3's early-token
  restructuring.
- **Reconstruction dominates the budget everywhere.** β·k3 stays ≈
  3–7×10⁻⁴/token, below the marginal reconstruction gain until t≈110 —
  essentially the entire explanation pays for itself. Cumulative over the
  response: ~0.59 reward gained vs ~0.05 KL penalty paid (**~12× in favor of
  reconstruction** — the doll v3 was ~7–11× in favor of *KL*).
- Sanity: k1 ≈ k3 ≈ exact full-vocab KL at every position band; mean
  full-length reward −0.289 (FVE ≈ 0.60 against var 0.72), consistent with
  the sweep's reported post-RL numbers. Past t≈125 the curves get noisy —
  few rollouts remain alive (n_alive ≤ 105 → 2).

Why so different from the matryoshka run: (1) no truncation pressure — the
reward never asks early prefixes to reconstruct, so RL has no reason to
restructure early tokens; (2) short RL (250 steps, LoRA r=16, lr 1e-5) from
an already-decent SFT vs 200 full-FT steps with a co-trained critic; (3) the
frozen critic can't co-adapt, capping how much reward the policy can extract
by drifting. Same GRPO-normalization caveat as the doll analysis: this is
the reward-vs-penalty budget in raw units, not the literal gradient ratio.
