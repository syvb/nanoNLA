"""Figures for the nanoNLA Qwen3-8B reward/KL per-token-position decomposition.
Local, matplotlib only — reads aggregate.csv (+ per_sample.npz for the
content-start marker) produced by sweep_reward_kl.py.

Both series are in the same units (reward per token at position t):
  reconstruction = marginal reward dr(t) = r(t) - r(t-1)
  KL penalty     = beta * k3(t)   (beta = 0.01, the sweep's --kl-beta)
Positions up to the median content start (the "<explanation>\n" opening) sit
on the -2.0 failed-extraction floor, so the first content token's dr is a
floor artifact — the marginal panel starts after the median content start.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("RESULTS", os.path.join(HERE, "results"))
BETA = float(os.environ.get("KL_BETA", "0.01"))
XCAP = int(os.environ.get("XCAP", "160"))
WIN = int(os.environ.get("WIN", "5"))

C_REC = "#1f77b4"
C_KL = "#d62728"
C_KLX = "#ff9896"

rows = list(csv.DictReader(open(os.path.join(R, "aggregate.csv"))))
t = np.array([int(r["t"]) for r in rows])
r_t = np.array([float(r["mean_reward"]) for r in rows])
dr = np.array([float(r["mean_marginal"]) for r in rows])
k1 = np.array([float(r["mean_k1"]) for r in rows])
k3 = np.array([float(r["mean_k3"]) for r in rows])
klx = np.array([float(r["mean_kl_exact"]) for r in rows])
n_alive = np.array([int(r["n_alive"]) for r in rows])
z = np.load(os.path.join(R, "per_sample.npz"))
cs = z["content_start"]
cs_med = int(np.median(cs[cs >= 0]))


def roll(y, w=WIN):
    if w <= 1:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


m = t <= XCAP
m2 = m & (t > cs_med + 1)  # marginals: skip the opening-tag floor artifact

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(t[m2], roll(dr[m2]), color=C_REC, lw=2.0, label="reconstruction: marginal reward Δr(t)")
a1.plot(t[m], roll(BETA * k3[m]), color=C_KL, lw=2.0, label=f"KL penalty: β·k3(t), β={BETA}")
a1.plot(t[m], roll(BETA * klx[m]), color=C_KLX, lw=1.4, ls="--", label="β·KL_exact(t) (check)")
a1.axvline(cs_med, color="gray", lw=0.8, ls=":", label=f"median content start (t={cs_med})")
a1.axhline(0, color="gray", lw=0.8)
a1.set_xlabel("token position t in the rollout")
a1.set_ylabel("reward units per token")
a1.set_title(f"per-position signal ({WIN}-token rolling mean)")
a1.grid(alpha=.3)
a1.legend(fontsize=9)

a2.plot(t[m], r_t[m], color=C_REC, lw=2.0, label="reward r(t) of the first-t-token prefix")
a2.axvline(cs_med, color="gray", lw=0.8, ls=":")
a2.axhline(0, color="gray", lw=0.8)
a2.set_xlabel("prefix length t (response tokens)")
a2.set_ylabel("reward = −mse_nrm")
a2.set_title("prefix reward curve")
a2.grid(alpha=.3)
a2.legend(fontsize=9)
fig.suptitle("nanoNLA Qwen3-8B (p0.0 RL): reconstruction vs KL contribution per token position", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_reward_kl_pertoken.png"), dpi=130, bbox_inches="tight")

fig2, b2 = plt.subplots(figsize=(7, 5))
cum_rec = np.cumsum(np.where(t > cs_med + 1, dr, 0.0))
cum_kl = BETA * np.cumsum(k3)
b2.plot(t[m], cum_rec[m], color=C_REC, lw=2.0, label="Σ Δr (reward gained past content start)")
b2.plot(t[m], cum_kl[m], color=C_KL, lw=2.0, label="Σ β·k3 (total KL penalty paid)")
b2.plot(t[m], (cum_rec - cum_kl)[m], color="k", lw=1.4, ls=":", label="net")
b2.axhline(0, color="gray", lw=0.8)
b2.set_xlabel("rollout length t")
b2.set_ylabel("cumulative reward units")
b2.set_title("nanoNLA Qwen3-8B: cumulative decomposition")
b2.grid(alpha=.3)
b2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(R, "fig_reward_kl_cumulative.png"), dpi=130, bbox_inches="tight")

xover = next((int(tt) for tt, d, k in zip(t[m2], roll(dr[m2]), roll(BETA * k3[m2]))
              if d < k), None)
print(f"median content start: {cs_med}")
print(f"r(cs+1)={r_t[cs_med]:.3f}  r({XCAP})={r_t[m][-1]:.3f}")
print(f"mean beta*k3 per token (t<={XCAP}): {BETA * k3[m].mean():.5f}")
print(f"first position where smoothed beta*k3 exceeds marginal reconstruction: {xover}")
print(f"n_alive at t={XCAP}: {n_alive[m][-1]}")
