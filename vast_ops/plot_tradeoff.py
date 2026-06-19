"""Plot the length<->FVE tradeoff curve from the held-out summaries.
Saves a PNG. Pure read of *.summary.json — the jsonl dumps already contain
everything needed to regenerate this offline.
"""

import argparse
import glob
import json
import os


def pen_of(tag):
    if tag == "base":
        return -1.0
    try:
        return float(tag[1:]) if tag.startswith("p") else float(tag)
    except ValueError:
        return 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = [json.load(open(f)) for f in glob.glob(os.path.join(args.heldout_dir, "*.summary.json"))]

    def lbl(x):  # nice label from the penalty
        return "base" if x["tag"] == "base" else f"λ={pen_of(x['tag']):g}"

    # base (no RL) is a STANDALONE point — not on the line. The line connects the
    # RL models (control λ=0 through the largest penalty), sorted by length.
    base = next((x for x in s if x["tag"] == "base"), None)
    rl = sorted([x for x in s if x["tag"] != "base"], key=lambda x: x["mean_tokens"])

    # 95% CI on the means (1.96 * SEM) if the summaries carry the SEM fields.
    def ci(x, key):
        v = x.get(key)
        return 1.96 * v if isinstance(v, (int, float)) else None
    # Only FVE (y) error bars — the length (x) SEM is sub-token, so x bars would be
    # invisible and only clutter the legend with a horizontal stub.
    have_err = all(ci(x, "fve_sem") is not None for x in s)

    fig, ax = plt.subplots(figsize=(7, 5))
    rt, rf = [x["mean_tokens"] for x in rl], [x["fve"] for x in rl]
    if have_err:
        ax.errorbar(rt, rf, yerr=[ci(x, "fve_sem") for x in rl], fmt="o-", color="#3060c0",
                    ecolor="#3060c0", elinewidth=1, capsize=3, label="RL (length penalty)", zorder=3)
    else:
        ax.plot(rt, rf, "o-", color="#3060c0", label="RL (length penalty)", zorder=3)
    # per-point label offsets (dx, dy, ha) chosen to sit clear of the line
    OFF = {"p0.03": (12, -4, "left"), "p0.015": (12, -13, "left"), "p0.006": (12, -16, "left"),
           "p0.002": (-2, 11, "right"), "p0.001": (4, 12, "left"), "p0.0": (-12, 11, "right")}
    for x in rl:
        dx, dy, ha = OFF.get(x["tag"], (8, 8, "left"))
        ax.annotate(lbl(x), (x["mean_tokens"], x["fve"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=11)
    if base is not None:
        bx, by = base["mean_tokens"], base["fve"]
        if have_err:
            ax.errorbar([bx], [by], yerr=[ci(base, "fve_sem")],
                        fmt="*", markersize=10, color="#d04020", ecolor="#d04020",
                        elinewidth=1, capsize=3, linestyle="none", label="base (no RL)", zorder=4)
        else:
            ax.plot([bx], [by], marker="*", markersize=10, color="#d04020",
                    linestyle="none", label="base (no RL)", zorder=4)
        ax.annotate("base", (bx, by), textcoords="offset points", xytext=(-12, -6),
                    ha="right", fontsize=11, color="#d04020")
    ax.set_xlabel("mean explanation length (generated tokens)")
    ax.set_ylabel("FVE (fraction of variance explained)")
    ttl = "Length penalty: explanation length vs reconstruction FVE"
    if have_err:
        ttl += "\n(error bars = 95% CI on FVE, n=1000)"
    ax.set_title(ttl)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
