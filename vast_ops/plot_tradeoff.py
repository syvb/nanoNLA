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

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([x["mean_tokens"] for x in rl], [x["fve"] for x in rl], "o-",
            color="#3060c0", label="RL (length penalty)", zorder=3)
    for x in rl:
        ax.annotate(lbl(x), (x["mean_tokens"], x["fve"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    if base is not None:
        ax.plot([base["mean_tokens"]], [base["fve"]], marker="*", markersize=16,
                color="#d04020", linestyle="none", label="base (no RL)", zorder=4)
        ax.annotate("base", (base["mean_tokens"], base["fve"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8, color="#d04020")
    ax.set_xlabel("mean explanation length (generated tokens)")
    ax.set_ylabel("FVE (fraction of variance explained)")
    ax.set_title("Length penalty: explanation length vs reconstruction FVE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
