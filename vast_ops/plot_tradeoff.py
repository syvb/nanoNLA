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
    s.sort(key=lambda x: x["mean_tokens"])
    toks = [x["mean_tokens"] for x in s]
    fves = [x["fve"] for x in s]
    tags = [x["tag"] for x in s]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(toks, fves, "o-", color="#3060c0")
    for t, f, tag in zip(toks, fves, tags):
        ax.annotate(tag, (t, f), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("mean explanation length (generated tokens)")
    ax.set_ylabel("FVE (fraction of variance explained)")
    ax.set_title("Length penalty: explanation length vs reconstruction FVE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
