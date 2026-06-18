"""Aggregate the held-out summaries into RESULTS.md: how much FVE you sacrifice
per token of length reduction (the length<->FVE tradeoff). Mirrors the main
repo's RESULTS.md. Reads the *.summary.json written by eval_heldout.py.
"""

import argparse
import glob
import json
import os
from pathlib import Path


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
    p.add_argument("--base", default="base")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    summaries = [json.load(open(f)) for f in glob.glob(os.path.join(args.heldout_dir, "*.summary.json"))]
    if not summaries:
        raise SystemExit(f"no *.summary.json in {args.heldout_dir}")
    # order by penalty (base first), which is also ~descending token count
    summaries.sort(key=lambda s: pen_of(s["tag"]))
    # drop models that produced no scorable samples (fve is null) from the
    # quantitative tables — they'd poison the arithmetic; note them instead.
    broken = [s["tag"] for s in summaries if not isinstance(s.get("fve"), (int, float))]
    summaries = [s for s in summaries if isinstance(s.get("fve"), (int, float))]
    if not summaries:
        raise SystemExit("no models with a valid FVE")
    base = next((s for s in summaries if s["tag"] == args.base), summaries[0])
    base_fve = base["fve"]

    L = []
    L.append("# Length penalty vs reconstruction — from-scratch nanoNLA (Qwen3-8B, L24)\n")
    L.append("Held-out reconstruction quality vs the AV's explanation length, swept "
             "over the per-token length penalty `λ` (reward −= λ·response_tokens). "
             "FVE = fraction of variance explained (1 − NMSE); baseline = "
             f"predict-the-mean (mse_nrm = {base['fve_baseline']:.4f}). "
             f"Each model evaluated on {base['n']} held-out prompts.\n")

    L.append("## Per-model held-out")
    L.append("| model (λ) | mean tok | FVE | NMSE | FVE retained vs base | extraction |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for s in summaries:
        ret = (s["fve"] / base_fve) if base_fve else float("nan")
        L.append(f"| {s['tag']} | {s['mean_tokens']:.0f} | {s['fve']:.3f} | {s['nmse']:.3f} | "
                 f"{ret:.0%} | {s['extraction_rate']:.0%} |")
    L.append("")

    # marginal tradeoff between consecutive models (sorted longest -> shortest)
    chain = sorted(summaries, key=lambda s: -s["mean_tokens"])
    L.append("## Marginal tradeoff (each step = next-shorter model)")
    L.append("| from → to | Δtok (saved) | Δlen | ΔFVE (lost) | FVE per token saved |")
    L.append("|---|--:|--:|--:|--:|")
    for a, b in zip(chain, chain[1:]):
        dtok = a["mean_tokens"] - b["mean_tokens"]
        dlen = (dtok / a["mean_tokens"]) if a["mean_tokens"] else float("nan")
        dfve = a["fve"] - b["fve"]
        per = (dfve / dtok) if dtok else float("nan")
        L.append(f"| {a['tag']} → {b['tag']} | {dtok:.0f} | −{dlen:.0%} | {dfve:+.3f} | {per:.4f} |")
    L.append("")

    # base -> shortest overall
    if len(chain) >= 2:
        a, b = chain[0], chain[-1]
        dtok = a["mean_tokens"] - b["mean_tokens"]
        dfve = a["fve"] - b["fve"]
        L.append("## Endpoints")
        L.append(f"- Longest model **{a['tag']}**: {a['mean_tokens']:.0f} tok, FVE {a['fve']:.3f}")
        L.append(f"- Shortest model **{b['tag']}**: {b['mean_tokens']:.0f} tok, FVE {b['fve']:.3f}")
        if dtok:
            L.append(f"- Overall: −{dtok:.0f} tok ({dtok/a['mean_tokens']:.0%}) costs "
                     f"{dfve:+.3f} FVE → {dfve/dtok:.4f} FVE per token saved on average.")
        L.append("")

    if broken:
        L.append(f"> ⚠️ models with no scorable samples (extraction collapsed), excluded: {', '.join(broken)}\n")
    L.append("## Notes")
    L.append("- `λ` is per **token** (GRPO normalizes advantages within each prompt "
             "group, so only the length spread matters — a single coefficient).")
    L.append("- `base` = AV-SFT + AR-SFT with **no RL**. The `p0.0` row (if present) "
             "is control RL with zero penalty — isolates the penalty's effect from RL itself.")
    L.append("- From-scratch nanoNLA sits at lower absolute FVE than continue-RL from a "
             "released checkpoint; the tradeoff *shape* is the result of interest.")
    L.append("")

    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out}  ({len(summaries)} models)")


if __name__ == "__main__":
    main()
