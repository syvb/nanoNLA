"""Co-train vs frozen comparison report.

Reads the FROZEN sweep summaries (experiment_results/heldout/) and the CO-TRAINED
summaries (experiment_results/heldout_ct/) and emits a markdown table isolating
the effect of co-training the AR critic alongside the AV during RL.

Two co-trained evals per penalty:
  *_ct_baseAR : scored by the FROZEN base AR (same ruler as the frozen sweep)
                -> "did the AV itself get better?" — directly comparable.
  *_ct_ownAR  : scored by the model's OWN co-trained AR
                -> the real co-trained NLA system FVE (AV+AR moved together).

All FVE on the same predict-the-mean baseline (0.6704), all n=1000 held-out
(rows 25k:26k), doc-disjoint from RL train. Deltas flagged significant when the
gap exceeds the 95% CI of the difference (1.96*sqrt(sem_a^2+sem_b^2)).
"""
import argparse
import glob
import json
import math
import os


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.summary.json")):
        s = json.load(open(f))
        out[s["tag"]] = s
    return out


def sig(a, b):
    """Is a['fve'] - b['fve'] significant at 95%? Returns (delta, ci, flag)."""
    da, db = a.get("fve_sem"), b.get("fve_sem")
    delta = a["fve"] - b["fve"]
    if not (isinstance(da, (int, float)) and isinstance(db, (int, float))):
        return delta, None, ""
    ci = 1.96 * math.sqrt(da * da + db * db)
    flag = "**sig**" if abs(delta) > ci else "ns"
    return delta, ci, flag


def fmt(s):
    t = f"{s['mean_tokens']:.1f}"
    ts = s.get("mean_tokens_sem")
    if isinstance(ts, (int, float)):
        t += f"±{1.96*ts:.1f}"
    f = f"{s['fve']:.3f}"
    fs = s.get("fve_sem")
    if isinstance(fs, (int, float)):
        f += f"±{1.96*fs:.3f}"
    return t, f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-dir", default="experiment_results/heldout")
    p.add_argument("--cotrain-dir", default="experiment_results/heldout_ct")
    p.add_argument("--out", default="experiment_results/COTRAIN_COMPARISON.md")
    args = p.parse_args()

    fz = load(args.frozen_dir)
    ct = load(args.cotrain_dir)

    L = []
    w = L.append
    w("# Co-training the AR critic during RL — does it matter?\n")
    w("Re-ran two length-penalty points (λ=0.006, λ=0.015) with `--train-critic` "
      "(AR co-trained alongside the AV), against the original **frozen-critic** sweep. "
      "Everything else identical. All numbers: held-out n=1000 (rows 25k:26k, doc-disjoint "
      "from RL train), FVE on the shared predict-the-mean baseline 0.6704. "
      "± = 95% CI on the mean.\n")
    w("Two rulers for the co-trained models:\n")
    w("- **base-AR**: scored by the *frozen* base AR — same ruler as the frozen sweep → "
      "isolates whether the **AV itself** improved (directly comparable to frozen).")
    w("- **own-AR**: scored by the model's *own co-trained* AR → the real co-trained "
      "**NLA system** FVE (AV+AR moved together).\n")

    w("| λ | model | mean tokens | FVE | ΔFVE vs frozen | sig? |")
    w("|---|-------|-------------|-----|----------------|------|")
    for pen in ["0.006", "0.015"]:
        slug = f"p{pen}"
        frow = fz.get(slug)
        if frow:
            t, f = fmt(frow)
            w(f"| {pen} | frozen (base AR) | {t} | {f} | — | — |")
        for ruler in ["baseAR", "ownAR"]:
            tag = f"{slug}_ct_{ruler}"
            crow = ct.get(tag)
            if not crow:
                w(f"| {pen} | co-train ({ruler}) | _pending_ | _pending_ | | |")
                continue
            t, f = fmt(crow)
            delta, ci, flag = sig(crow, frow) if frow else (None, None, "")
            ds = f"{delta:+.3f}" + (f" (±{ci:.3f})" if ci is not None else "") if delta is not None else ""
            w(f"| {pen} | co-train ({ruler}) | {t} | {f} | {ds} | {flag} |")
        w("| | | | | | |")

    # base (no-RL) anchor for context
    if "base" in fz:
        t, f = fmt(fz["base"])
        w(f"\n_Context: base NLA (no RL) = {t} tokens, FVE {f}._\n")

    w("\n## Reading this\n")
    w("- **base-AR column** answers the user's question directly: with the same frozen "
      "ruler, did co-training the critic produce a better AV at the same length? "
      "A significant +ΔFVE here = co-training helps the policy.")
    w("- **own-AR column** is the co-trained system as it would actually be deployed "
      "(both halves trained). It can read higher partly because the AR adapted to this "
      "AV's style — legitimate as an absolute held-out number, but not a like-for-like "
      "isolation of the AV.")
    w("\n### Caveats (from pre-run review)\n")
    w("- The co-trained arm used `--gradient-checkpointing` (memory headroom for the "
      "co-trained 5.9B critic); the frozen arm did not. GC is a recompute/memory trade-off "
      "and is numerically neutral up to bf16 rounding, so it cannot account for an FVE "
      "delta — noted for full transparency.")
    w("- The wandb in-loop FVE *curves* are not comparable between arms (the co-trained "
      "arm's curve is scored by a moving critic). Only these final held-out snapshots are.")

    txt = "\n".join(L) + "\n"
    open(args.out, "w").write(txt)
    print(txt)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
