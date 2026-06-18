"""Build a side-by-side markdown comparing base NLA vs each length-penalty model
on the SAME held-out prompts. Reads the jsonl dumps from eval_heldout.py and
matches samples by `idx` (every model scored the same rows in the same order).

Per example: source context, then one row per model with token count, NMSE, FVE,
and the explanation text — so you can see length shrink as quality degrades.
"""

import argparse
import json
from pathlib import Path


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def pen_of(tag):
    # "base" sorts first; "p0.002" -> 0.002
    if tag == "base":
        return -1.0
    try:
        return float(tag[1:]) if tag.startswith("p") else float(tag)
    except ValueError:
        return 1e9


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout-dir", required=True)
    p.add_argument("--base", default="base")
    p.add_argument("--out", required=True)
    p.add_argument("--n-examples", type=int, default=100)
    args = p.parse_args()

    d = Path(args.heldout_dir)
    files = sorted(d.glob("*.samples.jsonl"), key=lambda f: pen_of(f.name.split(".samples")[0]))
    models = []
    for f in files:
        tag = f.name.split(".samples")[0]
        rows = {r["idx"]: r for r in load(f)}
        # FVE baseline read straight from the summary (same value across models) —
        # robust vs back-solving mse/nmse from a row (which breaks when nmse==0).
        sfile = Path(str(f).replace(".jsonl", ".summary.json"))
        baseline = json.load(open(sfile))["fve_baseline"] if sfile.exists() else None
        models.append((tag, rows, baseline))
    if not models:
        raise SystemExit(f"no *.samples.jsonl in {d}")

    # common idx set across all models
    common = sorted(set.intersection(*[set(r.keys()) for _, r, _ in models]))
    n = min(args.n_examples, len(common))

    lines = []
    lines.append("# Base NLA vs length-penalty — held-out comparison\n")
    lines.append(f"Same {len(common)} held-out prompts scored by every model; "
                 f"{n} examples shown. Metrics: **NMSE** = mse / predict-the-mean "
                 "baseline (= 1 − FVE), **FVE** = fraction of variance explained, "
                 "**tok** = AV's generated token count. Lower NMSE / higher FVE = "
                 "better reconstruction; fewer tokens = shorter explanation.\n")

    # aggregate header
    lines.append("## Aggregate (all common prompts)\n")
    lines.append("| model | mean tok | NMSE | FVE | extraction |")
    lines.append("|---|--:|--:|--:|--:|")
    for tag, rows, baseline in models:
        rr = [rows[i] for i in common]
        valid = [r for r in rr if r["mse"] is not None]
        mean_tok = sum(r["n_tokens"] for r in rr) / len(rr)
        if valid and baseline:
            mean_mse = sum(r["mse"] for r in valid) / len(valid)
            nmse = mean_mse / baseline
            fve = 1 - nmse
        else:
            nmse = fve = None
        lines.append(f"| {tag} | {mean_tok:.0f} | {fmt(nmse)} | {fmt(fve)} | "
                     f"{len(valid)/len(rr):.0%} |")
    lines.append("")

    lines.append("## Examples\n")
    for k in range(n):
        idx = common[k]
        src = models[0][1][idx].get("source_text") or ""
        lines.append(f"### Example {k+1}")
        if src:
            lines.append(f"> _source ctx:_ {src[:300]}\n")
        lines.append("| model | tok | NMSE | FVE | explanation |")
        lines.append("|---|--:|--:|--:|---|")
        for tag, rows, _ in models:
            r = rows[idx]
            expl = (r.get("explanation") or "_<extraction failed>_").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {tag} | {r['n_tokens']} | {fmt(r['nmse'])} | {fmt(r['fve'])} | {expl[:300]} |")
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}  ({n} examples, {len(models)} models)")


if __name__ == "__main__":
    main()
