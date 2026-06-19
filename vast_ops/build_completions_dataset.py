"""Consolidate the per-model held-out completions into ONE analysis dataset and
upload it to HF. Each row = one AV's generated explanation for one held-out
sample, with reconstruction quality (mse/nmse/fve), length, and the source text.

Source: the held-out eval jsonls (base = AV-SFT only; pXXX = AV-SFT + RL LoRA),
1000 completions per model on the SAME fixed random held-out slice (doc-disjoint
from training) so models are directly comparable row-for-row by `idx`.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

COLS = ["model", "length_penalty", "idx", "source_text", "explanation",
        "n_tokens", "mse", "nmse", "fve", "reward", "extracted", "cjk"]


def pen_of(tag):
    if tag == "base":
        return 0.0
    try:
        return float(tag[1:]) if tag.startswith("p") else float(tag)
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout-dir", required=True)
    p.add_argument("--out", required=True, help="output parquet")
    p.add_argument("--repo", required=True, help="HF dataset repo id")
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.heldout_dir, "*.samples.jsonl")))
    rows = {c: [] for c in COLS}
    summary_lines = []
    for f in files:
        tag = os.path.basename(f).split(".samples")[0]
        pen = pen_of(tag)
        n = 0
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            rows["model"].append(tag)
            rows["length_penalty"].append(pen)
            rows["idx"].append(r.get("idx"))
            rows["source_text"].append(r.get("source_text"))
            rows["explanation"].append(r.get("explanation"))
            rows["n_tokens"].append(r.get("n_tokens"))
            rows["mse"].append(r.get("mse"))
            rows["nmse"].append(r.get("nmse"))
            rows["fve"].append(r.get("fve"))
            rows["reward"].append(r.get("reward"))
            rows["extracted"].append(bool(r.get("extracted")))
            rows["cjk"].append(r.get("cjk"))
            n += 1
        # aggregate row for the README
        sfile = f.replace(".jsonl", ".summary.json")
        if os.path.exists(sfile):
            s = json.load(open(sfile))
            fve = s.get("fve")
            summary_lines.append(
                f"| {tag} | {pen} | {s.get('mean_tokens', float('nan')):.0f} | "
                f"{(fve if isinstance(fve,(int,float)) else float('nan')):.3f} | "
                f"{s.get('extraction_rate', float('nan')):.0%} | {n} |")
        print(f"  {tag}: {n} completions", flush=True)

    table = pa.table({c: rows[c] for c in COLS})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(f"wrote {table.num_rows} rows -> {args.out}", flush=True)

    if args.no_upload:
        return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    card = (
        "---\nlicense: apache-2.0\ntags: [nla, qwen3, interpretability, length-penalty]\n"
        "configs:\n  - config_name: default\n    data_files: completions.parquet\n---\n\n"
        "# Qwen3-8B NLA — held-out completions across the length-penalty sweep\n\n"
        "1000 held-out completions per model (the **A**ctivation **V**erbalizer's generated "
        "explanation of a layer-24 activation), for the from-scratch base NLA and each "
        "RL length-penalty (`reward -= length_penalty * tokens`). Same fixed random held-out "
        "slice across all models (doc-disjoint from training), so rows are comparable by `idx`.\n\n"
        "Columns: `model`, `length_penalty`, `idx`, `source_text` (the activation's source "
        "context), `explanation` (the generated text), `n_tokens` (generated length), "
        "`mse`/`nmse`/`fve` (reconstruction quality from the AR critic; FVE = 1 - mse/baseline), "
        "`reward` (=-mse), `extracted`, `cjk`.\n\n"
        "## Aggregate\n\n| model | λ | mean tok | FVE | extraction | n |\n|---|--:|--:|--:|--:|--:|\n"
        + "\n".join(summary_lines) + "\n"
    )
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=args.out, path_in_repo="completions.parquet",
                    repo_id=args.repo, repo_type="dataset")
    print(f"COMPLETIONS_DATASET_DONE -> https://huggingface.co/datasets/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
