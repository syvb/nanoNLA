"""Persist the co-train comparison artifacts to HuggingFace (public) so they
survive box teardown:

  models:  {prefix}-rl-cotrain   — per penalty: <slug>/av/ (actor LoRA) + <slug>/ar/ (co-trained critic)
  dataset: {prefix}-cotrain-heldout — the 4 held-out sample jsonls + summaries + COTRAIN_COMPARISON.md

Idempotent (exist_ok). Run on the box (where ckpts/rl_ct/* and experiment_results/heldout_ct/* live).
"""
import argparse
import glob
import os
from pathlib import Path

from huggingface_hub import HfApi


def latest_iter(d):
    its = sorted(glob.glob(os.path.join(d, "iter_*")))
    return its[-1] if its else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--ct-base", required=True, help="dir holding p<pen>/ subdirs (iter_* + critic/)")
    p.add_argument("--heldout-dir", required=True, help="experiment_results/heldout_ct")
    p.add_argument("--report", required=True, help="COTRAIN_COMPARISON.md path")
    p.add_argument("--penalties", required=True, help="space-separated, e.g. '0.006 0.015'")
    p.add_argument("--private", action="store_true", default=False)  # public by default
    args = p.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    # ---- models: one repo, <slug>/av + <slug>/ar per penalty ----
    mrepo = f"{args.owner}/{args.prefix}-rl-cotrain"
    api.create_repo(mrepo, repo_type="model", private=args.private, exist_ok=True)
    for pen in args.penalties.split():
        slug = f"p{pen}"
        d = os.path.join(args.ct_base, slug)
        it = latest_iter(d)
        critic = os.path.join(d, "critic")
        if it:  # actor LoRA — ignore peft's auto README (base_model=local path → HF rejects)
            api.upload_folder(folder_path=it, repo_id=mrepo, repo_type="model",
                              path_in_repo=f"{slug}/av", ignore_patterns=["README.md"])
            print(f"uploaded co-train AV LoRA {slug} <- {it}", flush=True)
        else:
            print(f"skip AV {slug}: no iter dir in {d}", flush=True)
        if Path(critic).is_dir():
            api.upload_folder(folder_path=critic, repo_id=mrepo, repo_type="model",
                              path_in_repo=f"{slug}/ar", ignore_patterns=["README.md"])
            print(f"uploaded co-train AR critic {slug} <- {critic}", flush=True)
        else:
            print(f"skip AR {slug}: no critic dir in {d}", flush=True)
    mcard = (
        "---\nlicense: apache-2.0\ntags: [nla, qwen3, length-penalty, co-training]\n---\n\n"
        "# NLA length-penalty RL — co-trained AV + AR\n\n"
        "Companion to `" + f"{args.owner}/{args.prefix}-rl-lora" + "` (frozen-critic sweep). Here the AR "
        "critic was **co-trained** alongside the AV during RL (`--train-critic`). One subfolder per "
        "length penalty, each with both halves of the NLA:\n\n"
        "- `p<λ>/av/` — actor LoRA adapter (load onto `" + f"{args.owner}/{args.prefix}-av" + "`).\n"
        "- `p<λ>/ar/` — the co-trained AR critic (`NLACriticModel.from_pretrained`): truncated backbone "
        "shards + `value_head.safetensors`.\n\n"
        "See `" + f"{args.owner}/{args.prefix}-cotrain-heldout" + "` for held-out samples + the "
        "frozen-vs-co-train comparison.\n")
    api.upload_file(path_or_fileobj=mcard.encode(), path_in_repo="README.md",
                    repo_id=mrepo, repo_type="model")
    print(f"models repo: {mrepo}", flush=True)

    # ---- dataset: held-out samples + summaries + comparison md ----
    drepo = f"{args.owner}/{args.prefix}-cotrain-heldout"
    api.create_repo(drepo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.heldout_dir, repo_id=drepo, repo_type="dataset",
                      path_in_repo="heldout_ct",
                      allow_patterns=["*.samples.jsonl", "*.summary.json"])
    if Path(args.report).is_file():
        api.upload_file(path_or_fileobj=args.report, path_in_repo="COTRAIN_COMPARISON.md",
                        repo_id=drepo, repo_type="dataset")
    # scope viewer to the per-sample jsonls (else jsonl+summary schema merge CastErrors)
    dcard = (
        "---\nlicense: apache-2.0\ntags: [nla, qwen3, length-penalty, co-training]\n"
        "configs:\n  - config_name: heldout\n    data_files: heldout_ct/*.samples.jsonl\n---\n\n"
        "# NLA co-train comparison — held-out samples\n\n"
        "Held-out completions (n=1000, rows 25k:26k, doc-disjoint from RL train) for the co-trained "
        "λ=0.006 / λ=0.015 models, each scored two ways:\n\n"
        "- `*_ct_baseAR` — scored by the frozen base AR (same ruler as the frozen sweep).\n"
        "- `*_ct_ownAR` — scored by the model's own co-trained AR (real co-trained system FVE).\n\n"
        "Per-sample rows: `idx, n_tokens, fve, nmse, mse, reward, extracted, cjk, explanation, "
        "source_text`. Matched by `idx` across models and against the frozen sweep's "
        "`-results` dataset. `COTRAIN_COMPARISON.md` has the headline table.\n")
    api.upload_file(path_or_fileobj=dcard.encode(), path_in_repo="README.md",
                    repo_id=drepo, repo_type="dataset")
    print(f"dataset repo: {drepo}", flush=True)
    print("ALL_COTRAIN_UPLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()
