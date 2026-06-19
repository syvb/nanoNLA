"""Persist the from-scratch NLA artifacts to HuggingFace so they survive box
teardown: the AV + AR merged checkpoints, the per-penalty RL LoRA adapters, and
the held-out results. Private repos by default. Idempotent (exist_ok)."""

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
    p.add_argument("--av", required=True)
    p.add_argument("--ar", required=True)
    p.add_argument("--rl-base", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--penalties", required=True, help="space-separated")
    p.add_argument("--private", action="store_true", default=False)  # public by default
    args = p.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    def push_folder(repo, src, repo_type, **kw):
        api.create_repo(repo, repo_type=repo_type, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=src, repo_id=repo, repo_type=repo_type, **kw)
        print(f"uploaded {repo_type}: {repo}  <- {src}", flush=True)

    # AV / AR merged checkpoints. Ignore any auto-generated README.md — peft/HF
    # write a card with base_model set to the LOCAL path, which HF's YAML
    # validator rejects ("not a valid model id"). The weights/config are what matter.
    for role, src in [("av", args.av), ("ar", args.ar)]:
        if Path(src).is_dir():
            push_folder(f"{args.owner}/{args.prefix}-{role}", src, "model",
                        ignore_patterns=["README.md"])
        else:
            print(f"skip {role}: {src} missing", flush=True)

    # RL LoRA adapters — one repo, a subfolder per penalty
    rl_repo = f"{args.owner}/{args.prefix}-rl-lora"
    api.create_repo(rl_repo, repo_type="model", private=args.private, exist_ok=True)
    for pen in args.penalties.split():
        slug = f"p{pen}"
        it = latest_iter(os.path.join(args.rl_base, slug))
        if it:
            # ignore peft's auto README (base_model = local path -> HF rejects it)
            api.upload_folder(folder_path=it, repo_id=rl_repo, repo_type="model",
                              path_in_repo=slug, ignore_patterns=["README.md"])
            print(f"uploaded LoRA {slug} <- {it}", flush=True)
        else:
            print(f"skip LoRA {slug}: no iter dir", flush=True)

    # Results (jsonl + markdown + plot) as a dataset
    res_repo = f"{args.owner}/{args.prefix}-results"
    push_folder(res_repo, args.results, "dataset",
                allow_patterns=["heldout/*.jsonl", "heldout/*.summary.json", "*.md", "*.png"])
    # Scope the dataset-viewer to the per-sample jsonls only — otherwise HF tries
    # to merge them with the differently-shaped summary.json files and CastErrors.
    res_card = ("---\nlicense: apache-2.0\ntags: [nla, qwen3, length-penalty]\n"
                "configs:\n  - config_name: completions\n    data_files: heldout/*.samples.jsonl\n---\n\n"
                "# NLA length-penalty results\n\nHeld-out completions (matched by `idx` across models) + "
                "per-model `*.summary.json` aggregates + RESULTS.md / comparison / tradeoff.png.\n")
    api.upload_file(path_or_fileobj=res_card.encode(), path_in_repo="README.md",
                    repo_id=res_repo, repo_type="dataset")
    print("ALL_UPLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()
