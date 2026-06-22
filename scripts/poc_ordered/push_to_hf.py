#!/usr/bin/env python3
"""Push ordered-features PoC artifacts to HuggingFace and add them to a collection.

Uploads:
  - one DATASET repo  : the built av_sft/ar_sft/rl parquets (+ sidecars)
  - three MODEL repos : av-sft, ar-sft, rl (final iter_* checkpoint dirs)
All are made public and added to --collection-slug.

Auth: --token or $HF_TOKEN.
"""
import argparse
import glob
import os

from huggingface_hub import HfApi, add_collection_item

_CARD = """---
license: mit
tags:
- natural-language-autoencoders
- mechanistic-interpretability
- nla
---

# {title}

{blurb}

Part of the **NLA ordered-features (nested-dropout RL)** PoC: an Activation
Verbalizer trained so its ~10 next-token-prediction features are ordered
most->least important, via random feature-prefix truncation (nested dropout)
during GRPO RL. Code: `feat/nla-ordered-features-rl` on
https://github.com/syvb/nanoNLA . Base model: Qwen/Qwen3-8B, layer 24.

Collection: https://huggingface.co/collections/{collection}
"""


def latest_iter(d: str):
    its = sorted(glob.glob(os.path.join(d, "iter_*")))
    return its[-1] if its else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="$WORK/ckpts")
    ap.add_argument("--build-dir", required=True, help="$WORK/data/build")
    ap.add_argument("--collection-slug", required=True)
    ap.add_argument("--namespace", default="syvb")
    ap.add_argument("--prefix", default="nla-ordered-features")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    api = HfApi(token=args.token)
    ns, pre, coll = args.namespace, args.prefix, args.collection_slug
    added = []

    def card(repo_id, title, blurb):
        api.upload_file(
            path_or_fileobj=_CARD.format(title=title, blurb=blurb, collection=coll).encode(),
            path_in_repo="README.md", repo_id=repo_id,
            repo_type="dataset" if "data" in repo_id else "model",
        )

    def add_to_collection(repo_id, repo_type):
        try:
            add_collection_item(coll, repo_id, repo_type, token=args.token, exists_ok=True)
        except Exception as e:  # noqa: BLE001
            print("  add_collection_item warning:", e)
        added.append((repo_type, repo_id))

    # ---- dataset repo: built training splits ----
    ds_repo = f"{ns}/{pre}-poc-data"
    api.create_repo(ds_repo, repo_type="dataset", private=False, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(args.build_dir, "*"))):
        if os.path.isfile(f):
            api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                            repo_id=ds_repo, repo_type="dataset")
            print("uploaded", f)
    card(ds_repo, "NLA ordered-features — PoC training splits",
         "Built av_sft / ar_sft / rl parquets (+ `.nla_meta.yaml` sidecars) with "
         "raw activation vectors, used to warm-start and RL-train the PoC model.")
    add_to_collection(ds_repo, "dataset")

    # ---- model repos ----
    for sub, name, blurb in [
        ("av_sft", "av-sft", "Activation Verbalizer after warm-start SFT (LoRA on Qwen3-8B, 4-bit)."),
        ("ar_sft", "ar-sft", "Activation Reconstructor (critic) after warm-start SFT (LoRA + value head)."),
        ("rl_ordered", "rl", "Final ordered-features model after generate-K GRPO RL (AV + co-trained critic)."),
    ]:
        ck = latest_iter(os.path.join(args.ckpt_dir, sub))
        if not ck:
            print("  no checkpoint for", sub, "- skipping")
            continue
        repo = f"{ns}/{pre}-{name}"
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(folder_path=ck, repo_id=repo, repo_type="model")
        print("uploaded", ck, "->", repo)
        card(repo, f"NLA ordered-features — {name}", blurb)
        add_to_collection(repo, "model")

    print("\nDONE. Items added to collection:")
    for t, r in added:
        print(f"  {t}: https://huggingface.co/{'datasets/' if t == 'dataset' else ''}{r}")


if __name__ == "__main__":
    main()
