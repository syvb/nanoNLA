#!/usr/bin/env python3
"""Push ONE artifact (a checkpoint dir or a folder of files) to an HF repo and
add it to a collection. Best-effort: prints errors and exits 0 unless --strict,
so a transient push failure never aborts the training pipeline.

Used by run_pipeline.sh to upload each SFT checkpoint the moment it is written
(so a host failure mid-run doesn't lose completed work).
"""
import argparse
import os

from huggingface_hub import HfApi, add_collection_item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="dir (upload_folder) or file")
    ap.add_argument("--repo", required=True, help="e.g. syvb/nla-ordered-features-av-sft")
    ap.add_argument("--type", default="model", choices=["model", "dataset"])
    ap.add_argument("--collection", default=os.environ.get("COLLECTION_SLUG"))
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    try:
        api = HfApi(token=args.token)
        api.create_repo(args.repo, repo_type=args.type, private=False, exist_ok=True)
        if os.path.isdir(args.path):
            api.upload_folder(folder_path=args.path, repo_id=args.repo, repo_type=args.type)
        else:
            api.upload_file(path_or_fileobj=args.path,
                            path_in_repo=os.path.basename(args.path),
                            repo_id=args.repo, repo_type=args.type)
        print(f"[push_ckpt] pushed {args.path} -> {args.repo} ({args.type})")
        if args.collection:
            try:
                add_collection_item(args.collection, args.repo, args.type,
                                    token=args.token, exists_ok=True)
                print(f"[push_ckpt] added {args.repo} to collection")
            except Exception as e:  # noqa: BLE001
                print("[push_ckpt] collection add warning:", e)
    except Exception as e:  # noqa: BLE001
        print("[push_ckpt] PUSH FAILED (continuing):", repr(e))
        if args.strict:
            raise


if __name__ == "__main__":
    main()
