"""Build a small stage-2 input parquet from the existing warm-start dataset.

Stage 2 only needs `detokenized_text_truncated` (the source text) plus a valid
NLA sidecar. We reuse the *texts* from the already-published Sonnet warm-start
dataset so the DeepSeek-Flash pipeline test runs against the exact same inputs,
then drop the Sonnet `api_explanation` so stage 2 regenerates it.

No GPU: this is pure text + sidecar plumbing. Output is a stage-1-style "base"
parquet (stage 2 does not require token meta for a non-training stage).
"""

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from nla.datagen.sidecar import NLAExtractionMeta, NLADatasetMeta, write_sidecar_local

# Carried through to stage 2's output untouched; stage 2 keys/joins only on
# detokenized_text_truncated, but downstream stages expect these columns.
_KEEP = ["detokenized_text_truncated", "doc_id", "n_raw_tokens", "activation_layer"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-repo", default="syvb/nla-warmstart-explanations-finefineweb-sonnet46")
    p.add_argument("--src-file", default="data/train-00000-of-00001.parquet")
    p.add_argument("--n", type=int, default=100, help="number of rows to take")
    p.add_argument("--output", required=True, help="stage-2 input parquet path")
    # Sidecar fields — match the base-model extraction config of the warm-start run.
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--layer-index", type=int, default=24)
    p.add_argument("--positions-per-doc", type=int, default=10)
    args = p.parse_args()

    src = hf_hub_download(args.src_repo, args.src_file, repo_type="dataset")
    table = pq.read_table(src)
    missing = [c for c in _KEEP if c not in table.column_names]
    assert not missing, f"source missing columns {missing}; has {table.column_names}"
    sub = table.select(_KEEP).slice(0, args.n)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sub, out)

    meta = NLADatasetMeta(
        dataset_id="deepseek_flash_pipeline_test",
        stage="base",  # non-training stage → no token meta required by serializer
        row_count=sub.num_rows,
        extraction=NLAExtractionMeta(
            base_model=args.base_model,
            d_model=args.d_model,
            layer_index=args.layer_index,
            norm="none",
            corpus=args.src_repo,
            corpus_slice={"start": 0, "stop": args.n},
            positions_per_doc=args.positions_per_doc,
        ),
        created_by="scripts/datagen_deepseek/build_test_input.py",
    )
    write_sidecar_local(out, meta)
    print(f"wrote {sub.num_rows} rows -> {out}")
    print(f"  sidecar -> {out}.nla_meta.yaml")


if __name__ == "__main__":
    main()
