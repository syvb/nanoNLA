#!/usr/bin/env python3
"""Synthesize a `base`-stage NLA sidecar for a regenerated activation parquet.

The published slim dataset ships no `.nla_meta.yaml`, and
tools/regenerate_activations.py writes none — but stage1_split and stage3_build
both require one (read_sidecar -> extraction.{d_model,layer_index,norm}). The
extraction metadata for this corpus (Qwen3-8B / layer 24 / FineFineWeb-100k, raw
vectors) is known and constant (mse_scale / injection_scale are resolved at
training time from the raw vectors, NOT stored here), so we write it next to the
regenerated parquet. stage1_split asserts stage=="base"; stage3_build then emits
the real av_sft/ar_sft/rl sidecars with token metadata.
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq

from nla.datagen.sidecar import (
    NLADatasetMeta,
    NLAExtractionMeta,
    write_sidecar_local,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True,
                   help="regenerated parquet (activation_vector present)")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--layer-index", type=int, default=24)
    p.add_argument("--corpus",
                   default="syvb/nla-warmstart-explanations-finefineweb-sonnet46")
    p.add_argument("--positions-per-doc", type=int, default=10)
    p.add_argument("--dataset-id", default="nla_ordered_slim_Qwen3-8B_L24_base")
    args = p.parse_args()

    pf = pq.ParquetFile(args.parquet)
    n = pf.metadata.num_rows
    # Verify d_model matches the actual activation_vector width.
    av_field = pf.schema_arrow.field("activation_vector")
    list_size = getattr(av_field.type, "list_size", None)
    if list_size is not None and list_size != args.d_model:
        raise SystemExit(
            f"--d-model {args.d_model} != activation_vector width {list_size} "
            f"in {args.parquet}"
        )

    meta = NLADatasetMeta(
        dataset_id=args.dataset_id,
        stage="base",
        row_count=n,
        extraction=NLAExtractionMeta(
            base_model=args.base_model,
            d_model=args.d_model,
            layer_index=args.layer_index,
            norm="none",
            corpus=args.corpus,
            corpus_slice={"start": 0, "length": 100000},
            positions_per_doc=args.positions_per_doc,
        ),
        created_by="scripts/poc_ordered/synth_base_sidecar.py",
    )
    write_sidecar_local(Path(args.parquet), meta)
    print(f"wrote base sidecar for {args.parquet} "
          f"(rows={n}, d_model={args.d_model}, layer={args.layer_index}, norm=none)")


if __name__ == "__main__":
    main()
