"""Regenerate the activation_vector column for a slim NLA dataset parquet.

The published NLA datasets (e.g. ceselder/qwen3-8b-nla-L24-finefineweb-100k)
ship WITHOUT the activation_vector column to stay small — the raw vectors are
deterministic given the input text + layer, so they are regenerated here rather
than stored (~7.6 GB of float arrays -> ~0.8 GB of text/provenance).

For each row we re-run `detokenized_text_truncated` through the base model and
take the layer-K hidden state at the FINAL token — which is the original
extraction position by construction, since stage 0 truncated the text to end
exactly at it. Output matches the original stage-0 schema (activation_vector
re-added, raw / unnormalized — norm="none").

Usage:
    python tools/regenerate_activations.py \\
        --in av_sft_shuf.parquet --out av_sft_shuf.full.parquet \\
        --base-model Qwen/Qwen3-8B
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen.extractors import HFExtractor


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", required=True, help="slim parquet (no activation_vector)")
    p.add_argument("--out", required=True, help="output parquet (activation_vector re-added)")
    p.add_argument("--base-model", required=True, help="HF base model, e.g. Qwen/Qwen3-8B")
    p.add_argument("--chunk-size", type=int, default=512, help="rows per write (bounds memory)")
    p.add_argument("--batch-size", type=int, default=16, help="model forward batch size")
    p.add_argument("--max-length", type=int, default=4096,
                   help="extractor context cap — MUST match the original stage-0 "
                        "extraction (4096 for the published datasets; "
                        "HFExtractor's bare default is 2048, which would "
                        "right-truncate long rows and silently regenerate the "
                        "activation at the wrong position)")
    p.add_argument("--drop-mismatch", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Drop rows whose decode->re-encode token count != "
                        "n_raw_tokens (their activation would land at the wrong "
                        "position) instead of aborting. Logs how many. Use for "
                        "slim datasets where a few rows aren't faithfully "
                        "regenerable via detok->retok.")
    args = p.parse_args()

    pf = pq.ParquetFile(args.inp)
    assert "activation_vector" not in pf.schema_arrow.names, (
        "input already has an activation_vector column — nothing to regenerate"
    )
    assert "detokenized_text_truncated" in pf.schema_arrow.names, (
        "input lacks detokenized_text_truncated — cannot regenerate without the source text"
    )

    ext = HFExtractor(model_name=args.base_model, batch_size=args.batch_size,
                      max_length=args.max_length)
    vec_type = pa.list_(pa.float32(), ext.d_model)

    writer = None
    done = 0
    n_dropped = 0
    for batch in pf.iter_batches(batch_size=args.chunk_size):
        tbl = pa.Table.from_batches([batch])
        texts = tbl.column("detokenized_text_truncated").to_pylist()
        layers = set(tbl.column("activation_layer").to_pylist())
        assert len(layers) == 1, f"chunk mixes activation_layer values: {layers}"
        layer = layers.pop()

        results = ext.extract(texts, layer)
        # Round-trip check: re-encoding decode(token_ids[:pos+1]) must
        # reproduce the original token count, or the final token (= the
        # extraction position) is not the one the explanation describes.
        # n_raw_tokens ships in the slim parquet exactly for this.
        keep = [True] * len(results)
        if "n_raw_tokens" in tbl.schema.names:
            n_raw = tbl.column("n_raw_tokens").to_pylist()
            bad = [(done + i, len(r.token_ids), n)
                   for i, (r, n) in enumerate(zip(results, n_raw))
                   if len(r.token_ids) != n]
            if bad and args.drop_mismatch:
                keep = [len(r.token_ids) == n for r, n in zip(results, n_raw)]
                n_dropped += len(bad)
                print(f"  dropped {len(bad)} round-trip-mismatch rows "
                      f"(running total {n_dropped})", flush=True)
            else:
                assert not bad, (
                    f"{len(bad)} rows fail the tokenization round-trip "
                    f"(first: row {bad[0][0]} re-encoded to {bad[0][1]} tokens, "
                    f"stage-0 had {bad[0][2]}). These rows are not faithfully "
                    f"regenerable — check --max-length matches the original "
                    f"extraction, or pass --drop-mismatch to skip them."
                )
        # the prefix ends exactly at the extraction token -> it is the last token
        vecs = [r.hidden_states[-1].tolist() for r in results]
        tbl = tbl.append_column("activation_vector", pa.array(vecs, type=vec_type))
        if not all(keep):
            tbl = tbl.filter(pa.array(keep, type=pa.bool_()))

        if writer is None:
            writer = pq.ParquetWriter(args.out, tbl.schema)
        writer.write_table(tbl)
        done += tbl.num_rows
        print(f"  {done} rows", flush=True)

    if writer is not None:
        writer.close()
    print(f"wrote {done} rows with activation_vector -> {args.out} "
          f"({n_dropped} dropped for round-trip mismatch)")


if __name__ == "__main__":
    main()
