"""Build ONE JSON file of all Claude API requests for a full Sonnet-4.6 warm-start run.

The full NLA warm-start data-gen explains every AV-SFT + AR-SFT source text with
an LLM (the RL split needs no API — the actor generates during rollout). This
script reads the *texts* from the published full slim splits and emits every
request as a single JSON array, each element matching the Anthropic Message
Batches API request schema:

    {"custom_id": "...", "params": {"model", "max_tokens", "temperature", "messages"}}

It DOES NOT call any API — it only writes the request file. Submitting it (and
paying for it) is a separate, deliberate step.

Source texts: ceselder/qwen3-8b-nla-L24-finefineweb-100k (av_sft_shuf, ar_sft_shuf).
Prompt: the 10-feature importance-ordered prompt (scripts/datagen_deepseek/new_prompt.txt).
"""

import argparse
from pathlib import Path

import orjson
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SRC_REPO = "ceselder/qwen3-8b-nla-L24-finefineweb-100k"
SUBSETS = [("av", "av_sft_shuf.parquet"), ("ar", "ar_sft_shuf.parquet")]


def _docidx(doc_id: str) -> str:
    # doc_id = ".../finefineweb_100k.parquet:train:87484" -> "87484"
    return doc_id.rsplit(":", 1)[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-file", default="scripts/datagen_deepseek/next_prompt.txt")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output", default="scripts/datagen_batch/out/full_run_sonnet46_requests.json")
    p.add_argument("--cache-dir", default="scripts/datagen_batch/cache_ce")
    p.add_argument("--limit", type=int, default=0, help="cap rows per subset (0 = all; for testing)")
    args = p.parse_args()

    template = Path(args.prompt_file).read_text()
    # Robust placeholder substitution: next_prompt.txt writes {{text}}, older
    # prompts write {text}. Use str.replace (NOT .format) so the literal double
    # braces aren't collapsed to a no-op and any other braces can't break it.
    if "{{text}}" in template:
        placeholder = "{{text}}"
    elif "{text}" in template:
        placeholder = "{text}"
    else:
        raise SystemExit("prompt template must contain a {text} or {{text}} placeholder")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n = 0
    total_bytes = 0
    approx_prompt_tokens = 0  # rough: chars/4

    with open(out, "wb") as fh:
        fh.write(b"[\n")
        first = True
        for subset, fname in SUBSETS:
            path = hf_hub_download(SRC_REPO, fname, repo_type="dataset", local_dir=args.cache_dir)
            pf = pq.ParquetFile(path)
            sub_n = 0
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["detokenized_text_truncated", "doc_id", "n_raw_tokens"])
                texts = tbl.column("detokenized_text_truncated").to_pylist()
                docs = tbl.column("doc_id").to_pylist()
                ntoks = tbl.column("n_raw_tokens").to_pylist()
                for text, doc, ntok in zip(texts, docs, ntoks):
                    if args.limit and sub_n >= args.limit:
                        break
                    cid = f"{subset}-{_docidx(doc)}-{ntok}"
                    assert cid not in seen, f"duplicate custom_id {cid}"
                    seen.add(cid)
                    content = template.replace(placeholder, text)
                    approx_prompt_tokens += len(content) // 4
                    req = {
                        "custom_id": cid,
                        "params": {
                            "model": args.model,
                            "max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                            "messages": [{"role": "user", "content": content}],
                        },
                    }
                    line = orjson.dumps(req)
                    if not first:
                        fh.write(b",\n")
                    fh.write(line)
                    first = False
                    total_bytes += len(line)
                    n += 1
                    sub_n += 1
                if args.limit and sub_n >= args.limit:
                    break
            print(f"  {subset}: {sub_n} requests from {fname}")
        fh.write(b"\n]\n")

    # Cost estimate (Sonnet 4.6 batch = 50% off: $1.50/$7.50 per 1M in batch mode).
    out_tokens = n * args.max_tokens  # worst case (most stop earlier)
    in_cost = approx_prompt_tokens / 1e6 * 1.50
    out_cost = out_tokens / 1e6 * 7.50
    print(f"\nwrote {n} requests -> {out} ({out.stat().st_size/1e9:.2f} GB)")
    print(f"  approx input tokens : {approx_prompt_tokens/1e6:.1f}M")
    print(f"  max output tokens   : {out_tokens/1e6:.1f}M (ceiling; real << this)")
    print(f"  est. BATCH cost     : ${in_cost + out_cost:,.0f}  (input ${in_cost:,.0f} + output<=${out_cost:,.0f})")
    print(f"  est. STANDARD cost  : ${(in_cost + out_cost) * 2:,.0f}  (2x of batch)")


if __name__ == "__main__":
    main()
