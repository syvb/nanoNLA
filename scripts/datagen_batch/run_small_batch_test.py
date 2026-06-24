"""Submit a SMALL real Anthropic batch from the generated request file.

Pulls the first N request objects straight out of full_run_sonnet46_requests.json
and submits them to the Anthropic Message Batches API exactly as-is — the truest
test that the file's request schema works end to end. Polls, fetches results,
reports success/extract rates and a sample.

This DOES spend money, but only on N requests (default 30). It must NEVER be
pointed at the full file.
"""

import argparse
import re
import sys
import time

import anthropic
import orjson

ANALYSIS_RE = re.compile(r"<analysis>\s*(.*?)\s*</analysis>", re.DOTALL)


def load_first_n(path: str, n: int) -> list[dict]:
    """Stream the leading N request objects from the JSON-array file (one per line)."""
    out: list[dict] = []
    with open(path, "rb") as f:
        first = f.readline()  # b"[\n"
        assert first.strip() == b"[", first
        for line in f:
            s = line.strip()
            if s in (b"]", b""):
                break
            if s.endswith(b","):
                s = s[:-1]
            out.append(orjson.loads(s))
            if len(out) >= n:
                break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests-file", default="scripts/datagen_batch/out/full_run_sonnet46_requests.json")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--poll-interval", type=float, default=10.0)
    p.add_argument("--timeout", type=float, default=900.0)
    args = p.parse_args()

    if args.n > 100:
        sys.exit(f"refusing n={args.n}: this is a SMALL-batch smoke test, keep n<=100")

    reqs = load_first_n(args.requests_file, args.n)
    print(f"loaded {len(reqs)} requests; custom_ids: {[r['custom_id'] for r in reqs][:5]} ...")
    model = reqs[0]["params"]["model"]
    print(f"model={model} max_tokens={reqs[0]['params']['max_tokens']} temp={reqs[0]['params']['temperature']}")

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=reqs)
    print(f"submitted batch {batch.id} (status={batch.processing_status})")

    t0 = time.monotonic()
    while True:
        time.sleep(args.poll_interval)
        b = client.messages.batches.retrieve(batch.id)
        elapsed = time.monotonic() - t0
        print(f"  [{elapsed:5.0f}s] status={b.processing_status} counts={b.request_counts}")
        if b.processing_status == "ended":
            break
        if elapsed > args.timeout:
            sys.exit(f"timed out after {elapsed:.0f}s")

    n_ok = n_extract = 0
    samples = []
    for entry in client.messages.batches.results(batch.id):
        r = entry.result
        if r.type != "succeeded":
            print(f"  {entry.custom_id}: NON-SUCCESS {r.type}")
            continue
        n_ok += 1
        text = r.message.content[0].text if r.message.content else ""
        m = ANALYSIS_RE.search(text)
        feats = [ln for ln in m.group(1).split("\n") if ln.strip()] if m else []
        if m:
            n_extract += 1
        if len(samples) < 3:
            samples.append((entry.custom_id, len(feats), text))

    print(f"\n=== RESULT: {n_ok}/{len(reqs)} succeeded | {n_extract}/{n_ok} parsed <analysis> ===")
    for cid, nf, text in samples:
        print(f"\n----- {cid} ({nf} features) -----")
        print(text[:700])


if __name__ == "__main__":
    main()
