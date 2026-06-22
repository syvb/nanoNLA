#!/usr/bin/env python3
"""Generate FULL AV explanations on HELD-OUT activations from an NLA AV checkpoint.

Loads base + AV LoRA (4-bit), registers the Karvonen injection hook, and decodes
the AV's complete ordered-feature explanation for N held-out rows. "Held-out" =
doc-disjoint from the first --skip-rows (the RL training cursor), matching the
RL trainer's eval split. Prints the full <explanation> (NOT truncated like the
training-log eval lines). Reuses the trainer's exact inject/rollout code so the
output matches what the model produces in training/eval.
"""
import argparse

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from nla.config import load_nla_config
from nla.schema import extract_explanation
from nla.train_rl_self_contained import (
    _register_karvonen_hook,
    build_prompt_text,
    rollout_one_prompt,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True, help="AV LoRA adapter dir (e.g. RL actor/)")
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    ap.add_argument("--rl-parquet", required=True,
                    help="held-out data: prompt + activation_vector + doc_id")
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--skip-rows", type=int, default=3000,
                    help="rows before this were the RL train cursor; held-out is "
                         "doc-disjoint from these")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tok)

    qc = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=qc, device_map={"": 0},
    )
    actor = PeftModel.from_pretrained(base, args.av_ckpt, adapter_name="default")
    actor.set_adapter("default")
    actor.eval()

    vectors_ref = [None]
    _register_karvonen_hook(
        actor, vectors_ref, cfg.injection_token_id,
        cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
        layer_idx=1,
    )
    eos_ids = {tok.eos_token_id}

    # Held-out rows: doc-disjoint from the first --skip-rows (RL train cursor).
    pf = pq.ParquetFile(args.rl_parquet)
    rows = []
    for b in pf.iter_batches(batch_size=4096,
                             columns=["prompt", "activation_vector", "doc_id"]):
        rows.extend(b.to_pylist())
    train_docs = {r["doc_id"] for r in rows[:args.skip_rows]}
    heldout = [r for r in rows[args.skip_rows:] if r["doc_id"] not in train_docs]
    picks = heldout[:args.n]
    print(f"[generate_av] {len(rows)} rows | {len(train_docs)} train docs | "
          f"{len(heldout)} doc-disjoint held-out | generating {len(picks)} "
          f"(temp={args.temperature}, max_new={args.max_new_tokens})", flush=True)

    for i, row in enumerate(picks):
        prompt_text = build_prompt_text(row["prompt"], cfg.injection_char, tok)
        act = torch.tensor(row["activation_vector"], dtype=torch.float32)
        resp = rollout_one_prompt(
            actor, tok, prompt_text, act, vectors_ref, cfg.injection_token_id,
            1, args.max_new_tokens, args.temperature, device, eos_ids=eos_ids,
        )
        text = resp[0]["text"]
        expl = extract_explanation(text)
        print(f"\n================ held-out AV gen {i + 1}/{len(picks)} "
              f"(doc {row['doc_id']}) ================", flush=True)
        print(expl if expl else f"[no <explanation> parsed]\n{text}", flush=True)


if __name__ == "__main__":
    main()
