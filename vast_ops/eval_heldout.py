"""Generate + score N held-out samples for one actor (base AV-SFT, or +RL LoRA),
dumping a jsonl with per-sample reconstruction quality AND response length.

This is the length-penalty analogue of the main repo's extract_heldout.py: it
records the AV's generated token count per sample (what the length penalty acts
on) alongside mse / nmse / fve, so a length<->FVE tradeoff chart is recoverable
from the persisted jsonl alone.

reward = -mse_nrm from the AR critic (failed extraction -> -2.0).
FVE = 1 - mse / baseline, baseline = predict-the-mean normalized MSE.
NMSE = mse / baseline = 1 - FVE.

The held-out rows are a FIXED slice (--skip-rows past the RL training cursor),
so every model scores the SAME prompts in the SAME order -> match across models
by `idx` downstream.
"""

import argparse
import json
import os
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_RE, normalize_activation, resolve_target_scale


def cjk_frac(text):
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


# Karvonen injection hook — same mechanism as launch/eval_post_rl.py, inlined
# here so this script has no cross-directory import dependency.
def register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        if isinstance(output, tuple):
            resid, *rest = output
        else:
            resid, rest = output, None
        input_ids = state["input_ids"]
        if input_ids is None or resid.shape[1] < 2:
            return output
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0:
            return output
        if (input_ids == inj_id).sum().item() == 0:
            return output
        injected = karvonen_inject_in_residual(input_ids, resid, v, inj_id, left_id, right_id)
        return injected if rest is None else (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    target = model.base_model if hasattr(model, "base_model") else model
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    target.layers[layer_idx].register_forward_hook(layer_hook)


def user_text(prompt_msgs, inject_char):
    """Best-effort: the user turn's text, with the inject marker stripped, for display."""
    for m in prompt_msgs:
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].replace("<INJECT>", "").replace(inject_char, "").strip()
    return ""


def n_generated_tokens(resp_ids, eos_id):
    """Generated token count, matching the RL trainer's len(old_logp) exactly:
    one step per generated token INCLUDING the terminal EOS. Stops at the first
    EOS so any post-EOS padding isn't counted. This is the quantity the length
    penalty acts on, so eval-reported length is comparable to training."""
    for i, t in enumerate(resp_ids):
        if t == eos_id:
            return i + 1
    return len(resp_ids)


def fve_baseline(parquet, mse_scale_f, n=2000):
    """Predict-the-mean normalized-MSE ceiling (same as launch/compute_fve_baseline.py)."""
    pf = pq.ParquetFile(parquet)
    rg = pf.read_row_group(0, columns=["activation_vector"]).slice(0, n)
    acts = torch.tensor(rg.column("activation_vector").to_pylist(), dtype=torch.float32)
    mu_n = normalize_activation(acts.mean(dim=0, keepdim=True), mse_scale_f)
    acts_n = normalize_activation(acts, mse_scale_f)
    return ((mu_n - acts_n) ** 2).mean(dim=-1).mean().item()


def load_rows(parquet, skip_rows, n_rows):
    pf = pq.ParquetFile(parquet)
    rows, skipped = [], 0
    for rg_idx in range(pf.num_row_groups):
        if len(rows) >= n_rows:
            break
        rg = pf.read_row_group(rg_idx, columns=["prompt", "activation_vector"])
        n = rg.num_rows
        if skipped + n <= skip_rows:
            skipped += n
            continue
        start = max(0, skip_rows - skipped)
        skipped += start
        take = min(n_rows - len(rows), n - start)
        rg = rg.slice(start, take)
        rows.extend({"prompt": p_, "activation": a}
                    for p_, a in zip(rg.column("prompt").to_pylist(),
                                     rg.column("activation_vector").to_pylist()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True, help="HF AV-SFT checkpoint (the base actor)")
    p.add_argument("--ar-ckpt", required=True, help="HF AR critic checkpoint")
    p.add_argument("--sidecar", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--rl-lora", default=None,
                   help="RL LoRA adapter dir. Omit to evaluate the BASE actor (no RL).")
    p.add_argument("--n-rows", type=int, default=1000)
    p.add_argument("--skip-rows", type=int, default=25000)
    p.add_argument("--max-new", type=int, default=150)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--fve-baseline", type=float, default=None,
                   help="Use this baseline mse for FVE (keeps it identical across models). "
                        "If unset, computed from --val-parquet.")
    p.add_argument("--tag", required=True, help="model tag, e.g. base / p0.002")
    p.add_argument("--out", required=True, help="output .jsonl path")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/nla/hf"))
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tokenizer)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    baseline = args.fve_baseline if args.fve_baseline is not None else fve_baseline(args.val_parquet, mse_scale_f)
    print(f"[{args.tag}] fve baseline mse_nrm = {baseline:.4f}")

    rows = load_rows(args.val_parquet, args.skip_rows, args.n_rows)
    print(f"[{args.tag}] {len(rows)} held-out prompts (skip {args.skip_rows})")

    critic = NLACriticModel.from_pretrained(args.ar_ckpt, torch_dtype=torch.bfloat16).to(device).eval()
    for pr in critic.parameters():
        pr.requires_grad_(False)

    actor = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    if args.rl_lora:
        actor = PeftModel.from_pretrained(actor, args.rl_lora)
        print(f"[{args.tag}] loaded LoRA {args.rl_lora}")
    actor.eval()

    inj_id = cfg.injection_token_id
    inject_char = cfg.injection_char
    template = cfg.critic_prompt_template
    eos_id = tokenizer.eos_token_id
    vectors_ref = [None]
    register_karvonen_hook(actor, vectors_ref, inj_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    sum_mse = 0.0
    sum_tok = 0
    with open(args.out, "w") as fout:
        for i, row in enumerate(rows):
            msgs = [{**m, "content": m["content"].replace("<INJECT>", inject_char)}
                    if isinstance(m.get("content"), str) else m for m in row["prompt"]]
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_t = torch.tensor([ids], dtype=torch.long, device=device)
            activation = torch.tensor(row["activation"], dtype=torch.float32).unsqueeze(0).to(device)
            vectors_ref[0] = activation
            try:
                with torch.no_grad():
                    out = actor.generate(
                        input_ids=prompt_t, attention_mask=torch.ones_like(prompt_t),
                        max_new_tokens=args.max_new, do_sample=args.temperature > 0,
                        temperature=args.temperature, pad_token_id=eos_id,
                        return_dict_in_generate=True)
            finally:
                vectors_ref[0] = None
            resp_ids = out.sequences[0, prompt_t.shape[1]:].tolist()
            response = tokenizer.decode(resp_ids, skip_special_tokens=True)
            n_tok = n_generated_tokens(resp_ids, eos_id)
            m = EXPLANATION_RE.search(response)
            expl = m.group(1).strip() if m else None

            mse = None
            reward = -2.0
            if expl is not None:
                crit_text = template.format(explanation=expl)
                crit_ids = tokenizer.encode(crit_text, add_special_tokens=False)
                if len(crit_ids) <= 1024:
                    x = torch.tensor([crit_ids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        pred = critic(input_ids=x).values[0, -1].float()
                    pn = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
                    gn = normalize_activation(activation[0].float().unsqueeze(0), mse_scale_f)[0]
                    mse_val = F.mse_loss(pn, gn).item()
                    if np.isfinite(mse_val):
                        mse = mse_val
                        reward = -mse_val
            rec = {
                "idx": i,
                "tag": args.tag,
                "n_tokens": n_tok,
                "mse": mse,                                   # None if extraction failed
                "nmse": (mse / baseline) if mse is not None else None,
                "fve": (1.0 - mse / baseline) if mse is not None else None,
                "reward": reward,
                "extracted": expl is not None,
                "cjk": cjk_frac(response),
                "explanation": expl,
                "source_text": user_text(row["prompt"], inject_char)[:400],
            }
            fout.write(json.dumps(rec) + "\n")
            if mse is not None:
                n_valid += 1
                sum_mse += mse
            sum_tok += n_tok
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(rows)}] mean_tok={sum_tok/(i+1):.1f} "
                      f"valid_fve={1 - (sum_mse/max(n_valid,1))/baseline:.3f}")

    mean_mse = sum_mse / max(n_valid, 1)
    summary = {
        "tag": args.tag,
        "n": len(rows),
        "extraction_rate": n_valid / max(len(rows), 1),
        "mean_tokens": sum_tok / max(len(rows), 1),
        "mean_mse": mean_mse,
        "nmse": mean_mse / baseline,
        "fve": 1.0 - mean_mse / baseline,
        "fve_baseline": baseline,
        "rl_lora": args.rl_lora,
    }
    with open(args.out.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.tag}] DONE  fve={summary['fve']:.4f}  nmse={summary['nmse']:.4f}  "
          f"mean_tokens={summary['mean_tokens']:.1f}  ext={summary['extraction_rate']:.0%}")


if __name__ == "__main__":
    main()
