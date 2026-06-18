"""Self-contained LoRA warm-start SFT for NLA — no Miles, no SGLang.

Two roles, one script:

  --role av : train the verbalizer. A LoRA on the base causal LM, trained with
              cross-entropy on the assistant explanation, with the gold
              activation injected at the ㊗ marker via the Karvonen layer-1
              residual hook (the SAME hook the RL trainer and eval use).

  --role ar : train the reconstructor. A LoRA on the truncated NLACriticModel
              backbone (value head frozen identity), trained with normalized-MSE
              between the last-token prediction and the gold activation —
              the same objective as nla/loss.py / the RL critic scoring.

Reuses nla.{injection,models,config,schema}: the Miles-free primitives the
validated path already uses. Saves a merged HF checkpoint so the self-contained
RL trainer and eval load it with from_pretrained — no DCP conversion.
"""

import argparse
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import INJECT_PLACEHOLDER, normalize_activation, resolve_target_scale

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ----------------------------- data -----------------------------
def load_samples(parquet, input_key, inj_char, max_rows, want_response):
    """Parquet -> list of {prompt, response?, activation}. Replicates the
    numpy fast-path from nla/data_source.py (no Miles, no per-float PyObjects)."""
    pf = pq.ParquetFile(parquet)
    cols = pf.schema_arrow.names
    assert "activation_vector" in cols, f"{parquet}: no activation_vector column"
    assert input_key in cols, f"{parquet}: no {input_key!r} column (have {cols})"
    if want_response:
        assert "response" in cols, f"{parquet}: AV SFT needs a 'response' column (have {cols})"
    other = [c for c in cols if c != "activation_vector"]
    samples = []
    for batch in pf.iter_batches(batch_size=16384):
        av_col = batch.column("activation_vector")
        av = av_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(av_col), -1)
        rest = batch.select(other).to_pylist()
        for row, vec in zip(rest, av):
            prompt = row[input_key]
            if isinstance(prompt, list):
                prompt = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in prompt]
            samples.append({"prompt": prompt, "response": row.get("response"), "activation": vec})
            if max_rows and len(samples) >= max_rows:
                return samples
    return samples


class Batcher:
    """Deterministic shuffled stream over samples; reshuffles each epoch."""
    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.ptr = n, seed, 0, 0
        self._reshuffle()

    def _reshuffle(self):
        r = random.Random(self.seed + self.epoch)
        self.order = list(range(self.n))
        r.shuffle(self.order)
        self.ptr = 0

    def next(self, k):
        out = []
        while len(out) < k:
            if self.ptr >= self.n:
                self.epoch += 1
                self._reshuffle()
            out.append(self.order[self.ptr])
            self.ptr += 1
        return out


# ----------------------------- injection hook (AV) -----------------------------
def register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        resid, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        ids = state["input_ids"]
        v = vectors_ref[0]
        if ids is None or v is None or resid.shape[1] < 2:
            return output
        if (ids == inj_id).sum().item() == 0:
            return output
        injected = karvonen_inject_in_residual(ids, resid, v, inj_id, left_id, right_id)
        return injected if rest is None else (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    target = model.base_model if hasattr(model, "base_model") else model
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    target.layers[layer_idx].register_forward_hook(layer_hook)


# ----------------------------- AV SFT -----------------------------
def build_av_example(tokenizer, msgs, response, max_seq_len):
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": response}], add_generation_prompt=False, tokenize=True)
    full_ids = full_ids[:max_seq_len]
    # loss only on the assistant turn (everything after the prompt prefix)
    mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    mask = mask[:len(full_ids)]
    return full_ids, mask


def train_av(args, tokenizer, cfg, samples, device):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=LORA_TARGETS, task_type=TaskType.CAUSAL_LM, use_rslora=True))
    model.print_trainable_parameters()
    vectors_ref = [None]
    register_karvonen_hook(model, vectors_ref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)
    pad_id = tokenizer.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = cosine_with_warmup(opt, args.warmup, args.steps)
    batcher = Batcher(len(samples), args.seed)
    model.train()

    for step in range(args.steps):
        opt.zero_grad()
        tot_loss = 0.0
        tot_tok = 0
        for _ in range(args.grad_accum):
            idxs = batcher.next(args.micro_batch)
            exs, vecs = [], []
            for i in idxs:
                ids, m = build_av_example(tokenizer, samples[i]["prompt"], samples[i]["response"], args.max_seq_len)
                exs.append((ids, m))
                vecs.append(torch.tensor(samples[i]["activation"], dtype=torch.float32))
            L = max(len(ids) for ids, _ in exs)
            B = len(exs)
            batch_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((B, L), dtype=torch.long, device=device)
            labels = torch.full((B, L), -100, dtype=torch.long, device=device)
            for r, (ids, m) in enumerate(exs):
                n = len(ids)
                t = torch.tensor(ids, device=device)
                batch_ids[r, :n] = t
                attn[r, :n] = 1
                mm = torch.tensor(m, dtype=torch.bool, device=device)
                labels[r, :n] = torch.where(mm, t, torch.full_like(t, -100))
            v_batch = torch.stack(vecs).to(device)
            vectors_ref[0] = v_batch
            try:
                logits = model(input_ids=batch_ids, attention_mask=attn).logits
            finally:
                vectors_ref[0] = None
            # next-token CE on response positions
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
            shift_labels = labels[:, 1:].reshape(-1)
            ntok = (shift_labels != -100).sum().clamp(min=1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum") / ntok
            (loss / args.grad_accum).backward()
            tot_loss += loss.item()
            tot_tok += int(ntok.item())
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == args.steps - 1:
            log = {"step": step, "loss": tot_loss / args.grad_accum, "lr": sched.get_last_lr()[0],
                   "grad_norm": float(gnorm), "resp_tok": tot_tok}
            print(f"[av] step {step:04d} loss {log['loss']:.4f} lr {log['lr']:.2e} gnorm {log['grad_norm']:.2f}", flush=True)
            wandb_log(args, log)
        maybe_save(args, model, tokenizer, cfg, step, role="av")
    save_merged(args, model, tokenizer, cfg, role="av")


# ----------------------------- AR SFT -----------------------------
def train_ar(args, tokenizer, cfg, samples, device):
    os.environ["NLA_FREEZE_VALUE_HEAD"] = "1"  # identity value head; train backbone only
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    model = NLACriticModel.from_pretrained(args.base_ckpt, torch_dtype=torch.bfloat16).to(device)
    for p in model.value_head.parameters():
        p.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=LORA_TARGETS, task_type=None, use_rslora=True))
    model.print_trainable_parameters()
    pad_id = tokenizer.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = cosine_with_warmup(opt, args.warmup, args.steps)
    batcher = Batcher(len(samples), args.seed)
    model.train()

    for step in range(args.steps):
        opt.zero_grad()
        tot_loss = 0.0
        for _ in range(args.grad_accum):
            idxs = batcher.next(args.micro_batch)
            id_lists, golds = [], []
            for i in idxs:
                ids = tokenizer(samples[i]["prompt"], add_special_tokens=True)["input_ids"][:args.max_seq_len]
                id_lists.append(ids)
                golds.append(torch.tensor(samples[i]["activation"], dtype=torch.float32))
            L = max(len(x) for x in id_lists)
            B = len(id_lists)
            batch_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((B, L), dtype=torch.long, device=device)
            last_idx = torch.empty(B, dtype=torch.long, device=device)
            for r, ids in enumerate(id_lists):
                n = len(ids)
                batch_ids[r, :n] = torch.tensor(ids, device=device)
                attn[r, :n] = 1
                last_idx[r] = n - 1
            gold = torch.stack(golds).to(device).float()
            h = model(input_ids=batch_ids, attention_mask=attn).backbone_last_hidden  # [B,T,d]
            pred = h[torch.arange(B, device=device), last_idx].float()
            loss = F.mse_loss(normalize_activation(pred, mse_scale_f),
                              normalize_activation(gold, mse_scale_f), reduction="none").mean(dim=-1).mean()
            (loss / args.grad_accum).backward()
            tot_loss += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == args.steps - 1:
            mean_mse = tot_loss / args.grad_accum
            log = {"step": step, "mse": mean_mse, "lr": sched.get_last_lr()[0], "grad_norm": float(gnorm)}
            print(f"[ar] step {step:04d} mse {mean_mse:.4f} lr {log['lr']:.2e} gnorm {log['grad_norm']:.2f}", flush=True)
            wandb_log(args, log)
        maybe_save(args, model, tokenizer, cfg, step, role="ar")
    save_merged(args, model, tokenizer, cfg, role="ar")


# ----------------------------- shared utils -----------------------------
def cosine_with_warmup(opt, warmup, total):
    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def maybe_save(args, model, tokenizer, cfg, step, role):
    if args.save_every and (step + 1) % args.save_every == 0 and (step + 1) < args.steps:
        save_merged(args, model, tokenizer, cfg, role=role, suffix=f"_step{step+1}")


def save_merged(args, model, tokenizer, cfg, role, suffix=""):
    out = args.save_dir + suffix
    print(f"[{role}] merging LoRA + saving -> {out}", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)
    # carry the NLA sidecar so config.py resolves the model sidecar downstream
    src = Path(args.sidecar + ".nla_meta.yaml")
    if not src.exists():
        src = Path(args.base_ckpt) / "nla_meta.yaml"
    if src.exists():
        shutil.copy(src, Path(out) / "nla_meta.yaml")
    print(f"[{role}] saved {out}", flush=True)


_WANDB = {"run": None, "init": False}


def wandb_log(args, d):
    if args.no_wandb:
        return
    if not _WANDB["init"]:
        try:
            import wandb
            _WANDB["run"] = wandb.init(project=args.wandb_project, name=args.wandb_name or f"sft_{args.role}",
                                       config=vars(args), reinit=True)
        except Exception as e:
            print(f"[wandb] disabled: {e}", flush=True)
            args.no_wandb = True
            return
        _WANDB["init"] = True
    if _WANDB["run"] is not None:
        _WANDB["run"].log(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["av", "ar"])
    p.add_argument("--base-ckpt", required=True, help="AV: base causal LM; AR: critic_init dir")
    p.add_argument("--parquet", required=True)
    p.add_argument("--sidecar", required=True, help="parquet whose .nla_meta.yaml has the NLA constants")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--max-rows", type=int, default=80000, help="cap parquet rows loaded (memory)")
    p.add_argument("--save-every", type=int, default=0, help="0 = only at end")
    p.add_argument("--wandb-project", default="nla-lenpen")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cfg = load_nla_config(args.sidecar, tokenizer)
    t0 = time.time()
    samples = load_samples(args.parquet, "prompt", cfg.injection_char, args.max_rows,
                           want_response=(args.role == "av"))
    print(f"[{args.role}] loaded {len(samples)} samples in {time.time()-t0:.1f}s", flush=True)

    if args.role == "av":
        train_av(args, tokenizer, cfg, samples, device)
    else:
        train_ar(args, tokenizer, cfg, samples, device)


if __name__ == "__main__":
    main()
