"""
UCIP-09R Neural Language Intelligence Recovery Trainer.

Key Enhancements:
1. Pure Natural Prompt Stems (No artificial suffix leakage or synthetic memorization triggers).
2. Clean Mathematical Alignment: Direct canonical arithmetic pairs (e.g. 'What is 2 + 2?' -> '4').
3. Verbatim Exact Copying Pairs.
4. Strict Autoregressive Shift & Target Masking.
5. Periodic Diagnostic Rollouts on Owner Verification Prompts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from scripts.run_m3_evaluations import DirectNeuralRunner


def collate_example(example: Dict[str, str], tokenizer, max_seq_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt = example["prompt"]
    target = example["target"]

    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)

    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]

    labels = [-100] * len(prompt_ids) + target_ids
    if len(labels) > max_seq_len:
        labels = labels[:max_seq_len]

    return torch.tensor([input_ids], dtype=torch.long), torch.tensor([labels], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=======================================================")
    print(f"Starting UCIP-09R Natural Recovery Training")
    print(f"Steps={args.steps} BatchSize={args.batch_size} LR={args.lr} Seed={args.seed}")
    print("=======================================================")

    tokenizer_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)

    train_path = REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_train.jsonl"
    val_path = REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_val.jsonl"

    train_data = [json.loads(line) for line in open(train_path, encoding="utf-8") if line.strip()]
    val_data = [json.loads(line) for line in open(val_path, encoding="utf-8") if line.strip()]
    print(f"Loaded {len(train_data)} train samples and {len(val_data)} validation samples.")

    parent_ckpt = REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt"
    print(f"Loading parent weights from: {parent_ckpt.name}")

    runner = DirectNeuralRunner(parent_ckpt)
    model = runner.model
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.03)

    save_dir = REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_candidate_path = save_dir / "candidate_best.pt"
    best_val_loss = float("inf")

    indices = list(range(len(train_data)))
    random.shuffle(indices)
    data_idx = 0
    accum_loss = 0.0
    t_start = time.time()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        b_loss = 0.0

        for _ in range(args.batch_size):
            if data_idx >= len(indices):
                random.shuffle(indices)
                data_idx = 0

            ex = train_data[indices[data_idx]]
            data_idx += 1

            inp, tgt = collate_example(ex, tokenizer, max_seq_len=256)
            logits = model(inp)

            # Strict causal shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = tgt[:, 1:].contiguous()

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
            loss = loss / args.batch_size
            loss.backward()
            b_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        accum_loss += b_loss

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_loss = 0.0
            val_steps = min(32, len(val_data))
            with torch.no_grad():
                for v_i in range(val_steps):
                    v_ex = val_data[v_i]
                    v_inp, v_tgt = collate_example(v_ex, tokenizer, max_seq_len=256)
                    v_logits = model(v_inp)
                    v_shift_logits = v_logits[:, :-1, :].contiguous()
                    v_shift_targets = v_tgt[:, 1:].contiguous()
                    v_l = criterion(v_shift_logits.view(-1, v_shift_logits.size(-1)), v_shift_targets.view(-1))
                    val_loss += v_l.item()

            avg_val_loss = val_loss / val_steps
            avg_train_loss = accum_loss / args.eval_every
            accum_loss = 0.0
            elapsed = time.time() - t_start

            print(f"\nStep {step:4d}/{args.steps} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Elapsed: {elapsed:.1f}s")

            # Check rollouts on representative prompts
            test_prompts = [
                ("GREETING", "User: Hello ChakrView. How are you?\nAssistant:"),
                ("CPU", "User: What does a central processing unit (CPU) do?\nAssistant:"),
                ("MATH_2+2", "User: What is 2 + 2?\nAssistant:"),
                ("MATH_5-2", "User: What is 5 - 2?\nAssistant:"),
                ("COPY", "User: Say exactly: The sky is blue.\nAssistant:"),
                ("HINDI", "User: नमस्ते! आप कौन हैं?\nAssistant:")
            ]

            for label, p_str in test_prompts:
                p_toks = tokenizer.encode(p_str, add_bos=True, add_eos=False)
                c_toks = list(p_toks)
                g_toks = []
                with torch.no_grad():
                    for _ in range(25):
                        inp_t = torch.tensor([c_toks], dtype=torch.long)
                        if inp_t.shape[1] > 256:
                            inp_t = inp_t[:, -256:]
                        l = model(inp_t)[:, -1, :].squeeze(0)
                        nxt = int(torch.argmax(l).item())
                        if nxt == 1:
                            break
                        g_toks.append(nxt)
                        c_toks.append(nxt)
                gen_s = tokenizer.decode(g_toks, errors="replace").strip()
                print(f"  [{label}] {repr(gen_s[:80])}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": model.config.__dict__ if hasattr(model, "config") else {},
                    "architecture": "ChakrMedium",
                    "parent_checkpoint": str(parent_ckpt.name)
                }, best_candidate_path)
                print(f"  >>> Saved new best candidate model to: {best_candidate_path.name} (Val Loss: {avg_val_loss:.4f})")

            model.train()

    print(f"\nTraining completed! Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
