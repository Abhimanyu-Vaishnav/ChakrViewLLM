"""
UCIP-09S Controlled Balanced Replay Trainer.

Evaluates Condition C (Balanced Replay):
- 70% Conversational, Multilingual, and Instruction Replay Anchors
- 30% Diverse Arithmetic and Exact Copying
- Learning Rate: 1.8e-5 (Conservative)
- Label Smoothing: 0.05 (Prevents single-digit collapse)
- Checkpoint validation on balanced split
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.8e-5)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=======================================================")
    print(f"Starting UCIP-09S Condition C (Balanced Replay) Training")
    print(f"Steps={args.steps} BatchSize={args.batch_size} LR={args.lr} Seed={args.seed}")
    print("=======================================================")

    tokenizer_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)

    train_path = REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_train.jsonl"
    val_path = REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_val.jsonl"

    train_data = [json.loads(line) for line in open(train_path, encoding="utf-8") if line.strip()]
    val_data = [json.loads(line) for line in open(val_path, encoding="utf-8") if line.strip()]
    print(f"Loaded {len(train_data)} balanced train samples and {len(val_data)} validation samples.")

    parent_ckpt = REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt"
    print(f"Loading parent weights from: {parent_ckpt.name}")

    parent_raw = torch.load(parent_ckpt, map_location="cpu")
    saved_cfg = parent_raw.get("config", {})

    runner = DirectNeuralRunner(parent_ckpt)
    model = runner.model
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    save_dir = REPO_ROOT / "experiments" / "ucip09s" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_candidate_path = save_dir / "ucip09s_balanced_pilot.pt"
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

            print(f"Step {step:3d}/{args.steps} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Elapsed: {elapsed:.1f}s")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"  --> New best validation loss: {best_val_loss:.4f}. Saving checkpoint.")
                save_dict = {
                    "architecture": "ChakrMedium",
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "val_loss": best_val_loss,
                    "config": saved_cfg,
                    "parent_checkpoint": str(parent_ckpt.name)
                }
                torch.save(save_dict, best_candidate_path)

            model.train()

    print(f"\nTraining complete. Best checkpoint saved to: {best_candidate_path}")

if __name__ == "__main__":
    main()
