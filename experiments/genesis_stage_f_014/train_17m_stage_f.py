"""
ChakrView Stage-F Multi-Domain Balanced Training Pipeline.
Mission CV-GENESIS-NR-014 Phase 6.

Initializes from:
`experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`

Training Data:
`data/stage_f_corpus_014/stage_f_train.jsonl` (5,388 samples: 87% conversational, 13% diverse coding)
`data/stage_f_corpus_014/stage_f_val.jsonl` (570 samples)

Hyperparameters:
- Shifted autoregressive alignment: logits[..., :-1, :] vs labels[..., 1:]
- Prompt token loss masking (-100)
- Moderate peak learning rate: 4e-5 with cosine decay (preserving conversational foundation)
- Weight decay 0.01, gradient clipping 1.0
- Mixed precision AMP FP16 on CUDA
"""

from __future__ import annotations
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


class StageFDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 256):
        self.examples = []
        eos_id = 1
        pad_id = 2

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                p_text = item.get("prompt", "")
                r_text = item.get("response", "")

                p_ids = tokenizer.encode(p_text)
                r_ids = tokenizer.encode(" " + r_text) + [eos_id]

                input_ids = p_ids + r_ids
                labels = [-100] * len(p_ids) + r_ids

                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                    labels = labels[:max_seq_len]

                pad_len = max_seq_len - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [pad_id] * pad_len
                    labels = labels + [-100] * pad_len

                self.examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def compute_eval_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _ = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            valid_tokens = (shift_labels != -100).sum().item()
            if valid_tokens > 0:
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def train_stage_f(pilot: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-5)
    args, _ = parser.parse_known_args()

    is_pilot = pilot or args.pilot

    print("==========================================================")
    print(f"  CHAKRVIEW STAGE-F BALANCED SFT [{'PILOT' if is_pilot else 'FULL CAMPAIGN'}]")
    print("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active training device: {device}")

    # Load canonical tokenizer
    vocab_dir = "data/experiments/vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(vocab_dir, verify_checksums=False)

    train_file = "data/stage_f_corpus_014/stage_f_train.jsonl"
    val_file = "data/stage_f_corpus_014/stage_f_val.jsonl"
    train_dataset = StageFDataset(train_file, tokenizer, max_seq_len=256)
    val_dataset = StageFDataset(val_file, tokenizer, max_seq_len=256)
    print(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val examples.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize from verified Stage-D Best
    ckpt_path = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    print(f"Initializing model weights from: {ckpt_path}")
    config = Genesis17MConfig()
    model = ChakrGenesis17M(config)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = st.get("model_state_dict", st)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * (1 if is_pilot else args.epochs)
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    init_val_loss, init_val_ppl = compute_eval_loss(model, val_loader, criterion, device)
    print(f"[INITIAL] Baseline Stage-D Val Loss: {init_val_loss:.4f} | Val PPL: {init_val_ppl:.2f}")

    out_dir = Path("experiments/genesis_stage_f_014/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("experiments/genesis_stage_f_014/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = init_val_loss
    best_ckpt = out_dir / ("genesis_17m_stage_f_pilot_best.pt" if is_pilot else "genesis_17m_stage_f_best.pt")
    final_ckpt = out_dir / ("genesis_17m_stage_f_pilot_final.pt" if is_pilot else "genesis_17m_stage_f_final.pt")

    t0 = time.perf_counter()
    global_step = 0
    max_steps = 150 if is_pilot else total_steps

    for epoch in range(1 if is_pilot else args.epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _ = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if global_step % 50 == 0 or global_step == max_steps:
                elapsed = time.perf_counter() - t0
                tok_sec = (global_step * args.batch_size * 256) / elapsed
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step:4d}/{max_steps} | Loss: {loss.item():.4f} | LR: {cur_lr:.2e} | Speed: {tok_sec:.1f} tok/s")

            if global_step >= max_steps:
                break

        # Eval at epoch or pilot end
        val_loss, val_ppl = compute_eval_loss(model, val_loader, criterion, device)
        print(f"\n[EVAL] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, best_ckpt)
            print(f"  [+] Saved new best model to {best_ckpt}")

        if global_step >= max_steps:
            break

    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_ckpt)
    print(f"Final model saved to {final_ckpt}")

    with open(log_dir / ("pilot_log.json" if is_pilot else "stage_f_train_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "is_pilot": is_pilot,
            "initial_val_loss": init_val_loss,
            "best_val_loss": best_val_loss,
            "total_steps": global_step
        }, f, indent=2)


if __name__ == "__main__":
    is_p = "--pilot" in sys.argv
    train_stage_f(pilot=is_p)
