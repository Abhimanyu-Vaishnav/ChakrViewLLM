"""
train_v011.py - Training Pipeline for ChakrView Conversational Model v0.11

Lineage:
- Parent: releases/v0.10.1_conversational_model/chakrview_v0101_model.pt (SHA-256: 548b20a7...)
- Dataset: cr21_train.jsonl (1,080 high-diversity dialogues across 100+ entities and multilingual QA)
- Target: releases/v0.11_conversational_model/chakrview_v011_model.pt
"""

import os
import sys
import math
import json
import time
import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.conversation.cr_exp_17.conversation_formatter import ConversationFormatter
from experiments.conversation.cr_exp_17.conversation_loss import ConversationLoss

TOKENIZER_DIR = REPO_ROOT / "data/experiments/vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

PARENT_CKPT = REPO_ROOT / "releases/v0.10.1_conversational_model/chakrview_v0101_model.pt"

class ConversationalDataset(Dataset):
    def __init__(self, jsonl_path: Path, formatter: ConversationFormatter):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                prompt = rec["prompt"]
                response = rec["response"]
                inp_ids, labels = formatter.prepare_training_pair(prompt, response)
                self.samples.append((inp_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    max_len = min(512, max(len(inp) for inp, _ in batch))
    input_ids = []
    labels = []
    for inp, lab in batch:
        inp = inp[:max_len]
        lab = lab[:max_len]
        pad_len = max_len - len(inp)
        input_ids.append(inp + [2] * pad_len)
        labels.append(lab + [-100] * pad_len)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss, acc = loss_fn(logits, batch_y)
            total_loss += loss.item()
            total_acc += acc.item()
            steps += 1
    model.train()
    return total_loss / max(1, steps), total_acc / max(1, steps)

def train(seed: int = 5050, total_steps: int = 400, lr_max: float = 4.5e-5, lr_min: float = 4.5e-6, batch_size: int = 4):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    torch.set_num_threads(14)

    base_dir = REPO_ROOT / "experiments" / "cognition" / "cr_exp_21"
    train_dir = base_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_dir = base_dir / "data"
    train_file = data_dir / "cr21_train.jsonl"
    val_file = data_dir / "cr21_validation.jsonl"

    formatter = ConversationFormatter(tokenizer, max_seq_len=512)
    train_ds = ConversationalDataset(train_file, formatter)
    val_ds = ConversationalDataset(val_file, formatter)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512
    )
    model = ChakrMicro(cfg)

    print(f"Loading base checkpoint from: {PARENT_CKPT}")
    parent_sha = get_file_sha256(PARENT_CKPT)
    print(f"Parent Checkpoint SHA-256: {parent_sha}")
    assert parent_sha == "548b20a73d3cb3a662bcbfe74ab4f108d7bfdc4c60e56abd76049b260fa16a3b"

    ckpt_data = torch.load(PARENT_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01, betas=(0.9, 0.95))
    loss_fn = ConversationLoss(ignore_index=-100)

    val_loss_init, val_acc_init = evaluate(model, val_loader, loss_fn, device)
    print(f"Initial Validation: Loss = {val_loss_init:.4f} | Token Accuracy = {val_acc_init*100:.2f}%\n")

    global_step = 0
    best_val_loss = float("inf")
    best_step = 0
    best_ckpt_file = train_dir / "v011_candidate_best.pt"

    t0 = time.time()
    warmup_steps = int(total_steps * 0.10)
    train_logs = []

    while global_step < total_steps:
        for batch_x, batch_y in train_loader:
            global_step += 1
            if global_step > total_steps:
                break

            if global_step < warmup_steps:
                cur_lr = lr_min + (lr_max - lr_min) * (global_step / max(1, warmup_steps))
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cur_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss, token_acc = loss_fn(logits, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if global_step % 25 == 0 or global_step == total_steps:
                val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
                train_logs.append({
                    "step": global_step,
                    "train_loss": round(loss.item(), 4),
                    "train_acc": round(token_acc.item(), 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                    "lr": cur_lr
                })
                print(f"Step {global_step:3d}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {cur_lr:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = global_step
                    torch.save({
                        "step": best_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_acc": val_acc,
                        "seed": seed,
                        "config": cfg.__dict__,
                        "parent_sha256": parent_sha,
                        "train_file_sha256": get_file_sha256(train_file)
                    }, best_ckpt_file)

    elapsed = time.time() - t0
    final_sha = get_file_sha256(best_ckpt_file)
    print(f"\nv0.11 Training Complete in {elapsed:.1f}s. Best Step: {best_step} (Val Loss: {best_val_loss:.4f})")
    print(f"Best Candidate Saved: {best_ckpt_file.name} | SHA-256: {final_sha}")

    summary = {
        "candidate": "v011_candidate",
        "seed": seed,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "checkpoint_file": best_ckpt_file.name,
        "sha256": final_sha,
        "elapsed_sec": round(elapsed, 2),
        "history": train_logs
    }

    with open(results_dir / "v011_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    train(seed=5050, total_steps=400)

