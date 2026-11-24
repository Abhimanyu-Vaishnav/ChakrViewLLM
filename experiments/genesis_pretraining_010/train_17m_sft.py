"""
Mission CV-GENESIS-NR-010: ChakrGenesis-17M Conversational SFT Trainer.
Initializes strictly from genesis_17m_pretrain_best.pt.
Fine-tunes on the balanced multilingual corpus (experiments/genesis_balanced_sft_008/data/).

Contract Invariants:
- Architecture: ChakrGenesis-17M (17,063,808 parameters)
- EOS ID: 1 (<EOS>)
- PAD ID: 2 (<PAD>)
- Masked loss over assistant response tokens only (-100 for prompt)
- Automatic mixed precision (torch.amp)
- Output checkpoints stored in experiments/genesis_pretraining_010/sft_checkpoints/
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"


class SFT17MDataset(Dataset):
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


def evaluate_sft_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

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
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    model.train()
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint", type=str, default="experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt")
    parser.add_argument("--data_dir", type=str, default="experiments/genesis_balanced_sft_008/data")
    parser.add_argument("--output_dir", type=str, default="experiments/genesis_pretraining_010/sft_checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RUNTIME] Training 17M SFT on {device}")

    # Tokenizer
    tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

    # Load 17M pretrained weights
    init_path = Path(args.init_checkpoint)
    assert init_path.exists(), f"Pretrained checkpoint {init_path} missing!"
    print(f"[INIT] Loading 17M pretrained weights from {init_path.name}")
    
    cfg = Genesis17MConfig(max_seq_len=256)
    model = ChakrGenesis17M(cfg).to(device)
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Loaders
    train_ds = SFT17MDataset(os.path.join(args.data_dir, "train.jsonl"), tok)
    val_ds = SFT17MDataset(os.path.join(args.data_dir, "validation.jsonl"), tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    init_val_loss, init_val_ppl = evaluate_sft_loss(model, val_loader, device)
    print(f"[INITIAL] Val Loss: {init_val_loss:.4f} | Val PPL: {init_val_ppl:.2f}")

    best_val_loss = init_val_loss
    best_ckpt = os.path.join(args.output_dir, "genesis_17m_sft_best.pt")
    final_ckpt = os.path.join(args.output_dir, "genesis_17m_sft_final.pt")

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
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

            if (step + 1) % 100 == 0 or step == len(train_loader) - 1:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {cur_lr:.2e}")

        val_loss, val_ppl = evaluate_sft_loss(model, val_loader, device)
        print(f"\n[EPOCH {epoch+1} SUMMARY] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_ppl": val_ppl
            }, best_ckpt)
            print(f"--> Saved NEW BEST 17M SFT checkpoint to {best_ckpt}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": args.epochs,
        "val_loss": val_loss,
        "val_ppl": val_ppl
    }, final_ckpt)
    print(f"--> Saved FINAL 17M SFT checkpoint to {final_ckpt}")


if __name__ == "__main__":
    main()
