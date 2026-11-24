"""
Mission CV-GENESIS-NR-010: ChakrGenesis-17M Foundation Pretraining Script.

Architecture:
- 17,063,808 parameters (15,490,944 non-tied)
- 10 layers, d_model 384, 12 query heads, 3 KV heads (GQA 4:1)
- SwiGLU FFN (hidden_dim 1024), RoPE, Pre-RMSNorm

Data:
- Canonical Stage-C Tokenized Shards (data/tokenized/stage_c/train, 27 shards, 6.65M tokens)
- Validation Shards (data/tokenized/stage_c/validation, 2 shards, 465k tokens)
- Sequence Length: 256 tokens per context window
- Batch Size: 16 (effective batch = 4,096 tokens/step)
- Optimizer: AdamW (lr 4e-4, cosine annealing to 4e-5, weight decay 0.01)
- Automatic Mixed Precision (torch.amp)
- Gradient Clipping: 1.0
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"


class StageCDataset(Dataset):
    def __init__(self, shard_files: list[str], seq_len: int = 256):
        self.seq_len = seq_len
        tokens_list = []
        for sf in sorted(shard_files):
            data = np.fromfile(sf, dtype=np.uint16)
            tokens_list.append(data)
        self.all_tokens = np.concatenate(tokens_list)
        self.total_tokens = len(self.all_tokens)
        self.num_samples = (self.total_tokens - 1) // self.seq_len
        print(f"Loaded {len(shard_files)} shards -> {self.total_tokens:,} tokens ({self.num_samples} samples of seq_len={seq_len})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.all_tokens[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def evaluate_validation(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, loss = model(x, targets=y)
            total_loss += loss.item() * len(x)
            total_samples += len(x)
    model.train()
    avg_loss = total_loss / max(1, total_samples)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/tokenized/stage_c/train")
    parser.add_argument("--val_dir", type=str, default="data/tokenized/stage_c/validation")
    parser.add_argument("--output_dir", type=str, default="experiments/genesis_pretraining_010/checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RUNTIME] Training ChakrGenesis-17M on {device}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Tokenizer validation
    vocab_path = Path("data/experiments/vocab_4096/vocab.json")
    with open(vocab_path, "rb") as f:
        vh = hashlib.sha256(f.read()).hexdigest()
    assert vh == EXPECTED_VOCAB_HASH, "Vocab mismatch!"
    print(f"[VERIFIED] Tokenizer Vocab SHA-256: {vh}")

    # 2. Shards
    train_shards = glob.glob(os.path.join(args.train_dir, "*.bin"))
    val_shards = glob.glob(os.path.join(args.val_dir, "*.bin"))
    assert len(train_shards) > 0, "No training shards found!"
    assert len(val_shards) > 0, "No validation shards found!"

    train_ds = StageCDataset(train_shards, seq_len=args.seq_len)
    val_ds = StageCDataset(val_shards, seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Model instantiation
    cfg = Genesis17MConfig(max_seq_len=args.seq_len)
    model = ChakrGenesis17M(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ChakrGenesis-17M instantiated with {total_params:,} parameters.")

    # 4. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Initial loss
    init_val_loss, init_val_ppl = evaluate_validation(model, val_loader, device)
    print(f"[INITIAL] Val Loss: {init_val_loss:.4f} | Val Perplexity: {init_val_ppl:.2f}")

    best_val_loss = init_val_loss
    best_ckpt = os.path.join(args.output_dir, "genesis_17m_pretrain_best.pt")
    final_ckpt = os.path.join(args.output_dir, "genesis_17m_pretrain_final.pt")

    global_step = 0
    start_time = time.time()
    training_log = []

    for epoch in range(args.epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, loss = model(x, targets=y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            if global_step % 200 == 0 or step == len(train_loader) - 1:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {cur_lr:.2e}")
                training_log.append({
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "global_step": global_step,
                    "loss": loss.item(),
                    "lr": cur_lr
                })

        # Epoch end validation
        val_loss, val_ppl = evaluate_validation(model, val_loader, device)
        print(f"\n[EPOCH {epoch+1} EVAL] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_ppl": val_ppl
            }, best_ckpt)
            print(f"--> Saved NEW BEST checkpoint to {best_ckpt}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": args.epochs,
        "val_loss": val_loss,
        "val_ppl": val_ppl
    }, final_ckpt)
    print(f"--> Saved FINAL checkpoint to {final_ckpt}")

    duration = time.time() - start_time
    print(f"\n[COMPLETE] 17M Pretraining complete in {duration:.1f}s | Best Val Loss: {best_val_loss:.4f}")

    # Save log
    log_file = Path("experiments/genesis_pretraining_010/training_log.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({"best_val_loss": best_val_loss, "duration_seconds": duration, "steps": training_log}, f, indent=2)


if __name__ == "__main__":
    main()
