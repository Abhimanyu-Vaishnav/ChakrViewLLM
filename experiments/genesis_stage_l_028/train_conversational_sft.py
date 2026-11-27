"""
experiments/genesis_stage_l_028/train_conversational_sft.py
Fine-tunes ChakrGenesis-17M into the ChakrView 1.0 Conversational Release Candidate Model:
- Initializes weights from verified Stage-D Canonical Best (2c21632e...) to preserve pretraining base.
- Trains on deduplicated, leakage-free Master Conversational Corpus V1 (5,826 train samples across 22 capability areas).
- Uses AdamW with Cosine Annealing, FP16 mixed precision, gradient clipping (1.0), and eval validation.
- Saves best release checkpoint to experiments/genesis_stage_l_028/checkpoints/chakr_1_0_rc1_best.pt.
"""

import sys
import os
import json
import time
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sys.path.insert(0, '.')

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class ConversationalDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                p = d.get("prompt", "").strip()
                r = d.get("response", "").strip()
                if p and r:
                    self.samples.append((p, r))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        # Standard ChakrView conversational format
        formatted = f"<user>\n{prompt}\n<assistant>\n{response}<eos>"
        tokens = self.tokenizer.encode(formatted)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [2] * pad_len
            target_ids = target_ids + [-100] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def train_sft():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== TRAINING CHAKRVIEW 1.0 CONVERSATIONAL MODEL ON {device.upper()} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    
    # Load canonical Stage-D best checkpoint as initialization base
    init_path = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    st = torch.load(init_path, map_location="cpu", weights_only=False)
    raw_cfg = st["config"]
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)

    model = ChakrGenesis17M(cfg)
    model.load_state_dict(st["model_state_dict"])
    model.to(device)

    train_ds = ConversationalDataset("data/corpus_v1/master_corpus_train.jsonl", tokenizer, max_seq_len=256)
    val_ds = ConversationalDataset("data/corpus_v1/master_corpus_val.jsonl", tokenizer, max_seq_len=256)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset Loaded: {len(train_ds)} train samples, {len(val_ds)} val samples | Batch Size: {batch_size}")

    epochs = 2
    total_steps = len(train_loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    out_dir = Path("experiments/genesis_stage_l_028/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_checkpoint_path = out_dir / "chakr_1_0_rc1_best.pt"

    t0 = time.time()
    step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for b_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, loss = model(x, targets=y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if (b_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [Epoch {epoch+1}/{epochs} | Step {step}/{total_steps}] Train Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Elapsed: {elapsed:.1f}s", flush=True)

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    _, vl = model(vx, targets=vy)
                val_loss += vl.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        val_ppl = math.exp(min(15.0, avg_val_loss))
        print(f"--> [Epoch {epoch+1} Completed] Validation Loss: {avg_val_loss:.4f} | Validation PPL: {val_ppl:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch + 1,
                "step": step,
                "config": cfg,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "val_ppl": val_ppl,
                "vocab_size": cfg.vocab_size
            }, best_checkpoint_path)
            print(f"    *** New Best Checkpoint Saved: {best_checkpoint_path} (Val Loss: {avg_val_loss:.4f}) ***")

    print(f"\nTraining Complete in {time.time()-t0:.1f}s! Best Validation Loss: {best_val_loss:.4f}")
    return str(best_checkpoint_path)

if __name__ == "__main__":
    train_sft()
