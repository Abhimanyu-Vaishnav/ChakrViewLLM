"""
experiments/genesis_stage_rc2_046/train_rc2_sft.py
Fine-tunes ChakrGenesis-17M into the ChakrView 1.0 RC2 Candidate:
- Initialized strictly from canonical Stage-D best (2c21632e...) to preserve pretraining base.
- Trained on isolated Corpus V2 (data/corpus_v2/corpus_v2_train.jsonl) with verified response-only supervision.
- Prompt tokens masked with -100 (0 gradient on prompt tokens).
- First response token supervised from final prompt token.
- Evaluates both full-sequence and response-only validation loss.
- Saves best candidate checkpoint to experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt.
"""

import sys
import os
import json
import time
import math
import random
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Deterministic seed
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sys.path.insert(0, ".")

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class ResponseMaskedDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                p = d.get("prompt", "").strip()
                r = d.get("response", "").strip()
                if p and r:
                    self.samples.append((p, r))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        
        prompt_str = f"<user>\n{prompt}\n<assistant>\n"
        response_str = f"{response}<eos>"
        
        prompt_tokens = self.tokenizer.encode(prompt_str)
        response_tokens = self.tokenizer.encode(response_str)
        
        full_tokens = prompt_tokens + response_tokens
        if len(full_tokens) > self.max_seq_len:
            full_tokens = full_tokens[:self.max_seq_len]
            
        P = len(prompt_tokens)
        labels = []
        for i in range(1, len(full_tokens)):
            if i < P:
                labels.append(-100)
            else:
                labels.append(full_tokens[i])
                
        input_ids = full_tokens[:-1]
        target_ids = labels
        
        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [2] * pad_len
            target_ids = target_ids + [-100] * pad_len
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            num_tokens = (targets != -100).sum().item()
            if num_tokens > 0:
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
    model.train()
    return total_loss / max(1, total_tokens)

def train_rc2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== TRAINING CHAKRVIEW 1.0 RC2 ON {device.upper()} ===")

    # 1. Load canonical tokenizer
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    # 2. Load canonical Stage-D best checkpoint as initialization base
    init_path = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    init_hash = file_sha256(init_path)
    print(f"Loading Stage-D initialization base: {init_path} (SHA-256: {init_hash})")
    assert init_hash == "2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269", "Base hash mismatch!"

    st = torch.load(init_path, map_location="cpu", weights_only=False)
    raw_cfg = st["config"]
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)

    model = ChakrGenesis17M(cfg)
    model.load_state_dict(st["model_state_dict"])
    model.to(device)

    # 3. Load Corpus V2
    train_path = "data/corpus_v2/corpus_v2_train.jsonl"
    val_path = "data/corpus_v2/corpus_v2_val.jsonl"
    
    train_ds = ResponseMaskedDataset(train_path, tokenizer, max_seq_len=256)
    val_ds = ResponseMaskedDataset(val_path, tokenizer, max_seq_len=256)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset Loaded: {len(train_ds)} train samples, {len(val_ds)} val samples | Batch Size: {batch_size}")

    epochs = 2
    total_steps = len(train_loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    out_dir = Path("experiments/genesis_stage_rc2_046")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "chakr_1_0_rc2_best.pt"
    
    best_val_loss = float("inf")
    step_count = 0
    history = []

    # Initial validation loss
    init_val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Stage-D Zero-Shot Validation Loss on Corpus V2: {init_val_loss:.4f} (Perplexity: {math.exp(min(20, init_val_loss)):.2f})")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        tokens_seen = 0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            num_tokens = (targets != -100).sum().item()
            epoch_loss += loss.item() * num_tokens
            tokens_seen += num_tokens
            step_count += 1

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                cur_train_loss = epoch_loss / max(1, tokens_seen)
                val_loss = evaluate(model, val_loader, criterion, device)
                ppl = math.exp(min(20, val_loss))
                print(f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] | Train Loss: {cur_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {ppl:.2f}")
                
                history.append({
                    "epoch": epoch,
                    "step": step + 1,
                    "train_loss": cur_train_loss,
                    "val_loss": val_loss,
                    "val_ppl": ppl
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "epoch": epoch,
                        "step": step_count,
                        "val_loss": best_val_loss,
                        "stage": "RC2_Conversational_SFT",
                        "base_checkpoint": "genesis_17m_staged_best.pt",
                        "base_hash": init_hash
                    }, best_ckpt_path)
                    print(f"  * Saved new best RC2 checkpoint: {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Save training log
    log_path = out_dir / "training_history.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_time_seconds": total_time,
            "best_val_loss": best_val_loss,
            "best_checkpoint_sha256": file_sha256(best_ckpt_path),
            "history": history
        }, f, indent=2)
    print(f"Training log written to {log_path}")

if __name__ == "__main__":
    train_rc2()
