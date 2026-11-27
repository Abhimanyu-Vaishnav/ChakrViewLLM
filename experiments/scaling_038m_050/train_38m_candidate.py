"""
experiments/scaling_038m_050/train_38m_candidate.py
Controlled Scaling Experiment (NR-050):
- Model: ChakrGenesis-38M (37.50M unique trainable parameters)
- Architecture: 12 layers, d_model=512, 8 Q-heads, 4 KV-heads (GQA), hidden_dim=1408, RoPE, RMSNorm, SwiGLU.
- Dataset: Corpus V3 (data/corpus_v3/corpus_v3_train.jsonl) - identical to RC3.
- Tokenizer: Canonical Byte-Level BPE 4096 (data/experiments/vocab_4096).
- Supervision: Verified response-only cross-entropy loss (prompt tokens masked with -100).
- Optimizer: AdamW (lr=1.5e-4, weight_decay=0.01), Cosine Annealing, FP16 AMP.
- Checkpoint output: experiments/scaling_038m_050/chakr_genesis_38m_best.pt
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

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sys.path.insert(0, ".")

from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
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
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            num_tokens = (targets != -100).sum().item()
            if num_tokens > 0:
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
    model.train()
    return total_loss / max(1, total_tokens)

def train_38m():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== TRAINING CHAKRGENESIS-38M CANDIDATE (NR-050) ON {device.upper()} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    cfg = Genesis38MConfig(
        vocab_size=4096,
        d_model=512,
        n_layers=12,
        n_q_heads=8,
        n_kv_heads=4,
        hidden_dim=1408,
        max_seq_len=256,
        tie_embeddings=True
    )

    model = ChakrGenesis38M(cfg)
    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Model Initialized: {unique_params:,} unique trainable parameters ({unique_params/1e6:.2f}M)")
    model.to(device)

    train_path = "data/corpus_v3/corpus_v3_train.jsonl"
    val_path = "data/corpus_v3/corpus_v3_val.jsonl"

    train_ds = ResponseMaskedDataset(train_path, tokenizer, max_seq_len=256)
    val_ds = ResponseMaskedDataset(val_path, tokenizer, max_seq_len=256)

    # Use micro-batch size 8 with gradient accumulation steps 2 for identical effective batch size 16
    micro_batch = 8
    accum_steps = 2
    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=micro_batch, shuffle=False)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val | Effective Batch Size: {micro_batch * accum_steps}")

    epochs = 2
    total_steps = (len(train_loader) // accum_steps) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    out_dir = Path("experiments/scaling_038m_050")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "chakr_genesis_38m_best.pt"

    best_val_loss = float("inf")
    step_count = 0
    history = []

    start_time = time.time()
    optimizer.zero_grad()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        tokens_seen = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()

            num_tokens = (targets != -100).sum().item()
            epoch_loss += loss.item() * num_tokens
            tokens_seen += num_tokens

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

                if step_count % 50 == 0 or (batch_idx + 1) == len(train_loader):
                    cur_train_loss = epoch_loss / max(1, tokens_seen)
                    val_loss = evaluate(model, val_loader, criterion, device)
                    ppl = math.exp(min(20, val_loss))
                    print(f"Epoch [{epoch}/{epochs}] Step [{step_count}/{total_steps}] | Train Loss: {cur_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {ppl:.2f}")

                    history.append({
                        "epoch": epoch,
                        "step": step_count,
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
                            "unique_params": unique_params,
                            "architecture": "ChakrGenesis-38M",
                            "stage": "NR050_Controlled_Scaling_Study"
                        }, best_ckpt_path)
                        print(f"  * Saved new best 38M checkpoint: {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\n38M Training completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    log_path = out_dir / "NR050_38M_TRAINING_LOG.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "ChakrGenesis-38M",
            "unique_parameters": unique_params,
            "total_time_seconds": total_time,
            "best_val_loss": best_val_loss,
            "checkpoint_sha256": file_sha256(best_ckpt_path),
            "peak_vram_allocated_mb": torch.cuda.max_memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0,
            "peak_vram_reserved_mb": torch.cuda.max_memory_reserved() / (1024*1024) if torch.cuda.is_available() else 0,
            "history": history
        }, f, indent=2)
    print(f"Training log written to {log_path}")

if __name__ == "__main__":
    train_38m()
