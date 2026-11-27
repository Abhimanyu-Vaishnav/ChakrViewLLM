"""
experiments/pretraining_nr052/run_controlled_pretraining_v4.py

Phase 5: Controlled Pretraining Campaign on Corpus V4
Trains both ChakrGenesis-17M and ChakrGenesis-38M under strictly identical conditions:
- Random initialization (seed=42)
- Same canonical tokenizer (vocab_size=4096)
- Same Corpus V4 train and validation splits (data/corpus_v4/corpus_v4_train.jsonl)
- Matched token sequences, batch size, learning rate schedule, and cumulative step budget
- Evaluates validation loss, perplexity, next-token accuracy, GPU memory, and throughput
- Saves checkpoints to experiments/pretraining_nr052/
- Outputs experiments/pretraining_nr052/NR052_PRETRAINING_SUMMARY.json
"""

import sys
import os
import json
import time
import math
import hashlib
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sys.path.insert(0, ".")

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class PretrainDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                text = obj["text"]
                tokens = tokenizer.encode(text)
                if len(tokens) == 0: continue
                # Append EOS (id=2)
                tokens.append(2)
                # Chunk if longer than max_len
                for i in range(0, len(tokens), max_len):
                    chunk = tokens[i:i + max_len]
                    if len(chunk) >= 16:
                        self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        if len(chunk) < self.max_len:
            pad_len = self.max_len - len(chunk)
            input_ids = chunk + [0] * pad_len
        else:
            input_ids = chunk[:self.max_len]
        return torch.tensor(input_ids, dtype=torch.long)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            inp = batch[:, :-1].contiguous()
            tgt = batch[:, 1:].contiguous()
            mask = (tgt != 0)
            tgt_loss = tgt.clone()
            tgt_loss[~mask] = -100

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(inp)
                loss = criterion(logits.reshape(-1, 4096), tgt_loss.reshape(-1))

            n_active = mask.sum().item()
            total_loss += loss.item() * n_active
            total_tokens += n_active

            preds = logits.argmax(dim=-1)
            correct_tokens += ((preds == tgt) & mask).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    acc = (correct_tokens / max(1, total_tokens)) * 100.0
    return avg_loss, ppl, acc

def train_model(model_name, model, train_loader, val_loader, epochs=2, lr=5e-4, device="cuda", out_dir=Path("experiments/pretraining_nr052")):
    print(f"\n=======================================================", flush=True)
    print(f"TRAINING {model_name.upper()} ON CORPUS V4 ({device.upper()})", flush=True)
    print(f"=======================================================", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = min(100, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")
    best_ppl = float("inf")
    best_acc = 0.0
    best_ckpt_path = out_dir / f"chakr_{model_name.lower()}_v4_base.pt"

    history = []
    start_time = time.time()
    tokens_seen = 0
    step = 0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            batch = batch.to(device)
            inp = batch[:, :-1].contiguous()
            tgt = batch[:, 1:].contiguous()
            mask = (tgt != 0)
            tgt_loss = tgt.clone()
            tgt_loss[~mask] = -100

            tokens_seen += mask.sum().item()

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(inp)
                loss = criterion(logits.reshape(-1, 4096), tgt_loss.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 75 == 0 or step == total_steps:
                val_loss, val_ppl, val_acc = evaluate(model, val_loader, criterion, device)
                model.train()
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:4d}/{total_steps} | Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val Acc: {val_acc:.1f}% | LR: {cur_lr:.2e} | Tokens: {tokens_seen:,}", flush=True)

                history.append({
                    "step": step,
                    "epoch": epoch,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(val_loss, 4),
                    "val_ppl": round(val_ppl, 2),
                    "val_acc": round(val_acc, 2),
                    "tokens_seen": tokens_seen
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ppl = val_ppl
                    best_acc = val_acc
                    cfg_to_save = getattr(model, "config", getattr(model, "cfg", None))
                    torch.save({
                        "model_name": model_name,
                        "config": cfg_to_save,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "val_acc": val_acc,
                        "step": step,
                        "tokens_seen": tokens_seen
                    }, best_ckpt_path)

    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    print(f"\n[{model_name.upper()}] Best Val Loss: {best_val_loss:.4f} | Best PPL: {best_ppl:.2f} | Best Acc: {best_acc:.1f}%")
    print(f"Saved Checkpoint: {best_ckpt_path}")
    print(f"Training Time: {total_time:.2f} s ({total_time/60:.2f} min) | Peak VRAM: {peak_vram:.1f} MB")

    ckpt_sha = sha256_file(best_ckpt_path)

    return {
        "model_name": model_name,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_ppl,
        "best_val_acc": best_acc,
        "total_time_seconds": total_time,
        "peak_vram_mb": peak_vram,
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "tokens_seen": tokens_seen,
        "history": history
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("experiments/pretraining_nr052")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    train_dataset = PretrainDataset("data/corpus_v4/corpus_v4_train.jsonl", tokenizer, max_len=256)
    val_dataset = PretrainDataset("data/corpus_v4/corpus_v4_val.jsonl", tokenizer, max_len=256)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Dataset Loaded: {len(train_dataset)} train chunks, {len(val_dataset)} val chunks.", flush=True)

    # 1. Train 17M
    torch.manual_seed(42)
    cfg17 = Genesis17MConfig(vocab_size=4096, max_seq_len=256)
    model17 = ChakrGenesis17M(cfg17).to(device)
    res17 = train_model("17M", model17, train_loader, val_loader, epochs=2, lr=5e-4, device=device, out_dir=out_dir)
    del model17
    torch.cuda.empty_cache()

    # 2. Train 38M
    torch.manual_seed(42)
    cfg38 = Genesis38MConfig(vocab_size=4096, max_seq_len=256)
    model38 = ChakrGenesis38M(cfg38).to(device)
    res38 = train_model("38M", model38, train_loader, val_loader, epochs=2, lr=5e-4, device=device, out_dir=out_dir)
    del model38
    torch.cuda.empty_cache()

    # Save summary
    summary = {
        "17M": res17,
        "38M": res38,
        "comparison": {
            "loss_delta_38M_vs_17M": round(res38["best_val_loss"] - res17["best_val_loss"], 4),
            "ppl_delta_38M_vs_17M": round(res38["best_val_ppl"] - res17["best_val_ppl"], 2),
            "acc_delta_38M_vs_17M": round(res38["best_val_acc"] - res17["best_val_acc"], 2)
        }
    }

    out_summary = out_dir / "NR052_PRETRAINING_SUMMARY.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=======================================================")
    print(f"PRETRAINING COMPARISON SUMMARY SAVED TO {out_summary}")
    print(f"17M Best Val Loss: {res17['best_val_loss']:.4f} (PPL: {res17['best_val_ppl']:.2f}, Acc: {res17['best_val_acc']:.1f}%)")
    print(f"38M Best Val Loss: {res38['best_val_loss']:.4f} (PPL: {res38['best_val_ppl']:.2f}, Acc: {res38['best_val_acc']:.1f}%)")
    print(f"Scaling Advantage: {summary['comparison']['loss_delta_38M_vs_17M']} loss, {summary['comparison']['ppl_delta_38M_vs_17M']} PPL")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
