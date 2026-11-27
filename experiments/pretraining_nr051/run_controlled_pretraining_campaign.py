"""
experiments/pretraining_nr051/run_controlled_pretraining_campaign.py
Executes Phase 4 & 5: Controlled Pretraining Comparison between 17M and 38M:
- Both models initialized from fresh independent random initialization (seed 42).
- Both models trained on identical Pretraining Corpus (data/pretraining_nr051/pretrain_train.jsonl).
- Same sequence length policy (256), same optimizer (AdamW), same cosine learning rate schedule.
- Same token budget: exactly 1 epoch (~1.02M tokens) across identical batches.
- Tracks per-language validation loss, per-token accuracy, throughput, and peak CUDA VRAM.
- Saves checkpoints:
  * experiments/pretraining_nr051/chakr_17m_pretrained_base.pt
  * experiments/pretraining_nr051/chakr_38m_pretrained_base.pt
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

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class CausalPretrainingDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                txt = d.get("text", "").strip()
                lang = d.get("language", "en")
                if txt:
                    self.samples.append((txt, lang))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, lang = self.samples[idx]
        tokens = self.tokenizer.encode(text)
        tokens = [0] + tokens + [1]
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [2] * pad_len
            target_ids = target_ids + [-100] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long), lang

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    lang_stats = {}

    with torch.no_grad():
        for inputs, targets, langs in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                preds = torch.argmax(logits, dim=-1)

            mask = (targets != -100)
            num_tokens = mask.sum().item()
            correct = (preds[mask] == targets[mask]).sum().item()

            if num_tokens > 0:
                total_loss += loss.item() * num_tokens
                total_correct += correct
                total_tokens += num_tokens

            # Per-language stats
            for i, lang in enumerate(langs):
                if lang not in lang_stats:
                    lang_stats[lang] = {"loss": 0.0, "tokens": 0}
                t_mask = (targets[i] != -100)
                n_t = t_mask.sum().item()
                if n_t > 0:
                    l_loss = criterion(logits[i].view(-1, logits.size(-1)), targets[i].view(-1)).item()
                    lang_stats[lang]["loss"] += l_loss * n_t
                    lang_stats[lang]["tokens"] += n_t

    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    accuracy = (total_correct / max(1, total_tokens)) * 100
    ppl = math.exp(min(20, mean_loss))

    per_lang_results = {}
    for lang, s in lang_stats.items():
        if s["tokens"] > 0:
            per_lang_results[lang] = round(s["loss"] / s["tokens"], 4)

    return mean_loss, accuracy, ppl, per_lang_results

def train_candidate(model_name, model, train_loader, val_loader, out_dir, device="cuda"):
    print(f"\n=======================================================")
    print(f"TRAINING {model_name.upper()} FROM SCRATCH (SEED 42)")
    print(f"=======================================================")

    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Unique Parameters: {unique_params:,} ({unique_params/1e6:.2f}M)")

    model.to(device)
    total_steps = len(train_loader)
    lr = 1.5e-4 if "38m" in model_name.lower() else 2.0e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_ckpt_path = out_dir / f"chakr_{model_name.lower()}_pretrained_base.pt"
    best_val_loss = float("inf")
    history = []

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    tokens_processed = 0

    model.train()
    for step, (inputs, targets, _) in enumerate(train_loader):
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
        tokens_processed += num_tokens

        if (step + 1) % 100 == 0 or (step + 1) == total_steps:
            val_loss, val_acc, val_ppl, per_lang = evaluate_model(model, val_loader, criterion, device)
            elapsed = time.time() - start_time
            tok_per_sec = tokens_processed / max(1, elapsed)
            print(f"Step [{step+1}/{total_steps}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | PPL: {val_ppl:.2f} | Tok/s: {tok_per_sec:.0f}")

            history.append({
                "step": step + 1,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_ppl": val_ppl,
                "per_language_loss": per_lang,
                "tokens_seen": tokens_processed
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_config = getattr(model, "config", getattr(model, "cfg", None))
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "step": step + 1,
                    "val_loss": best_val_loss,
                    "val_acc": val_acc,
                    "unique_params": unique_params,
                    "model_family": model_name,
                    "stage": "NR051_Pretraining_Base"
                }, best_ckpt_path)

    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0

    print(f"\n{model_name} Pretraining Complete in {total_time/60:.2f} minutes.")
    print(f"Best Val Loss: {best_val_loss:.4f} | Peak VRAM: {peak_vram:.1f} MB")
    print(f"Saved to: {best_ckpt_path} (SHA-256: {file_sha256(best_ckpt_path)})")

    return {
        "model_name": model_name,
        "unique_parameters": unique_params,
        "total_time_seconds": total_time,
        "tokens_processed": tokens_processed,
        "best_val_loss": best_val_loss,
        "final_accuracy": history[-1]["val_accuracy"],
        "final_ppl": history[-1]["val_ppl"],
        "peak_vram_mb": peak_vram,
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_sha256": file_sha256(best_ckpt_path),
        "history": history
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== NR-051 CONTROLLED PRETRAINING CAMPAIGN ({device.upper()}) ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    out_dir = Path("experiments/pretraining_nr051")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = "data/pretraining_nr051/pretrain_train.jsonl"
    val_path = "data/pretraining_nr051/pretrain_val.jsonl"

    train_ds = CausalPretrainingDataset(train_path, tokenizer, max_seq_len=256)
    val_ds = CausalPretrainingDataset(val_path, tokenizer, max_seq_len=256)

    # Batch size 16 for both models
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset Ready: {len(train_ds):,} train samples, {len(val_ds):,} val samples.")

    # 1. Initialize ChakrGenesis-17M from fresh random seed 42
    torch.manual_seed(42)
    cfg_17m = Genesis17MConfig(max_seq_len=256)
    model_17m = ChakrGenesis17M(cfg_17m)

    res_17m = train_candidate("17M", model_17m, train_loader, val_loader, out_dir, device=device)

    # Clean memory between runs
    del model_17m
    torch.cuda.empty_cache()

    # 2. Initialize ChakrGenesis-38M from fresh random seed 42
    torch.manual_seed(42)
    cfg_38m = Genesis38MConfig(max_seq_len=256)
    model_38m = ChakrGenesis38M(cfg_38m)

    res_38m = train_candidate("38M", model_38m, train_loader, val_loader, out_dir, device=device)

    # Save comparative log
    summary_path = out_dir / "NR051_PRETRAINING_SUMMARY.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"17M": res_17m, "38M": res_38m}, f, indent=2)

    print(f"\nPretraining summary written to {summary_path}")

if __name__ == "__main__":
    main()
