"""
experiments/pretraining_nr052/run_controlled_sft_campaign.py

Phase 6: Controlled Conversational SFT Campaign
Fine-tunes both Corpus V4 pretrained candidates (chakr_17m_v4_base.pt and chakr_38m_v4_base.pt) on Corpus V3:
- Strict response-only supervision (prompt tokens and padding tokens masked with ignore_index=-100)
- Identical batch size (16), optimizer (AdamW, lr=1e-4 with cosine decay), and 2 full epochs
- Evaluates validation loss and perplexity on response tokens only
- Saves SFT checkpoints: chakr_17m_nr052_sft.pt and chakr_38m_nr052_sft.pt
- Produces experiments/pretraining_nr052/NR052_SFT_SUMMARY.json
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

class ResponseMaskedDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                p = obj.get("prompt", "").strip()
                r = obj.get("response", "").strip()
                if not p or not r: continue

                p_fmt = f"<user>\n{p}\n<assistant>\n"
                r_fmt = f"{r}<eos>"

                p_ids = tokenizer.encode(p_fmt)
                r_ids = tokenizer.encode(r_fmt)

                full_ids = p_ids + r_ids
                if len(full_ids) > max_len:
                    full_ids = full_ids[:max_len]

                pad_len = max_len - len(full_ids)
                input_ids = full_ids + [0] * pad_len

                # Mask prompt tokens and pad tokens with -100
                targets = [-100] * max_len
                resp_start = len(p_ids)
                resp_end = min(len(full_ids), max_len)
                for i in range(resp_start, resp_end):
                    targets[i - 1] = full_ids[i]

                self.samples.append({
                    "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
                    "targets": torch.tensor(targets[:-1], dtype=torch.long)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()

def evaluate_sft(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input_ids"].to(device)
            tgt = batch["targets"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(inp)
                loss = criterion(logits.reshape(-1, 4096), tgt.reshape(-1))

            n_active = (tgt != -100).sum().item()
            if n_active > 0:
                total_loss += loss.item() * n_active
                total_tokens += n_active

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, ppl

def train_sft(model_name, model, train_loader, val_loader, epochs=2, lr=1e-4, device="cuda", out_dir=Path("experiments/pretraining_nr052")):
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING CONVERSATIONAL SFT: {model_name.upper()} ({device.upper()})", flush=True)
    print(f"=======================================================", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = min(50, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")
    best_ppl = float("inf")
    best_ckpt_path = out_dir / f"chakr_{model_name.lower()}_nr052_sft.pt"

    history = []
    start_time = time.time()
    step = 0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            inp = batch["input_ids"].to(device)
            tgt = batch["targets"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(inp)
                loss = criterion(logits.reshape(-1, 4096), tgt.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 50 == 0 or step == total_steps:
                val_loss, val_ppl = evaluate_sft(model, val_loader, criterion, device)
                model.train()
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:4d}/{total_steps} | Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Response Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} | LR: {cur_lr:.2e}", flush=True)

                history.append({
                    "step": step,
                    "epoch": epoch,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(val_loss, 4),
                    "val_ppl": round(val_ppl, 2)
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ppl = val_ppl
                    cfg_to_save = getattr(model, "config", getattr(model, "cfg", None))
                    torch.save({
                        "model_name": model_name,
                        "config": cfg_to_save,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "step": step
                    }, best_ckpt_path)

    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    print(f"\n[{model_name.upper()} SFT] Best Val Loss: {best_val_loss:.4f} | Best PPL: {best_ppl:.2f}")
    print(f"Saved SFT Checkpoint: {best_ckpt_path}")
    print(f"SFT Training Time: {total_time:.2f} s ({total_time/60:.2f} min) | Peak VRAM: {peak_vram:.1f} MB")

    ckpt_sha = sha256_file(best_ckpt_path)

    return {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_ppl,
        "total_time_seconds": total_time,
        "peak_vram_mb": peak_vram,
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "history": history
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("experiments/pretraining_nr052")
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    train_dataset = ResponseMaskedDataset("data/corpus_v3/corpus_v3_train.jsonl", tokenizer, max_len=256)
    val_dataset = ResponseMaskedDataset("data/corpus_v3/corpus_v3_val.jsonl", tokenizer, max_len=256)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"SFT Dataset Loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.", flush=True)

    # 1. Load 17M Pretrained Base Checkpoint
    ckpt17 = torch.load("experiments/pretraining_nr052/chakr_17m_v4_base.pt", map_location="cpu", weights_only=False)
    cfg17 = ckpt17.get("config", ckpt17.get("cfg"))
    cfg17 = cfg17 if isinstance(cfg17, Genesis17MConfig) else Genesis17MConfig(**cfg17)
    model17 = ChakrGenesis17M(cfg17)
    model17.load_state_dict(ckpt17["model_state_dict"])
    model17.to(device)

    res17 = train_sft("17M", model17, train_loader, val_loader, epochs=2, lr=1e-4, device=device, out_dir=out_dir)
    del model17
    torch.cuda.empty_cache()

    # 2. Load 38M Pretrained Base Checkpoint
    ckpt38 = torch.load("experiments/pretraining_nr052/chakr_38m_v4_base.pt", map_location="cpu", weights_only=False)
    cfg38 = ckpt38.get("config", ckpt38.get("cfg"))
    cfg38 = cfg38 if isinstance(cfg38, Genesis38MConfig) else Genesis38MConfig(**cfg38)
    model38 = ChakrGenesis38M(cfg38)
    model38.load_state_dict(ckpt38["model_state_dict"])
    model38.to(device)

    res38 = train_sft("38M", model38, train_loader, val_loader, epochs=2, lr=1e-4, device=device, out_dir=out_dir)
    del model38
    torch.cuda.empty_cache()

    # Save SFT Summary
    sft_summary = {
        "17M": res17,
        "38M": res38,
        "comparison": {
            "loss_delta_38M_vs_17M": round(res38["best_val_loss"] - res17["best_val_loss"], 4),
            "ppl_delta_38M_vs_17M": round(res38["best_val_ppl"] - res17["best_val_ppl"], 2)
        }
    }

    out_json = out_dir / "NR052_SFT_SUMMARY.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sft_summary, f, indent=2)

    print(f"\nSFT Summary written to: {out_json}")
    print(f"17M SFT Best Val Loss: {res17['best_val_loss']:.4f} (PPL: {res17['best_val_ppl']:.2f})")
    print(f"38M SFT Best Val Loss: {res38['best_val_loss']:.4f} (PPL: {res38['best_val_ppl']:.2f})")
    print(f"38M Advantage: {sft_summary['comparison']['loss_delta_38M_vs_17M']} loss, {sft_summary['comparison']['ppl_delta_38M_vs_17M']} PPL")

if __name__ == "__main__":
    main()
