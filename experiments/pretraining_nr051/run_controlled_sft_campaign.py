"""
experiments/pretraining_nr051/run_controlled_sft_campaign.py
Executes Phase 6: Identical Conversational SFT on both Pretrained Candidates:
- Candidate A: chakr_17m_pretrained_base.pt (17.06M) -> chakr_17m_sft_candidate.pt
- Candidate B: chakr_38m_pretrained_base.pt (37.50M) -> chakr_38m_sft_candidate.pt
- Both fine-tuned on identical Corpus V3 (data/corpus_v3/corpus_v3_train.jsonl)
- Identical response-only cross-entropy loss (prompt tokens masked with -100)
- Identical optimizer settings: AdamW (lr=1e-4), Cosine Annealing, 2 epochs, batch size 16
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

class ResponseMaskedDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
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

def sft_candidate(model_name, model_class, base_ckpt_path, out_dir, train_loader, val_loader, device="cuda"):
    print(f"\n=======================================================")
    print(f"RUNNING CONVERSATIONAL SFT ON {model_name} (BASE: {base_ckpt_path})")
    print(f"=======================================================")

    st = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    cfg = st["config"]
    model = model_class(cfg)
    model.load_state_dict(st["model_state_dict"])
    model.to(device)

    unique_params = sum(p.numel() for p in set(model.parameters()))
    epochs = 2
    total_steps = len(train_loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_ckpt_path = out_dir / f"chakr_{model_name.lower()}_sft_candidate.pt"
    best_val_loss = float("inf")
    history = []

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
                        "val_loss": best_val_loss,
                        "unique_params": unique_params,
                        "model_family": model_name,
                        "base_checkpoint": str(base_ckpt_path),
                        "base_sha256": file_sha256(base_ckpt_path),
                        "stage": "NR051_Conversational_SFT"
                    }, best_ckpt_path)

    total_time = time.time() - start_time
    print(f"{model_name} SFT Complete in {total_time/60:.2f} minutes. Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved: {best_ckpt_path} (SHA-256: {file_sha256(best_ckpt_path)})")

    del model
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "base_checkpoint": str(base_ckpt_path),
        "sft_checkpoint": str(best_ckpt_path),
        "sft_sha256": file_sha256(best_ckpt_path),
        "best_val_loss": best_val_loss,
        "total_time_seconds": total_time,
        "history": history
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== NR-051 PHASE 6 CONVERSATIONAL SFT ({device.upper()}) ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    out_dir = Path("experiments/pretraining_nr051")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ResponseMaskedDataset("data/corpus_v3/corpus_v3_train.jsonl", tokenizer, max_seq_len=256)
    val_ds = ResponseMaskedDataset("data/corpus_v3/corpus_v3_val.jsonl", tokenizer, max_seq_len=256)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    print(f"SFT Dataset Loaded: {len(train_ds)} train, {len(val_ds)} val samples.")

    # 1. SFT on 17M Pretrained Base
    res_17m = sft_candidate(
        "17M",
        ChakrGenesis17M,
        "experiments/pretraining_nr051/chakr_17m_pretrained_base.pt",
        out_dir,
        train_loader,
        val_loader,
        device=device
    )

    # 2. SFT on 38M Pretrained Base
    res_38m = sft_candidate(
        "38M",
        ChakrGenesis38M,
        "experiments/pretraining_nr051/chakr_38m_pretrained_base.pt",
        out_dir,
        train_loader,
        val_loader,
        device=device
    )

    summary_path = out_dir / "NR051_SFT_SUMMARY.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"17M": res_17m, "38M": res_38m}, f, indent=2)

    print(f"\nSFT Summary written to: {summary_path}")

if __name__ == "__main__":
    main()
