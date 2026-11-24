"""
Mission CV-GENESIS-NR-011: ChakrGenesis-17M Stage-D Training Pipeline.
Trains ChakrGenesis-17M initialized from genesis_17m_pretrain_best.pt.
Fine-tunes on the newly curated Stage-D dataset (5,513 unique samples).

Contract:
- Model: ChakrGenesis-17M (17,063,808 params)
- Pretrained weights: experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt
- Training dataset: data/stage_d/splits/stage_d_train.jsonl (4,686 examples)
- Validation dataset: data/stage_d/splits/stage_d_val.jsonl (551 examples)
- Special tokens: EOS=1, PAD=2
- Output: experiments/genesis_stage_d_011/checkpoints/
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
EXPECTED_17M_PRETRAIN_HASH = "96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0"


class StageDDataset(Dataset):
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


def compute_eval_loss(model, dataloader, device):
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
    parser.add_argument("--train_file", type=str, default="data/stage_d/splits/stage_d_train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/stage_d/splits/stage_d_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="experiments/genesis_stage_d_011/checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RUNTIME] Training ChakrGenesis-17M Stage-D on {device}")

    # 1. Cryptographic Gates
    vocab_path = Path("data/experiments/vocab_4096/vocab.json")
    with open(vocab_path, "rb") as f:
        vh = hashlib.sha256(f.read()).hexdigest()
    assert vh == EXPECTED_VOCAB_HASH, "Vocab mismatch!"
    tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
    print(f"[VERIFIED] Tokenizer Vocab SHA-256 match: {vh}")

    init_path = Path(args.init_checkpoint)
    with open(init_path, "rb") as f:
        ih = hashlib.sha256(f.read()).hexdigest()
    assert ih == EXPECTED_17M_PRETRAIN_HASH, f"Pretrain hash mismatch! Expected {EXPECTED_17M_PRETRAIN_HASH}, got {ih}"
    print(f"[VERIFIED] Checkpoint lineage: {init_path.name} (SHA-256: {ih})")

    # 2. Load model
    cfg = Genesis17MConfig(max_seq_len=256)
    model = ChakrGenesis17M(cfg).to(device)
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[MODEL] ChakrGenesis-17M instantiated ({sum(p.numel() for p in model.parameters()):,} params).")

    # 3. Data Loaders
    train_ds = StageDDataset(args.train_file, tok)
    val_ds = StageDDataset(args.val_file, tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[DATA] Train: {len(train_ds)} samples ({len(train_loader)} batches) | Val: {len(val_ds)} samples")

    # 4. Optimization
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

    init_val_loss, init_val_ppl = compute_eval_loss(model, val_loader, device)
    print(f"[INITIAL] Stage-D Val Loss: {init_val_loss:.4f} | Val PPL: {init_val_ppl:.2f}")

    best_val_loss = init_val_loss
    best_ckpt = os.path.join(args.output_dir, "genesis_17m_staged_best.pt")
    final_ckpt = os.path.join(args.output_dir, "genesis_17m_staged_final.pt")

    start_time = time.time()
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

        val_loss, val_ppl = compute_eval_loss(model, val_loader, device)
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
            print(f"--> Saved NEW BEST Stage-D candidate to {best_ckpt}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": args.epochs,
        "val_loss": val_loss,
        "val_ppl": val_ppl
    }, final_ckpt)
    print(f"--> Saved FINAL Stage-D candidate to {final_ckpt}")

    duration = time.time() - start_time
    print(f"\n[COMPLETE] Stage-D 17M training finished in {duration:.1f}s | Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
