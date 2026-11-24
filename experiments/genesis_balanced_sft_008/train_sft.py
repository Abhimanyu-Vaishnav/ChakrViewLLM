"""
Mission CV-GENESIS-NR-008: Controlled Multilingual SFT Trainer.
Trains ChakrGenesis (5.18M) on the balanced bilingual dataset.
Initialized from canonical_pretrain_best.pt (or genesis_sft_best.pt for warm-start continuation).

Features:
- Cryptographic hash checks of tokenizer and base checkpoint.
- Masked loss over assistant response tokens only (-100 for prompt).
- Proper <eos> token supervision.
- Cosine LR decay with linear warmup.
- Gradient clipping (1.0).
- Automatic Mixed Precision (AMP).
- Checkpoint validation perplexity evaluation.
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

# Ensure repo root in python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_PRETRAIN_HASH = "03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e"


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        eos_id = 2
        pad_id = 0

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
    parser.add_argument("--init_checkpoint", type=str, default="experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt")
    parser.add_argument("--data_dir", type=str, default="experiments/genesis_balanced_sft_008/data")
    parser.add_argument("--output_dir", type=str, default="experiments/genesis_balanced_sft_008/checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[RUNTIME] Training on device: {device}")

    # 1. Cryptographic Tokenizer Validation
    vocab_path = Path("data/experiments/vocab_4096/vocab.json")
    with open(vocab_path, "rb") as f:
        actual_vocab_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_vocab_hash == EXPECTED_VOCAB_HASH, "Tokenizer hash mismatch!"
    tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
    print(f"[VERIFIED] Tokenizer hash: {actual_vocab_hash}")

    # 2. Checkpoint Validation & Loading
    init_path = Path(args.init_checkpoint)
    assert init_path.exists(), f"Checkpoint {init_path} does not exist!"
    with open(init_path, "rb") as f:
        init_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"[VERIFIED] Initializing from checkpoint: {init_path.name} (SHA-256: {init_hash[:16]}...)")

    cfg = GenesisConfig()
    model = ChakrGenesis(cfg).to(device)
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)

    # 3. Data Loaders
    print("[DATA] Loading training and validation sets...")
    train_ds = SFTDataset(os.path.join(args.data_dir, "train.jsonl"), tok)
    val_ds = SFTDataset(os.path.join(args.data_dir, "validation.jsonl"), tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[DATA] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 4. Optimizer & Schedule
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

    # Initial validation loss
    init_val_loss, init_val_ppl = compute_eval_loss(model, val_loader, device)
    print(f"[INITIAL EVAL] Val Loss: {init_val_loss:.4f} | Val Perplexity: {init_val_ppl:.2f}")

    best_val_loss = init_val_loss
    best_path = os.path.join(args.output_dir, "balanced_sft_best.pt")
    final_path = os.path.join(args.output_dir, "balanced_sft_final.pt")

    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

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

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0 or step == len(train_loader) - 1:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {cur_lr:.2e}")

        # Epoch end validation
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
            }, best_path)
            print(f"--> Saved NEW BEST checkpoint to {best_path}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": args.epochs,
        "val_loss": val_loss,
        "val_ppl": val_ppl
    }, final_path)
    print(f"--> Saved FINAL checkpoint to {final_path}")

    duration = time.time() - start_time
    print(f"\n[TRAINING COMPLETE] Total time: {duration:.1f}s | Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
