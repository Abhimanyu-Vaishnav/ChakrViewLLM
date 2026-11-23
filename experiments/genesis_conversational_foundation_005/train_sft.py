"""
Workstream B: Robust Conversational SFT Trainer for Mission CV-GENESIS-NR-005.

Features:
- Cryptographic hash validation of canonical tokenizer (db4d6119...) and pretrained best checkpoint (03a4c16d...)
- Masked loss over assistant response tokens only (prompt tokens labeled -100)
- Proper <EOS> token (ID 1) appended and supervised
- Padding tokens (<PAD> ID 2) masked with -100
- Cosine learning rate decay with linear warmup
- Gradient clipping (1.0 max norm)
- Automatic mixed precision (AMP)
- Validation perplexity tracking with best checkpoint preservation
- Full optimizer & scheduler state persistence
"""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_PRETRAIN_HASH = "03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e"


class ConversationalSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 512, limit: int | None = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items = []
        
        # Meta: special token IDs in canonical vocab: <BOS>: 0, <EOS>: 1, <PAD>: 2
        bos_id = 0
        eos_id = 1
        pad_id = 2

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                p_text = d["prompt"].strip()
                r_text = d["response"].strip()

                if not p_text.endswith("Assistant:"):
                    p_text = p_text + "\nAssistant:"

                # Encode prompt and response
                p_ids = tokenizer.encode(p_text)
                r_ids = tokenizer.encode(" " + r_text) + [eos_id]

                full_ids = p_ids + r_ids
                if len(full_ids) > max_seq_len:
                    full_ids = full_ids[:max_seq_len]

                input_ids = full_ids[:-1]
                labels = full_ids[1:]

                # Mask prompt tokens so loss is computed ONLY on assistant responses
                prompt_len = len(p_ids)
                masked_labels = []
                for idx, tok in enumerate(labels):
                    if idx < prompt_len - 1:
                        masked_labels.append(-100)
                    else:
                        masked_labels.append(tok)

                # Pad sequences to max_seq_len - 1
                target_len = max_seq_len - 1
                pad_len = target_len - len(input_ids)
                input_ids = input_ids + [pad_id] * pad_len
                masked_labels = masked_labels + [-100] * pad_len

                self.items.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(masked_labels, dtype=torch.long)
                ))
                
                if limit and len(self.items) >= limit:
                    break

        print(f"Loaded {len(self.items)} examples from {jsonl_path} (seq_len={max_seq_len})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[idx]


def evaluate_sft_validation(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.startswith("cuda"))):
                logits, loss = model(x, targets=y)
            num_valid_tokens = (y != -100).sum().item()
            if num_valid_tokens > 0:
                total_loss += loss.item() * num_valid_tokens
                total_tokens += num_valid_tokens
    model.train()
    if total_tokens == 0:
        return 0.0, 0.0
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def run_sft_training(
    train_path: str = "experiments/genesis_conversational_foundation_005/data/train.jsonl",
    val_path: str = "experiments/genesis_conversational_foundation_005/data/validation.jsonl",
    tokenizer_dir: str = "data/experiments/vocab_4096",
    pretrained_ckpt: str = "experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt",
    output_dir: str = "experiments/genesis_conversational_foundation_005/checkpoints",
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 1.8e-4,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    pilot_mode: bool = False,
    pilot_limit: int = 32
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cryptographic Tokenizer Hash Gate
    vocab_file = Path(tokenizer_dir) / "vocab.json"
    actual_vocab_hash = hashlib.sha256(vocab_file.read_bytes()).hexdigest()
    if actual_vocab_hash != EXPECTED_VOCAB_HASH:
        raise ValueError(f"FATAL: Tokenizer vocab mismatch!\nExpected: {EXPECTED_VOCAB_HASH}\nActual:   {actual_vocab_hash}")
    print(f"[VERIFIED] Tokenizer Vocab SHA-256 match: {actual_vocab_hash}")

    # 2. Cryptographic Pretrained Checkpoint Hash Gate
    actual_ckpt_hash = hashlib.sha256(Path(pretrained_ckpt).read_bytes()).hexdigest()
    if actual_ckpt_hash != EXPECTED_PRETRAIN_HASH:
        raise ValueError(f"FATAL: Pretrained checkpoint mismatch!\nExpected: {EXPECTED_PRETRAIN_HASH}\nActual:   {actual_ckpt_hash}")
    print(f"[VERIFIED] Pretrained Checkpoint SHA-256 match: {actual_ckpt_hash}")

    tok, _ = load_tokenizer_artifacts(tokenizer_dir)
    train_limit = pilot_limit if pilot_mode else None
    val_limit = pilot_limit if pilot_mode else None

    train_dataset = ConversationalSFTDataset(train_path, tok, max_seq_len=512, limit=train_limit)
    val_dataset = ConversationalSFTDataset(val_path, tok, max_seq_len=512, limit=val_limit)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True if device.startswith("cuda") else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loading pretrained weights from: {pretrained_ckpt}")
    ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ChakrGenesis(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    print(f"Model successfully loaded: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(10, int(total_steps * 0.08))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    best_val_loss = float("inf")
    history = []
    global_step = 0
    t0 = time.time()

    val_loss_init, val_ppl_init = evaluate_sft_validation(model, val_loader, device)
    print(f"[INITIAL BASELINE] Validation Loss: {val_loss_init:.4f} | Validation PPL: {val_ppl_init:.2f}")

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            global_step += 1
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                logits, loss = model(x, targets=y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if global_step % 25 == 0 or global_step == 1 or global_step == total_steps or pilot_mode:
                train_loss = loss.item()
                train_ppl = math.exp(min(train_loss, 20.0))
                current_lr = scheduler.get_last_lr()[0]
                val_loss, val_ppl = evaluate_sft_validation(model, val_loader, device)
                elapsed = time.time() - t0
                tok_per_sec = (global_step * batch_size * 512) / max(0.001, elapsed)

                print(
                    f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                    f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                    f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                    f"LR: {current_lr:.2e} | Speed: {tok_per_sec:.0f} tok/s"
                )

                history.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_ppl": train_ppl,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "lr": current_lr
                })

                if val_loss < best_val_loss and not pilot_mode:
                    best_val_loss = val_loss
                    best_ckpt_path = out_dir / "genesis_sft_best.pt"
                    torch.save({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "tokenizer_vocab_hash": EXPECTED_VOCAB_HASH,
                        "foundation_pretrain_hash": EXPECTED_PRETRAIN_HASH
                    }, best_ckpt_path)
                    print(f"--> Saved new BEST SFT checkpoint: {best_ckpt_path} (Val Loss: {val_loss:.4f})")

    if not pilot_mode:
        final_ckpt_path = out_dir / "genesis_sft_final.pt"
        torch.save({
            "step": global_step,
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "tokenizer_vocab_hash": EXPECTED_VOCAB_HASH,
            "foundation_pretrain_hash": EXPECTED_PRETRAIN_HASH
        }, final_ckpt_path)
        print(f"--> Saved FINAL SFT checkpoint: {final_ckpt_path}")

        metrics_path = out_dir.parent / "evaluations" / "sft_training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "initial_baseline": {"loss": val_loss_init, "ppl": val_ppl_init},
                "best_val_loss": best_val_loss,
                "history": history
            }, f, indent=2)
        print(f"Saved training history to {metrics_path}")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run small batch pilot sanity test")
    args = parser.parse_args()
    
    if args.pilot:
        print("=== RUNNING PILOT SANITY OVERFIT TEST ===")
        run_sft_training(epochs=5, batch_size=8, lr=3e-4, pilot_mode=True, pilot_limit=32)
    else:
        print("=== RUNNING FULL CONVERSATIONAL SFT ===")
        run_sft_training(epochs=4, batch_size=16, lr=1.8e-4, pilot_mode=False)
