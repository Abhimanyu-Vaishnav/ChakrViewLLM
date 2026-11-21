"""
Training pipeline for ChakrUltra Neural Core.
Features:
- AdamW optimizer with warmup and cosine decay.
- Assistant-only loss masking (-100 on prompt tokens).
- Dynamic gradient clipping.
- Epoch-wise validation and early checkpointing.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.ultimate.chakr_ultra import ChakrUltra
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence, normalize_prompt_response_strings


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class UltimateDataset:
    def __init__(self, jsonl_path: Path, tokenizer: Any):
        self.tokenizer = tokenizer
        self.records = [json.loads(line) for line in open(jsonl_path, encoding="utf-8")]

    def __len__(self) -> int:
        return len(self.records)

    def encode_record(self, prompt: str, response: str) -> Tuple[List[int], List[int]]:
        p_str, r_str = normalize_prompt_response_strings(prompt, response, strip_delimiters=True)
        p_toks = self.tokenizer.encode(p_str)
        r_toks = self.tokenizer.encode(r_str)
        inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
        return inp, tgt

    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_inps = []
        batch_tgts = []
        for idx in indices:
            rec = self.records[idx]
            inp, tgt = self.encode_record(rec["prompt"], rec["response"])
            batch_inps.append(inp)
            batch_tgts.append(tgt)

        max_len = max(len(x) for x in batch_inps)
        pad_id = 2
        ignore_index = -100

        padded_inps = [x + [pad_id] * (max_len - len(x)) for x in batch_inps]
        padded_tgts = [y + [ignore_index] * (max_len - len(y)) for y in batch_tgts]

        return torch.tensor(padded_inps, dtype=torch.long), torch.tensor(padded_tgts, dtype=torch.long)


def evaluate_dataset(model: ChakrUltra, dataset: UltimateDataset, batch_size: int = 16) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_active_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = list(range(start_idx, min(start_idx + batch_size, len(dataset))))
            x, y = dataset.get_batch(batch_indices)
            logits = model(x)
            active_mask = (y != -100)
            n_tokens = active_mask.sum().item()
            if n_tokens > 0:
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
                total_active_tokens += n_tokens

    avg_nll = total_loss / max(1, total_active_tokens)
    ppl = math.exp(min(avg_nll, 20.0))
    return avg_nll, ppl


def train_chakr_ultra():
    print("=== Training ChakrUltra Neural Core ===")
    torch.manual_seed(42)
    tok, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")

    train_ds = UltimateDataset(Path("experiments/ultimate_brain/datasets/train.jsonl"), tok)
    val_ds = UltimateDataset(Path("experiments/ultimate_brain/datasets/val.jsonl"), tok)
    print(f"Loaded Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = ChakrUltra(
        vocab_size=4096,
        d_model=256,
        n_layers=6,
        n_q_heads=8,
        n_kv_heads=2,
        hidden_dim=682,
        max_seq_len=1024,
        tie_embeddings=True
    )
    total_p = sum(p.numel() for p in model.parameters())
    print(f"ChakrUltra parameters: {total_p:,}")

    base_nll, base_ppl = evaluate_dataset(model, val_ds)
    print(f"Baseline (Untrained): Val NLL = {base_nll:.4f}, Val PPL = {base_ppl:.2f}")

    batch_size = 16
    n_epochs = 8
    lr = 3.5e-4
    min_lr = 1e-5
    total_steps = n_epochs * (len(train_ds) // batch_size + 1)
    warmup_steps = 25

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    best_val_nll = float("inf")
    best_ckpt_path = Path("experiments/ultimate_brain/training/checkpoint_best.pt")
    training_log = []

    global_step = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        indices = list(range(len(train_ds)))
        torch.manual_seed(42 + epoch)
        perm = torch.randperm(len(indices)).tolist()
        epoch_loss = 0.0
        batches = 0

        for start_idx in range(0, len(indices), batch_size):
            global_step += 1
            curr_lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            batch_idx = [perm[i] for i in range(start_idx, min(start_idx + batch_size, len(indices)))]
            x, y = train_ds.get_batch(batch_idx)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        val_nll, val_ppl = evaluate_dataset(model, val_ds)
        train_nll = epoch_loss / batches
        print(f"Epoch {epoch:02d}/{n_epochs:02d} | Step {global_step:03d} | Train Loss: {train_nll:.4f} | Val NLL: {val_nll:.4f} | Val PPL: {val_ppl:.2f}")

        training_log.append({
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_nll,
            "val_nll": val_nll,
            "val_ppl": val_ppl,
            "lr": curr_lr
        })

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "val_nll": val_nll,
                "val_ppl": val_ppl,
                "config": {
                    "vocab_size": 4096,
                    "d_model": 256,
                    "n_layers": 6,
                    "n_q_heads": 8,
                    "n_kv_heads": 2,
                    "hidden_dim": 682,
                    "max_seq_len": 1024,
                    "tie_embeddings": True
                }
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint (Val NLL: {val_nll:.4f}, Val PPL: {val_ppl:.2f})")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    best_hash = compute_sha256(best_ckpt_path)
    summary = {
        "architecture": "ChakrUltra v1.0",
        "parameters": total_p,
        "total_epochs": n_epochs,
        "total_steps": global_step,
        "training_time_seconds": elapsed,
        "baseline_val_nll": base_nll,
        "baseline_val_ppl": base_ppl,
        "best_val_nll": best_val_nll,
        "best_val_ppl": math.exp(best_val_nll),
        "best_checkpoint_sha256": best_hash,
        "training_log": training_log
    }

    with open("experiments/ultimate_brain/training/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary. Checkpoint hash: {best_hash}")


if __name__ == "__main__":
    train_chakr_ultra()
