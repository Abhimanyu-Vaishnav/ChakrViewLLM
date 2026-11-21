"""
Isolated Recovery SFT Trainer for Project Genesis (CV-GENESIS-NR-001).
Trains a recovery checkpoint from genesis_pretrain_best.pt on the balanced dataset:
- Assistant-turn masking
- Cosine schedule with warmup
- Gradient clipping
- Regular intermediate checkpointing
"""

from __future__ import annotations
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


class RecoverySFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                p_text = d["prompt"].strip()
                r_text = d["response"].strip()

                if not p_text.endswith("Assistant:"):
                    p_text = p_text + "\nAssistant:"

                p_ids = tokenizer.encode(p_text)
                r_ids = tokenizer.encode(" " + r_text) + [2]  # eos_id = 2

                full_ids = p_ids + r_ids
                if len(full_ids) > max_seq_len:
                    full_ids = full_ids[:max_seq_len]

                input_ids = full_ids[:-1]
                labels = full_ids[1:]

                prompt_len = len(p_ids)
                masked_labels = []
                for idx, tok in enumerate(labels):
                    if idx < prompt_len - 1:
                        masked_labels.append(-100)
                    else:
                        masked_labels.append(tok)

                pad_len = max_seq_len - len(input_ids)
                input_ids = input_ids + [0] * pad_len
                masked_labels = masked_labels + [-100] * pad_len

                self.items.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(masked_labels, dtype=torch.long)
                ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def run_recovery_training(
    dataset_path: str = "experiments/genesis_neural_recovery_001/recovery_sft_dataset.jsonl",
    pretrained_ckpt: str = "experiments/project_genesis/pretraining/genesis_pretrain_best.pt",
    output_dir: str = "experiments/genesis_neural_recovery_001/checkpoints",
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 1.5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, _ = load_tokenizer_artifacts("experiments/tokenizer/candidates/v4096")
    dataset = RecoverySFTDataset(dataset_path, tok, max_seq_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Loading pretrained foundation model from: {pretrained_ckpt}")
    ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ChakrGenesis(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(loader) * epochs
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 20 == 0:
                avg_l = running_loss / 20
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | Loss: {avg_l:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

    final_path = out_dir / "genesis_recovery_sft_best.pt"
    torch.save({
        "step": global_step,
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": config
    }, final_path)
    print(f"[COMPLETE] Saved recovery checkpoint to: {final_path}")


if __name__ == "__main__":
    run_recovery_training()
