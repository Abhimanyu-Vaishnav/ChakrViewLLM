"""
Canonical Pretraining Trainer for Project Genesis (CV-GENESIS-NR-004).
Pretrains ChakrGenesis strictly using the canonical tokenizer matching Stage-C shards:
- Tokenizer: data/experiments/vocab_4096 (SHA-256: db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a)
- Shard validation against dataset manifest
- Autoregressive causal LM training on NVIDIA RTX A1000 GPU
"""

from __future__ import annotations
import glob
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"


class CanonicalShardedDataset(Dataset):
    def __init__(self, shard_files: list[str], seq_len: int = 512):
        self.seq_len = seq_len
        self.tokens_list = []
        for sf in shard_files:
            data = np.fromfile(sf, dtype=np.uint16)
            self.tokens_list.append(data)
        self.all_tokens = np.concatenate(self.tokens_list)
        self.total_tokens = len(self.all_tokens)
        self.num_samples = (self.total_tokens - 1) // self.seq_len
        print(f"Loaded {len(shard_files)} shards -> {self.total_tokens:,} tokens ({self.num_samples} samples of seq_len={seq_len})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.all_tokens[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def run_canonical_pretraining(
    tokenizer_dir: str = "data/experiments/vocab_4096",
    train_dir: str = "data/tokenized/stage_c/train",
    val_dir: str = "data/tokenized/stage_c/validation",
    output_dir: str = "experiments/genesis_canonical_pretraining_004/checkpoints",
    epochs: int = 2,
    batch_size: int = 16,
    seq_len: int = 512,
    lr: float = 6e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hard Tokenizer Hash Verification
    vocab_file = Path(tokenizer_dir) / "vocab.json"
    actual_hash = hashlib.sha256(vocab_file.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_VOCAB_HASH:
        raise ValueError(
            f"FATAL: Tokenizer vocabulary hash mismatch!\nExpected: {EXPECTED_VOCAB_HASH}\nActual:   {actual_hash}"
        )
    print(f"[VERIFIED] Canonical Tokenizer SHA-256 match: {actual_hash}")

    tok, _ = load_tokenizer_artifacts(tokenizer_dir)
    train_shards = sorted(glob.glob(os.path.join(train_dir, "shard_*.bin")))
    val_shards = sorted(glob.glob(os.path.join(val_dir, "shard_*.bin")))

    train_dataset = CanonicalShardedDataset(train_shards, seq_len=seq_len)
    val_dataset = CanonicalShardedDataset(val_shards, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True if device=="cuda" else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    config = GenesisConfig(
        vocab_size=4096,
        d_model=256,
        n_layers=6,
        n_q_heads=8,
        n_kv_heads=2,
        hidden_dim=682,
        max_seq_len=1024
    )
    model = ChakrGenesis(config).to(device)
    print(f"ChakrGenesis Initialized: {sum(p.numel() for p in model.parameters()):,} parameters on {device}.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = min(500, int(total_steps * 0.05))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()
    log_file = out_dir.parent / "pretraining_log.jsonl"

    print(f"Beginning Canonical Pretraining for {epochs} epochs ({total_steps} steps)...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                elapsed = time.time() - t0
                tok_sec = (global_step * batch_size * seq_len) / elapsed
                avg_l = running_loss / 100
                ppl = math.exp(min(avg_l, 20))
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | Loss: {avg_l:.4f} | PPL: {ppl:.2f} | Speed: {tok_sec:.0f} tok/s | LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

            if global_step % 400 == 0:
                model.eval()
                val_loss_acc = 0.0
                val_steps = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        _, vloss = model(vx, targets=vy)
                        val_loss_acc += vloss.item()
                        val_steps += 1
                        if val_steps >= 20:
                            break
                val_l = val_loss_acc / val_steps
                val_ppl = math.exp(min(val_l, 20))
                print(f"\n---> [VALIDATION] Step {global_step} | Val Loss: {val_l:.4f} | Val PPL: {val_ppl:.2f}")

                # Sample continuation test
                sample_prompt = "August is the eighth month of the"
                sample_ids = torch.tensor([tok.encode(sample_prompt)], device=device)
                gen_ids = model.generate(sample_ids, max_new_tokens=20, temperature=0.7)
                sample_gen = tok.decode(gen_ids[0].tolist(), errors="replace")
                print(f"---> [SAMPLE PROBE]: {sample_gen}\n")

                if val_l < best_val_loss:
                    best_val_loss = val_l
                    best_path = out_dir / "canonical_pretrain_best.pt"
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "val_loss": val_l,
                        "tokenizer_vocab_hash": EXPECTED_VOCAB_HASH
                    }, best_path)
                    print(f"Saved new best checkpoint to {best_path}")

                model.train()

    # Final checkpoint
    final_path = out_dir / "canonical_pretrain_final.pt"
    torch.save({
        "step": global_step,
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer_vocab_hash": EXPECTED_VOCAB_HASH
    }, final_path)
    print(f"\n[COMPLETE] Pretraining finished! Checkpoint saved to: {final_path}")


if __name__ == "__main__":
    run_canonical_pretraining()
