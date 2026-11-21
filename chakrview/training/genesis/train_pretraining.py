"""
ChakrGenesis Autoregressive Pretraining Engine.
Trains the foundation model on the Stage-C corpus using CUDA acceleration.
Saves checkpoints, logs perplexity, and validates intermediate free generation.
"""

from __future__ import annotations
import glob
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


class ShardedBinaryDataset(Dataset):
    def __init__(self, shard_files: list[str], seq_len: int = 512):
        self.seq_len = seq_len
        self.tokens_list = []
        print(f"Loading {len(shard_files)} binary shards...")
        for sf in shard_files:
            data = np.fromfile(sf, dtype=np.uint16)
            self.tokens_list.append(data)
        
        self.all_tokens = np.concatenate(self.tokens_list)
        self.total_tokens = len(self.all_tokens)
        self.num_samples = (self.total_tokens - 1) // self.seq_len
        print(f"Total tokens loaded: {self.total_tokens:,} -> {self.num_samples} samples of seq_len={seq_len}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.all_tokens[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def train_genesis_pretraining(
    data_dir: str = "data/tokenized/stage_c/train",
    val_dir: str = "data/tokenized/stage_c/validation",
    output_dir: str = "experiments/project_genesis/pretraining",
    epochs: int = 1,
    batch_size: int = 16,
    seq_len: int = 512,
    lr: float = 6e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  PROJECT GENESIS: FOUNDATION AUTOREGRESSIVE PRETRAINING         ")
    print(f"  Device: {device} | Batch Size: {batch_size} | Seq Len: {seq_len}")
    print("=================================================================\n")

    train_shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.bin")))
    val_shards = sorted(glob.glob(os.path.join(val_dir, "shard_*.bin")))

    train_dataset = ShardedBinaryDataset(train_shards, seq_len=seq_len)
    val_dataset = ShardedBinaryDataset(val_shards, seq_len=seq_len)

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
    param_count = sum(p.numel() for p in model.parameters())
    print(f"ChakrGenesis Model Initialized: {param_count:,} parameters.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = min(500, int(total_steps * 0.1))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load tokenizer for intermediate validation generation
    tok, _ = load_tokenizer_artifacts("experiments/tokenizer/candidates/v4096")

    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

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
                avg_loss = running_loss / 100
                ppl = math.exp(min(avg_loss, 20))
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Speed: {tok_sec:.0f} tok/s | LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

            if global_step % 500 == 0:
                # Validation evaluation
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
                val_loss = val_loss_acc / val_steps
                val_ppl = math.exp(min(val_loss, 20))
                print(f"\n---> [VALIDATION] Step {global_step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                # Sample test prompt generation
                test_prompt = "Photosynthesis is the process by which"
                test_ids = torch.tensor([tok.encode(test_prompt)], device=device)
                gen_ids = model.generate(test_ids, max_new_tokens=25, temperature=0.7)
                try:
                    gen_text = tok.decode(gen_ids[0].tolist(), errors="replace")
                except Exception:
                    gen_text = str(gen_ids[0].tolist())
                print(f"---> [SAMPLE GENERATION]: {gen_text}\n")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = out_path / "genesis_pretrain_best.pt"
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "val_loss": val_loss
                    }, best_ckpt_path)
                    print(f"Saved new best checkpoint to {best_ckpt_path}")

                model.train()

    # Final save
    final_ckpt_path = out_path / "genesis_pretrain_final.pt"
    torch.save({
        "step": global_step,
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": config,
        "final_loss": avg_loss
    }, final_ckpt_path)
    print(f"\n[COMPLETE] Pretraining finished! Checkpoint saved to {final_ckpt_path}")


if __name__ == "__main__":
    train_genesis_pretraining()
