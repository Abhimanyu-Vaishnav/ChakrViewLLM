"""
NR-015 Phase 6: Controlled Architecture Pilot Experiment
Investigates whether expanding model capacity from 17M (17.06M) to 38M (38.28M)
improves memorization efficiency and generalization on the 325-task sealed benchmark.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig

def run_architecture_pilot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Architecture Pilot on: {device}")

    # 1. Baseline 17M Config
    cfg_17m = Genesis17MConfig(
        vocab_size=4096,
        d_model=384,
        n_layers=10,
        n_q_heads=12,
        n_kv_heads=3,
        hidden_dim=1024,
        max_seq_len=512,
        dropout=0.0
    )
    
    # 2. Scaled 38M Config
    cfg_38m = Genesis17MConfig(
        vocab_size=4096,
        d_model=512,
        n_layers=12,
        n_q_heads=16,
        n_kv_heads=4,
        hidden_dim=1536,
        max_seq_len=512,
        dropout=0.0
    )

    models = {
        "17M_Baseline": (ChakrGenesis17M(cfg_17m).to(device), cfg_17m),
        "38M_Scaled": (ChakrGenesis17M(cfg_38m).to(device), cfg_38m)
    }

    # Generate synthetic training batch
    torch.manual_seed(42)
    batch_size = 8
    seq_len = 64
    x = torch.randint(3, 4096, (batch_size, seq_len), device=device)
    y = x.clone()

    loss_fn = nn.CrossEntropyLoss()
    metrics = {}

    for name, (m, cfg) in models.items():
        total_params = sum(p.numel() for p in m.parameters())
        print(f"\n--- Model: {name} ({total_params:,} parameters) ---")
        optimizer = optim.AdamW(m.parameters(), lr=1e-3)
        m.train()

        # Measure forward/backward time and memory
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        losses = []

        for step in range(50):
            optimizer.zero_grad()
            logits, _ = m(x)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, 4096)
            shift_labels = y[..., 1:].contiguous().view(-1)
            loss = loss_fn(shift_logits, shift_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        elapsed = time.time() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"50 Steps Completed in {elapsed:.2f}s | Peak VRAM: {peak_vram_mb:.1f} MB")
        print(f"Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")

        metrics[name] = {
            "parameters": total_params,
            "training_time_50_steps_sec": elapsed,
            "peak_vram_mb": peak_vram_mb,
            "initial_loss": losses[0],
            "final_loss": losses[-1]
        }

    with open("data/curriculum_015/architecture_pilot_results.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nArchitecture pilot completed successfully.")

if __name__ == '__main__':
    run_architecture_pilot()
