# Project Genesis: Compute and Training Budget Engineering

**Date:** 21 November 2026  
**Project:** ChakrView (ULTIMATE-BRAIN-003)  
**Hardware Platform:** Intel Core (14 physical cores, 20 logical threads), 32 GB RAM, NVIDIA RTX A1000 6GB Laptop GPU  
**Environment:** `.venv_gpu` (PyTorch 2.5.1+cu124 / CUDA 12.6)

---

## 1. Measured Benchmarks on Target Hardware

All benchmarks were measured directly on local hardware via forward-backward-optimizer steps with AdamW, batch size = 8, sequence length = 512 tokens:

| Candidate Model | Hidden Dim / Layers / Heads | Parameters | GPU Throughput (tok/sec) | Throughput (tok/hour) | Peak VRAM | Feasibility Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Candidate A (ChakrGenesis-Micro)** | $d=256, L=6, H_q=8, H_{kv}=2, d_{ff}=682$ | **5.18M** | **24,396 tok/s** | **87.83M tok/hr** | **1.14 GB** | **Optimal for rapid pretraining + full epoch coverage** |
| **Candidate B (ChakrGenesis-Base)** | $d=384, L=8, H_q=12, H_{kv}=4, d_{ff}=1024$ | **14.16M** | **12,018 tok/s** | **43.27M tok/hr** | **2.33 GB** | **Viable, fits comfortably within 6GB VRAM** |
| **Candidate C (ChakrGenesis-Large)** | $d=512, L=12, H_q=16, H_{kv}=4, d_{ff}=1365$ | **31.5M** | ~5,200 tok/s | ~18.7M tok/hr | ~4.85 GB | High thermal / memory stress on 6GB GPU |

---

## 2. Selected Foundation Architecture: ChakrGenesis-Micro (5.18M) with Option for Base (14.2M)

- **Primary Foundation Model:** **`ChakrGenesis-Micro` (5,177,612 parameters)**
  - At **87.8M tokens/hour**, the complete 6.65M token Stage-C pretraining corpus can be completed in **~4.5 minutes per epoch**.
  - A comprehensive 3-epoch pretraining campaign (20M tokens processed) takes **~14 minutes**.
  - Peak VRAM footprint of **1.14 GB** guarantees zero out-of-memory errors and prevents GPU thermal throttling.
  - Allows full parameter updates with AdamW, cosine learning rate schedule, and linear warmup.

---

## 3. Storage and Checkpoint Budget

- **Checkpoint Size (Weights only):** ~20.7 MB per checkpoint (`.pt`).
- **Checkpoint Size (Weights + Optimizer + Scheduler):** ~62.1 MB.
- **Available Disk Space:** >100 GB on `D:\Project\ChakrView`.
- **Checkpointing Strategy:** Save best validation checkpoint (`genesis_pretrain_best.pt`) and final checkpoint (`genesis_pretrain_final.pt`), with periodic step checkpoints every 1,000 steps.
