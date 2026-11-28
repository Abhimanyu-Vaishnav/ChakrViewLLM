# NR-057-R1 REAL MULTI-COMPONENT TRAINING THROUGHPUT REPORT

**Date:** November 28, 2026  
**Mission:** NR-057-R1 Corpus Expansion Completion & Training Gate Revalidation  
**Artifact ID:** `NR057R1_REAL_TRAINING_PROFILE`  
**Profile JSON:** [training_profile.json](file:///d:/Project/ChakrView/experiments/corpus_nr057_r1/training_profile.json)

---

## 1. Executive Summary

A critical mandate of NR-057-R1 was to conduct an independent, decoupled empirical benchmark that does not extrapolate full training duration from forward-only inference measurements. This audit explicitly separated forward evaluation, backward gradient computation, full optimizer parameter updates, dataloader transfer, checkpoint serialization, and peak memory allocations.

### Key Empirical Findings:
1. **Full Optimizer Step Throughput:**
   - **ChakrGenesis17M (Batch 16):** **26,039.6 tokens/sec** (Step latency: 314.6 ms)
   - **ChakrGenesis38M (Batch 8):** **21,206.3 tokens/sec** (Step latency: 193.15 ms)
   - **ChakrGenesis38M (Batch 16):** **23,118.6 tokens/sec** (Step latency: 354.35 ms)
2. **Divergence Between Forward-Only vs Full Training:**
   - Forward-only inference reaches ~75,000–80,000 tokens/sec.
   - True training throughput (including backward pass, AdamW momentum states, and gradient updates) is approximately **3.2x slower** than forward-only passes.
   - **Extrapolating from forward-only measurements would have severely underestimated training time by over 68%.**
3. **Exact Training Duration over Corpus V6 Train (4,163,555 tokens):**
   - 17M Model: **2.66 minutes per epoch** (5 epochs = 13.3 minutes).
   - 38M Model: **3.00 minutes per epoch** (5 epochs = 15.0 minutes).
4. **VRAM and System Stability:**
   - 38M at batch 16 consumes **3.565 GB VRAM** (59.4% of 6.00 GB GPU capacity), leaving **2.435 GB of headroom**.
   - Host RAM consumption is negligible at **1.51 GB**.

---

## 2. Multi-Component Performance Decomposition

| Architecture | Batch Size | Forward-Only (tok/s) | Forward + Backward (tok/s) | Full Optimizer Step (tok/s) | Peak VRAM | VRAM Headroom | Checkpoint Latency | Size on Disk |
|---|---|---|---|---|---|---|---|---|
| **ChakrGenesis17M** | 16 | 74,587.4 | 26,817.8 | **26,039.6** | 3.204 GB | 2.796 GB (46.6%) | 95.7 ms | 65.1 MB |
| **ChakrGenesis38M** | 8 | 76,049.3 | 23,551.1 | **21,206.3** | 2.097 GB | 3.903 GB (65.0%) | 158.2 ms | 143.1 MB |
| **ChakrGenesis38M** | 16 | 79,878.7 | 24,451.6 | **23,118.6** | 3.565 GB | 2.435 GB (40.6%) | 150.8 ms | 143.1 MB |

---

## 3. Training Feasibility & Seed Comparison Realistic Cost

For the upcoming **NR-058** controlled pretraining mission:
- A 3-epoch pretraining run on `Corpus V6 Train` (12.49M tokens seen):
  - 17M: **7.98 minutes** per run.
  - 38M: **9.00 minutes** per run.
- Running **three distinct seeds** (`seed=42, 1337, 2026`):
  - 17M across 3 seeds: **~24 minutes total**.
  - 38M across 3 seeds: **~27 minutes total**.
- **Conclusion:** Full three-seed replication is computationally practical on this local NVIDIA RTX A1000 GPU without exceeding thermal limits or session timeouts.
