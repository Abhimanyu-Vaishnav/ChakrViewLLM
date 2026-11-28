# NR-057 HARDWARE PROFILING & RUNTIME BENCHMARK REPORT

**Date:** November 28, 2026  
**Mission:** NR-057 Clean Corpus Expansion & Training Readiness Engineering  
**Artifact ID:** `NR057_HARDWARE_PROFILE`  
**Profile Results JSON:** [hardware_profile.json](file:///d:/Project/ChakrView/experiments/corpus_nr057/hardware_profile.json)

---

## 1. Executive Summary

This report provides empirical profiling of the training runtime, VRAM footprints, step latency, throughput (tokens/second), and checkpoint serialization speeds for both physical model families (`ChakrGenesis17M` and `ChakrGenesis38M`).

### Hardware Platform Profile:
- **Processor:** Intel Core i9-13900H (14 cores, 20 threads)
- **Host Memory:** 34.0 GB physical DDR5 RAM
- **Dedicated Accelerator:** NVIDIA RTX A1000 Laptop GPU (Ampere architecture, 6.00 GB total GDDR6 VRAM)
- **Software Runtime:** Python 3.11, PyTorch 2.14.1+cu126, CUDA 12.6, AMP fp16 autocast

---

## 2. Empirical Benchmark Matrix

Profiling was executed using disposable 50-step forward and backward optimization passes with AdamW optimizer, CrossEntropy loss, and sequence length 512.

| Architecture | Parameters | Batch Size | Sequence Length | Step Latency | Throughput (Tokens/s) | Peak VRAM | VRAM Utilization (%) | Checkpoint Save Latency | Checkpoint Size |
|---|---|---|---|---|---|---|---|---|---|
| **ChakrGenesis17M** | 17,063,808 | 16 | 512 | 313.07 ms | **26,166.8** | 3.204 GB | 53.4% of 6.0 GB | 59.0 ms | 65.1 MB |
| **ChakrGenesis38M** | 37,499,392 | 8 | 512 | 194.10 ms | **21,102.6** | 2.096 GB | 34.9% of 6.0 GB | 114.4 ms | 143.1 MB |
| **ChakrGenesis38M** | 37,499,392 | 16 | 512 | 353.22 ms | **23,192.6** | 3.563 GB | 59.4% of 6.0 GB | 115.7 ms | 143.1 MB |

---

## 3. Training Feasibility & Epoch Duration Projections

Based on measured GPU throughput, we project exact training durations across `Corpus V5 Train` (3,187,865 tokens) and future scaled token thresholds:

| Model Family | Batch Size | Throughput | 1 Full Epoch (Corpus V5) | 5 Epochs (Corpus V5) | 50M Token Milestone |
|---|---|---|---|---|---|
| **17M Architecture** | 16 | 26,166.8 tok/s | **2.03 minutes** | 10.15 minutes | **0.53 hours (31.8 min)** |
| **38M Architecture** | 8 | 21,102.6 tok/s | **2.52 minutes** | 12.60 minutes | **0.66 hours (39.6 min)** |
| **38M Architecture** | 16 | 23,192.6 tok/s | **2.29 minutes** | 11.45 minutes | **0.60 hours (36.0 min)** |

### Critical Hardware Findings:
1. **No OOM Danger:** Peak VRAM for 38M at batch size 16 is **3.56 GB**, well within the 6.00 GB hardware limit (2.44 GB headroom remaining for KV cache or larger batch sizes).
2. **High Efficiency Throughput:** Both models achieve over **21,000 to 26,000 tokens/sec**, allowing full pretraining runs over millions of tokens in under one hour.
3. **Rapid Checkpoint Serialization:** Model checkpoints save in under 120 ms, enabling frequent non-disruptive checkpointing every 500 or 1,000 steps.
