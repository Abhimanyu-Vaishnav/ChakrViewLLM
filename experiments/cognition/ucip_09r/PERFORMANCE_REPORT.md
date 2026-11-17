# UCIP-09R: Inference Performance, Latency & Resource Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Target:** Local CPU Execution Performance  
**Date:** November 17, 2026  

---

## 1. System Environment

- **Host Platform:** Windows 11 Pro (64-bit)
- **CPU:** Intel Core i9-13900H (32 GB RAM)
- **Runtime:** CPython 3.14.7 + PyTorch 2.14.0 (CPU Native)
- **Checkpoint Footprint:** 51.8 MB (51,756,113 bytes)
- **Model Parameters:** 12,932,736 (ChakrMedium)

---

## 2. Measured Latency & Throughput

| Operation | Total Tokens | Wall Time (s) | Speed (tok/s) | Peak RSS Memory |
|---|---|---|---|---|
| Single Token Forward Pass | 1 | 0.0068 s | 147.0 tok/s | ~68 MB |
| Full 48-Token Rollout | 48 | 0.380 s | 126.3 tok/s | ~74 MB |
| Full 500-Probe Benchmark Run | ~13,500 tokens | 2 min 41 s | 83.8 tok/s | 121 MB |
| Full 100-Probe Reality Benchmark | ~2,600 tokens | 23.2 s | 112.0 tok/s | 89 MB |

Generation speeds exceeding 125 tokens/second on raw CPU maintain fluid, real-time interactivity with minimal resource overhead.
