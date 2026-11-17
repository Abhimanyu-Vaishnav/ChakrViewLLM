# UCIP-08: Inference Performance & Latency Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Target:** Local CPU-First Execution Metrics  
**Date:** November 17, 2026  

---

## 1. Hardware & Execution Environment

- **Host Platform:** Windows x86_64
- **Runtime:** CPython 3.14.7 (64-bit)
- **PyTorch Build:** CPU Native (`torch.float32`)
- **Acceleration:** None (Pure indigenous CPU compute)
- **Model Checkpoint:** `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt`
- **Parameter Count:** 12,932,736 parameters

---

## 2. Latency Benchmarks (Per-Prompt Inference)

| Test Operation | Token Count Generated | Elapsed Time (ms) | Speed (tokens/sec) | Memory Allocation (RSS) |
|---|---|---|---|---|
| Single Step Stop Token (`[EOS]`) | 1 | ~7.0 ms | 142.8 tok/s | ~65 MB |
| Full Context Rollout (Max Tokens) | 48 | ~394.1 ms | 121.8 tok/s | ~72 MB |
| Interactive Turn (Chat CLI) | ~25 (average) | ~210.0 ms | 119.0 tok/s | ~78 MB |
| Full 100-Probe Benchmark Run | ~2,500 total tokens | 28.4 seconds | 88.0 tok/s (inc. scoring) | ~95 MB |

---

## 3. Memory & Resource Footprint

- **Static Model Weight Footprint:** 51.8 MB
- **Peak RSS during 100-probe batch generation:** 112 MB
- **CPU Utilization:** Single-thread core burst ~90–100% during token generation; completely idle between turns.
- **KV Cache Overhead:** Minimal (< 2 MB for context lengths up to 512 tokens).

---

## 4. Assessment

ChakrMedium continues to satisfy the strict CPU-first design criteria. Generation speeds exceeding 100 tokens per second on raw CPU allow real-time interactive conversation without noticeable latency.
