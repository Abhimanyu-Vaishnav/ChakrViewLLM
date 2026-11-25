# CHAKRVIEW RESEARCH REPORT: NR-012 PERFORMANCE REPORT
**Mission ID:** CV-GENESIS-NR-012  
**Date:** 2026-11-25  
**Hardware:** Intel Core i9-13900H (14C/20T) | 32 GB DDR5 | NVIDIA RTX A1000 6GB Laptop GPU  

---

## 1. System Latency & Cold Start Profile

| Component / Metric | Cold Start (First Load) | Warm Inference |
| :--- | :--- | :--- |
| **Model Weight Load Time** | 185 ms | Instant (in-memory) |
| **Tokenizer Artifact Load Time** | 22 ms | Instant |
| **First Token Latency (TTFT)** | ~327 ms (CUDA initial kernel init) | **10.4 ms** (CUDA warm) |
| **Tokens per Second (FP32 CUDA)** | — | **82.9 – 98.6 tokens/sec** |
| **Tokens per Second (FP16 CUDA)** | — | **65.2 tokens/sec** |
| **Tokens per Second (INT8 CPU)** | — | **38.9 tokens/sec** |
| **Tokens per Second (FP32 CPU)** | — | **93.7 tokens/sec** |

---

## 2. Memory Footprint
- Model VRAM (FP32): 76.1 MB allocated
- Model VRAM (FP16): 107.9 MB peak during generation
- Host RAM overhead: ~240 MB
- Zero memory leakage observed during 450+ continuous generation turns.
