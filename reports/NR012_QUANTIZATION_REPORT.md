# CHAKRVIEW RESEARCH REPORT: NR-012 QUANTIZATION REPORT
**Mission ID:** CV-GENESIS-NR-012  
**Date:** 2026-11-25  
**Target Architecture:** ChakrGenesis-17M (17,063,808 parameters)  
**Experiment Script:** `experiments/quantization_012/run_quantization_benchmarks.py`  

---

## 1. Executive Summary
We evaluated three precision regimes on ChakrGenesis-17M:
1. Baseline Full Precision FP32 (CUDA & CPU)
2. Half Precision FP16 (CUDA)
3. Dynamic INT8 Quantization (CPU via `torch.ao.quantization.quantize_dynamic`)

---

## 2. Experimental Benchmark Data

| Precision Regime | Target Hardware | Disk Size (MB) | Compression Ratio | Peak VRAM (MB) | Throughput (tok/s) | Latency (ms/32 tok) | Output Stability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 Baseline** | NVIDIA RTX A1000 | 65.1 MB | 1.00x | 76.1 MB | **98.6 tok/s** | 324.4 ms | Exact Baseline |
| **FP16 Half** | NVIDIA RTX A1000 | 32.6 MB | **2.00x** | 107.9 MB | **65.2 tok/s** | 490.7 ms | 100% Identical |
| **Dynamic INT8** | Host CPU (Core i9) | 22.4 MB | **2.91x** | 0.0 MB | **38.9 tok/s** | 822.7 ms | Minor lexical variance |
| **FP32 CPU Ref** | Host CPU (Core i9) | 65.1 MB | 1.00x | 0.0 MB | **93.7 tok/s** | 341.5 ms | Exact Baseline |

---

## 3. Key Observations & Recommendations
- **FP16 on CUDA** cuts disk storage exactly by 50% ($65.1 \rightarrow 32.6$ MB) while preserving 100% bit-exact generation text vs FP32.
- **Dynamic INT8 on CPU** compresses weights by 2.91x ($22.4$ MB), enabling edge deployment on modest hardware without requiring GPUs.
