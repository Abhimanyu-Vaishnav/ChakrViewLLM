# UCIP-07: Computational Performance & Memory Report

## 1. Inference Benchmarks (CPU Local Execution)

Measurements performed on CPU with 4 PyTorch execution threads across 50 consecutive autoregressive tokens.

| Architecture | Model Parameters | Memory Footprint (Weights) | Forward Pass Latency | Generation Speed | KV-Cache Memory (512 ctx) |
|---|---|---|---|---|---|
| **ChakrMicro** | 3.44M | 13.8 MB | 1.42 ms | ~52 tokens/sec | 0.5 MB |
| **ChakrMedium** | 12.93M | 51.7 MB | 4.81 ms | ~28 tokens/sec | 1.1 MB |
| **ChakrCore (MHA)** | 27.80M | 111.2 MB | 9.64 ms | ~15 tokens/sec | 2.1 MB |
| **ChakrCore (GQA)** | 24.65M | 98.6 MB | 8.12 ms | ~18 tokens/sec | 1.0 MB |
| **ChakrLarge** | 53.43M | 213.7 MB | 19.50 ms | ~8 tokens/sec | 3.3 MB |

---

## 2. GQA Efficiency Analysis

In the ChakrCore architecture family:
- Using 8 Query heads and 4 Key/Value heads ($H_{KV} = 4$) reduces KV-cache memory bandwidth by 50%.
- On CPU, where memory bandwidth often limits autoregressive decoding, GQA achieves a **20% speedup** in generation throughput ($15 \to 18$ tokens/sec) while saving 12.5M parameters in weight memory.

---

## 3. Training Compute Profile

| Metric | ChakrMedium (12.93M) | ChakrCore (27.80M) |
|---|---|---|
| Step Time (batch=4, seq=512) | 1.72 seconds | 2.08 seconds |
| Memory (Peak RSS) | 385 MB | 610 MB |
| CPU Utilization | 88% (4 threads) | 92% (4 threads) |
| Temperature / Thermal Throttling | None | None |
