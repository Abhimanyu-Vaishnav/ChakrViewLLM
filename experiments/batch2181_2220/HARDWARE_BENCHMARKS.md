# HARDWARE BENCHMARKS

**Batch:** 2181–2220  
**Model:** ChakrMicro Canonical Baseline (3,443,136 parameters, float32, tied embeddings)  
**Hardware:** Intel(R) Core(TM) i5-13500H (16 CPUs) + NVIDIA RTX A1000 6GB Laptop GPU  
**Environment:** PyTorch 2.14.1+cu126 on Windows 11  

---

## 1. Benchmarking Matrix

All benchmarks were executed using identical test prompts, greedy decoding, and warm-up cycles.

| Workload Configuration | Metric | CPU Execution | CUDA GPU Execution | Speedup / Winner |
| :--- | :--- | :---: | :---: | :---: |
| **Model Initialization** | Checkpoint Load & Init | **71.52 ms** | 198.43 ms | **CPU (2.77x faster)** |
| **Interactive Prefill (B=1)**| Average Prefill Latency | **3.68 ms** | 13.69 ms | **CPU (3.72x faster)** |
| **Interactive Prefill (B=1)**| Prefill Throughput | **5,815.5 tok/s**| 1,563.4 tok/s | **CPU (3.72x faster)** |
| **Time to First Token (TTFT)**| Initial Output Token Latency| **3.82 ms** | 9.43 ms | **CPU (2.47x faster)** |
| **Interactive Generation (B=1)**| 32-Token Generation Latency| **133.58 ms** | 266.60 ms | **CPU (2.00x faster)** |
| **Interactive Generation (B=1)**| Generation Throughput | **239.55 tok/s**| 120.03 tok/s | **CPU (2.00x faster)** |
| **Parallel Batch Forward (B=4)**| 256-Seq Latency | 35.70 ms | **10.11 ms** | **GPU (3.53x faster)** |
| **High-Throughput Batch (B=32)**| 256-Seq Latency | 377.06 ms | **52.69 ms** | **GPU (7.16x faster)** |
| **Resource Allocation** | Peak System RAM | 1,016 MB | 980 MB | Comparable |
| **Resource Allocation** | Peak GPU VRAM | 0.0 MB | 69.68 MB | Negligible footprint (< 1.2% of 6GB) |

---

## 2. Key Insights & System Guidance

1. **Why Small-Batch Generation Favors CPU:**
   - ChakrMicro is an ultralight core (3.44M parameters, ~13.8 MB).
   - In single-token autoregressive decoding at batch=1, kernel launches across the PCI-e bus take ~0.3 ms each. With 6 layers and multiple transformer operations per layer, driver dispatch time dominates raw compute.
   - On the Intel i5-13500H CPU with AVX2/AVX512 vectorization, L3 cache locality allows immediate execution at 239 tokens/second.
2. **Why Batched Operations Favor GPU:**
   - As soon as batch size reaches 4 or 32 (or during CPT training sweeps), GPU parallel compute completely overcomes launch overhead, yielding a **7.16x acceleration**.
3. **Adaptive Selection Strategy:**
   - For interactive chat CLI: Run on CPU.
   - For CPT training or evaluation of large document splits: Run on GPU.
