# HARDWARE VALIDATION REPORT

**Batch:** 2221–2260  
**Hardware Environment:** Intel(R) Core(TM) i5-13500H (20 logical processors) + NVIDIA RTX A1000 6GB Laptop GPU (31.64 GB RAM)  
**Software Environment:** PyTorch 2.14.1+cu126, CUDA 12.6, Windows 11 AMD64  
**Benchmark Script:** `experiments/batch2221_2260/run_inference_benchmarks.py`  
**Output Data:** `experiments/batch2221_2260/HARDWARE_BENCHMARKS.json`  

---

## 1. Rigorous Benchmarking Methodology

To ensure scientifically valid performance measurements, the benchmarking harness was implemented with strict inference-appropriate controls:
1. **Inference Context:** Enforced via `torch.inference_mode()`, disabling autograd tracking and intermediate graph allocations.
2. **CUDA Synchronization:** Pre- and post-timed invocations synchronize CUDA streams (`torch.cuda.synchronize()`) to prevent timer under-reporting from asynchronous GPU execution.
3. **Warm-up Cycles:** Executed 5 full forward warm-up cycles before timing to ensure JIT kernel compilation and memory caches were primed.
4. **Repeated Measurements:** Collected multiple repetitions across 5 distinct prompt domains, calculating median, minimum, maximum, and standard deviation.
5. **Separation of Concerns:** Explicitly isolated **prompt prefill**, **time-to-first-token (TTFT)**, **single-token autoregressive generation**, and **batched forward passes**.

---

## 2. Benchmark Results Summary

| Workload & Metric | CPU Execution (i5-13500H) | CUDA GPU Execution (RTX A1000) | Winning Device | Rationale & Ratio |
| :--- | :---: | :---: | :---: | :--- |
| **Model Load & Init Latency** | **100.46 ms** (median) | 185.08 ms (median) | **CPU** | 1.84x faster; avoids CUDA context creation overhead. |
| **Time to First Token (TTFT)** | **6.47 ms** (median) | 8.64 ms (median) | **CPU** | 1.34x faster; CPU avoids host-to-device kernel launch latency. |
| **Prefill Throughput (Prompt)** | **3,344.96 tok/s** (median) | 2,184.70 tok/s (median) | **CPU** | 1.53x faster on short interactive prompts (< 25 tokens). |
| **Autoregressive Gen (Batch=1)** | **157.80 tok/s** (median) | 110.37 tok/s (median) | **CPU** | 1.43x faster; sequential single-token PCI-e dispatch overhead dominates GPU. |
| **32-Token Generation Latency** | **202.79 ms** (median) | 289.93 ms (median) | **CPU** | 1.43x faster for interactive single-user chat. |
| **Batched Forward (Batch=1)** | 18.49 ms | **11.91 ms** | **GPU** | 1.55x faster on GPU at full sequence length 256. |
| **Batched Forward (Batch=4)** | 37.14 ms | **8.76 ms** | **GPU** | **4.24x faster on GPU.** |
| **Batched Forward (Batch=16)**| 127.85 ms | **27.72 ms** | **GPU** | **4.61x faster on GPU.** |
| **Batched Forward (Batch=32)**| 291.29 ms | **52.80 ms** | **GPU** | **5.52x faster on GPU** (155,147 tok/s throughput). |
| **VRAM Footprint** | 0.0 MB | 208.26 MB | **Both Safe** | Only 3.4% of 6 GB dedicated VRAM allocated. |

---

## 3. Operational Rules for Adaptive Execution

The empirical data demonstrates that **neither device universally wins across all workloads**:
1. **Interactive Chat & Single Prompts (Batch 1–2):** CPU execution provides lower latency (6.47 ms TTFT vs 8.64 ms) and higher autoregressive throughput (158 tok/s vs 110 tok/s) because ChakrMicro is small (3.44M parameters) and runs efficiently in CPU L3 cache.
2. **Batched Ingestion, Evaluation & Training (Batch $\ge$ 4):** GPU parallel compute achieves massive speedups (**4.2x to 5.5x faster**), processing 155,000 tokens/second at batch 32.
3. **Adaptive Policy:** `AdaptiveExecutionPolicy` seamlessly switches between CPU and GPU based on workload size, while maintaining guaranteed CPU fallback if CUDA is unavailable.
