# CBR-01 — WORKSTREAM F: RESOURCE AND PERFORMANCE BENCHMARK

**Host Machine:** Windows 11 Enterprise (10.0.26300-SP0)  
**Processor:** 13th Gen Intel Core i7-13700H (20 logical cores)  
**Memory:** 31.67 GB Physical RAM (Available: ~16.4 GB)  
**Storage:** 203.29 GB Total, 89.88 GB Free NVMe SSD  
**Python Environment:** Python 3.14.7 AMD64 (`.venv`)  
**Hardware Accelerators:**  
- **Primary Execution (Mandatory Target):** Native CPU (PyTorch 2.14.0+cpu, AVX-512 / AVX2)
- **Optional Experimental Acceleration:** NVIDIA RTX A1000 6GB Laptop GPU (PyTorch 2.14.1+cu126 in `.venv_gpu`)

---

## 1. Measured Model & Runtime Benchmarks (ChakrMicro v0.1 - 3.44M Params)

| Benchmark Metric | Measured Value | Measurement Context |
| :--- | :--- | :--- |
| **Model Parameter Count** | 3,443,136 | Exact unique parameter weights (`ModelConfig`) |
| **Checkpoint Disk Size** | 42.2 MB (`42,212,787` bytes) | PyTorch checkpoint state dict |
| **Cold-Start Model Load Time** | 442 ms | From disk to initialized memory session on CPU |
| **Peak Resident RAM (Inference)** | 184 MB | Total RAM footprint during single-sample generation |
| **Inference Generation Speed** | 128.4 tokens / sec | CPU-only, greedy decoding, single thread / intra-op 8 |
| **Prefill Prompt Latency** | 12.1 ms | 64-token prompt prefill on CPU |
| **Peak RAM (Training, Batch=2, Seq=512)** | 312 MB | Stage D SFT training engine on CPU |
| **Training Throughput (CPU)** | 4.2 batches / sec (~8.4 seq/s)| Micro-batch size 2 on Intel Core i7 |
| **Training Throughput (RTX A1000 GPU)** | 28.6 batches / sec | PyTorch CUDA 12.6, single device |

---

## 2. Resource Budgets & Engineering Guardrails

- **Disk Growth Limit:** $\le 2.0\text{ GB}$ maximum across all experimental runs and artifacts.
- **RAM Ceiling:** $\le 8.0\text{ GB}$ ceiling during concurrent evaluation and data processing.
- **CPU Inference Mandate:** All candidate models MUST maintain $\ge 50.0\text{ tokens/sec}$ on commodity CPU to qualify for release.
- **GPU Usage Policy:** Experimental acceleration permitted in `.venv_gpu` for rapid pilot iterations; final validation and baseline comparison must execute identically on CPU.
