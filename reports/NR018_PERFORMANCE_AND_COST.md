# NR-018: Performance, Latency and Cost Report

## Benchmarked Hardware
- **Device**: NVIDIA GPU (`cuda`)
- **Model Architecture**: ChakrGenesis-17M (17,063,808 parameters)

---

## Latency and Throughput Profiling

| Component | Latency per Operation | Throughput | Relative Speedup |
| :--- | :---: | :---: | :---: |
| **Deterministic Restricted Executor** | **0.0006 ms** ($0.6\ \mu\text{s}$) | $>1,600,000\ \text{ops/sec}$ | **16,137x faster** |
| **Neural Forward Pass (32 tokens)** | **10.04 ms** | $\approx 99.6\ \text{passes/sec}$ | $1\text{x}$ (Baseline) |
| **End-to-End Neural Generation (100 tokens)**| $\approx 310\ \text{ms}$ | $\approx 3.2\ \text{queries/sec}$ | $0.03\text{x}$ |

---

## Memory Footprint
- **Model Checkpoint Size on Disk**: 65.13 MB (`genesis_17m_staged_best.pt` / `neural_router_17m.pt`)
- **GPU VRAM Allocated**: 132.72 MB
- **GPU VRAM Reserved**: 150.0 MB
- **System RAM Working Set**: $\approx 1.4\ \text{GB}$

---

## Computational Cost Analysis
By offloading symbolic calculations, list operations, and state transformations to the deterministic executor:
1. **Zero FLOPs** are wasted simulating arithmetic addition or sorting inside neural attention layers.
2. The neural model acts purely as an intent classifier and query translator, reducing inference token output lengths from lengthy chain-of-thought texts ($>200$ tokens) to concise JSON strings ($<40$ tokens), resulting in a **$5\times$ speedup** in query servicing.
