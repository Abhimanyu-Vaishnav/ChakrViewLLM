# NR-019: Performance and Cost Report

## 1. Hardware Environment
- **Device**: NVIDIA GPU (`cuda`)
- **Foundation Model**: ChakrGenesis-17M (17,063,808 parameters)

---

## 2. Component Latency Benchmarks (Mean over $\ge 2,000$ iterations)

| Pipeline Stage | Mean Latency | Median Latency | Throughput |
| :--- | :---: | :---: | :---: |
| **Deterministic Rule Parser** | $0.005\ \text{ms}$ | $0.004\ \text{ms}$ | $200,000\ \text{ops/s}$ |
| **Source-Span Grounding Engine** | $0.011\ \text{ms}$ | $0.010\ \text{ms}$ | $90,000\ \text{ops/s}$ |
| **Multilingual Numeral Normalizer** | $0.006\ \text{ms}$ | $0.005\ \text{ms}$ | $160,000\ \text{ops/s}$ |
| **Restricted Runtime Executor** | $0.001\ \text{ms}$ | $0.001\ \text{ms}$ | $1,000,000\ \text{ops/s}$ |
| **Neural Intent Forward Pass (32 tokens)** | $8.06\ \text{ms}$ | $8.02\ \text{ms}$ | $124\ \text{passes/s}$ |
| **Complete System E Cascaded Turn** | **$8.08\ \text{ms}$** | **$8.04\ \text{ms}$** | **$123.7\ \text{turns/s}$** |

---

## 3. Memory & Resource Footprint
- **GPU VRAM Allocated**: 132.72 MB
- **GPU VRAM Reserved**: 150.0 MB
- **System RAM Working Set**: $\approx 1.05\ \text{GB}$
- **Checkpoint Disk Footprint**: 65.13 MB (`genesis_17m_staged_best.pt` / `neural_router_17m.pt`)
