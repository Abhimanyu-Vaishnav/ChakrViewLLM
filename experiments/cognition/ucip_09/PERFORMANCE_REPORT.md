# UCIP-09: Inference Performance, Latency & Resource Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Candidate Checkpoint:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt`  
**Date:** November 17, 2026  

---

## 1. System & Hardware Metrics

- **Host Platform:** Windows 11 Pro x86_64
- **Processor:** GenuineIntel Intel Core i9 (13th Gen)
- **CPython Build:** 3.14.7 64-bit
- **PyTorch Build:** 2.14.0+cpu (Native CPU execution)
- **Model Checkpoint Size:** 51,756,113 bytes (51.8 MB on disk)
- **Parameter Count:** 12,932,736 parameters

---

## 2. Autoregressive Inference Throughput & Latency

| Benchmark Operation | Total Generated Tokens | Elapsed Wall Time | Generation Speed | Peak Memory (RSS) |
|---|---|---|---|---|
| Single Step Forward Pass | 1 | 6.8 ms | 147.0 tok/s | ~68 MB |
| Full 48-Token Rollout | 48 | 382.4 ms | 125.5 tok/s | ~76 MB |
| Interactive Turn (Chat CLI) | ~25 tokens | ~198.0 ms | 126.2 tok/s | ~82 MB |
| Complete 500-Probe Benchmark Run | ~14,200 tokens | 2 minutes 47 seconds | 85.0 tok/s (inc. scoring) | 124 MB |

---

## 3. Resource Discipline Assessment

ChakrMedium strictly conforms to CPU-first edge device execution:
- Peak RAM usage remained under **125 MB** during the entire 500-probe independent evaluation run.
- Zero GPU dependencies or external services were invoked.
- Instantaneous token throughput of **>125 tokens/second** provides a fluid, real-time conversational experience in the terminal.
