# UCIP-13: Throughput & Resource Profile Report

**Mission ID:** UCIP-13 — ChakrMicro Foundational Neural Pretraining  
**Hardware Device:** CPU (Intel/AMD x86_64, Windows)  
**Precision:** PyTorch float32  

---

## 1. Computational Throughput & Profile

- **Batch Size:** 16 sequences
- **Context Length ($T$):** 512 tokens
- **Effective Tokens Per Optimizer Step:** **8,192 tokens**
- **Average Step Duration:** 1.35 seconds
- **Throughput:** **~5,800 to 6,600 tokens/second**
- **Total Optimizer Steps Executed:** 800 steps
- **Total Training Duration:** **1,318.89 seconds (21.98 minutes)**
- **Total Tokens Exposed:** **6,553,600 tokens**

---

## 2. Resource Consumption

- **Memory Usage:** Peak RSS stayed strictly under 1.2 GB RAM due to shard streaming via `StreamingTokenDataset`.
- **Numerical Stability:** 0 NaN or Inf values observed across all 800 steps. Gradient clipping threshold of 1.0 successfully contained gradient norms.
- **Validation Overhead:** Full evaluation over 57 validation batches (466,944 tokens) completed in ~23 seconds per validation epoch.
