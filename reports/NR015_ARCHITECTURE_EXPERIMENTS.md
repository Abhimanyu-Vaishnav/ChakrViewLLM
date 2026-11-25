# NR-015 Phase 6 — Controlled Architecture Experiment Report
**Mission ID:** CV-GENESIS-NR-015  
**Date:** 2026-11-25  
**Experiment:** Controlled Capacity Scaling (17M Baseline vs. 38M Scaled Architecture)  
**Artifacts:** `data/curriculum_015/architecture_pilot_results.json`

---

## 1. Architectural Configurations
To test the hypothesis of capacity-constrained generalization, a scaled configuration was designed preserving all causal invariants, RoPE, Pre-RMSNorm, and GQA 4:1 ratios:

| Dimension | ChakrGenesis-17M Baseline | ChakrGenesis-38M Scaled Pilot | Ratio / Change |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | $17,063,808$ | **$38,285,824$** | $+124.4\%$ ($2.24\times$) |
| **Layers ($n_{layers}$)** | 10 | 12 | $+2$ layers |
| **Model Dimension ($d_{model}$)** | 384 | 512 | $+33.3\%$ width |
| **Query Heads ($n_{q\_heads}$)** | 12 | 16 | $+33.3\%$ heads |
| **Key/Value Heads ($n_{kv\_heads}$)** | 3 | 4 | $4:1$ GQA preserved |
| **FFN Dimension ($d_{ff}$)** | 1024 | 1536 | $+50.0\%$ SwiGLU width |
| **Context Length** | 512 | 512 | Controlled |

---

## 2. Empirical Performance & Hardware Profiling

| Performance Metric | 17M Baseline | 38M Scaled Pilot | Feasibility Verdict |
| :--- | :---: | :---: | :--- |
| **Initial Cross-Entropy Loss** | $8.3827$ | $8.4385$ | Expected $\approx \ln(4096)$ |
| **50-Step Convergence Loss** | **$0.0837$** | **$0.0889$** | Fast overfit confirmed |
| **50-Step Wallclock Time** | $2.36\text{s}$ ($21.2\text{ steps/s}$) | $3.91\text{s}$ ($12.8\text{ steps/s}$) | Highly feasible on single GPU |
| **Peak VRAM Allocation** | $581.5\text{ MB}$ | $927.0\text{ MB}$ | Well within GPU headroom |

---

## 3. Scientific Findings & Model Promotion Decision
1. **Mathematical Feasibility:** The 38M architecture runs smoothly on GPU within $<1.0\text{ GB}$ VRAM, with identical loss convergence properties and zero numerical instability.
2. **Preservation Invariant:** In accordance with Global Rule 10 ("Never promote a checkpoint solely because validation loss improves") and Phase 6 rules ("Do not replace the canonical model"), `ChakrGenesis-17M Stage-D Best` is strictly maintained as the canonical production foundation. The 38M architecture is validated as a candidate blueprint for future full-corpus pretraining campaigns.
