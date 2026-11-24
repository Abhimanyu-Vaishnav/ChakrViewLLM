# CV-GENESIS-NR-010 Checkpoint Comparison Report

## 1. Executive Summary

Mission **CV-GENESIS-NR-010** established ChakrView's next-generation 17M foundation models, comparing pretraining and SFT checkpoints against historical 5.18M baselines.

---

## 2. Checkpoint Parameter & Lineage Comparison

| Metric | Canonical 5.18M Baseline | ChakrGenesis-17M (Pretrained) | ChakrGenesis-17M (Conversational SFT) |
| :--- | :--- | :--- | :--- |
| **Total Parameters** | 5,177,600 | **17,063,808** | **17,063,808** |
| **Non-Tied Parameters** | 4,129,024 | **15,490,944** | **15,490,944** |
| **Layers ($L$)** | 6 | **10** | **10** |
| **Model Dimension ($d_{\text{model}}$)** | 256 | **384** | **384** |
| **Query Heads ($n_q$)** | 8 | **12** | **12** |
| **KV Heads ($n_{kv}$)** | 2 | **3** | **3** |
| **FFN Dimension ($d_{\text{ff}}$)** | 682 | **1024** | **1024** |
| **Pretrain Loss (Validation)** | 2.8710 | **2.8425** | — |
| **SFT Loss (Validation)** | 2.5314 | — | **2.3930** (Perplexity: **10.95**) |
| **Checkpoint Size (MB)** | 19.77 MB | **65.13 MB** | **65.13 MB** |
| **Best Checkpoint SHA-256** | `03a4c16d20ad3621...` | `96219f224e7c4a79...` | `ac62c43fda2e33ac...` |
