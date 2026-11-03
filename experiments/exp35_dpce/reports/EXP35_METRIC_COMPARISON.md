# CHAKRVIEW — EXP35 METRIC COMPARISON REPORT
## Statistical Cross-Comparison Across Checkpoints, Training Arms & Benchmarks

**Experiment:** Exp35-DPCE  
**Date:** 2026-10-30  

---

## 1. COMPREHENSIVE METRIC COMPARISON TABLE

| Metric / Dimension | Canonical (Step 10k) | M2 Arm A (Step 50) | M3 Candidate (Step 250) | Exp35 Arm A (Step 500) | Exp35 Arm B (Step 500) |
|---|---|---|---|---|---|
| **Base Checkpoint** | Pretrained Core | Canonical | M2 Arm A Step 50 | M3 Step 250 | M2 Arm A Step 50 |
| **DPCE Initial Val Loss** | 5.2928 | 4.9529 | 5.2928 | 5.2928 | 4.9529 |
| **DPCE Final Train Loss** | — | — | — | 2.0855 | **1.0980** |
| **DPCE Final Val Loss** | — | — | — | 2.5216 | **1.4006** |
| **200-Task Grounded Extraction** | 0/200 (0.0%) | 0/200 (0.0%) | 0/200 (0.0%) | 0/200 (0.0%) | 0/200 (0.0%) |
| **Single-Entity Extraction (30)** | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) |
| **Multi-Entity Disambiguation (30)**| 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) |
| **Unseen Combinations (30)** | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) |
| **Harness Repetition Ratio** | 0.2540 | 0.1862 | 0.0223 | 0.0308 | **0.0207** |
| **Harness Unique Token Ratio** | 0.5314 | 0.5663 | 0.9036 | 0.9068 | **0.9033** |
| **Genuine EOS Rate (30 prompts)**| 2/30 (6.7%) | 2/30 (6.7%) | 28/30 (93.3%) | 26/30 (86.7%) | 26/30 (86.7%) |
| **CPU Throughput (TPS)** | 30.3 | 51.6 | 84.2 | 156.2 | 162.6 |

---

## 2. OBSERVATIONS ON ARM DYNAMICS

1. **Arm B (M2 Arm A base) Learned Faster and Reached Lower Loss:**
   - Arm B achieved a final train loss of **1.0980** and val loss of **1.4006**, significantly outperforming Arm A (train loss 2.0855, val loss 2.5216).
   - This indicates that M3 Step 250 had already begun specializing to the collapsed M3 conversational distribution, making it slightly more resistant to learning new template syntax than the earlier M2 checkpoint.
2. **Loss Divergence from Task Performance:**
   - Despite Arm B reaching an impressive cross-entropy loss of 1.0980 on the DPCE dataset, this reduction in loss did not translate to even a single correct extraction on unseen entity benchmarks.
   - Cross-entropy loss reduction in small causal language models can be entirely driven by learning sentence syntax and high-frequency structural tokens (`"At station"`, `"resides in"`) without acquiring factual variable binding.

