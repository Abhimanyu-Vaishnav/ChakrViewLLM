# ChakrView v0.9.0 — Comprehensive Evaluation Report

## 1. Multi-Suite Benchmark Results

Comprehensive evaluation executed across canonical checkpoints on native CPU:

| Benchmark Suite | Total Tasks | Canonical Step 10k Baseline | M3 Step 250 Checkpoint | M3 Step 500 Checkpoint |
| :--- | :---: | :---: | :---: | :---: |
| **Conversational Benchmark (M3)** | 60 | 0/60 (0.0%) | 1/60 (1.67%) | 0/60 (0.0%) |
| **Intelligence Suite** | 32 | 1/32 (3.1%) | 2/32 (6.2%) | 3/32 (9.4%) |
| **Coding Benchmark** | 12 | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) |
| — Syntax Validity Rate | 12 | 3/12 (25.0%) | 4/12 (33.3%) | 4/12 (33.3%) |
| **Neural Harness Accuracy** | 30 | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) |
| **Attractor Rate** | 32 | 53.1% | 15.6% | 15.6% |

---

## 2. Key Insights & Diagnostics
1. **Attractor Suppression Verified:** SFT training at Step 250 significantly reduced repetitive attractor collapse from 53.1% down to 15.6%.
2. **Coding Syntax Capability:** Syntax validity reached 33.3% on basic Python prompts, though full functional execution remained at 0%.
3. **Conversational Challenge:** Microscopic models require variable-binding data and higher capacity to generalize across held-out unseen domains.
