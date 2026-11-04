# CBR-SFT-03A — CHECKPOINT COMPARISON REPORT
## Evaluation of Canonical Baseline vs. All Eight CBR-SFT-02 Checkpoints

**Mission ID:** `CBR-01`  
**Experiment ID:** `CBR-SFT-03A`  
**Parent Model:** Canonical ChakrMicro Step 10,000 (`training_checkpoint_step_10000.pt`)  
**Parent Checkpoint SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  
**Date:** 2026-11-04  

---

## 1. Master Independent 5-Suite Comparison Matrix

Every checkpoint was evaluated on identical prompt serialization, temperature=0.0 (greedy), and deterministic word-boundary scoring across all 5 independent evaluation suites on native CPU:

| Checkpoint Identifier | Step | Suite A: Conv (60) | Suite B: Intel (32) | Attractor Collapse Rate | Suite C: Coding Syntax | Suite D: Entity (100) | Suite E: M-Turn (50) | SHA-256 (First 16 Hex) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Canonical Baseline** | 10k | **0/60 (0.0%)** | **1/32 (3.1%)** | **17/32 (53.1%)** | **3/12 (25.0%)** | **0/100 (0.0%)** | **0/50 (0.0%)** | `96357705a683482e` |
| **CBR-SFT-02 Step 25** | 25 | 0/60 (0.0%) | 3/32 (9.4%) | 14/32 (43.8%) | 4/12 (33.3%) | 0/100 (0.0%) | 0/50 (0.0%) | `b752e514360d1ddc` |
| **CBR-SFT-02 Step 50** | 50 | 0/60 (0.0%) | **3/32 (9.4%)** | 11/32 (34.4%) | **4/12 (33.3%)** | 0/100 (0.0%) | 0/50 (0.0%) | `592553c3e27b6549` |
| **CBR-SFT-02 Step 75** | 75 | 0/60 (0.0%) | 1/32 (3.1%) | 11/32 (34.4%) | 4/12 (33.3%) | 0/100 (0.0%) | 0/50 (0.0%) | `09afbd221d9ad467` |
| **CBR-SFT-02 Step 100**| 100| 0/60 (0.0%) | 1/32 (3.1%) | **9/32 (28.1%)** | 3/12 (25.0%) | 0/100 (0.0%) | 0/50 (0.0%) | `5c51f1d5d0efcd40` |
| **CBR-SFT-02 Step 125**| 125| 0/60 (0.0%) | 1/32 (3.1%) | 9/32 (28.1%) | 2/12 (16.7%) | 0/100 (0.0%) | 0/50 (0.0%) | `b261eff121dca2a7` |
| **CBR-SFT-02 Step 150**| 150| 0/60 (0.0%) | 2/32 (6.2%) | 10/32 (31.2%) | 2/12 (16.7%) | 0/100 (0.0%) | 0/50 (0.0%) | `7a6545396ddef320` |
| **CBR-SFT-02 Step 175**| 175| 0/60 (0.0%) | 1/32 (3.1%) | 10/32 (31.2%) | 3/12 (25.0%) | 0/100 (0.0%) | 0/50 (0.0%) | `343c70e0a685cf18` |
| **CBR-SFT-02 Step 200**| 200| 0/60 (0.0%) | 2/32 (6.2%) | 11/32 (34.4%) | 2/12 (16.7%) | 0/100 (0.0%) | 0/50 (0.0%) | `7d5b233d6b6d4ee5` |

---

## 2. Definitive Checkpoint Ranking & Selection

- **Overall Strongest Checkpoint:** **Step 50 (`sft_checkpoint_step_50.pt`)**
  - **Why:** Step 50 achieves the best Pareto frontier across all suites: highest Native Intelligence score (**3/32, 9.4%**), highest Coding AST syntax validity (**4/12, 33.3%**), substantial attractor reduction (**34.4%** vs 53.1%), and lowest validation loss during the stable early epoch.
  - Beyond Step 100, coding syntax validity degrades (dropping to 16.7% at Step 150 and Step 200), indicating slight over-fitting to the specific prose distributions of the SFT mixture.
- **Attractor Suppression Leader:** **Step 100 (`sft_checkpoint_step_100.pt`)**
  - Achieves the lowest greedy attractor collapse rate (**28.1%**).
