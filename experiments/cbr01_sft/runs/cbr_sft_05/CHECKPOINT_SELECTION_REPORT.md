# CBR-SFT-05 Checkpoint Selection Report

**Date:** 2026-11-04  
**Mission:** CBR-01 / CBR-SFT-05  
**Experiment:** Context Binding, Induction & Conversational Memory  
**Evaluator:** Antigravity Autonomous Research Agent  

---

## 1. Candidate Comparison Matrix

All candidates were evaluated deterministically using temperature 0.0 with identical prompt serializations and decoding configurations across all six test suites:
1. **CBR-SFT-05 Held-Out Suite (120 tasks):** Exact copy, key-value retrieval, multi-turn retention, and abstention.
2. **CBR-SFT-04 Held-Out Suite (80 tasks):** Time-conditioned greetings, identity, grounding, abstention.
3. **CBR-SFT-03B Held-Out Suite (70 tasks):** Free-form conversational greetings and ChakrView identity.
4. **Canonical M3 Conversational Suite (60 tasks):** Core conversational benchmark.
5. **Native Intelligence Suite (32 tasks):** Logic, reasoning, attractor collapse check.
6. **Coding Benchmark (12 tasks):** Syntax validity and functional execution.

### Full Benchmark Summary Table

| Candidate Checkpoint | SHA-256 (Prefix) | CBR-05 (120) | CBR-04 Sem (80) | CBR-03B (70) | M3 Conv (60) | Nat Intel (32) | Coding Syntax (12) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CBR-04 Step 200 (Parent)** | `327118ce` | 1 / 120 (0.8%) | **20 / 80 (25.0%)** | 18 / 70 (25.7%) | **4 / 60 (6.7%)** | **3 / 32 (9.4%)** | 5 / 12 (41.7%) |
| **CBR-03B Step 200 (Grandparent)**| `297dcc60` | 1 / 120 (0.8%) | 7 / 80 (8.8%) | **19 / 70 (27.1%)** | 3 / 60 (5.0%) | 1 / 32 (3.1%) | 4 / 12 (33.3%) |
| **CBR-05 Step 50** | `a619297f` | 1 / 120 (0.8%) | 0 / 80 (0.0%) | 0 / 70 (0.0%) | 0 / 60 (0.0%) | 1 / 32 (3.1%) | 5 / 12 (41.7%) |
| **CBR-05 Step 100** | `8653d0bb` | 5 / 120 (4.2%) | 4 / 80 (5.0%) | 3 / 70 (4.3%) | 1 / 60 (1.7%) | 1 / 32 (3.1%) | **6 / 12 (50.0%)** |
| **CBR-05 Step 150** | `fc5f33e2` | 5 / 120 (4.2%) | 1 / 80 (1.2%) | 1 / 70 (1.4%) | 0 / 60 (0.0%) | 1 / 32 (3.1%) | 4 / 12 (33.3%) |
| **CBR-05 Step 200** | `34bc6a12` | **6 / 120 (5.0%)** | 1 / 80 (1.2%) | 1 / 70 (1.4%) | 0 / 60 (0.0%) | 1 / 32 (3.1%) | 5 / 12 (41.7%) |
| **CBR-05 Step 250** | `46d820df` | **6 / 120 (5.0%)** | 1 / 80 (1.2%) | 1 / 70 (1.4%) | 0 / 60 (0.0%) | 1 / 32 (3.1%) | 4 / 12 (33.3%) |
| **CBR-05 Step 300** | `7dcb81dc` | **6 / 120 (5.0%)** | 1 / 80 (1.2%) | 1 / 70 (1.4%) | 0 / 60 (0.0%) | 1 / 32 (3.1%) | 4 / 12 (33.3%) |

---

## 2. Checkpoint Selection & Evaluation Decision

### Official Lineage Decision:
- **Selected Best CBR-SFT-05 Checkpoint:** **Step 100** (`sft_checkpoint_step_100.pt`, SHA-256: `8653d0bbe1b7ba3ea7eebeedf37a2b6b2e1d5d44d67743c4bcb7b7b065d1985f`)
- **Canonical Parent Retained as Production / Primary Lineage Head:** **CBR-SFT-04 Step 200** (`sft_checkpoint_step_200.pt`, SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`)

### Rationale:
1. **Catastrophic Conversational Regression in CBR-SFT-05:**
   - Training on dense copy-supervision and abstention caused the model to lose its conversational greetings and identity responses (dropping from 25.0% on CBR-04 and 25.7% on CBR-03B down to 1.2% - 5.0%).
   - Step 100 balances initial copy induction attempts and abstention recognition (4.2% on CBR-05) with partial retention of conversational identity (4/80 on CBR-04, 3/70 on CBR-03B, 1/60 on M3 Conv, and the highest coding syntax rate at 50.0%).
   - Later steps (200, 250, 300) collapse entirely into an overfit abstention basin ("That information is not provided in the context." emitted universally across key-value and conversational queries).
2. **Safety and Capability Integrity:**
   - Because CBR-SFT-05 models suffer a substantial conversational capability regression without establishing generalized copy induction, **CBR-SFT-04 Step 200 remains the superior overall model** and the official ancestor for conversational checkpoints.
   - CBR-SFT-05 Step 100 is archived as a valuable scientific experiment checkpoint highlighting the exact threshold where copy-induction pressure collides with conversational representations in small parameter regimes.
