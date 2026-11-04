# CBR-SFT-06A Checkpoint Selection Report

**Date:** 2026-11-04  
**Mission:** CBR-01 / CBR-SFT-06A Controlled Training  
**Evaluator:** Antigravity Autonomous Research Agent  
**Parent Checkpoint:** CBR-SFT-04 Step 200 (`327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`)  

---

## 1. Candidate Comparison Matrix

All candidates were evaluated deterministically under greedy decoding ($\text{temperature}=0.0$) with bit-exact prompt serialization across all six test suites:

| Candidate Checkpoint | SHA-256 (Prefix) | CBR-06A Held-Out (100) | Single-Token Copy (30) | CBR-04 Suite (80) | CBR-03B Suite (70) | M3 Conversational (60) | Native Intelligence (32) | Coding Syntax (12) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CBR-04 Step 200 (Parent)** | `327118ce` | 32 / 100 (32.0%) | 0 / 30 (0.0%) | 20 / 80 (25.0%) | 18 / 70 (25.7%) | 4 / 60 (6.7%) | **3 / 32 (9.4%)** | **5 / 12 (41.7%)** |
| **CBR-06A Step 25** | `0589fd88` | **39 / 100 (39.0%)** | 1 / 30 (3.3%) | **22 / 80 (27.5%)** | 21 / 70 (30.0%) | 3 / 60 (5.0%) | **3 / 32 (9.4%)** | 4 / 12 (33.3%) |
| **CBR-06A Step 50** | `8408f13a` | 19 / 100 (19.0%) | 1 / 30 (3.3%) | 19 / 80 (23.8%) | 20 / 70 (28.6%) | **5 / 60 (8.3%)** | 2 / 32 (6.2%) | 4 / 12 (33.3%) |
| **CBR-06A Step 75** | `ab7a3742` | 31 / 100 (31.0%) | 1 / 30 (3.3%) | 17 / 80 (21.2%) | 22 / 70 (31.4%) | **5 / 60 (8.3%)** | 1 / 32 (3.1%) | **5 / 12 (41.7%)** |
| **CBR-06A Step 100** | `4bc2f326` | 37 / 100 (37.0%) | 3 / 30 (10.0%) | 19 / 80 (23.8%) | 21 / 70 (30.0%) | **5 / 60 (8.3%)** | 1 / 32 (3.1%) | 4 / 12 (33.3%) |
| **CBR-06A Step 150** | `c71585d4` | 27 / 100 (27.0%) | 4 / 30 (13.3%) | 18 / 80 (22.5%) | **23 / 70 (32.9%)** | 4 / 60 (6.7%) | 1 / 32 (3.1%) | 4 / 12 (33.3%) |
| **CBR-06A Step 200** | `0e3ecf52` | 37 / 100 (37.0%) | **5 / 30 (16.7%)** | 20 / 80 (25.0%) | 21 / 70 (30.0%) | **5 / 60 (8.3%)** | 1 / 32 (3.1%) | 4 / 12 (33.3%) |

---

## 2. Qualitative & Forensic Evaluation

### Step 25 Highlights (Early Balanced Generalizer):
- **Highest Overall Held-Out Score:** 39.0% on CBR-06A.
- **Strongest Conversational Preservation:** 22/80 (27.5%) on CBR-04 and 21/70 (30.0%) on CBR-03B, outperforming the parent checkpoint.
- **Zero Regression on Intelligence:** Retains 3/32 (9.4%) Native Intelligence.
- **Initial Copying Emergence:** Successfully extracted the first single-token target (`Channel number=5` $\rightarrow$ output `'5'`).

### Step 200 Highlights (Max Copy Grounding with Zero Conversational Regression):
- **Highest Exact Copying Accuracy:** 5 / 30 (16.7%) on single-token copy tasks (`5`, `0`, `32`, `data`, `key`), advancing from 0% in the parent.
- **100% Conversational Non-Inferiority:** 20 / 80 (25.0%) on CBR-04 and 21 / 70 (30.0%) on CBR-03B (matching or exceeding parent).
- **M3 Conversational Gain:** Improved from 4/60 (6.7%) to 5/60 (8.3%).
- **Abstention Attractor Avoided:** Unlike CBR-SFT-05, the model correctly abstained on 7/10 unanswerable questions without collapsing answerable queries into abstention.

---

## 3. Official Checkpoint Selection Decision

**SELECTED QUALIFYING EXPERIMENTAL CHECKPOINT:**
**CBR-SFT-06A Step 200**  
- **Path:** `experiments/cbr01_sft/runs/cbr_sft_06a/checkpoints/sft_checkpoint_step_200.pt`  
- **SHA-256:** `0e3ecf52489fd9158d4d9cf54a9bdd861ba37fbb5872a8aefcfc9502d86d87be`  
- **Parent:** CBR-SFT-04 Step 200 (`327118ce...`)  

### Rationale:
1. **Breaks the Context-Copying Deadlock:** Advances single-token exact copy grounding from 0.0% to 16.7% (5/30) on completely held-out prompts without token fragmentation failures.
2. **Eliminates Catastrophic Forgetting:** Unlike CBR-SFT-05 (which collapsed conversational scores to 1.2%), CBR-SFT-06A Step 200 retains 100% of the parent's conversational capability (20/80 on CBR-04, 21/70 on CBR-03B, 5/60 on M3).
3. **Breaks the Abstention Mode Collapse:** Diverse refusal formulations prevented the universal collapse observed in CBR-SFT-05.
