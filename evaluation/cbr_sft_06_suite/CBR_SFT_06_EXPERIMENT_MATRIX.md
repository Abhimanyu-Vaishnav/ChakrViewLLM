# CBR-SFT-06 Proposed Experiment Matrix & Learning Strategies

**Date:** 2026-11-04  
**Mission:** CBR-SFT-06 Preparation  
**Parent Lineage Head:** CBR-SFT-04 Step 200 (`327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`)  

---

## 1. Controlled Experiment Matrix

We design three tightly controlled candidate runs to scientifically test the remediations identified in CBR-SFT-06 forensics:

| Run ID | Strategy Name | Replay Ratio | Retrieval Composition | Main Research Hypothesis |
| :--- | :--- | :---: | :--- | :--- |
| **CBR-06A** | **Anchored Replay Baseline** | **35% Replay** | Single-token copy & simple KV | Adding 35% CBR-04 conversational replay completely prevents greeting regression while enabling initial copy learning. |
| **CBR-06B** | **Balanced Curriculum** | **30% Replay** | Short entities (1-3 tokens) + diverse non-verbatim abstentions | Eliminating verbatim abstention repetitions breaks the abstention attractor basin and unlocks key-value extraction. |
| **CBR-06C** | **Staged Copy Induction** | **25% Replay** | Stage 1: Single-token copy $\rightarrow$ Stage 2: Short KV lookup | Curricular sequencing from 1-token to multi-token entities reduces cross-entropy variance in early steps. |

---

## 2. Recommended Primary Candidate: CBR-06A (Anchored Replay)

### Specification:
- **Parent Checkpoint:** `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` (SHA: `327118ce...`)
- **Total Examples:** 250 examples (225 train, 25 val)
- **Data Composition:**
  - 80 examples (35.6%): Conversational greetings & identity replay directly from CBR-SFT-04.
  - 70 examples (31.1%): Single-token exact copy tasks (ports, single digits, single status codes).
  - 55 examples (24.4%): Short key-value bindings (e.g. `PORT=80`, `ENV=prod`).
  - 20 examples (8.9%): Diverse, non-verbatim abstentions (using 5 distinct phrasing variants).
- **Hyperparameters:**
  - Effective Batch Size: 4 (micro-batch 2, grad accum 2)
  - Optimizer: AdamW (weight decay 0.01, grad clip 1.0)
  - Peak Learning Rate: $1.0 \times 10^{-4}$ (cosine decay to $1.0 \times 10^{-5}$)
  - Steps: 200 (checkpoints saved at steps 50, 100, 150, 200)
  - Early-stopping condition: Validation perplexity increase $> 15\%$ over 2 consecutive checkpoints.
- **Success Criteria:**
  - CBR-04 Greeting/Identity Suite: $\ge 20.0\%$ (no catastrophic forgetting).
  - CBR-05 Exact Copy (Single Token): $\ge 15.0\%$.
  - M3 Conversational: $\ge 5.0\%$.
  - Native Intelligence: $\ge 6.25\%$.
