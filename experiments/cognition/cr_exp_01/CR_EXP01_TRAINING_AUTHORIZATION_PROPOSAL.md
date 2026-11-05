# CR-EXP-01: TRAINING AUTHORIZATION PROPOSAL

This document constitutes the formal pre-execution request submitted for user review and explicit authorization.

---

## 1. Executive Summary & Objective

- **Experiment Identifier**: `CR-EXP-01`
- **Objective**: Conduct a controlled, staged experiment on ChakrMicro v0.1 to determine whether the model can learn in-context variable binding and generalize to previously unseen key-value pairs without parametric memorization.
- **Parent Checkpoint**: `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` (SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`).
- **Current Status**: **AUDIT & DESIGN COMPLETE — AWAITING EXPLICIT USER AUTHORIZATION**.

---

## 2. Staged Training Plan

### Stage A: Small Feasibility Pilot (100 Steps)
1. **Dataset**: 1,000 synthetic records with disjoint Train Pool entities, 10 distinct template formats, and prompt-masked targets.
2. **Hyperparameters**:
   - Learning Rate: $3\times 10^{-5}$ decaying to $3\times 10^{-6}$ via cosine schedule.
   - Effective Batch Size: 8 sequences (batch size 4, accum 2).
   - Sequence Length: 64 tokens.
   - Duration: ~1.2 minutes on CPU.
3. **Mid-Point Evaluation Gate (Step 100)**:
   - Checkpoint saved to `experiments/cognition/cr_exp_01/checkpoints/candidate_step_100.pt`.
   - Condition B (Novel Associations) accuracy must exceed **30.0%** (baseline is 0.0%).
   - If Condition B $< 30.0\%$, training stops immediately; do not proceed to Stage B.

### Stage B: Controlled Expansion (300 Steps)
- *Triggered ONLY if Stage A passes the 30% gate.*
- Extends training to 300 steps to solidify attention induction patterns across distractor scrambles.

### Stage C: Multi-Condition & Regression Evaluation
- Full evaluation across all 10 conditions (Novel entities, distractors, multi-turn recall, unseen syntax, and frozen syntactic probes).

---

## 3. Explicit Numerical Acceptance Gates

To mark CAP-03 (Variable Binding) as **PARTIALLY VALIDATED** or **VALIDATED** in the Capability Registry:

| Gate Identifier | Target Metric | Required Threshold | Scientific Justification |
|---|---|---|---|
| **GATE-1: Novel In-Context Binding** | Condition B Exact Match ($EM_B$) | $\mathbf{\ge 60.0\%}$ | Baseline is 0.0%. A 60% score on completely unseen keys and values proves generalized induction circuits rather than memorization. |
| **GATE-2: Memorized Baseline** | Condition A Exact Match ($EM_A$) | $\mathbf{\ge 90.0\%}$ | Confirms training loss converged on familiar training pairs. |
| **GATE-3: Distractor Robustness** | Condition D Exact Match at $k=3$ | $\mathbf{\ge 50.0\%}$ | Proves attention heads select the correct key rather than latching onto positional recency. |
| **GATE-4: Multi-Turn Memory** | Condition G Exact Match ($EM_G$) | $\mathbf{\ge 70.0\%}$ | Proves binding transfers across turn boundaries in conversational formats. |
| **GATE-5: Syntax Preservation** | Frozen 40 Syntactic Probes | $\mathbf{\ge 15.0\%}$ (Top-5) | Zero degradation compared to parent baseline (prevents catastrophic forgetting). |
| **GATE-6: Identity Preservation** | ChakrView Identity Probe | $\mathbf{100.0\%}$ Exact Match | Ensures model retains self-awareness without regression. |

---

## 4. Operational Protection Contract

During execution of CR-EXP-01:
- Zero canonical files (`model.py`, `session.py`, canonical checkpoints) will be touched.
- All candidate artifacts, checkpoints, and logs will be isolated inside `experiments/cognition/cr_exp_01/`.
- Zero Git commits, pushes, or branch switches will occur.
- Execution will strictly halt if any step encounters non-finite loss or memory exceeding 1 GB.

---

## 5. Authorization Request

We recommend: **`READY FOR REVIEW AND USER AUTHORIZATION`**.
Training will **NOT** begin until you provide explicit authorization.

