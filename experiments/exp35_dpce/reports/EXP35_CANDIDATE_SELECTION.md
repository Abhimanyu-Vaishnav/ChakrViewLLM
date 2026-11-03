# CHAKRVIEW — EXP35 CANDIDATE SELECTION REPORT
## Candidate Evaluation and Formal Verdict on Checkpoint Selection

**Experiment:** Exp35-DPCE  
**Date:** 2026-10-30  

---

## 1. PRE-DECLARED SELECTION SCORECARD REVIEW

Under Section 13 of the Frozen Exp35 Specification, candidate selection is governed by a strict, pre-declared priority order:
1. Exact grounded extraction accuracy on independent benchmark.
2. Unseen entity-combination accuracy.
3. Template generalization accuracy.
4. Changed-fact tracking.
5. Distractor resistance.
6. Multi-turn retention.
7. Unsupported answer rate.
8. EOS and repetition stability.
9. Validation loss.

**Explicit Rule:**  
*"If neither arm demonstrates meaningful improvement over baseline, do not select a winner."*

---

## 2. EVALUATION OF CANDIDATE CHECKPOINTS

1. **Exp35 Arm A Step 500 (`experiments/exp35_dpce/runs/arm_a_m3_step250/sft_checkpoint_step_500.pt`):**
   - Grounded Extraction: **0/200 (0.0%)** (Baseline: 0/200).
   - Val Loss: 2.5216.
   - Improvement over baseline: **NONE**.
2. **Exp35 Arm B Step 500 (`experiments/exp35_dpce/runs/arm_b_m2_arma_step50/sft_checkpoint_step_500.pt`):**
   - Grounded Extraction: **0/200 (0.0%)** (Baseline: 0/200).
   - Val Loss: 1.4006.
   - Improvement over baseline: **NONE**.

---

## 3. OFFICIAL CANDIDATE SELECTION VERDICT

### **NO WINNER SELECTED**

**Justification:**  
Neither Arm A nor Arm B demonstrated any measurable capability improvement on the 200-task independent extraction benchmark over their respective baselines. While both models achieved remarkable physical generation stability (repetition $< 0.03$, EOS termination $> 86\%$), selecting a checkpoint based purely on validation loss reduction would violate the scientific mandate of Milestone M5.

- **Status of Checkpoints:** Both Arm A and Arm B checkpoints are preserved as valuable negative experimental controls in the repository lineage.
- **Production / Baseline Recommendation:** ChakrView continues to recognize **M3 Step 250** (`91b0bae2...`) as the latest generation-stabilized experimental research prototype, without claiming conversational extraction capability.

