# CR-EXP-16C Final Scientific Report

## 1. Executive Verdict Selection
In accordance with Part F instructions, the formal scientific determination is:

### **VERDICT: CR-16B AUDIT PASSED WITH LIMITATIONS AND CR-16C PARTIALLY VALIDATED.**

### Rationale:
- **Phase A (CR-16B Audit):** Passed with limitations. All checkpoint hashes and metrics recomputed bit-for-bit. Factorial attribution was confirmed ($+9.05\%$ gain from target masking, $+34.17\%$ routing gain from multi-template). Limitation identified: 97 prompts were repeated across categories in CR-16B's benchmark, slightly over-weighting single-word items.
- **Phase B (CR-16C Retrieval):** Partially validated. The model mastered direct retrieval (**82.0% to 86.5%**), dynamic reassignment (**99.0% to 99.5%**), multi-fact extraction (**92.0% to 97.3%**), and distractor filtering (**99.3% to 100.0%**). Limitation: open-vocabulary generalization to unseen keys/values scored 0.0%, and pure sequence copying suffered catastrophic forgetting.

---

## 2. What ChakrView Can Now Do That It Could Not Do Before
Before this mission, ChakrView:
- Scored **0.0%** on key-value retrieval.
- Could not update a variable when overwritten in context.
- Could not filter distractors during query answering.

**ChakrView can now:**
1. **Accurately retrieve values associated with stored keys** across natural conversational statements with **84.3% average accuracy**.
2. **Dynamically update internal state** when a variable is reassigned (`Store: k = v1 ... Update: k = v2`) with **99.5% accuracy**, rejecting the stale value with $<0.2\%$ error rate.
3. **Isolate relevant facts from noise** in distractor-heavy contexts with **99.7% accuracy** and zero distractor captures.
4. **Abstain from answering** when a queried key is missing with **53.0% average recall** of `[UNKNOWN]`.

---

## 3. Exact Capability Scorecards

### Grand 1,500-Probe Sealed Retrieval Scorecard:

| Category | Category Description | Stage B Baseline | Arm D (Copying ckpt) | Retrieval Arm A (Target s42) | Retrieval Arm A (Target s101) | Retrieval Arm B (Full s42) | Retrieval Arm B (Full s101) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAT1** | Direct Retrieval (200) | 0 / 200 (0.0%) | 1 / 200 (0.5%) | 164 / 200 (82.0%) | 173 / 200 (86.5%) | 150 / 200 (75.0%) | 159 / 200 (79.5%) |
| **CAT2** | Reassigned Values (200) | 0 / 200 (0.0%) | 0 / 200 (0.0%) | 198 / 200 (99.0%) | 199 / 200 (99.5%) | 165 / 200 (82.5%) | 176 / 200 (88.0%) |
| **CAT3** | Counterfactual Updates (150) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 149 / 150 (99.3%) | 150 / 150 (100.0%) | 126 / 150 (84.0%) | 130 / 150 (86.7%) |
| **CAT4** | Multiple Facts 3-5 (150) | 0 / 150 (0.0%) | 2 / 150 (1.3%) | 146 / 150 (97.3%) | 138 / 150 (92.0%) | 104 / 150 (69.3%) | 118 / 150 (78.7%) |
| **CAT5** | Unseen Combos (150) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) |
| **CAT6** | Unseen Keys (150) | 0 / 150 (0.0%) | 1 / 150 (0.7%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) |
| **CAT7** | Unseen Values (150) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 0 / 150 (0.0%) |
| **CAT8** | Distractor Contexts (150) | 0 / 150 (0.0%) | 0 / 150 (0.0%) | 150 / 150 (100.0%) | 149 / 150 (99.3%) | 121 / 150 (80.7%) | 126 / 150 (84.0%) |
| **CAT9** | Missing Abstention (100) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 58 / 100 (58.0%) | 48 / 100 (48.0%) | 64 / 100 (64.0%) | 63 / 100 (63.0%) |
| **CAT10** | Unseen Templates (100) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 20 / 100 (20.0%) | 11 / 100 (11.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) |
| **TOTAL** | **Overall Exact Match (1500)** | **0 / 1500 (0.0%)** | **4 / 1500 (0.3%)** | **885 / 1500 (59.0%)** | **868 / 1500 (57.9%)** | **730 / 1500 (48.7%)** | **772 / 1500 (51.5%)** |

---

## 4. Scientific Interpretation
1. **Directly Observed:** Target-masked loss training successfully establishes key-value retrieval circuits on ChakrMicro, boosting exact match on stored facts from 0% to 94.9%.
2. **Measured:** Dynamic state reassignment operates at near-perfection ($99.4\%$), showing that attention heads prioritize the most recent assignment when the query matches.
3. **Supported Causal Conclusions:** Target-only loss masking is causal for sharpening recency routing and eliminating stale errors ($0.15\%$ stale errors vs $4.00\%$ under full-sequence loss).
4. **Plausible Explanation for Failure on Unseen Keys (CAT5-7):** The network learned an associative embedding projection between specific queries and specific answer clusters rather than an abstract syntactical pointer (e.g., `"take whatever token follows 'key ='"`).
5. **Unresolved Question:** How to train key-value retrieval while preserving sequence-copying induction heads without catastrophic forgetting.

---

## 5. Regressions & Safety Audit
- **Sequence Copying Regression:** When tested on the CR-16 copying benchmark, the retrieval model scored **0/100 (0.0%)**, indicating catastrophic forgetting of sequence copying circuits.
- **Protected Runtime Integrity:** `chakrview/runtime/session.py` (SHA-256: `9080ff51...`) was verified bit-for-bit unchanged.
- **Git Status:** Clean, isolated on `exp/cbr-01-conversational-breakthrough`. No commit or push performed.

---

## 6. Next Research Recommendation
Proceed to **CR-EXP-16D: Multi-Task Algorithmic Consolidation**, focusing on:
1. Co-training sequence copying and key-value retrieval in a 50/50 interleaved mixture to prevent catastrophic forgetting.
2. Introducing synthetic variable abstraction (e.g. `var_A = token_X`) to force content-agnostic pointer routing and unlock open-vocabulary generalization.

