# EXPERIMENT 33 PREREGISTRATION
## Relational Pointer Circuit, Key-to-Value Binding & Independent Validation

**Date:** 2026-10-27  
**Status:** PRE-REGISTERED (Before Training & Evaluation)  
**Investigator:** Principal ML Research Engineer & Repository Safety Owner  
**Repository:** `ChakrView` (Commit: `bcd67d97af5d826a481c706c1f23cd4b6840ad15`)  

---

### 1. Research Question
Does explicitly modeling key-to-query matching before routing pointer attention to associated value tokens (`ChakrRelationalPointerMicro`) resolve distractor slot misattributions and improve retrieval accuracy across variable-order contexts and unseen tokens, while maintaining reliable abstention on absent keys?

### 2. Hypotheses
- **Null Hypothesis ($H_0$):** `ChakrRelationalPointerMicro` achieves distractor retrieval performance indistinguishable from or worse than standard single-head `ChakrPointerMicro` (Group D accuracy $\le 26.0\%$), and distractor errors remain $>90\%$ of failure modes.
- **Alternative Hypothesis 1 ($H_1$ - Relational Distractor Reduction):** The relational pointer head significantly improves distractor robustness (Group D accuracy $\ge 40.0\%$, compared to $26.0\%$ in Exp32), and cuts distractor error frequency by at least $25\%$.
- **Alternative Hypothesis 2 ($H_2$ - Permutation & Held-Out Transfer):** Relational key alignment generalizes across slot permutations (Group C $\ge 50.0\%$) and held-out token identities (Group B $\ge 35.0\%$).
- **Alternative Hypothesis 3 ($H_3$ - Robust Abstention Induction):** The explicit key-affinity abstention gate improves missing-key abstention on Group F ($F1 \ge 0.50$, compared to $0.35$ in Exp32) while keeping false abstentions on valid retrieval below $10\%$.

### 3. Task Specification & Format
- **Retrieval Prompt:**
  ```text
  Context: Stream alpha: Value: [X]. Stream beta: Value: [Y]. Stream gamma: Value: [Z].
  Question: What is the beta value?
  Answer: [Y]
  ```
- **Abstention Prompt:**
  ```text
  Context: Stream alpha: Value: [X]. Stream beta: Value: [Y].
  Question: What is the delta value?
  Answer: ?
  ```

### 4. Dataset Splits & Leakage Elimination
- **Deduplication:** Strict global prompt set tracking. Zero prompt collisions across splits ($Split_i \cap Split_j = \emptyset, \forall i \neq j$).
- **Vocabulary Pools:**
  - $V_{train} = \{\text{A, B, C, D, E, F, G, H, I, J, K, L}\}$ (12 familiar single BPE tokens).
  - $V_{unseen} = \{\text{M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, 1, 2, 3, 4, 5, 6, 7, 8, 9, !}\}$ (24 held-out tokens).
  - $V_{train} \cap V_{unseen} = \emptyset$.
- **Splits:**
  - `train.jsonl`: 160 instances (120 positive retrieval + 40 negative abstention).
  - `val.jsonl`: 40 instances (30 positive + 10 negative).
  - `test_group_A_familiar.jsonl`: 50 instances.
  - `test_group_B_unseen_tokens.jsonl`: 50 instances.
  - `test_group_C_permutations.jsonl`: 50 instances.
  - `test_group_D_distractors.jsonl`: 50 instances (4 competing keys).
  - `test_group_E_counterfactuals.jsonl`: 100 instances (50 pairs).
  - `test_group_F_negative_abstain.jsonl`: 50 instances.
  - Total test set: 350 frozen, unique instances.

### 5. Model Variants
- **Arm 0 (Canonical Zero-Shot Baseline):** Step 10,000 checkpoint (`96357705...`), frozen weights, greedy decoding.
- **Arm 1 (Standard Pointer Baseline):** `ChakrPointerMicro` (Exp32 lineage) trained on balanced mixture (160 examples).
- **Arm 2 (ChakrRelationalPointerMicro):** Two-stage relational pointer head with key-matching and transition kernel, trained under identical budget (160 examples, 300 steps).

### 6. Resource Limits & Budget
- **Device:** CPU strictly (AMD64, bounded to 4 threads).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.01$).
- **Learning Rate:** $5 \times 10^{-5}$ with cosine decay to $5 \times 10^{-6}$.
- **Batch Size:** 4.
- **Steps:** 300 steps.
- **Max RAM:** $\le 3.0$ GB peak memory.
- **Seed:** Primary seed `42`.

### 7. Acceptance Criteria for Milestone Progression
- Group D (Distractor EM) $\ge 40.0\%$.
- Group F (Abstention F1) $\ge 0.50$.
- False Abstention Rate on Groups A–E $< 10.0\%$.
- Regression test suite (`test_v060_release_smoke.py`) remains 100% PASS.
- Zero modification to canonical checkpoint or protected files.
