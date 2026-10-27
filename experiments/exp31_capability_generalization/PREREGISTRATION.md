# EXPERIMENT 31 PREREGISTRATION
## Capability Improvement, Generalization Validation & Safe Next-Milestone Decision

**Date:** 2026-10-27  
**Status:** Pre-registered before training/eval execution  
**Repository:** `ChakrView` (Commit: `bcd67d97af5d826a481c706c1f23cd4b6840ad15`)  
**Investigator:** Principal ML Research Engineer & Experimental Auditor  

---

### 1. Primary Research Question
Can ChakrView's canonical neural architecture (or a minimally extended pointer-generator variant) achieve a measurable, reproducible improvement on a deterministic, context-dependent key-value copying/retrieval task that generalizes to genuinely unseen token identities, unseen slot arrangements, distractors, counterfactuals, and negative controls, without relying on memorization or positional shortcuts?

### 2. Hypotheses
- **Null Hypothesis ($H_0$):** Canonical ChakrMicro (both zero-shot and SFT-adapted) and ChakrPointerMicro achieve zero or near-chance accuracy on unseen token identities (Group B) and unseen arrangements (Group C), demonstrating that token-level retrieval in small transformers is bounded by memorized embedding bindings.
- **Alternative Hypothesis 1 ($H_1$ - Canonical SFT Transfer):** Canonical ChakrMicro fine-tuned with target-masked cross-entropy achieves high performance on familiar tokens (Group A > 80%) and statistically above-chance transfer to unseen token identities (Group B > 25%).
- **Alternative Hypothesis 2 ($H_2$ - Pointer Direct Copying):** ChakrPointerMicro (using copy-attention distributions directly over context token IDs) achieves near-perfect generalization to unseen token identities (Group B > 90%) and slot permutations (Group C > 80%), solving the token-identity binding barrier observed in standard autoregressive vocabulary heads.
- **Alternative Hypothesis 3 ($H_3$ - Robustness Failure):** Despite high performance on Groups A and B, both models degrade significantly under distractors (Group D), counterfactuals (Group E), or negative/absent key controls (Group F), revealing reliance on template shortcuts.

### 3. Task Specification & Format
- **Task:** Context-dependent Key-Value Retrieval with Abstention.
- **Template:**
  ```text
  Context: Stream alpha: Data: [X]. Stream beta: Value: [Y]. Stream gamma: Status: [Z].
  Question: What is the Value token?
  Answer: [Y]
  ```
- **Context Components:**
  - 2 to 4 key-value binding pairs per instance.
  - Standard keys: `Alpha`, `Beta`, `Gamma`, `Delta`, `Data`, `Value`, `Status`, `Metric`.
  - Target Token: Single token value associated with the queried key.
  - Abstention Case: When the queried key does not exist in context, the ground truth answer is `?`.

### 4. Evaluation Groups
- **Group A (Familiar Distribution):** 50 examples. Keys and value tokens drawn from training pool ($V_{train} = \{\text{A, B, C, D, E, F, G, H, I, J, K, L}\}$). Evaluates in-distribution task acquisition.
- **Group B (Unseen Token Identities):** 50 examples. Values strictly drawn from held-out pool ($V_{unseen} = \{\text{M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, 1, 2, 3, 4, 5, 6, 7, 8, 9, !}\}$). Strictly disjoint: $V_{train} \cap V_{unseen} = \emptyset$. Evaluates out-of-vocabulary identity generalization.
- **Group C (Unseen Slot Permutations):** 50 examples. Familiar tokens, but slot order inverted/randomized (e.g. queried slot placed first or last rather than standard middle position). Evaluates positional shortcut invariance.
- **Group D (Distractor Complexity):** 50 examples. Context expanded to 4 distinct key-value pairs with competing distractors. Evaluates retrieval attention selectivity.
- **Group E (Counterfactual Inversion):** 50 pairs (100 evaluations). Identical prompt prefix and keys, but the value of the target key is systematically altered. Evaluates whether prediction tracks the dynamic context or defaults to prior token bias.
- **Group F (Negative Control / Abstention):** 50 examples. The queried key is absent from the context. Correct output is `?`. Evaluates false-positive avoidance and abstention.

### 5. Model Variants
1. **Variant 0 (Baseline Zero-Shot):** Canonical `ChakrMicro` (Step 10,000 checkpoint, hash `96357705...`), frozen weights, temperature 0.0 greedy decode.
2. **Variant 1 (Canonical SFT):** `ChakrMicro` initialized from Step 10,000 checkpoint, fine-tuned on 120 training examples for 300 steps with target-masked cross-entropy loss.
3. **Variant 2 (ChakrPointerMicro):** Dual-head pointer-generator initialized from base weights with a copy-attention projection, trained under identical budget (120 examples, 300 steps).

### 6. Training Budget & Resource Limits
- **Device:** CPU strictly (AMD64, bounded to 4 threads).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.01$).
- **Learning Rate:** $1 \times 10^{-4}$ with cosine decay to $1 \times 10^{-5}$.
- **Batch Size:** 8 (gradient accumulation 1).
- **Max Steps:** 300 steps.
- **Max Time:** $\le 10$ minutes CPU wall-clock.
- **Max RAM:** $\le 4.0$ GB peak memory.
- **Seeds:** Preregistered primary seed `42`, confirmation seed `137`.

### 7. Evaluation Metrics & Success Criteria
- **Primary Metric:** Group B (Unseen Token Identity) Exact-Match (EM) Accuracy.
- **Secondary Metrics:**
  - Overall Test Set EM Accuracy across all groups (350 examples).
  - Group A (In-Distribution EM).
  - Group C (Permutation Invariance EM).
  - Group D (Distractor Robustness EM).
  - Group E (Counterfactual Responsiveness EM).
  - Group F (Abstention Precision & Recall).
- **Acceptance Criteria for Capability Claim:**
  - Model must achieve Group A EM $\ge 90\%$ (validating task acquisition).
  - Model must achieve Group B EM $\ge 80\%$ (validating generalization beyond token memorization).
  - Model must achieve Group C EM $\ge 80\%$ (validating invariance to slot order).
  - Regression smoke tests on canonical ChakrView must remain 100% passing.
  - Zero modification to canonical checkpoint, vocabulary, or protected core code.
- **Stopping Criteria:**
  - Training loss diverges ($\ge 10.0$ or NaN).
  - CPU memory exceeds 4.0 GB.
  - Any attempt to overwrite or mutate canonical artifacts.
