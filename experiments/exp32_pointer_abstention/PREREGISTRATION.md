# EXPERIMENT 32 PREREGISTRATION
## Pointer Retrieval, Distractor Resolution, Missing-Key Abstention & Leakage Elimination

**Date:** 2026-10-27  
**Status:** PRE-REGISTERED (Before Training & Evaluation)  
**Investigator:** Principal ML Research Engineer & Repository Safety Owner  
**Repository:** `ChakrView` (Commit: `bcd67d97af5d826a481c706c1f23cd4b6840ad15`)  

---

### 1. Research Question
Can an audited pointer-generator neural architecture (`ChakrPointerMicro`) reliably retrieve the correct value bound to a queried key in multi-slot, variable-order contexts with competing distractors, while successfully abstaining (emitting `?`) when the queried key is absent from the context?

Specifically:
1. Does training on an abstention-aware mixture (present keys + absent keys) allow the model to learn a dual objective: pointer copying when the key exists, and vocabulary fallback / abstention token emission when the key is absent?
2. Does distractor density degrade pointer selection, and what is the relative proportion of binding errors vs. distractor selection vs. false abstentions?

### 2. Hypotheses
- **Null Hypothesis ($H_0$):** Training on an abstention-balanced mixture fails to induce reliable abstention ($F1_{abstain} < 0.50$) or causes catastrophic false abstentions on valid retrieval queries (retrieval accuracy drops below zero-shot baseline).
- **Alternative Hypothesis 1 ($H_1$ - Balanced Retrieval & Abstention):** `ChakrPointerMicro` achieves $\ge 70\%$ retrieval accuracy on familiar and permuted contexts while simultaneously achieving $\ge 70\%$ precision and recall on missing-key abstention.
- **Alternative Hypothesis 2 ($H_2$ - Unseen Identity Generalization with Abstention):** `ChakrPointerMicro` maintains statistically significant unseen token copying on held-out tokens ($\ge 40\%$) while properly abstaining when queried for absent keys.

### 3. Task Specification & Format
- **Retrieval Template:**
  ```text
  Context: Stream alpha: Value: [X]. Stream beta: Value: [Y]. Stream gamma: Value: [Z].
  Question: What is the beta value?
  Answer: [Y]
  ```
- **Abstention Template:**
  ```text
  Context: Stream alpha: Value: [X]. Stream beta: Value: [Y].
  Question: What is the delta value?
  Answer: ?
  ```
- **Abstention Target:** `?` (Token ID 32 in canonical vocabulary).

### 4. Leakage Elimination & Disjoint Split Guarantees
- **Strict Set-Theoretic Deduplication:** All prompts across Train, Validation, and Test Groups are tracked in a global hash set. Zero prompt collisions permitted across any split ($Split_i \cap Split_j = \emptyset, \forall i \neq j$).
- **Disjoint Token Pools:**
  - $V_{train} = \{\text{A, B, C, D, E, F, G, H, I, J, K, L}\}$ (12 familiar tokens).
  - $V_{unseen} = \{\text{M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, 1, 2, 3, 4, 5, 6, 7, 8, 9, !}\}$ (24 held-out tokens).
  - Strict disjointness: $V_{train} \cap V_{unseen} = \emptyset$.
- **Training Mixture:**
  - Total: 160 training examples.
  - Composition: 120 positive retrieval examples (2–3 bindings) + 40 negative abstention examples (25% abstention ratio).

### 5. Evaluation Groups (Frozen Test Set: 350 Unique Examples)
1. **Group A — Familiar Basic Retrieval (50 examples):** Familiar tokens in standard 2–3 key contexts. Evaluates in-distribution retrieval.
2. **Group B — Unseen Token Identities (50 examples):** Held-out token values never seen in training. Evaluates token-copy generalization.
3. **Group C — Slot Permutation Invariance (50 examples):** Target key strictly placed at extreme edges (slot 1 or slot 3). Evaluates positional shortcut invariance.
4. **Group D — Distractor Robustness (50 examples):** 4 competing key-value pairs (maximum distractors). Evaluates selective attention.
5. **Group E — Counterfactual Consistency (50 pairs = 100 examples):** Identical context except target value swapped. Evaluates prompt conditioning.
6. **Group F — Missing-Key Abstention (50 examples):** Queried key absent from context. Ground truth is `?`. Evaluates abstention sensitivity.

### 6. Model Variants
- **Arm 0 (Canonical Zero-Shot Baseline):** Canonical `ChakrMicro` (Step 10,000 checkpoint, hash `96357705...`), frozen weights, greedy decode.
- **Arm 1 (Standard Pointer Baseline - Exp31 Lineage):** `ChakrPointerMicro` trained on 160 examples with pure cross-entropy / copy loss.
- **Arm 2 (Abstention-Tuned Pointer):** `ChakrPointerMicro` trained on the balanced present/absent mixture, evaluated for both retrieval accuracy and abstention F1.

### 7. Resource Bounds & Training Parameters
- **Hardware:** CPU strictly, bounded to 4 threads.
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.01$).
- **Learning Rate:** $5 \times 10^{-5}$ cosine decaying to $5 \times 10^{-6}$.
- **Batch Size:** 4.
- **Steps:** 300 steps.
- **Max RAM:** $\le 3.0$ GB peak memory.
- **Max Time:** $\le 10$ minutes CPU wall-clock.
- **Deterministic Seed:** Primary seed `42`.

### 8. Acceptance & Failure Criteria
- **Acceptance Criteria for Capability Claim:**
  - Group A (In-Distribution EM) $\ge 60.0\%$.
  - Group B (Unseen Token EM) $\ge 30.0\%$.
  - Group F (Abstention EM) $\ge 60.0\%$.
  - False Abstention Rate on Groups A–D $< 15.0\%$.
  - Bit-exact canonical artifact integrity preserved.
- **Stopping Criteria:**
  - Training loss diverges ($\ge 10.0$ or NaN).
  - CPU memory exceeds 3.0 GB.
