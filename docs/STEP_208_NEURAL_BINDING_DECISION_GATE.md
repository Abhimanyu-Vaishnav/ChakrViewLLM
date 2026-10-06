# STEP 208: MASTER NEURAL BINDING DECISION GATE

## Master Evaluation Matrix across Categories A through AB

| Category | Description | Classification | Empirical Result / Key Metric |
| :---: | :--- | :--- | :--- |
| **A** | Baseline Integrity | **EMPIRICALLY VERIFIED** | $3,443,136$ params, SHA-256 `c5571c...a282da`, $\Delta W = 0$ |
| **B** | Key Identity | **EMPIRICALLY VERIFIED** | Linear probe accuracy: $1.0000$ |
| **C** | Value Identity | **EMPIRICALLY VERIFIED** | Linear probe accuracy: $1.0000$ |
| **D** | Association Representation | **DIAGNOSTIC EVIDENCE** | Contrastive margin: $+0.0711$, probe acc: $0.6667$ |
| **E** | Relative Binding | **DIAGNOSTIC EVIDENCE** | Directional invariance: $1.0000$, pos bias: $0.0000$ |
| **F** | Contrastive Binding | **DIAGNOSTIC EVIDENCE** | Disjoint binding transfer: $0.1667$ with $\lambda = 0.5$ |
| **G** | Permutation Invariance | **EMPIRICALLY VERIFIED** | Variance across 5 layouts: $0.0000$ |
| **H** | Distractor Robustness | **DIAGNOSTIC EVIDENCE** | Smooth entropy scaling from $1.58 \to 2.18$ |
| **I** | Known/Known Binding | **DIAGNOSTIC EVIDENCE** | Diagnostic score: $0.2014$, median rank: $2,840$ |
| **J** | Known/Unseen Binding | **DIAGNOSTIC EVIDENCE** | Target prob: $1.85 \times 10^{-4}$ |
| **K** | Unseen/Known Binding | **DIAGNOSTIC EVIDENCE** | Target prob: $1.94 \times 10^{-4}$ |
| **L** | Unseen/Unseen Binding | **UNPROVEN** | Stage D token output: $0.0000$ |
| **M** | Disjoint Key Matching | **DIAGNOSTIC EVIDENCE** | Stage A matching signal observed |
| **N** | Disjoint Association | **DIAGNOSTIC EVIDENCE** | Stage B score: $0.1820$ |
| **O** | Disjoint Value Retrieval | **UNPROVEN** | Stage C value retrieval: $0.0000$ |
| **P** | Final Disjoint Output | **UNPROVEN** | Stage D final output: $0.0000$ |
| **Q** | Variable Binding | **UNPROVEN** | Disjoint role extraction: $0.0000$ |
| **R** | 1-Hop Retrieval | **DIAGNOSTIC EVIDENCE** | Target prob: $2.02 \times 10^{-4}$ |
| **S** | 2-Hop Reasoning | **REFUTED** | Accuracy = $0.0000$; failure at intermediate bridge |
| **T** | 3-Hop Reasoning | **REFUTED** | Accuracy = $0.0000$ |
| **U** | 4-Hop Reasoning | **REFUTED** | Accuracy = $0.0000$ |
| **V** | Anti-Shortcut Controls | **EMPIRICALLY VERIFIED** | Invariant across layouts and distractors |
| **W** | Multi-Seed Stability | **EMPIRICALLY VERIFIED** | Replicated identically across seeds $42, 101, 2026$ |
| **X** | Language Retention | **EMPIRICALLY VERIFIED** | Held-out loss = $8.4785$, zero linguistic regression |
| **Y** | CPU Reproducibility | **EMPIRICALLY VERIFIED** | 100% deterministic CPU execution |
| **Z** | Parameter Budget | **EMPIRICALLY VERIFIED** | Binding head adds $24,576$ params ($+0.71\%$ overhead) |
| **AA**| Historical Regression | **EMPIRICALLY VERIFIED** | $131 / 131$ passing tests across Steps 104 to 208 |
| **AB**| Baseline Immutability | **EMPIRICALLY VERIFIED** | Checkpoint untouched and bit-exact |

---

## Scientific Governance Scales

### Induction Scale Classification: `I2_HELDOUT_ASSOCIATIVE_RETRIEVAL`
- $I_0$: No associative retrieval — **PASSED**
- $I_1$: In-distribution associative recall — **PASSED**
- $I_2$: Held-out associative retrieval — **CURRENT LEVEL (VERIFIED)**
- $I_3$: Disjoint dynamic retrieval — **NOT PROVEN** (Stage D = $0.0000$)
- $I_4$: Robust dynamic variable binding — **NOT PROVEN**
- $I_5$: Induction-supported multi-hop reasoning — **NOT PROVEN**

### Official Neural Reasoning Scale: `LEVEL_1_MEMORIZED_OR_IN_DISTRIBUTION`
- LEVEL 1: Memorized / in-distribution — **CURRENT LEVEL (VERIFIED)**
- LEVEL 2: Direct held-out relational generalization — **UNPROVEN**
- LEVEL 3: Systematic compositional reasoning — **UNPROVEN**
- LEVEL 4: Multi-step compositional reasoning — **UNPROVEN**
- LEVEL 5: Robust distribution-shift reasoning — **UNPROVEN**

---

## Release Governance Decision
**RELEASE DECISION: DO NOT RELEASE YET**
