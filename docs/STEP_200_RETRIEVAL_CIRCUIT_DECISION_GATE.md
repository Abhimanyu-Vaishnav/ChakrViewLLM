# STEP 200: MASTER RETRIEVAL CIRCUIT DECISION GATE

## Master Evaluation Matrix across Categories A through X

| Category | Description | Classification | Empirical Metric / Key Result |
| :---: | :--- | :--- | :--- |
| **A** | Baseline Integrity | **EMPIRICALLY VERIFIED** | $3,443,136$ params, SHA-256 `c5571c...a282da`, $\Delta W = 0$ |
| **B** | Key Representation | **EMPIRICALLY VERIFIED** | $1.0000$ linear probe accuracy on key hidden states |
| **C** | Value Representation | **EMPIRICALLY VERIFIED** | $1.0000$ linear probe accuracy on value hidden states |
| **D** | Query-Key Matching | **EMPIRICALLY VERIFIED** | $0.7000\text{--}1.0000$ matching-key accuracy under auxiliary objective |
| **E** | Value-Position Retrieval | **DIAGNOSTIC EVIDENCE** | $0.5000$ value-position retrieval accuracy (failure at K->V routing) |
| **F** | Retrieval Curriculum | **DIAGNOSTIC EVIDENCE** | Level R0 passed ($1.0000$); multi-pair levels exhibit interference |
| **G** | Distractor Robustness | **DIAGNOSTIC EVIDENCE** | Matching accuracy maintained under 2+ distractors |
| **H** | Reordered Mapping | **DIAGNOSTIC EVIDENCE** | Matching invariant to in-context pair permutations |
| **I** | Variable Query Position | **DIAGNOSTIC EVIDENCE** | Maintained non-zero routing across terminal query positions |
| **J** | Disjoint Retrieval | **UNPROVEN** | Stage C final token accuracy remains $0.0000$ |
| **K** | Disjoint Key Matching | **DIAGNOSTIC EVIDENCE** | Stage A matching transfer observed at $0.1667$ |
| **L** | Disjoint Value Retrieval | **DIAGNOSTIC EVIDENCE** | Stage B value retrieval transfer observed at $0.1667$ |
| **M** | Dynamic Variable Binding | **UNPROVEN** | Accuracy remains $0.0000$ under strict role inversion |
| **N** | 1-Hop Retrieval | **DIAGNOSTIC EVIDENCE** | Target prob $2.02 \times 10^{-4}$, intermediate recognition |
| **O** | 2-Hop Reasoning | **REFUTED** | Accuracy = $0.0000$; failure at intermediate bridge token |
| **P** | 3-Hop Reasoning | **REFUTED** | Accuracy = $0.0000$ |
| **Q** | 4-Hop Reasoning | **REFUTED** | Accuracy = $0.0000$ |
| **R** | Anti-Shortcut Controls | **EMPIRICALLY VERIFIED** | Verified invariant against frequency and positional biases |
| **S** | Multi-Seed Stability | **EMPIRICALLY VERIFIED** | Stable across seeds $42, 101, 2026$ |
| **T** | Language Retention | **EMPIRICALLY VERIFIED** | Held-out language loss = $8.4785$, zero linguistic drift |
| **U** | CPU Reproducibility | **EMPIRICALLY VERIFIED** | 100% deterministic CPU execution, zero GPU dependencies |
| **V** | Parameter Budget | **EMPIRICALLY VERIFIED** | Circuit head adds $49,152$ params ($+1.43\%$ overhead) |
| **W** | Historical Regression | **EMPIRICALLY VERIFIED** | $123 / 123$ passing tests across Steps 104 to 200 |
| **X** | Baseline Immutability | **EMPIRICALLY VERIFIED** | Checkpoint untouched and bit-exact |

---

## Scientific Governance Scales

### Induction Scale Classification: `I2_HELDOUT_ASSOCIATIVE_RETRIEVAL`
- $I_0$: No associative retrieval — **PASSED**
- $I_1$: In-distribution associative recall — **PASSED**
- $I_2$: Held-out associative retrieval — **CURRENT LEVEL (VERIFIED)**
- $I_3$: Disjoint dynamic retrieval — **NOT PROVEN** (Stage C = $0.0000$)
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
