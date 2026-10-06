# STEP 216: MASTER ASSOCIATION DECISION GATE

## Master Evaluation Matrix across Categories A through AC

| Category | Description | Classification | Empirical Result / Key Metric |
| :---: | :--- | :--- | :--- |
| **A** | Baseline Integrity | **EMPIRICALLY VERIFIED** | $3,443,136$ params, SHA-256 `c5571c...a282da`, $\Delta W = 0$ |
| **B** | Key Identity | **EMPIRICALLY VERIFIED** | Linear probe accuracy: $1.0000$ |
| **C** | Value Identity | **EMPIRICALLY VERIFIED** | Linear probe accuracy: $1.0000$ |
| **D** | Pair Representation | **DIAGNOSTIC EVIDENCE** | Location D interaction product accuracy: $0.8889$ |
| **E** | Association Margin | **DIAGNOSTIC EVIDENCE** | Max contrastive margin: $+0.0711$ |
| **F** | Persistence | **DIAGNOSTIC EVIDENCE** | Persistence ratio: $0.9834$ ($98.3\%$ retention) |
| **G** | Interference Resistance | **DIAGNOSTIC EVIDENCE** | Interference degradation: $3.50\%$ across scaling |
| **H** | Rebinding | **DIAGNOSTIC EVIDENCE** | Mean overwrite ratio: $0.5428$ ($54.3\%$ preference for new) |
| **I** | Old/New Separation | **DIAGNOSTIC EVIDENCE** | Residual interference: $0.4572$ |
| **J** | Known-Known | **DIAGNOSTIC EVIDENCE** | Target probability: $1.87 \times 10^{-4}$ |
| **K** | Known-Unseen | **DIAGNOSTIC EVIDENCE** | Target probability: $1.70 \times 10^{-4}$ |
| **L** | Unseen-Known | **DIAGNOSTIC EVIDENCE** | Target probability: $1.96 \times 10^{-4}$ |
| **M** | Unseen-Unseen | **UNPROVEN** | Stage E final output accuracy: $0.0000$ |
| **N** | Disjoint Assoc Repr | **DIAGNOSTIC EVIDENCE** | Contrastive margin preserved at $+0.0450$ |
| **O** | Disjoint Retrieval | **UNPROVEN** | Zero-shot disjoint retrieval remains $0.0000$ |
| **P** | Variable Binding | **UNPROVEN** | Disjoint role extraction remains $0.0000$ |
| **Q** | Immediate Retrieval | **DIAGNOSTIC EVIDENCE** | Memory head achieves $0.5000$ on familiar pairs |
| **R** | Delayed Retrieval | **DIAGNOSTIC EVIDENCE** | Stable retrieval under delayed queries |
| **S** | Distractor Retrieval | **DIAGNOSTIC EVIDENCE** | Monotonic bounded entropy scaling |
| **T** | Multiple Associations | **DIAGNOSTIC EVIDENCE** | Evaluated across loads $[1, 2, 4, 8]$ |
| **U** | Rebinding Retrieval | **DIAGNOSTIC EVIDENCE** | Updated association preferred over obsolete |
| **V** | Association Ablation | **EMPIRICALLY VERIFIED** | Pair interaction confirmed as most critical component |
| **W** | Anti-Shortcut Controls | **EMPIRICALLY VERIFIED** | Invariant across layouts, sequence lengths, and distractors |
| **X** | Multi-Seed Stability | **EMPIRICALLY VERIFIED** | Replicated identically across seeds $42, 101, 2026$ |
| **Y** | Language Retention | **EMPIRICALLY VERIFIED** | Held-out loss = $8.4785$, zero linguistic regression |
| **Z** | CPU Reproducibility | **EMPIRICALLY VERIFIED** | 100% deterministic CPU execution |
| **AA**| Parameter Budget | **EMPIRICALLY VERIFIED** | Memory head adds $49,345$ params ($+1.43\%$ overhead) |
| **AB**| Historical Regression | **EMPIRICALLY VERIFIED** | $139 / 139$ passing tests across Steps 104 to 216 |
| **AC**| Baseline Immutability | **EMPIRICALLY VERIFIED** | Canonical baseline untouched and bit-exact |

---

## Scientific Governance Scales

### Induction Scale Classification: `I2_HELDOUT_ASSOCIATIVE_RETRIEVAL`
- $I_0$: No associative retrieval — **PASSED**
- $I_1$: In-distribution associative recall — **PASSED**
- $I_2$: Held-out associative retrieval — **CURRENT LEVEL (VERIFIED)**
- $I_3$: Disjoint dynamic associative retrieval — **NOT PROVEN** (Stage E = $0.0000$)
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
