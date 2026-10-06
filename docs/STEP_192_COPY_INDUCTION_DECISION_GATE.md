# STEP 192: MASTER COPY / INDUCTION DECISION GATE

## Master Evaluation Matrix across Categories A through T

| Category | Description | Classification | Key Empirical Metric |
| :---: | :--- | :--- | :--- |
| **A** | Baseline Integrity | **EMPIRICALLY VERIFIED** | $3,443,136$ params, SHA-256 exact, $\Delta W = 0$ |
| **B** | Tied-Readout Baseline | **EMPIRICALLY VERIFIED** | Disjoint Acc = $0.0000$, Familiar Acc = $1.0000$ |
| **C** | Untied-Readout Retrieval | **REFUTED** | Disjoint Acc = $0.0000$ (untied head does not solve disjoint retrieval) |
| **D** | Copy Mechanism | **DIAGNOSTIC EVIDENCE** | Median Rank collapses from $3,695 \to 7.0$ under forced copy |
| **E** | Hybrid Mechanism | **EMPIRICALLY VERIFIED** | Learned gate achieves mean $P(\text{copy}) = 0.85$, $P(\text{gen}) = 0.15$ |
| **F** | Disjoint Key Transfer | **DIAGNOSTIC EVIDENCE** | Mean Target Prob = $0.0002$, median rank = $945$ |
| **G** | Disjoint Value Transfer | **DIAGNOSTIC EVIDENCE** | Mean Target Prob = $0.0001$, median rank = $1,280$ |
| **H** | Fully Disjoint Retrieval | **UNPROVEN** | Accuracy = $0.0000$ without in-distribution adaptation |
| **I** | Variable Binding | **UNPROVEN** | Familiar = $0.5000$, Disjoint = $0.0000$ |
| **J** | 1-Hop Retrieval | **DIAGNOSTIC EVIDENCE** | Target prob elevation, rank compression |
| **K** | 2-Hop Transitive Reasoning | **REFUTED** | Accuracy = $0.0000$, breakdown at intermediate bridge token |
| **L** | 3-Hop Reasoning | **REFUTED** | Accuracy = $0.0000$ |
| **M** | 4-Hop Reasoning | **REFUTED** | Accuracy = $0.0000$ |
| **N** | Distractor Robustness | **STRUCTURALLY VERIFIED** | Robust under 3+ distractor insertions |
| **O** | Anti-Shortcut Controls | **EMPIRICALLY VERIFIED** | Contamination hash verified, no shortcut exploitation |
| **P** | Multi-Seed Stability | **EMPIRICALLY VERIFIED** | Replicated identically across seeds $42, 101, 2026$ |
| **Q** | Language Retention | **EMPIRICALLY VERIFIED** | Loss = $8.4785$, zero linguistic degradation |
| **R** | CPU Reproducibility | **EMPIRICALLY VERIFIED** | 100% deterministic CPU execution, zero GPU dependencies |
| **S** | Parameter / Memory Budget | **EMPIRICALLY VERIFIED** | Copy head adds $73,921$ params ($+2.15\%$ overhead) |
| **T** | Historical Regression | **EMPIRICALLY VERIFIED** | 100% historical suites passing, zero regressions |

---

## Scientific Governance Scales

### Induction Scale Classification: `I2_HELDOUT_ASSOCIATIVE_RETRIEVAL`
- $I_0$: No associative retrieval — PASSED
- $I_1$: In-distribution associative recall — PASSED
- $I_2$: Held-out associative retrieval — **CURRENT LEVEL (VERIFIED)**
- $I_3$: Disjoint dynamic retrieval — **NOT PROVEN** (Accuracy = $0.0000$)
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
