# CR-16D G1 Paired Counterfactual Audit

### 1. Motivation & Methodology
In CR-16D, high single-probe retrieval accuracy on Group G1 was interpreted as evidence of in-context variable binding. However, evaluating probes independently cannot distinguish between genuine counterfactual sensitivity and prompt/value priors.

We designed a diagnostic suite of **200 matched counterfactual pairs** (400 probes total):
- **Probe A:** `Store: KEY = VAL_X. Query: KEY.\nAnswer: ` $\to$ Expected: `VAL_X`
- **Probe B:** `Store: KEY = VAL_Y. Query: KEY.\nAnswer: ` $\to$ Expected: `VAL_Y`

The model must dynamically invert its prediction solely based on the updated in-context value.

---

### 2. Quantitative Results across Key Checkpoints

| Model Condition | Both Correct (Success) | A Only Correct | B Only Correct | Both Incorrect | Invariant Failure ($A=B$) | True Dynamic Sensitivity |
|---|---|---|---|---|---|---|
| **Arm C (50% Replay, s42)** | 24 / 200 (12.0%) | 49 / 200 | 46 / 200 | 81 / 200 | 44 / 200 (22.0%) | **WEAK** |
| **Arm C (50% Replay, s101)**| **76 / 200 (38.0%)** | 44 / 200 | 48 / 200 | 32 / 200 | **2 / 200 (1.0%)** | **MODERATE** |
| **Arm B (25% Replay, s101)**| 7 / 200 (3.5%) | 28 / 200 | 42 / 200 | 123 / 200 | 95 / 200 (47.5%) | **VERY WEAK** |
| **Arm D (75% Replay, s42)** | 54 / 200 (27.0%) | 48 / 200 | 48 / 200 | 50 / 200 | 13 / 200 (6.5%) | **FAIR** |

---

### 3. Key Findings
1. **Single-Probe Scores Inflate Apparent Capability:**
   - Although Arm C Seed 101 achieved 96.4% on standard G1 probes, its paired counterfactual consistency is **38.0% (76/200)**.
   - For an additional 46% of pairs, the model correctly answers one variant but fails the other, indicating that lexical affinity between key and value still influences the routing decision.
2. **Suppression of Invariant Outputs:**
   - In Arm C Seed 101, invariant failure (outputting the exact same word regardless of the context value) dropped to **1.0% (2/200)**. The model is genuinely sensitive to input perturbations, but does not always predict the exact intended target string.
3. **Conclusion for Variable Binding:**
   - While the model exhibits substantial in-context routing behavior, it does not possess an ideal, deterministic algebraic variable-binding operator. Counterfactual fidelity requires paired supervision during pretraining.

