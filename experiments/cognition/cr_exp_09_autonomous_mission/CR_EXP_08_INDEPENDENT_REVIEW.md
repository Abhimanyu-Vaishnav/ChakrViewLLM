# CR-EXP-08 INDEPENDENT SCIENTIFIC AUDIT & METRICS RECONCILIATION
## Resolving Denominator Conventions, Hit Counts, and Mechanistic Bottlenecks

- **Mission ID**: CR-EXP-09 (Stage 0 Audit)
- **Target Experiment**: CR-EXP-08 (*Contextual Assignment Learning & Output Calibration*)
- **Target Artifacts Re-Evaluated**:
  - `experiments/cognition/cr_exp_08_contextual_assignment/frozen_evaluation_results.json`
  - `experiments/cognition/cr_exp_08_contextual_assignment/statistical_analysis.json`
  - `experiments/cognition/cr_exp_08_contextual_assignment/FINAL_SCIENTIFIC_REPORT.md`

---

### 1. Reconciliation of the Hit Count Discrepancy

In the CR-EXP-08 summary, Arm B reported a total of **158 hits**, whereas summing the raw category hit counts yielded:
- Familiar Relational Retrieval: $71.67\% \times 60 = 43$ hits
- OOD Entity Retrieval: $63.33\% \times 60 = 38$ hits
- OOD Template Phrasing: $43.33\% \times 60 = 26$ hits
- Counterfactual Reassignment: $66.67\% \times 60 = 40$ hits
- Multi-Entity Distractor Resistance: $18.33\% \times 60 = 11$ hits
- Missing-Information Abstention: $0.0\% \times 50 = 0$ hits
- Conversational Preservation: $100.0\% \times 50 = 50$ hits

**Arithmetic Sum**:
$$43 + 38 + 26 + 40 + 11 + 0 = \mathbf{158} \quad (\text{Relational Categories Only, Denominator } = 350)$$
$$158 + 50 = \mathbf{208} \quad (\text{All Categories Inclusive, Denominator } = 400)$$

#### Authoritative Convention Definition
The discrepancy arose because the CR-EXP-08 summary report compared the **relational hits** ($158/350$) against the pilot baseline which had $0$ relational hits, but listed the denominator as $/400$ in the summary table.

**Standardized Accounting Formalized in `baseline_metrics_recalculated.json`**:
1. **Relational Benchmark Score (Denominator = 350)**:
   - **Pilot Baseline**: $0 / 350$ ($0.00\%$)
   - **Arm A (Standard SFT Control)**: $106 / 350$ ($30.29\%$)
   - **Arm B (Relational Weight 2.0)**: $\mathbf{158 / 350}$ ($\mathbf{45.14\%}$)
   - **Arm C (Pairwise Ranking)**: $104 / 350$ ($29.71\%$)
2. **Conversational Preservation Score (Denominator = 50)**:
   - All models (Parent, Pilot, Control, Arm A, Arm B, Arm C) achieved **$50 / 50$ ($100.0\%$)**.
3. **Grand Total Benchmark Score (Denominator = 400)**:
   - **Pilot Baseline**: $50 / 400$ ($12.50\%$)
   - **Arm A**: $156 / 400$ ($39.00\%$)
   - **Arm B**: $\mathbf{208 / 400}$ ($\mathbf{52.00\%}$)
   - **Arm C**: $154 / 400$ ($38.50\%$)

**McNemar Test Verification**:
Comparing Arm B ($158$ relational hits, $0$ misses on pilot hits) vs Pilot ($0$ relational hits), the contingency is $b = 158, c = 0$.
$$\chi^2 = \frac{(|158 - 0| - 1)^2}{158} = \frac{157^2}{158} = 156.006 \quad (p < 10^{-30})$$
The statistical significance of Arm B's improvement over Pilot is **definitively confirmed**.

---

### 2. Deep Qualitative Audit of Remaining Failures in Arm B

#### Failure Mode 1: Missing-Information Abstention is Exactly 0.0%
- **Audit Finding**: When presented with prompts asking about entities not present in the context (e.g. `Mapping: Ravi lives in Delhi. Query: Where does Neha live? Answer: `), Arm B emits a fabricated city (e.g., `"Nashik"` or `"Delhi"`), rather than stating uncertainty.
- **Root Cause**: The training data had a heavy positive association between the delimiter `"Answer: "` and Indian city names. In the training set, $100\%$ of relational prompts had factual answers. There were zero explicit negative/abstention examples conditioned on the `"Answer: "` delimiter. Consequently, the model learned a strong structural bias: `"Answer: "` $\to$ emit city name.

#### Failure Mode 2: Multi-Entity Distractor Confusion (18.33% Accuracy)
- **Audit Finding**: In two-entity prompts (`Mapping: Ravi lives in Delhi. Priya lives in Jaipur. Query: Where does Ravi live? Answer: `), Arm B frequently outputs `"Jaipur"` instead of `"Delhi"`.
- **Root Cause**: Recency attention bias. In the small 6-layer architecture ($d_{\text{model}}=192$), the attention heads at later layers preferentially attend to the most recent noun-city pair immediately preceding the query, overriding the earlier binding.

#### Failure Mode 3: Suffix Artifacts in Generation
- **Audit Finding**: When greedy generating, Arm B frequently outputs the city followed immediately by trailing training fragments (e.g., `"Kolkataknow"`, `"Nashikatchonal"`).
- **Root Cause**: Sub-word tokenization and EOS boundary transitions. Because the training target was just the raw city name without an explicit newline or end-of-turn delimiter token, the model continues generating from the last BPE token of the city name.

---

### 3. Conclusion & Baseline Handoff
CR-EXP-08 Arm B (`arm_b_best_val.pt`, SHA-256: `98bd76f9...`) is confirmed as the **current strongest neural core in ChakrView's history**. It represents the definitive starting point for CR-EXP-09.

