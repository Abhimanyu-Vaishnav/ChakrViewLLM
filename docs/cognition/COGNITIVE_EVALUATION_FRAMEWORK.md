# COGNITIVE EVALUATION FRAMEWORK (CR-001)

An objective, multi-metric cognitive benchmark specification covering 12 evaluation domains. Designed to measure genuine reasoning versus superficial token memorization.

---

## 1. The 12 Evaluation Domains & Pass Gates

| Domain ID | Category | Number of Probes | Evaluation Methodology | Memorization Control | Pass Gate Criteria |
|---|---|---|---|---|---|
| **CE-01** | Deductive Reasoning | 50 probes | Deterministic formal logic parsing | Synthetic variable symbols ($A, B, C$) | $\ge 75\%$ Validity |
| **CE-02** | Inductive Reasoning | 50 probes | Next-number & rule continuation | Random sequence generators | $\ge 70\%$ Accuracy |
| **CE-03** | Causal Reasoning | 40 probes | Cause-effect pair ranking | Counter-factual scenario probes | $\ge 70\%$ Consistency |
| **CE-04** | Mathematical Reasoning | 100 probes | Normalized numerical regex extraction | Large random integer operands | $\ge 60\%$ Exact Result |
| **CE-05** | Analogical Reasoning | 50 probes | Relational tuple completion ($A:B :: C:?$) | Cross-domain structural transfer | $\ge 65\%$ Correct Rel |
| **CE-06** | Multi-Step Reasoning | 60 probes | Step-by-step intermediate verification | 3 to 5 hop novel problem graphs | $\ge 50\%$ End-to-End |
| **CE-07** | Working Memory | 50 probes | Key-value variable recall | Arbitrary alphanumeric nonce values | $\ge 80\%$ Retrieval |
| **CE-08** | Planning & Constraints| 40 probes | Action dependency graph validator | Blocked-state & order constraints | $\ge 65\%$ Valid Plan |
| **CE-09** | Contradiction Detection| 40 probes | Premise consistency classification | Injected semantic contradictions | $\ge 75\%$ Detection |
| **CE-10** | Uncertainty Calibration| 50 probes | Binary abstention gate on unknowables | Fictional/future unrecorded events | $\ge 85\%$ Calibration |
| **CE-11** | Self-Correction | 40 probes | Two-pass critique-and-repair probe | Seeded step-1 arithmetic error | $\ge 55\%$ Repair Rate |
| **CE-12** | Multilingual Reasoning| 60 probes | Bilingual logic (Hindi & Hinglish) | Nonce words in Devanagari/Hinglish | $\ge 60\%$ Multilingual |

---

## 2. Evaluation Mechanics: The 4-Way Verification Metric

To prevent grading generated text merely on the final token, every reasoning evaluation generates four diagnostic signals:

1. **Final Answer Correctness ($A_{final}$)**:
   Does the final extracted value match the ground truth target?
2. **Intermediate Step Validity ($V_{steps}$)**:
   Does each step $k$ logically and mathematically follow from step $k-1$ and the input premises?
3. **Step-to-Answer Consistency ($C_{step\text{-}ans}$)**:
   Does the final answer actually match the output of the final intermediate step, or did the model execute valid reasoning and then emit an ungrounded conclusion?
4. **Premise Contradiction Flag ($P_{contra}$)**:
   If an impossible or contradictory premise was provided, did the model flag the contradiction rather than proceeding to compute an invalid result?

$$\text{Cognitive Score} = \frac{1}{N} \sum_{i=1}^{N} \Big( A_{final}^{(i)} \land V_{steps}^{(i)} \land C_{step\text{-}ans}^{(i)} \Big)$$

This formulation ensures that a model which hallucinates a correct final answer by sheer guessing receives a score of **0** if its intermediate reasoning chain is invalid.

