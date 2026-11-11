# Claim-Evidence Matrix — CR-EXP-06 Scientific Audit

**Audit Date**: 2026-11-06  
**Auditor**: Antigravity Autonomous Scientific Auditor  
**Scope**: Independent audit of all empirical assertions, hypotheses, and conclusions in CR-EXP-06.

---

## 1. Matrix of Core Scientific Claims

| # | Scientific Claim in CR-EXP-06 Report | Audit Status | Evidence & Quantitative Verification |
|---|---|---|---|
| **C1** | **Control Arm achieved 10% and Treatment Arm achieved 12% in-distribution retrieval.** | **VERIFIED (Reproduced)** | Recalculation from raw generations in `per_example_evaluation.json` yields exactly 10/100 (10.0%) and 12/100 (12.0%) string-matching target hits. |
| **C2** | **CR-EXP-06 demonstrates genuine neural variable binding.** | **REFUTED** | Qualitative audit in Audit E revealed that out of 10 hits in Control and 12 hits in Treatment, **EXACTLY 2 hits (2.0%)** correctly bound the prompt's subject to the prompt's city. The remaining 8–10 hits were **subject-substitution hallucinations** (e.g. asked about Deepa, the model generated `"Kabir lives in Chennai"`). |
| **C3** | **Familiar retrieval gains reflect memorization / static association rather than dynamic variable binding.** | **SUPPORTED** | Across 100 disjoint OOD entities and 100 disjoint templates, accuracy was **0.0%**. The model merely memorized familiar co-occurrence pairs (`"Kabir"`, `"Priya"`, `"Chennai"`) from the training set and emitted them upon seeing the `"lives in"` syntax. |
| **C4** | **Treatment Arm B (5.0x weighting) was statistically superior to Control Arm A in retrieval.** | **UNFOUNDED** | McNemar's paired test yields $\chi^2 = 0.125, p = 0.7237$. The 95% Wilson confidence intervals heavily overlap ([5.52%, 17.44%] vs [7.00%, 19.81%]). The 2-point difference is statistically indistinguishable from chance. |
| **C5** | **Treatment Arm B caused conversational degradation compared to Control Arm A.** | **WEAK EVIDENCE** | The difference in conversational preservation (58.33% vs 41.67%) corresponds to an absolute difference of **only 2 probes** (7/12 vs 5/12) on an $n=12$ sample. The 95% confidence intervals overlap by 36% ([31.95%, 80.67%] vs [19.33%, 68.05%]). |
| **C6** | **Standard SFT on dense 3.44M Transformer is insufficient for 2-hop multi-hop reasoning.** | **SUPPORTED** | All 4 models scored **0.0%** across all 60 strict 2-hop test cases. Autoregressive attention without architectural pointer routing cannot compose sequential hops. |
| **C7** | **Persona retention is preserved across both experimental arms.** | **VERIFIED** | Both Control and Treatment arms reliably generated the canonical identity string (*"I am ChakrView, an indigenous neural language model developed from first principles."*) without repetition collapse. |
| **C8** | **Both arms were trained under identical conditions with the sole intended difference being the loss multiplier.** | **VERIFIED** | Audit B verified identical parent checkpoint initialization, training samples, batch sizes, random seeds, cosine LR schedules, and step counts (150 steps each). |
| **C9** | **Dataset was 0% contaminated across OOD splits.** | **VERIFIED** | Audit D verified 0 prompt overlap and 0 entity overlap between training and `test_ood_entities` / `test_ood_templates`. |

---

## 2. Classification Summary

- **Directly Verified Observations**: Checkpoint hashes, training equivalence, loss weighting mathematical parity, zero OOD leakage, 10% and 12% raw string match rates, persona preservation.
- **Supported Interpretations**: Dense 3.44M Transformer fails zero-shot OOD variable binding and multi-hop reasoning; relational gains on familiar splits represent surface pattern memorization.
- **Refuted / Corrected Claims**: Claim that CR-EXP-06 demonstrated variable binding is refuted; 80% of "hits" were subject-substitution hallucinations. Claim that Treatment Arm B was superior is statistically unfounded ($p = 0.7237$).

