# CR-16E-R2 Phase H: Updated Capability Registry Scorecard

### Evaluation Standards & Capability Status Levels
- **NOT DEMONSTRATED:** Score is $0.0\%$ or at chance level; no reproducible mechanism identified.
- **UNDER EXPERIMENT:** Partial signal or isolated seed success, but lack of multi-seed reproducibility, strong confounders, or severe variance prevent formal validation.
- **PARTIALLY VALIDATED:** Statistically robust capability verified across multiple seeds in a specific distribution, but limited by vocabulary, prompt formatting, or sequence length.
- **VALIDATED:** Reproducible capability confirmed across multiple independent training runs meeting pre-specified research criteria.

---

### Grand Capability Registry Matrix (14 Evaluated Dimensions)

| Capability Dimension | Benchmark Suite Reference | CR-16E-R Status | Audited & Replicated CR-16E-R2 Status | Key Quantitative Evidence | Status Assessment |
|---|---|---|---|---|---|
| **1. Short Sequence Copying** | CR-16 Copy (1–3 tokens) | Degraded | **NOT DEMONSTRATED** | Parent achieved $66.0\%$; CR-16E models scored $0.3\% - 8.3\%$; G2 masked scored $1.0\%$. | **Severely degraded.** Task interference from retrieval fine-tuning erased copying. |
| **2. Long Sequence Copying** | CR-16 Copy (4–8 tokens) | Degraded | **NOT DEMONSTRATED** | Strict exact match on long copy sequences collapsed to $0.0\%$ across all intervention arms. | Not retained during multi-task pretraining. |
| **3. Multi-Token Span Generation** | Benchmark B3 (Len 3 & 4) | VALIDATED | **PARTIALLY VALIDATED** | $37.3\% - 40.2\%$ mean ($57.0\%$ peak) on familiar entities. Conditional completion is $>90\%$. However, unseen entities score strictly $0.0\%$. | **Qualified.** Robust on familiar entities, but vocabulary-bound. |
| **4. Direct Key-Value Retrieval** | CR-16C CAT1 | PARTIALLY VALIDATED | **PARTIALLY VALIDATED** | $25\% - 68\%$ across seeds when queried key matches Position 0 training template. | Stable when syntax and position match training. |
| **5. Non-Zero Position Retrieval** | Benchmark B1 (Slots 1, 2, 3) | PARTIALLY VALIDATED | **PARTIALLY VALIDATED** | Shuffled models achieve $24.0\% - 28.0\%$ across slots 1..3 (eliminating 0% fixed-position collapse). | Proven elimination of zero-collapse, but diffuse. |
| **6. Position-Invariant Retrieval** | Benchmark B1 ($\ge 70\%$) | NOT DEMONSTRATED | **NOT DEMONSTRATED** | Mean across shuffled models is $24.5\% - 27.3\%$; peak seed is $28.8\%$. Far below $70\%$ gate. | Gate 1 failed. Attention is uniformly spread, not invariant. |
| **7. Counterfactual Reassignment** | Benchmark B2 Counterfactual | Ambiguous | **UNDER EXPERIMENT** | Apparent $24.5\%$ in fixed models was a benchmark artifact. Shuffled models achieve $4.5\% - 12.7\%$. | Emergent in Arm D s4040, but low mean across seeds. |
| **8. Missing-Key Abstention** | Benchmark B4 ($N=300$) | VALIDATED | **PARTIALLY VALIDATED** | Mean recall $75.7\% - 86.7\%$ in enabled models, but recall drops to $0.0\%$ when loss is masked. | Verified trainable, but susceptible to severe over-abstention. |
| **9. Calibrated Abstention** | Risk-Coverage / False Abstain | PARTIALLY VALIDATED | **UNDER EXPERIMENT** | High-recall models falsely reject $\approx 50\% - 60\%$ of answerable queries. Only Arm D s4040 achieved balance ($4.5\%$ FA). | **Single-seed operating point.** Not yet multi-seed reproducible. |
| **10. Answerable Question Coverage**| CR-16C Answerable ($N=1400$) | Not separated | **PARTIALLY VALIDATED** | Coverage was suppressed to $38.6\% - 57.5\%$ in high-abstention seeds, but recovered to $88.8\%$ in Arm D s4040 and $100\%$ in G2. | Coverage is severely traded off against abstention recall. |
| **11. Unseen-Value Generalization** | CR-16C CAT7 ($N=150$) | Failed | **NOT DEMONSTRATED** | Strictly **0 / 150 (0.0%)** across all models. Models only retrieve values seen in pretraining. | Transformer acts as a associative memory, not an abstract binding engine. |
| **12. Unseen-Template Generalization**| CR-16C CAT10 ($N=100$) | Low | **NOT DEMONSTRATED** | Accuracy strictly $< 5\%$ when syntax deviates from trained delimiters. | Delimiter-bound. |
| **13. Multi-Turn State Tracking** | CR-16C CAT2 Reassignment | PARTIALLY VALIDATED | **PARTIALLY VALIDATED** | Functional when formatted as single-prompt sequential updates; unverified in conversational multi-turn session. | Syntax-dependent sequential update verified. |
| **14. Compositional Generalization** | CR-16C CAT5 ($N=150$) | Failed | **NOT DEMONSTRATED** | Strictly $< 4\%$ on novel key-value bindings. | Unresolved within current objective. |

---

### Summary of Status Distribution
- **VALIDATED (0):** No capability currently satisfies full multi-seed, out-of-distribution, generalizable validation without trade-offs.
- **PARTIALLY VALIDATED (6):** Multi-Token Span Generation (familiar), Direct Key-Value Retrieval, Non-Zero Position Retrieval, Missing-Key Abstention, Answerable Question Coverage, Multi-Turn State Tracking.
- **UNDER EXPERIMENT (2):** Counterfactual Reassignment, Calibrated Abstention.
- **NOT DEMONSTRATED (6):** Short Sequence Copying, Long Sequence Copying, Position-Invariant Retrieval ($\ge 70\%$), Unseen-Value Generalization, Unseen-Template Generalization, Compositional Generalization.

