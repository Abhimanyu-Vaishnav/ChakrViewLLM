# CR-EXP-09: BASELINE BENCHMARK SCORECARD
## Quantitative Evaluation of Current Best Checkpoint (`arm_b_best_val_copy.pt`) across 600 Probes

- **Evaluated Checkpoint**: `experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/baseline/cr_exp_08_arm_b_best_val_copy.pt`
- **SHA-256 Digest**: `98bd76f99d43dedd10c31a1d290c95e452eba299f511ecf72e5ecc4075fb0956`
- **Benchmark Suite**: 600 Frozen Multi-Tier Probes across 8 Categories

---

### Category Performance Baseline

| Category ID | Cognitive Dimension | Probe Count | Exact Hits | Baseline Accuracy | Current State Assessment |
|---|---|---|---|---|---|
| **CAT-1** | **Conversational Understanding & Persona** | 75 | 62 | **82.67%** | **Strong**: Retains persona and coherent English/Hinglish greetings. |
| **CAT-2** | **Single-Hop & Counterfactual Binding** | 75 | 39 | **52.00%** | **Moderate**: Follows factual updates on single entities well. |
| **CAT-3** | **Multi-Entity Distractor Disambiguation** | 75 | 21 | **28.00%** | **Weak**: Severe recency bias in 2-entity contexts. |
| **CAT-4** | **Missing-Information Abstention** | 75 | 0 | **0.00%** | **Complete Failure**: Compulsive fabrication; zero epistemic abstention. |
| **CAT-5** | **Multi-Turn State Tracking & Memory** | 75 | 3 | **4.00%** | **Critical Deficit**: Fails to track entity moves across conversational dialogue turns. |
| **CAT-6** | **Structured State Transformation & Arithmetic** | 75 | 0 | **0.00%** | **Unlearned**: Zero arithmetic or inventory transfer capability. |
| **CAT-7** | **Conditional & Syllogistic Logical Reasoning** | 75 | 0 | **0.00%** | **Unlearned**: Zero deductive boolean reasoning. |
| **CAT-8** | **Constraint Satisfaction & Stability** | 75 | 32 | **42.67%** | **Moderate**: Emits single words inconsistently; appends trailing BPE tokens. |
| **TOTAL** | **Full Cognitive Capability Scorecard** | **600** | **157** | **26.17%** | **Authoritative Mission Baseline** |

---

### Failure Classification & Engineering Priority Order

1. **Top Priority 1 (Epistemic Abstention & Delimiter Calibration)**: 0.0% accuracy on CAT-4. Teach the model to output `"The context does not state..."` when unmentioned entities are queried.
2. **Top Priority 2 (Multi-Turn State Tracking & Dialogue Memory)**: 4.0% accuracy on CAT-5. Teach the model to update internal entity state across multi-turn conversational dialogue.
3. **Top Priority 3 (Distractor Resistance & Multi-Entity Grounding)**: 28.0% accuracy on CAT-3. Provide multi-entity contexts with counterbalanced positional ordering to extinguish recency bias.
4. **Top Priority 4 (Structured State & Elementary Reasoning)**: 0.0% accuracy on CAT-6 & CAT-7. Introduce elementary item arithmetic and simple boolean deduction.

