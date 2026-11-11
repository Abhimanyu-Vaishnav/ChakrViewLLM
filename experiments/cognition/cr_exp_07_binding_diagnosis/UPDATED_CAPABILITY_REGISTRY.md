# ChakrView Capability Registry — Post-CR-EXP-07 Diagnostic Audit

**Evaluation Date**: 2026-11-06  
**Audited Checkpoints**:
- Frozen Conversational Parent: `sft_checkpoint_step_200.pt` (`327118ce...`)
- Conversational Pilot: `conversational_pilot_best_val.pt` (`ea6ee570...`)
- CR-EXP-06 Control Arm Candidate: `control_arm_best_val.pt` (`729d5812...`)
- CR-EXP-06 Treatment Arm Candidate: `treatment_arm_best_val.pt` (`2d8cea93...`)
- Canonical Baseline: `ChakrMicro` (`c5571c9c...`, $\Delta W = 0$)

---

## 1. Authoritative Capability Matrix

| Capability ID | Domain / Function | Status | Evidence & Diagnostic Verification |
|---|---|---|---|
| **CAP-01** | **Deterministic CPU-First Runtime** | **VALIDATED** | Loads and executes with bit-exact determinism on CPU across all 4 checkpoints without external CUDA/GPU dependencies. |
| **CAP-02** | **Persona Identity & Self-Introduction** | **VALIDATED** | Prompt `"Who are you?"` reliably generates `"I am ChakrView, an indigenous neural language model developed from first principles."` across all candidates. |
| **CAP-03** | **Turn Boundary Discipline** | **VALIDATED** | Clean generation termination at `User:` / `Assistant:` stop tokens; 0 infinite repetition loops across 870+ test generations. |
| **CAP-04** | **Intermediate Subject & Value Representation** | **VALIDATED** | Internal hidden states measurably differentiate context subjects (15.4% divergence) and values (7.5% divergence) across Layers 0 to 5. |
| **CAP-05** | **Algebraic Subject-Value Association** | **PARTIALLY VALIDATED** | Difference-in-Differences interaction tensor is non-zero (2.29% interaction ratio), confirming internal geometric coupling of subject and value. |
| **CAP-06** | **Single-Hop Contextual Variable Binding** | **NOT DEMONSTRATED** | 0.0% exact match across 450 independent test cases; 100% of apparent relational hits in CR-EXP-06 were subject-substitution hallucinations. |
| **CAP-07** | **Counterfactual Reassignment** | **NOT DEMONSTRATED** | 0.0% accuracy; updating a subject's city across 5 distinct values produced identical static conversational text. |
| **CAP-08** | **Distractor Resistance** | **NOT DEMONSTRATED** | 0.0% accuracy when irrelevant background facts are inserted into the prompt. |
| **CAP-09** | **Missing-Information Epistemic Abstention** | **NOT DEMONSTRATED** | When queried about unmentioned subjects, model hallucinates training entities or emits canned filler rather than principled abstention. |
| **CAP-10** | **Strict Out-of-Distribution Entity Binding** | **NOT DEMONSTRATED** | 0.0% accuracy on disjoint entity pools (names/cities never present in training data). |
| **CAP-11** | **Out-of-Distribution Template Generalization** | **NOT DEMONSTRATED** | 0.0% accuracy on unseen syntactic phrasing (e.g. `"Record: Resident X resides at Y."`). |
| **CAP-12** | **Multi-Hop Relational Reasoning** | **NOT DEMONSTRATED** | 0.0% accuracy across two-hop compositional queries. |

---

## 2. Invariant Protection Rules
- Canonical baseline `c5571c...` remains completely untouched ($\Delta W = 0$).
- No experimental checkpoints promoted to production runtime.
- Never equate internal representation divergence with behavioral capability.

