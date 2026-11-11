# ChakrView Capability Registry — Post-CR-EXP-06 Audit

**Audit Date**: 2026-11-06  
**Audited Models**:
- Parent: `sft_checkpoint_step_200.pt` (`327118ce...`)
- Current Baseline Candidate: `conversational_pilot_best_val.pt` (`ea6ee570...`)
- Arm A (Control): `control_arm_best_val.pt`
- Arm B (Treatment): `treatment_arm_best_val.pt`
- Canonical Baseline: `ChakrMicro` (`c5571c9c...`)

---

## 1. Authoritative Capability Matrix

| Capability ID | Domain / Function | Status | Evidence & Frozen Evaluation Results |
|---|---|---|---|
| **CAP-01** | **Deterministic CPU-First Runtime** | **VALIDATED** | Loads and runs in strictly deterministic mode on CPU across all 4 models ($\Delta W = 0$ on canonical baseline). |
| **CAP-02** | **Persona Adherence & Identity** | **VALIDATED** | Prompt `"Who are you?"` reliably yields `"I am ChakrView, an indigenous neural language model developed from first principles."` across all candidates. |
| **CAP-03** | **Turn Boundary Termination** | **VALIDATED** | Models reliably terminate at `User:` or `Assistant:` stop tokens; 0 infinite repetition loops detected across 420+ frozen generations per model. |
| **CAP-04** | **In-Distribution Relational Retrieval** | **PARTIALLY VALIDATED** | Models improved from 0.0% (Parent & Pilot) to 10.0% (Arm A) and 12.0% (Arm B) on in-distribution single-hop fact extraction. However, errors reflect frequent entity substitutions (e.g. outputting `"Chennai"` while substituting `"Priya"` for `"Deepa"`). |
| **CAP-05** | **Strict Out-of-Distribution Entity Retrieval** | **NOT DEMONSTRATED** | 0.0% across all 4 models on disjoint entity pools (100 frozen test cases). |
| **CAP-06** | **Strict Out-of-Distribution Template Transfer** | **NOT DEMONSTRATED** | 0.0% across all 4 models on disjoint relational phrasing (100 frozen test cases). |
| **CAP-07** | **Multi-Hop Relational Reasoning** | **NOT DEMONSTRATED** | 0.0% across all 4 models on strict 2-hop compositional queries (60 frozen test cases). |
| **CAP-08** | **Multi-Turn Variable Binding (>2 turns)** | **NOT DEMONSTRATED** | 0.0% accuracy across 3-turn, 5-turn, and 8-turn dialogue histories. |
| **CAP-09** | **Epistemic Abstention** | **PARTIALLY VALIDATED** | Canned abstention strings emitted on familiar test sets, but uncalibrated on novel unanswerable queries. |
| **CAP-10** | **Arithmetic & Multi-Step Logic** | **NOT DEMONSTRATED** | 0.0% accuracy; parameter capacity and architecture lack arithmetic reasoning circuits. |
| **CAP-11** | **Pure Hindi Generation** | **NOT DEMONSTRATED** | 0.0% accuracy; BPE vocabulary lacks merged Devanagari tokens, causing byte fragmentation. |
| **CAP-12** | **Persistent Memory** | **NOT DEMONSTRATED** | Architecture operates strictly within current context window; no persistent state across sessions. |

---

## 2. Invariant Protection Rules
1. **Never promote an experimental candidate to canonical baseline**: Canonical baseline `c5571c...` remains immutable.
2. **Never claim general intelligence or memory from in-context pattern matching**: Even within a 35-token window, true relational binding requires explicit architectural routing mechanisms beyond standard autoregressive attention.

