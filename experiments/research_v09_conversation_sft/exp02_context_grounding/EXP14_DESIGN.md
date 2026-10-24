# ChakrView Experiment 14: Context Grounding and Evaluation Repair
**Directory**: `experiments/research_v09_conversation_sft/exp02_context_grounding/`  
**Date**: October 22, 2026  
**Auditor / Researcher**: Antigravity AI Engine

---

## 1. Executive Mission & Context

ChakrView Experiment 13 demonstrated stable 120-step CPU optimization on multi-turn dialogue, but failed to demonstrate generalized conversational reasoning. More critically, the forensic audit of Exp13 exposed that its reported "18.33% response relevance" metric was built on a flawed single-token-overlap heuristic, incorrectly classifying gibberish outputs containing non-stopwords like `"of"` or `"be"` as successful responses.

**Experiment 14 Objectives**:
1. **Evaluation Repair**: Completely eliminate unigram token-overlap heuristics and replace them with 10 deterministic, task-specific, mathematically rigorous verification criteria.
2. **Context-Grounding Benchmark**: Design a versioned benchmark testing dynamic entity binding, multi-turn anaphora, user corrections, updated instructions, distractor resistance, ambiguity clarification, and explicit missing-value abstention across strictly disjoint splits.
3. **Field-Conditioned Pointer Research Design**: Author an exhaustive architectural specification for a future isolated pointer experiment (Exp15) that resolves slot entanglement and surface memorization without modifying the frozen 3.44M causal LM backbone.
4. **Preservation Invariant**: Absolute zero-training, zero-Git-mutation, zero-production-modification protocol during this evaluation foundation phase.

---

## 2. Invariants & Isolation Protocol

- **Canonical Checkpoint Step-10000**: SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Unchanged).
- **Canonical Vocab & Merges**: `db4d6119...` & `ada4b1dc...` (Unchanged).
- **Protected Runtime**: `chakrview/runtime/session.py` (`9080ff51...`, Unchanged).
- **Historical Experiments**: `exp01_native_dialogue/` files and checkpoints preserved byte-identically.
- **Pure CPU & Local Execution**: No external models, hosted LLM APIs, or hidden evaluators.

---

## 3. Evaluation Methodology Repair Plan

| Evaluation Domain | Flawed Exp13 Heuristic | Repaired Exp14 Metric & Scoring Contract |
| :--- | :--- | :--- |
| **Entity Binding** | Single word in response matches target | **Exact Match of Dynamic Target Entity** (Case-sensitive / normalized substring match of prompt entity). |
| **Follow-up & Reference** | Unigram overlap | **Resolution Precision**: Output must accurately bind the specific entity referred to by "that one" or "earlier unit". |
| **User Correction** | Unigram overlap | **Correction Adherence**: Output must bind the *new* corrected entity and explicitly drop/omit the superseded entity. |
| **Distractor Resistance** | Unigram overlap | **Target Selection vs. Distractor Rejection**: Must contain target value and 0 distractor values. |
| **Missing Info / Uncertainty** | Output contains "weather" | **Explicit Abstention Accuracy**: Must contain explicit refusal tokens (`"does not state"`, `"cannot be predicted"`, `"none"`). |
| **Coding Tasks** | Contained `def` or `return` | **AST Parse & Name Match**: Python `ast.parse()` valid AND exact function name matches prompt request. |
| **Repetition / Malformation** | Trigram repetition | **Repetition Rejection**: Any repeated n-gram loop ($N \ge 3$) automatically scores 0 across all metrics. |

---

## 4. Benchmark Categories (10 Core Skills)

The benchmark assesses 10 core context-grounding tasks across 60 held-out evaluation scenarios ($N=60$):
1. `dynamic_entity_binding`: Binding unseen entity tokens from context.
2. `turn_dependent_retrieval`: Retrieving numerical/state facts given in turn 1.
3. `anaphora_reference_resolution`: Disambiguating "that one" between multiple units.
4. `superseded_instruction_adherence`: Following turn 3 command while ignoring superseded turn 1 command.
5. `user_correction_handling`: Updating target immediately upon user correction.
6. `distractor_resistance`: Identifying target amid 2 adjacent distractors.
7. `ambiguity_clarification`: Asking clarifying questions when key arguments are missing.
8. `missing_context_abstention`: Acknowledging that context lacks requested information.
9. `factual_uncertainty_handling`: Explicitly acknowledging epistemic uncertainty.
10. `constrained_code_synthesis`: Generating syntactically valid Python with exact requested name.
