# WP 941–946: Pipeline Diagnostic Audit

## 1. Verified Starting Point & Lineage
- **Branch**: `dev/v0.6-v0.7-validation`
- **Base Commit**: `ce0ebc21d8cdc5e429ba2ab1f01296b1b74eca0d`
- **Canonical Checkpoint Hash**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified intact)
- **User Code Protection**: `chakrview/runtime/session.py` remains untracked/unstaged and preserved.

---

## 2. Reproduction of Reported Baselines across 6 Configurations

Re-evaluating the 6 configurations on the frozen $N=70$ benchmark (`cabf7daa...`) reproduced the exact reported metrics:

| Configuration | Syntax Validity | Exact Plan Match | Slot Correctness |
| :--- | :---: | :---: | :---: |
| **Config A: Base AR Planner** | 37/70 (52.9%) | 6/70 (8.6%) | 6/70 (8.6%) |
| **Config B: Raw Span Model** | 49/70 (70.0%) | 0/70 (0.0%) | 23/70 (32.9%) |
| **Config C: Unified Fusion** | **70/70 (100.0%)** | **2/70 (2.9%)** | **3/70 (4.3%)** |
| **Config D: Oracle Struct + Pred Spans** | 70/70 (100.0%) | 0/70 (0.0%) | 23/70 (32.9%) |
| **Config E: Pred Struct + Oracle Spans** | 70/70 (100.0%) | **2/70 (2.9%)** | **70/70 (100.0%)** |
| **Config F: Oracle Struct + Oracle Spans**| 70/70 (100.0%) | **70/70 (100.0%)**| 70/70 (100.0%) |

---

## 3. Investigation of Key Diagnostic Discrepancies

### 3.1 Why Did Config E Report 70/70 Slot Correctness but Only 2/70 Exact Plan Match?
- **Definitive Finding**: In `evaluate_comparative_fusion.py` line 133, `metrics["Config_E_Pred_Struct_Oracle_Span"]["slot_correct"] += 1` was recorded unconditionally as an assumption of the oracle span injection definition, rather than verifying that the injected slots matched the target plan.
- **Topological Mismatch**: Config E used `fusion_pipeline.predict_plan_topology(prompt)` to generate the plan structure. The heuristic generator predicted 2-step plans for 52 of the 70 items, whereas the ground-truth benchmark contains 21 three-step plans, 12 four-step plans, 12 one-step/clarification plans, and 25 two-step plans. Because the number of steps and operations did not match, exact plan match was only 2/70!

### 3.2 Does 100% Syntax Validity Conceal Failures?
- **Yes.** `UnifiedPlan.materialize_dsl()` strictly enforces grammar tokens, indentation, and delimiters, guaranteeing 100% syntactic validity. However, when the underlying topology is wrong (e.g. producing a 2-step plan for a 3-step prompt) or spans are bound to the wrong operands (e.g. `expr=14*14`), the plan is syntactically pristine but semantically incorrect.
