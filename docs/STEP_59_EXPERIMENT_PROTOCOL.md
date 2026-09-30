# ChakrView Step 59: Experiment Protocol

- **Document Version**: 1.0.0
- **Status**: Ratified Protocol (Step 59)
- **Scope**: Controlled Empirical Evaluation of Repository-Level Problem Solving, Dependency Graph Reasoning, and Multi-File Verification

---

## 1. Scientific Objective

The primary question is:
> **Can the existing ChakrView Cognitive Workspace solve a bounded multi-file repository problem by reasoning over dependencies, interacting with ChakrKshetra, observing execution/test feedback, revising its plan, and verifying the final repository state?**

To isolate and prove causality, the experiment tests four conditions with memory ablation and negative transfer checks.

---

## 2. Experimental Conditions

- **Condition A (Cold Start)**:
  - Repository initialized with the defective `tax_service.py`.
  - Cognitive workspace starts with empty episodic/semantic memory.
  - Measures baseline actions, dependency tracing steps, repair attempts, and time to converge.
- **Condition B (Memory Assisted)**:
  - Repository task re-encountered or structurally related transfer task executed with verified episodic/semantic memory active.
  - Measures whether prior memory of the dependency defect accelerates diagnosis and reduces attempts.
- **Condition C (Memory Ablation)**:
  - Identical task executed with memory explicitly unmounted/disabled.
  - Proves causal utility ($\Delta \text{actions} = \text{actions}_{\text{ablated}} - \text{actions}_{\text{memory}}$).
- **Condition D (Negative Transfer Safety)**:
  - Unrelated multi-file repository task (e.g. string formatting repository or state machine repository).
  - Verifies that billing/tax memories are not misapplied across orthogonal domains.

---

## 3. Required Metrics

1. `task_success`: Boolean indicating 100% test pass and repository integrity.
2. `attempt_count`: Number of repair cycles dispatched to ChakrKshetra.
3. `action_count`: Total individual actions (`INSPECT`, `TEST`, `PATCH`, `VERIFY`).
4. `inspection_count`: Number of files/modules analyzed via AST inspection.
5. `patch_count`: Number of diff modifications applied.
6. `rollback_count`: Number of rejected or reversed modifications.
7. `test_runs`: Subprocess pytest executions in ChakrKshetra sandbox.
8. `targeted_test_pass`: Level 1 pass rate.
9. `regression_test_pass`: Level 2 pass rate.
10. `repository_verification`: Level 3 pass rate (all tests in repository green).
11. `final_diff_integrity`: Level 4 confirmation (only intended files modified).
12. `dependency_graph_accuracy`: Precision and recall of discovered inter-module edges.
13. `memory_retrieved`: Number of candidate memories retrieved.
14. `memory_relevance`: Relevance ratio of retrieved memories.
15. `memory_ablation_delta`: Quantitative causal delta between Condition B and Condition C.
16. `negative_transfer`: Flag indicating whether any irrelevant memory corrupted execution.
17. `baseline_hash_immutability`: Verification that $\Delta W_{\text{baseline}} \equiv 0$.
