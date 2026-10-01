# Step 62: Dynamic Branching Specification

## 1. Overview
Step 62 extends ChakrView's multi-step refactoring engine from linear execution sequences into an observation-driven, branching refactoring workflow. When an aggressive or speculative refactoring strategy produces intermediate regressions or fails repository verification, the system rolls back cleanly, restores the base cryptographic fingerprint, and pivots safely to alternative candidate branches.

```text
                  [Candidate Strategies: Priority Order]
                                    │
                                    ▼
                         Branch A Preconditions
                                    │
                         ┌──────────┴──────────┐
                   (Met) │                     │ (Not Met)
                         ▼                     ▼
                  Execute Steps           Reject Branch
                         │                     │
                Intermediate Checks            ▼
                         │               Evaluate Next
            ┌────────────┴────────────┐
    (Healthy)                         (Failure)
            ▼                                 ▼
       Continue Steps                   Rollback Branch
            │                                 │
     Final Verification             Verify State Fingerprint
            │                                 │
     (Pass) │                                 ▼
            ▼                       Reroute to Branch B
       Consolidate Experience                 │
                                    ┌─────────┴─────────┐
                                (Exhausted)         (Viable)
                                    ▼                   ▼
                              Safe Abstention     Execute Steps
```

## 2. Core Abstractions
1. **`RefactoringBranch`**: Strategy definition containing `branch_id`, `steps`, `allowed_files`, `priority`, `precondition_fn`, and lifecycle `status`.
2. **`BranchObservation`**: Observation collected after every step (`targeted_test_passed`, `regression_passed`, `diff_integrity_passed`, `state_fingerprint_before`, `state_fingerprint_after`, `highest_diff_category`, `is_healthy`).
3. **`BranchSelectionDecision`**: Deterministic record documenting candidate branch selection or precondition rejection.
4. **`RecoveryDecision`**: Detailed audit record verifying transaction rollback and pre-branch state fingerprint restoration before pivoting.
5. **`ObservationDrivenBranchingCoordinator`**: Orchestrates branch ordering, execution, rollback, recovery rerouting, and experience consolidation.

## 3. Safety Invariants & Operational Bounds
1. **Precondition Gating**: Branches are evaluated against `allowed_files` containment and optional custom preconditions before execution.
2. **Deterministic Priority**: Candidate branches execute strictly in sorted `(priority, branch_id)` order.
3. **Rollback Fingerprint Match**: If a branch fails, all transactions in that branch are rolled back, and the restored workspace fingerprint is compared with the base fingerprint. A mismatch causes an immediate hard abort.
4. **Operational Bounds**: `max_branches` (default: 5) and `max_recovery_transitions` (default: 3) prevent infinite execution loops or combinatorial explosion.
5. **Fail-Closed Abstention**: If all candidate branches fail or limits are reached, the coordinator abstains (`abstained=True`, `overall_success=False`).
