# ChakrView Step 70 Specification: Safe Patch Execution, Approval Boundary & Deterministic Rollback

## 1. Architectural Purpose

Step 70 defines the controlled execution layer of ChakrView. While Step 69 builds and independently audits patch plans, Step 70 enforces the strict execution boundary:
1. **Explicit Approval Boundary**: A plan in `READY_FOR_EXECUTION_REVIEW` state can NEVER be executed without explicit authorization (`approved=True` and a valid `approval_token`).
2. **Zero Neural Execution Authority**: The neural core never has write or execution permissions on the filesystem.
3. **Stale State Defense**: Re-reads repository manifest immediately before write; aborts if the repository fingerprint has drifted.
4. **Atomicity & In-Memory Staging**: Staging occurs in-memory; any error causes immediate zero-write rollback.
5. **Post-Application Verification**: Verifies that intended changes took effect and unrelated files remained byte-identical.
6. **Deterministic Rollback**: Captures pre- and post-application states in a [PatchRollbackRecord](file:///d:/Project/ChakrView/chakrview/cognition/repository/patch_execution.py) for reversible restoration without running arbitrary shell scripts.

## 2. Execution State Machine

```
PLANNED
   ↓
VALIDATED
   ↓
READY_FOR_EXECUTION_REVIEW
   ↓ (Explicit Approval Token)
APPROVED
   ↓
APPLYING (In-Memory Staging -> Workspace Write)
   ↓
APPLIED
   ↓ (Post-Application Verification)
VERIFIED
   ↓ (Optional Reversion)
ROLLED_BACK
```

Failure states:
- `REJECTED`: Validation failure or security violation
- `CONFLICTED`: Contradiction detected
- `STALE`: Repository modified between planning and execution
- `FAILED`: IO error during write (triggers atomic rollback)

## 3. Data Structures

### 3.1 PatchExecutionRequest
- `plan`: Validated `PatchPlan`
- `approved`: Explicit boolean flag
- `approval_token`: Authorization token string
- `approver_identity`: Identifier of approving authority

### 3.2 PatchRollbackRecord
- `plan_id`: Identifier of applied plan
- `plan_fingerprint`: Canonical plan hash
- `affected_files`: Tuple of applied file paths
- `pre_contents`: Mapping of `rel_path -> original_content` (or `None` for newly created files)
- `post_contents`: Mapping of `rel_path -> applied_content`
- `pre_repo_fingerprint`: Repository hash before mutation
- `applied_repo_fingerprint`: Repository hash after mutation

### 3.3 PatchExecutionResult
- `plan_id`: Identifier of plan
- `plan_fingerprint`: Hash of plan
- `status`: Final `PatchExecutionStatus` (`VERIFIED`, `APPLIED`, `STALE`, `REJECTED`, `FAILED`)
- `applied_files`: Tuple of paths actually written
- `unaffected_files_preserved`: True if all other repository files remained byte-identical
- `post_apply_verification_passed`: True if target contents match expected values
- `rollback_record`: Reversion record if patch was applied
- `messages`: Diagnostic messages and audit log

## 4. Security Guarantees & Non-Claims

### What is Enforced
- **Strict Workspace Confinement**: All writes route through `IsolatedWorkspace._assert_within_workspace()`.
- **No Command Execution**: Rollback and application do not invoke `subprocess`, `os.system`, or shell commands.
- **Fail-Closed Drift Protection**: Any divergence in repository fingerprint aborts execution with zero mutation.

### Explicit Non-Claims
- Does not protect against malicious processes running outside the Python process modifying files concurrently.
- Does not provide OS-level kernel filesystem transaction guarantees; uses staging and clean compensating writes.
