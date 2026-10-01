# ChakrView Step 69 Specification: Deterministic Patch Planning & Independent Validation

## 1. Architectural Purpose

Step 69 bridges the gap between passive, validated neural proposals ([ProposalContract](file:///d:/Project/ChakrView/chakrview/cognition/repository/neural_proposal.py)) and executable repository modifications by introducing strongly typed, deterministic patch structures ([PatchPlan](file:///d:/Project/ChakrView/chakrview/cognition/repository/patch_planning.py), [PatchOperation](file:///d:/Project/ChakrView/chakrview/cognition/repository/patch_planning.py)) and an authoritative independent validation gate ([PatchPlanValidator](file:///d:/Project/ChakrView/chakrview/cognition/repository/patch_planning.py)).

The neural core proposes text and intentions; Step 69 ensures that:
1. No plan can be formed from unaccepted or conflicted proposals (`fail-closed`).
2. Every operation preserves 100% provenance back to repository evidence and recalled episodic memories.
3. Every plan has a canonical, deterministic SHA-256 fingerprint independent of system clocks or process environments.
4. Human-readable, deterministic unified diffs are generated for review without executing or mutating any file.
5. An independent validator audits the plan against the live repository state and security boundaries before any execution consideration.

## 2. Core Data Models

### 2.1 PatchOperationType
- `CREATE_FILE`: Introduction of a new file inside the workspace root.
- `MODIFY_FILE`: Modification of an existing tracked source file.
- `DELETE_FILE`: Safe removal of an existing tracked file.
- `REPLACE`: Deterministic whole-file or range replacement.

### 2.2 PatchOperation
A frozen, immutable record defining a discrete mutation:
- `operation_type`: `PatchOperationType`
- `target_file`: Sanitized relative file path
- `target_symbol`: Optional targeted function or class symbol
- `original_content`: Optional pre-mutation content for diff calculation
- `new_content`: Exact replacement string
- `operation_fingerprint`: Canonical SHA-256 digest computed across operation type, path, symbol, and contents
- `provenance_evidence_ids`: Grounded repository evidence IDs
- `provenance_memory_ids`: Grounded episodic memory IDs
- `provenance_negative_boundary_ids`: Negative boundary references

### 2.3 PatchPlanStatus
State machine governing plan readiness:
- `PLANNED`: Initial deterministic composition complete.
- `VALIDATED`: Intermediate sanity checks passed.
- `READY_FOR_EXECUTION_REVIEW`: Successfully passed independent validation by `PatchPlanValidator`.
- `REJECTED`: Violation of security bounds, unallowed scope, or missing target files.
- `CONFLICTED`: Plan conflicts with task constraints or negative boundaries.
- `STALE`: Repository fingerprint changed between proposal and plan generation.

### 2.4 PatchPlan
Canonical plan specification:
- `plan_id`: Unique deterministic identifier (`plan_<16-hex-digest>`)
- `proposal_id`: Originating `ProposalContract` ID
- `context_fingerprint`: Originating cognitive context hash
- `repository_fingerprint`: Snapshot repository state hash
- `target_files`: Sorted tuple of target files
- `operations`: Sorted tuple of `PatchOperation`
- `unified_diff`: Standardized human-readable diff text
- `plan_fingerprint`: Canonical SHA-256 digest of proposal ID, context hash, repository hash, and operation fingerprints
- `affected_files_fingerprint`: Canonical digest of target files

## 3. Path Security & Sanitization

[sanitize_relative_path](file:///d:/Project/ChakrView/chakrview/cognition/repository/patch_planning.py) rejects:
- Relative path traversal (`..` or `.`)
- Drive-letter paths (`C:`, `D:`)
- Absolute root paths (`/etc`, `\\`)
- UNC paths (`//server/share`)
- Empty or whitespace strings

## 4. Independent Validation Rules

The [PatchPlanValidator](file:///d:/Project/ChakrView/chakrview/cognition/repository/patch_planning.py) does not trust the planner and verifies:
1. **Freshness Invariant**: `plan.repository_fingerprint == live_repo_state.state_fingerprint`.
2. **Digest Integrity**: Canonical recalculation of `plan_fingerprint` matches stored digest.
3. **Scope Containment**: All target files must reside within `allowed_files`.
4. **Existence Invariant**: For `MODIFY_FILE` or `DELETE_FILE`, target files must exist in repository context.
5. **Execution Keyword Prohibition**: Blocks dangerous handles (`os.system`, `subprocess`, `eval`, `exec`, `shutil.rmtree`).
6. **Provenance Presence**: Operations lacking supporting evidence or memory IDs are rejected.
