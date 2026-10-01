# ChakrView Steps 69 & 70 Experiment Protocol

## 1. Objective
Demonstrate an end-to-end, experimentally reproducible path from a grounded neural proposal to a safely planned, independently validated, deterministically diffed, explicitly approved, applied, and verified repository patch, with clean rollback capability and zero neural core drift.

## 2. Environment & Neural Baseline Verification
- Model: `ChakrMicro`
- Configuration: `ModelConfig()`
- Random Seed: `42`
- Required SHA-256 Hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Required Parameter Count: `3,443,136`
- Core Invariant: $\Delta W = 0$ throughout all execution steps.

## 3. Test Fixture Repository
A minimal isolated workspace representing a billing subsystem:
- `billing/discount.py`: Defective function with unbounded discount.
- `billing/taxes.py`: Unrelated module to test non-interference.

## 4. Execution Pipeline
1. **Task Specification**: "Fix discount computation to prevent discounts exceeding order total."
2. **Grounding & Episodic Recall**: Inspect AST, dependencies, and recall past clamp pattern.
3. **Cognitive Context Composition**: Assemble passive context bundle with budget boundaries.
4. **ChakrMicro Proposal**: Neural core generates structured `ProposalContract`.
5. **Proposal Validation**: `ProposalValidator` accepts grounded proposal.
6. **Patch Planning**: `PatchPlanner` compiles `PatchPlan` with deterministic fingerprints.
7. **Independent Validation**: `PatchPlanValidator` confirms integrity and security rules.
8. **Deterministic Diff**: Generate unified diff (`--- a/... +++ b/...`).
9. **Approval Check**: Test rejection of unapproved requests (`approved=False`).
10. **Application**: Apply staged patch to `billing/discount.py`.
11. **Post-Application Verification**: Assert `billing/discount.py` has expected content and `billing/taxes.py` is byte-identical.
12. **Rollback**: Restore original contents and verify identical hash.

## 5. Pass Criteria
All 21 conditions (A through U) must pass without mocks or synthetic shortcuts.
