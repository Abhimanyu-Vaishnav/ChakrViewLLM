# Experiment Protocol — Batches 1301–1340
## Refutation Safety Audit, Runtime Reliability and Neural Capability Assessment

### 1. Mission Objectives
1. **Audit Refutation Implementation**: Critically inspect the contradiction detection heuristics added in Batch 1261–1300 to determine if they rely on brittle lexical/numerical patterns.
2. **Build an Adversarial Decision Benchmark**: Construct a versioned, balanced 100-item benchmark containing paired examples, valid "why" queries, permission queries, unit differences, and distractors.
3. **Establish Conservative Decision Policy**: Tighten contradiction detection to require subject/action alignment and attribute-specific numeric bounds, preventing false refutations while preserving valid answers and honest abstentions.
4. **Prevent Retrieval-Induced Refutation Errors**: Ensure cross-domain or irrelevant retrieved passages cannot authorize false refutations.
5. **Neural Capability Inventory & Next Core Experiment**: Produce an honest capability inventory of ChakrView's neural core and define a small, measurable next-core experiment.

### 2. Verified Invariants
- **Canonical Step-10000 Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (Expected SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Protected User File**: `chakrview/runtime/session.py` preserved unmodified.
- **Branch**: `dev/v0.6-v0.7-validation`.
