# Step 145: Neural Learning Observability & Baseline/Candidate Protocol

## 1. Overview
Step 145 establishes rigorous governance and lineage tracking for neural learning experiments on the indigenous ChakrMicro core. It defines an immutable protocol enforcing candidate isolation and proving that candidate parameters diverge while the canonical neural baseline remains bit-exact and untouched ($\Delta W_{baseline} \equiv 0$).

## 2. Key Architecture Components

- `NeuralLearningExperiment`: Formal metadata record capturing:
  `experiment_id`, `parent_model_id`, `baseline_hash`, `candidate_hash`, `dataset_identity`, `dataset_sha256`, `tokenizer_identity`, `tokenizer_sha256`, `curriculum_identity`, `seed`, `optimizer_identity`, `learning_rate`, `batch_size`, `sequence_length`, `training_steps`, `training_device`, `parameter_count`, `train_loss`, `validation_loss`, `held_out_score`, `reasoning_score`, `language_score`, `generalization_score`, `regression_score`, `final_decision`.
- `CandidateIsolationManager`:
  - Branches mutable candidates via deep copy initialized from the frozen baseline.
  - Verifies before/after checkpoint hashes to ensure that:
    1. Candidate weights diverge (`candidate_hash != baseline_hash`).
    2. Canonical baseline hash matches `EXPECTED_WEIGHT_HASH` (`c5571c...`).
  - Records full experiment lineage into SQLite tables.

## 3. Empirical Verification
- Candidate isolation verified under unit test and master benchmark.
- Baseline immutability verified before and after candidate training cycles.
