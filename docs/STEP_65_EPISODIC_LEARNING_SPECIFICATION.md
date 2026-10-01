# Step 65 Episodic Learning Specification: Active Episodic Learning Loop & Verified Semantic Memory Admission

## 1. Overview & Architectural Role

Step 65 introduces an active episodic learning loop that updates semantic repository memory from verified observations and execution traces without modifying neural core weights ($\Delta W = 0$).

In the ChakrView universal neural brain architecture:
- **Neural proposals propose candidate solution patterns or hypotheses.**
- **Deterministic verification gates validate patches across 4 verification tiers.**
- **Deterministic admission gates decide whether derived lessons are eligible for persistent semantic memory.**
- **The neural core remains bit-exact and immutable.**

---

## 2. Core Schemas & Data Contracts

### 2.1 Episodic Experience (`EpisodicExperience`)
Captures the complete end-to-end trajectory of a repository task:
- `episode_id`: Unique identifier for the task run.
- `task_description`: Human/neural description of the objective.
- `task_family`: Domain classification for scoped retrieval and negative transfer protection.
- `initial_state_fingerprint`: Cryptographic state fingerprint before execution.
- `final_state_fingerprint`: Cryptographic state fingerprint after execution.
- `retrieved_evidence`: Context and evidence records provided during planning.
- `neural_proposal`: The structured candidate patch emitted by the neural model adapter.
- `grounding_decision`: Verification from the HallucinationContainmentGate (`is_grounded`, `epistemic_state`).
- `execution_result`: Execution and verification record from the branching/refactoring coordinator.
- `outcome`: Terminal status of the episode (`SUCCESS_VERIFIED`, `PROPOSAL_HALLUCINATED`, `EXECUTION_FAILED_ROLLED_BACK`, `UNVERIFIED_ABORT`, `REJECTED_SCOPE`).
- `derived_observations`: List of `LearnedObservation` derived from the execution trace.
- `rollback_performed`: Flag indicating whether workspace state was rolled back.
- `timestamp`: Monotonic timestamp of episode execution.

### 2.2 Learned Observation (`LearnedObservation`)
A discrete cognitive lesson derived from an episode:
- `observation_id`: Unique observation identifier.
- `observation_type`: Type of lesson (`SOLUTION_PATTERN`, `NEGATIVE_BOUNDARY`, `FAILURE_AVOIDANCE`).
- `task_family`: Domain tag.
- `target_files`: Target files associated with the lesson.
- `lesson_summary`: Concise summary of what succeeded or failed.
- `is_positive`: Boolean indicating whether the observation represents an admissible solution or a failure mode.
- `evidence_fingerprint`: State fingerprint of the evidence supporting the lesson.
- `confidence`: Confidence score in [0.0, 1.0].
- `supporting_episode_id`: Provenance back to source episode.
- `verification_passed`: Boolean flag denoting whether tests passed.
- `failure_class`: Typed failure class (`HALLUCINATED_FILE`, `HALLUCINATED_SYMBOL`, `SCOPE_VIOLATION`, `STALE_MEMORY`, `TEST_FAILURE`, `UNVERIFIED_CLAIM`).

### 2.3 Deterministic Admission Gate (`DeterministicAdmissionGate`)
Evaluates candidate observations against 4 strict invariants before allowing admission into `RepositoryMemoryIndex`:
1. `verification_passed == True`: Observation must be supported by passing test verification.
2. `episode_success == True`: The source episode outcome must be `EpisodeOutcome.SUCCESS_VERIFIED`.
3. `grounding_confirmed == True`: Grounding gate must confirm zero hallucinated entities.
4. `no_unverified_claims == True`: Unverified claims cannot be admitted as positive records.

Observations failing these invariants are **REJECTED** from positive memory.
Verified failure modes are admitted strictly as negative boundary constraints (`ADMITTED_NEGATIVE`) with solution pattern `"DO_NOT_APPLY"`.

---

## 3. Memory Lifecycle & Conflict Handling

### 3.1 Positive Record Admission
- Verified successful episodes are transformed into `RepositorySemanticRecord`.
- Provenance is bound via `source_episode_ids` and state fingerprint signatures.

### 3.2 Negative Experience Handling
- Concrete verified failures (such as confirmed hallucinations or regression failures) are recorded as `known_boundaries` on records.
- Prevents recurrent neural hallucinations from re-entering active execution branches.

### 3.3 Conflict Resolution & Superseding
- When a newer verified episode addresses the identical scope (`task_family` and `target_files`), the admission coordinator:
  1. Identifies the older active record.
  2. Marks the older record with `active_version = False` and sets `superseded_by = new_memory_id`.
  3. Inserts the newer record with `supersedes = old_memory_id` and `active_version = True`.
  4. Retains older records in the index for historical traceability and auditability.
