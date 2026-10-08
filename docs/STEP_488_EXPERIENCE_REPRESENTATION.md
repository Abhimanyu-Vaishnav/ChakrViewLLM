# STEP 488: Reasoning Experience Representation Specification

## Purpose
Define a formal, structured, auditable schema for cognitive reasoning experiences (`ReasoningExperience`) within ChakrView's governed RIL (Reinforcement / Iterative Learning) framework. Experiences bridge past reasoning trajectories with governed candidate updates without raw memorization or benchmark leakage.

---

## 1. Concrete Reasoning Experience Schema

Each record contains the following mandatory fields:

- `experience_id`: Unique deterministic SHA-256 hash computed over task, model version, and trajectory.
- `task_id`: Distinct identifier of the task episode.
- `task_family`: Relational task category (e.g. `1_hop`, `2_hop`, `3_hop`, `4_hop`, `distractor`, `inversion`, `bilingual_hindi`).
- `input_representation`: Compact summary of the premise sequence and query prompt.
- `trajectory_summary`: Sequence of executed computation cycles, attention focuses, and query refinements.
- `intermediate_state_metadata`: Compact norm, step deltas, and divergence stats across states ($s_1, s_2, \dots, s_n$).
- `reasoning_depth`: Total computation cycles executed before halting.
- `outcome`: Predicted answer and epistemic status.
- `expected_outcome`: Ground truth target answer and expected epistemic status.
- `success`: Boolean indicating exact outcome match.
- `confidence`: Controller confidence logit/probability at halting.
- `uncertainty`: Entropy or margin between top candidate logits.
- `failure_category`: Governed taxonomy classification (e.g., `NONE`, `STATE_DEGRADATION`, `PREMATURE_HALT`, `RETRIEVAL_FAILURE`, `CONFLICTING_PROPOSITION`).
- `resource_summary`: Peak RAM, execution latency in milliseconds, and cycles consumed.
- `model_version`: Exact checkpoint version used during generation.
- `candidate_version`: Active candidate ID.
- `provenance`: Origin trace (subsystem author, parent experience ID, sequence timestamp).
- `evaluation_split`: Strict separation among `TRAINING_EXPERIENCE`, `VALIDATION_EXPERIENCE`, and `HELDOUT_EXPERIENCE`.

---

## 2. Epistemic Separation & Anti-Leakage Rules
1. **No Benchmark Contamination**: Experiences marked `HELDOUT_EXPERIENCE` must never enter the candidate optimization loop or training dataset.
2. **Compact Representation**: High-dimensional weight matrices and large activations are summarized via L2 norms and SHA-256 digests rather than storing bloated multi-gigabyte tensors.
3. **Reproducibility Guarantee**: Given the seed, checkpoint, and input representation, the trajectory must be deterministically replayable.
