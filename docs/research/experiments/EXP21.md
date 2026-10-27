# Research Evidence Dossier: Experiment 21 (Exp21)
## Synthetic Recombined Copying Evaluation

- **Archive Status**: COMMITTED_IN_GIT
- **Artifact Status**: COMMITTED (Git ancestry) / LOCAL_ONLY (Checkpoints)
- **Primary Evidence Directory**: experiments/overnight_runs/exp21_synthetic_recombined_copy_cpu_300_20261024_205000/

---

### 1. Purpose and Hypothesis
- **Purpose**: Test whether synthetic recombination of copy tokens reduces positional bias during token extraction tasks.
- **Hypothesis**: Exposing the model to diverse token offset distances enables length-invariant token copying.

### 2. Empirical Results
- Candidate copying evaluation and statistical analysis (STATISTICAL_ANALYSIS.json) demonstrated improved short-span copying (<4 tokens), but failed to stabilize copy accuracy on sequences exceeding 8 tokens without explicit attention guidance.
