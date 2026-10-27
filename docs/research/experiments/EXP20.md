# Research Evidence Dossier: Experiment 20 (Exp20)
## Single-Task Contextual Binding & Generalization Limits

- **Archive Status**: EMPIRICALLY_BOUNDED
- **Artifact Status**: LOCAL_ONLY (Audit Dossiers & Baseline Evaluations)
- **Primary Evidence Directory**: experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/

---

### 1. Purpose and Hypothesis
- **Purpose**: Measure whether a 3.44M model can reliably bind contextual key-value pairs (e.g. name-attribute slots) when tested across both seen and unseen vocabulary tokens.
- **Hypothesis**: Attention heads in early layers can perform robust binding if loss is focused on slot-binding targets.

### 2. Empirical Findings & Boundary
- **Results**: The model achieved high accuracy (>85%) on in-distribution binding with seen token sequences, but suffered severe degradation when slot tokens were swapped with out-of-vocabulary or unseen distractor tokens.
- **Mandatory Claim Boundary**: Proves that contextual binding in this configuration functions via positional memorization and seen-token association rather than invariant symbolic indirection.

### 3. Supporting Artifacts & Availability
- Comparison Report: experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/EXP20_COMPARISON.md
- Forensic Audit: experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_20261024_134923/
