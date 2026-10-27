# Research Evidence Dossier: Experiment 29 (Exp29)
## Independent Replication Audit & Robustness Evaluation

- **Archive Status**: INDEPENDENT_AUDIT_PASS
- **Artifact Status**: LOCAL_ONLY (Audit Dossier & Robustness Metrics)
- **Primary Evidence Directory**: experiments/overnight_runs/exp29_independent_exp28_audit_20261025_212249/

---

### 1. Purpose and Protocol
- **Purpose**: Blind independent verification of Exp28 findings using an external evaluation harness, stress-testing distractor distance, sequence length variance, and token noise.
- **Verification Harness**: Evaluated across 12 distinct data splits under randomized distractor positioning and bracket variations.

### 2. Audit Findings & Recomputation Results
- **Metric Recomputation**: Recomputed all split metrics from raw predictions with zero numerical discrepancies (audit_status: RECOMPUTATION_SUCCESSFUL_ZERO_DISCREPANCIES).
- **Leakage Verification**: Confirmed strict dataset sequestration and absence of prompt-target overlap in Exp28 evaluation splits.
- **Robustness Limitations**: Under multi-distractor queries and natural language paraphrasing, pointer precision showed vulnerability, proving it acts as an equivariant positional lookup head rather than general language understanding.
- **Release Readiness Finding**: Confirmed that the canonical 3.44M Transformer core satisfies release criteria for v0.8.0, while ChakrPointerMicro remains safely quarantined as an experimental research artifact.
