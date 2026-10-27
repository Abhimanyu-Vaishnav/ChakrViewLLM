# Research Evidence Dossier: Experiment 27 (Exp27)
## Pointer Network Forensic Audit & Gradient Diagnostics

- **Archive Status**: CRITICAL_FORENSIC_AUDIT_READY_FOR_ARCHIVE
- **Artifact Status**: COMMITTED (Diagnostic Tools) / LOCAL_ONLY (Forensic Audits)
- **Primary Evidence Directory**: experiments/overnight_runs/exp27_forensic_pointer_research_20261025_203725/ and exp27_independent_validation_20261025_211125/

---

### 1. Purpose and Investigation
- **Purpose**: Rigorous forensic audit of previously reported 98%+ accuracy claims for pointer-augmented copy heads.
- **Investigative Findings**:
  - Uncovered synthetic split leakage in prior evaluation scripts where distractors shared identical prefix structure.
  - Gradient diagnostic (gradient_diagnostic.py) revealed vanishing gradient flow into pointer routing weights when vocabulary logits dominate.
  - Recomputed metrics dropped substantially when strict randomized distractor positions were enforced.

### 2. Supporting Diagnostic Tools (Committed)
- scripts/research_tools/audit_leakage_and_splits.py
- scripts/research_tools/test_gradient_diagnostic.py
- scripts/research_tools/run_bounded_pilot.py
- scripts/research_tools/recompute_metrics.py
