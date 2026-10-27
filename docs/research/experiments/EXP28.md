# Research Evidence Dossier: Experiment 28 (Exp28)
## Fair Controlled Comparison: Baseline vs Pointer Mechanism

- **Archive Status**: PIVOTAL_GOVERNANCE_DECISION_AUDIT
- **Artifact Status**: COMMITTED (Comparison Tool) / LOCAL_ONLY (Full Traces)
- **Primary Evidence Directory**: experiments/overnight_runs/exp28_fair_baseline_vs_pointer_20261025_211555/

---

### 1. Purpose and Design
- **Purpose**: Conduct an head-to-head controlled experiment comparing the baseline Transformer against the Pointer-augmented architecture under strictly matched compute, randomized distractor positioning, and leak-free validation splits.
- **Controlled Protocol**: Identical parameter budgets, identical training step counts, and independently sampled distractor sequences.

### 2. Empirical Verdict
- **Empirical Metrics**:
  - Baseline: 64.2% copy accuracy on randomized test sets.
  - Pointer Network: 65.1% copy accuracy (p > 0.05, no statistically significant difference).
- **Governance Impact**: Formally established that the pointer mechanism in its current form does not justify production architectural complexity or risk. Justified quarantining the pointer head from the canonical v0.8.0 release.

### 3. Supporting Tools & Reports
- Fair Comparison Script: scripts/research_tools/run_exp28_fair_comparison.py (COMMITTED)
- Recomputed Metrics: METRICS_RECOMPUTED.json (LOCAL_ARCHIVE)
