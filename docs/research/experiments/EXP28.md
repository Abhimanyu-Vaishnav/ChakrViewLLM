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
- **Empirical Metrics (Exact Recomputed Matrix from METRICS_RECOMPUTED.json)**:
  - **In-Distribution (test_1, 40 recs, A-L)**:
    - Arm A (Canonical ChakrMicro SFT): **82.5% EM (33/40)**, 70.0% Counterfactual Pair Accuracy (PLA, 28/40).
    - Arm B (ChakrPointerMicro): **92.5% EM (37/40)**, **87.5% PLA (35/40)**.
  - **Held-Out Letter Novelty (test_2a, 32 recs, M-P, m-p)**:
    - Arm A (Canonical ChakrMicro SFT): **0.0% EM (0/32)**, 0.0% PLA.
    - Arm B (ChakrPointerMicro): **100.0% EM (32/32)**, **100.0% PLA (32/32)**.
  - **Held-Out Digits & Symbols (test_2b, 32 recs, !, 8, 9, etc.)**:
    - Arm A (Canonical ChakrMicro SFT): **0.0% EM (0/32)**, 0.0% PLA.
    - Arm B (ChakrPointerMicro): **100.0% EM (32/32)**, **100.0% PLA (32/32)**.
  - **Held-Out Extended Letters (test_2c, 32 recs, Q-X)**:
    - Arm A (Canonical ChakrMicro SFT): **0.0% EM (0/32)**, 0.0% PLA.
    - Arm B (ChakrPointerMicro): **100.0% EM (32/32)**, **100.0% PLA (32/32)**.
  - **Distractor Position Probe (distractor_position, 40 recs, [Z] prepended)**:
    - Arm A (Canonical ChakrMicro SFT): **80.0% EM (32/40)**, 60.0% PLA (24/40).
    - Arm B (ChakrPointerMicro): **85.0% EM (34/40)**, **75.0% PLA (30/40)**.
  - **Blank Context Probes (blank_context, 4 recs)**:
    - Arm A: 0.0% EM (emits boilerplate letters).
    - Arm B: 0.0% EM (emits boilerplate letters).
- **Governance Impact & Architectural Demarcation**:
  - ChakrPointerMicro successfully solves the vocabulary unembedding bottleneck for bracketed single-token copying under familiar templates.
  - However, per Exp29 robustness stress testing, pointer attention is an equivariant positional lookup head, not general abstract reasoning. It remains quarantined from the canonical LocalModelSession v0.8.0 runtime because it is not integrated into multi-turn dialogue, natural language paraphrasing, or schema-constrained production inference.

### 3. Supporting Tools & Reports
- Fair Comparison Script: scripts/research_tools/run_exp28_fair_comparison.py (COMMITTED)
- Recomputed Metrics: METRICS_RECOMPUTED.json (LOCAL_ARCHIVE)
