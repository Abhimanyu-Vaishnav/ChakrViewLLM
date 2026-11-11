# CR-EXP-16D-R Git Delivery Report & Repository Audit

### 1. Repository Safety & Protocol Adherence
- **Branch:** `exp/cbr-01-conversational-breakthrough`
- **Initial Commit Integrity:** Preserved without rebase, reset, or history rewrites.
- **Protected File Check:**
  `chakrview/runtime/session.py` (SHA-256: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`)
  **Status: 100% UNTOUCHED.**
- **Commit Guardrail:** In strict compliance with mission guardrails, **no Git commit or push** has been executed. All experimental outputs reside cleanly in the working tree.

---

### 2. File Inventory of Delivered Artifacts

```
experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/
├── ARTIFACT_HASHES.txt
├── CR16D_ABSTENTION_REANALYSIS.md
├── CR16D_G1_COUNTERFACTUAL_AUDIT.md
├── CR16D_G2_UNSEEN_VALUE_DIAGNOSIS.md
├── CR16D_G3_KEY_GENERALIZATION_AUDIT.md
├── CR16D_LINEAGE_AUDIT.md
├── CR16D_RAW_METRIC_RECOMPUTATION.md
├── CR16D_SEED_VARIANCE_ANALYSIS.md
├── CR16DR_ARCHITECTURE_DECISION.md
├── CR16DR_CAPABILITY_SCORECARD.md
├── CR16DR_COMPOSITIONAL_GENERALIZATION.md
├── CR16DR_COPYING_RETENTION.md
├── CR16DR_EXECUTIVE_SUMMARY.md
├── CR16DR_FAILURE_ANALYSIS.md
├── CR16DR_FINAL_SCIENTIFIC_REPORT.md
├── CR16DR_REPLAY_RATIO_ANALYSIS.md
├── CR16DR_REPLICATION_DESIGN.md
├── CR16DR_RETRIEVAL_ACQUISITION.md
├── CR16DR_SEED_REPRODUCIBILITY.md
├── CR16DR_TRAINING_COMPARABILITY.md
├── GIT_DELIVERY_REPORT.md
├── generate_part_b_diagnostics.py
├── run_phase_a_diagnostic_eval.py
├── run_part_b_diagnostics.py
├── train_cr16dr_replication.py
├── evaluate_replication_seeds.py
├── data/
│   ├── cr16dr_counterfactual_pairs_200.json
│   ├── cr16dr_position_pairs_100.json
│   └── cr16dr_distractor_probes_100.json
├── checkpoints/
│   ├── arm_a_replay00_seed202/
│   ├── arm_a_replay00_seed303/
│   ├── arm_a_replay00_seed404/
│   ├── arm_b_replay25_seed202/
│   ├── arm_b_replay25_seed303/
│   ├── arm_b_replay25_seed404/
│   ├── arm_c_replay50_seed202/
│   ├── arm_c_replay50_seed303/
│   ├── arm_c_replay50_seed404/
│   ├── arm_d_replay75_seed202/
│   ├── arm_d_replay75_seed303/
│   └── arm_d_replay75_seed404/
└── eval_outputs/
    ├── cr16dr_seeds_202_303_404_eval_summary.json
    ├── arm_c_replay50_s101_part_b_diagnostics.json
    └── ... (per-model diagnostic JSONs)
```

---

### 3. Checkpoint Hash Verification
Every generated replication checkpoint and diagnostic file has its SHA-256 recorded in `ARTIFACT_HASHES.txt`. All hashes match on-disk bytes bit-for-bit.

