# CR-EXP-16E Git Delivery Report & Branch Audit

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
experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/
├── ARTIFACT_HASHES.txt
├── CR16DR_FORENSIC_AUDIT.md
├── CR16E_ABSTENTION_BENCHMARK.md
├── CR16E_ARCHITECTURE_DECISION.md
├── CR16E_CAPABILITY_SCORECARD.md
├── CR16E_COUNTERFACTUAL_ANALYSIS.md
├── CR16E_DATASET_SPECIFICATION.md
├── CR16E_EXECUTIVE_SUMMARY.md
├── CR16E_FACTORIAL_DESIGN.md
├── CR16E_FINAL_SCIENTIFIC_REPORT.md
├── CR16E_POSITION_BIAS_ANALYSIS.md
├── CR16E_SPAN_GENERATION_ANALYSIS.md
├── GIT_DELIVERY_REPORT.md
├── generate_cr16e_benchmarks.py
├── generate_cr16e_training_data.py
├── train_cr16e_factorial.py
├── evaluate_cr16e_grand.py
├── data/
│   ├── arm_a_fixed_stdspan_train.json
│   ├── arm_b_shuffled_stdspan_train.json
│   ├── arm_c_fixed_balancedspan_train.json
│   ├── arm_d_shuffled_balancedspan_train.json
│   ├── cr16e_multitask_val.json
│   ├── cr16e_b1_position_invariance_600.json
│   ├── cr16e_b2_permutation_pairs_150.json
│   ├── cr16e_b2_counterfactual_pairs_150.json
│   ├── cr16e_b3_multitoken_span_500.json
│   └── cr16e_b4_abstention_300.json
├── checkpoints/
│   ├── arm_a_fixed_stdspan_seed2026/
│   ├── arm_a_fixed_stdspan_seed3030/
│   ├── arm_a_fixed_stdspan_seed4040/
│   ├── arm_b_shuffled_stdspan_seed2026/
│   ├── arm_b_shuffled_stdspan_seed3030/
│   ├── arm_b_shuffled_stdspan_seed4040/
│   ├── arm_c_fixed_balancedspan_seed2026/
│   ├── arm_c_fixed_balancedspan_seed3030/
│   ├── arm_c_fixed_balancedspan_seed4040/
│   ├── arm_d_shuffled_balancedspan_seed2026/
│   ├── arm_d_shuffled_balancedspan_seed3030/
│   └── arm_d_shuffled_balancedspan_seed4040/
├── eval_outputs/
│   └── cr16e_grand_evaluation_summary.json
└── logs/
    └── (12 training execution logs)
```

---

### 3. Verification Hashes Summary
Every generated benchmark suite, script, and checkpoint has its SHA-256 recorded in `ARTIFACT_HASHES.txt`. Repository remains clean, reproducible, and ready for user inspection.

