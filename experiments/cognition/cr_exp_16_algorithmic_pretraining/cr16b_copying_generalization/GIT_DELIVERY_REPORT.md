# CR-EXP-16B Git Delivery Report

## 1. Compliance and Integrity Verification
- Target Branch: `exp/cbr-01-conversational-breakthrough`
- Protected Runtime: `chakrview/runtime/session.py` (SHA-256: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`) — Verified bit-for-bit unchanged.
- Canonical Checkpoint: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`) — Verified read-only and preserved.
- Sealed Benchmark: `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/data/cr16b_sealed_benchmark_1500.json` (SHA-256: `44aba606abfee0beac5b129eb7e5199b8f178c8b96314b0f7314ae75a6d56e1c`) — Generated, audited, and sealed.
- External Dependencies: None. Zero external weights, zero LLM APIs used.

## 2. Inventory of Delivered Deliverables
All work is strictly contained in:
`experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/`

### A. Reports (17/17 Deliverables)
1. `CR16B_EXECUTIVE_SUMMARY.md`
2. `CR16A_INDEPENDENT_REPRODUCTION.md`
3. `CR16B_BENCHMARK_SPECIFICATION.md`
4. `CR16B_LEAKAGE_AUDIT.md`
5. `CR16B_OBJECTIVE_FACTORIAL_DESIGN.md`
6. `CR16B_TRAINING_COMPARABILITY.md`
7. `CR16B_NAME_COPYING_ANALYSIS.md`
8. `CR16B_TOKENIZATION_ANALYSIS.md`
9. `CR16B_TEACHER_FORCED_ANALYSIS.md`
10. `CR16B_FREE_GENERATION_ANALYSIS.md`
11. `CR16B_SEED_REPRODUCIBILITY.md`
12. `CR16B_CAPABILITY_SCORECARD.md`
13. `CR16B_FAILURE_ANALYSIS.md`
14. `CR16B_ARCHITECTURE_DECISION.md`
15. `CR16B_FINAL_SCIENTIFIC_REPORT.md`
16. `ARTIFACT_HASHES.txt`
17. `GIT_DELIVERY_REPORT.md`

### B. Implementation Scripts & Datasets
- `reproduce_cr16a_scorecard.py`: Reproduction test suite
- `generate_cr16b_benchmark.py`: Sealed 1,500 benchmark generator
- `generate_factorial_data.py`: Matched factorial dataset synthesizer
- `train_factorial_arms.py`: 2x2 Factorial training engine (8 runs across 2 seeds)
- `evaluate_sealed_benchmark.py`: Complete evaluation harness across all models
- `data/cr16b_sealed_benchmark_1500.json`: Sealed 1,500-probe benchmark
- `data/factorial_single_template_train.json`: Matched single-template dataset
- `data/factorial_multi_template_train.json`: Matched multi-template dataset
- `data/factorial_val.json`: Validation dataset
- `eval_outputs/all_models_reproduction_summary.json`: Independent reproduction output
- `eval_outputs/cr16b_sealed1500_all_models_summary.json`: Grand 1,500 evaluation matrix

## 3. Git Operations Status
In accordance with instructions, **no git commit or git push has been executed**.
All files remain untracked / unstaged in the working directory awaiting user review.

