# WP 979–980: Reproducibility Review — Batches 941–980

## 1. Canonical Checkpoint Integrity
- **Canonical Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Expected SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Verified SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Match confirmed)

---

## 2. Frozen Test Datasets Checksums
1. **Original Frozen Benchmark ($N=70$)**:
   - Path: `experiments/batch781_820/data/test_plan_dsl_frozen.jsonl`
   - SHA-256: `cabf7daa3c6e453de1ad06e99c801fbba9b669f2130746a7b34bf022cd21ec2a` (Verified untouched)
2. **Focused Task Family Frozen Test ($N=33$)**:
   - Path: `experiments/batch941_980/data/frozen_test_focused_family.jsonl`
   - SHA-256: `2705e02b51c88f6ab473989ffe48a8d387eff711afd012c9ef32416b5cd61ea9`

---

## 3. Regression Suite Health
- **Total Historical Tests**: 51
- **Passed**: 51 / 51 (100.0%) in 5.59s.
