# WP 894–897: Reproducibility Review & Lineage Verification

## 1. Canonical Checkpoint Invariance
- **Canonical Step 10,000 Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Expected SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Verified Status**: Verified immutable at mission initialization, during model training, and post-experimentation. Zero modifications occurred.

---

## 2. Candidate Checkpoint Lineage

All candidate checkpoints produced during Batches 861–900 have been registered with verifiable cryptographic checksums:
1. **Candidate Seed 42**:
   - Path: `experiments/batch861_900/checkpoints/candidate_step250_seed42_span_grounding.pt`
   - SHA-256: `78ee192b005fe4379cae56ca2c395fb42171c696b994ee31dd3543d463b211d0`
2. **Candidate Seed 101**:
   - Path: `experiments/batch861_900/checkpoints/candidate_step250_seed101_span_grounding.pt`
   - SHA-256: `6fa7a4e69b56f8f55331ae8772bf6d13262e336215eb375d8d067c8be19ca709`
3. **Candidate Ablation (No Type Head)**:
   - Path: `experiments/batch861_900/checkpoints/candidate_step250_seed42_no_type_span_grounding.pt`
   - SHA-256: `00c0f8e97f0a7863cb6d8fa87f3418579ca6f5787f94bb2e1f2b622caefbe3ec`

---

## 3. Regression Suite Health
- **Total Test Cases**: 48 across Batches 541 to 900.
- **Pass Rate**: **48/48 passed (100.0%)** in 5.62s.
- **Zero Regressions**: All pre-existing capabilities (neural intent, span pointer, memory resolver, hierarchical planner, grammar checker, and pointer-generator) remain fully operational.
