# CR-EXP-16F: Git Delivery & Safety Audit Report

## 1. Non-Negotiable Safety Checklist Audit

1. **ChakrMicro v0.1 Architecture Invariant:**
   - Parameter count: Exactly 3,443,136 parameters ($L=6, d=192, H=6, d_{\text{mlp}}=512, V=4096$).
   - `seed=42` model parameter initialization SHA-256 digest: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` (**EXACT MATCH**).
   - Architectural configuration strictly frozen.
2. **Protected Runtime Protection:**
   - Target file: `chakrview/runtime/session.py`.
   - File SHA-256 digest: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (**STRICTLY UNMODIFIED**).
3. **Canonical Checkpoint Lineage Protection:**
   - Stage B canonical checkpoint: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`) (**VERIFIED**).
   - Parent checkpoint CR-16A Arm D: `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_best.pt` (`9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5`) (**VERIFIED**).
4. **Git Operations:**
   - No `git commit` or `git push` executed. All deliverables remain unstaged in the local working tree awaiting explicit user direction.

---

## 2. Inventory of Delivered Artifacts

Location: `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16f_multicapability_retention/`

- **Core Code & Protocols:**
  - `train_cr16f_factorial.py`: Factorial training harness for all 12 models.
  - `evaluate_cr16f.py`: Full evaluation pipeline covering D1 through D6 benchmarks.
  - `CR16F_PHASE_A_PREFLIGHT_AUDIT.md`: Preflight lineage and parameter verification audit.
  - `CR16F_TRAINING_PROTOCOL.md`: Exact weighted cross-entropy loss formulation and training protocol.
- **Scientific Analysis & Scorecards:**
  - `CR16F_TRAINING_REPORT.md`: Training dynamics, loss trajectories, and checkpoint digests.
  - `CR16F_EVALUATOR_AUDIT.md`: Benchmark validation and evaluation audit.
  - `CR16F_STATISTICAL_ANALYSIS.md`: Two-way factorial ANOVA, cell means, standard deviations, and effect sizes.
  - `CR16F_CAPABILITY_TRADEOFF_ANALYSIS.md`: Multi-capability trade-offs and forgetting analysis against ancestral lineage.
  - `CR16F_MECHANISM_ANALYSIS.md`: Detailed mechanistic breakdown of gradient conflict and capacity bottlenecks.
  - `CR16F_CAPABILITY_SCORECARD.md`: 14-dimension capability scorecard with integer numerators and denominators.
  - `CR16F_ARCHITECTURE_DECISION.md`: Formal characterization of ChakrMicro v0.1 capacity bounds and future roadmap.
  - `CR16F_FINAL_SCIENTIFIC_REPORT.md`: Master comprehensive research report answering all 10 user questions.
  - `ARTIFACT_HASHES.txt`: SHA-256 digests for all documentation, scripts, and 12 model checkpoints.
  - `EXPERIMENT_MANIFEST.json`: Machine-readable metadata and checkpoint manifest.
  - `cr16f_evaluation_master_summary.json`: Raw aggregated evaluation metrics.

