# WP 938–940: Reproducibility Review — Batches 901–940

## 1. Canonical Checkpoint Invariance
- **Canonical Step 10,000 Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Expected SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Verified SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Integrity**: Verified strictly unchanged before and after all evaluations.

---

## 2. Regression Test Results
- **Full Historical + New Regression Suite**: 51 tests across Batches 541 to 940.
- **Result**: **51 passed, 0 failed** in 6.75 seconds.
- **Coverage**:
  - Neural intent router & security boundaries (Batches 541–580)
  - Span pointer & text utility extraction (Batches 581–620)
  - Multi-turn context binding & session memory (Batches 621–660)
  - Hierarchical dialogue & episodic planning (Batches 661–700)
  - Neural task graph decomposition (Batches 701–740)
  - Plan DSL serialization & validation (Batches 741–780)
  - Grammar decoder state machine (Batches 781–820)
  - Pointer generator & optimized decoder equivalence (Batches 821–860)
  - Span grounding head & boundary preservation (Batches 861–900)
  - Unified plan representation & span-to-slot fusion (Batches 901–940)
