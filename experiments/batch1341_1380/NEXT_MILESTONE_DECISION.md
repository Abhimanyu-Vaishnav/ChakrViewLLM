# Next Milestone Decision

## Batches 1341–1380 -> Batches 1381+

### 1. Strategic Protocol Handoff

The language modeling benchmark completed in Batches 1341–1380 establishes that:
1. `ChakrMicro` possesses valid learned internal representations, beating empirical unigram baselines by a 79% margin in perplexity.
2. The evaluation harness, dataset split, causal masking, and metric aggregation are fully verified, unit-tested, and reproducible in sub-second execution on CPU.
3. The codebase is now prepared for a controlled, measurable training run.

**Approved Next Milestone (Batches 1381–1420):**
- Execute the controlled continual pre-training experiment proposed in [TRAINING_READINESS_DECISION.md](file:///D:/Project/ChakrView/experiments/batch1341_1380/TRAINING_READINESS_DECISION.md).
- Target: Reduce held-out technical test perplexity from $158.07$ to $< 90.0$.
- Guardrail: Train on a dedicated checkpoint path (`chakrmicro_domain_adapted_step_10500.pt`) and never overwrite the canonical step 10,000 checkpoint.

---

### 2. Checkpoint & Artifact Lineage

- **Canonical Model Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **LM Dataset Split Manifest:** `experiments/batch1341_1380/data/corpus_lm_split_manifest.json`
- **Evaluation Runner:** `experiments/batch1341_1380/run_lm_evaluation.py`
- **Machine-Readable Benchmark Results:** `experiments/batch1341_1380/results/LM_EVALUATION_RESULTS.json`
- **LM Regression Unit Tests:** `tests/test_batch1341_1380_lm_evaluation.py` (6 tests passing)
- **Total Milestone Regression Pass Rate:** 42 / 42 tests passing across Batches 1021–1380.
