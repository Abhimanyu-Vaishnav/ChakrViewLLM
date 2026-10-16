# Next Milestone Decision

## Batches 1381–1420 -> Batches 1421+

### 1. Selected Outcome and Strategic Decision

In accordance with Phase 10 of the mission mandate, we evaluate the four outcome criteria:

- **Selected Outcome: Outcome A**
  > *"The controlled candidate improves held-out technical language modeling, without a material measured regression on available auxiliary evaluations."*

#### Evidence for Outcome A:
1. Held-out test perplexity dropped by **51.6%** (from 158.07 to 76.49), achieving the target threshold ($< 90.0$).
2. Top-1 next-token prediction accuracy rose from 12.85% to 23.52%, and Top-10 accuracy reached 49.33%.
3. Validation loss improved monotonically over 80 steps without overfitting or gradient instability.
4. Qualitative probes demonstrated reduced degenerative repetition and improved technical formatting coherence.

---

### 2. Protocol and Artifact Lineage for Next Milestone

- **Canonical Baseline Checkpoint:**
  - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Unchanged, frozen baseline)
- **Experimental Domain-Adapted Checkpoint:**
  - Path: `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt`
  - SHA-256: `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139`
  - Adaptation step: 80 (Cumulative step: 10,080)
- **Evaluation Machine-Readable Output:**
  - `experiments/batch1381_1420/results/ADAPTATION_EVALUATION_RESULTS.json`
- **Regression Suite:**
  - `tests/test_batch1381_1420_adaptation.py` (5 tests passing)
  - Total repository regression pass rate: 47 / 47 tests passing.

---

### 3. Recommended Next Experiment (Batches 1421+)

Now that `ChakrMicro` has proven it can learn technical domain structure without overfitting or catastrophic forgetting:
- Evaluate whether the adapted checkpoint can be used for **constrained generative answer synthesis** (e.g. generating fluent 1-sentence answers conditioned on BM25 retrieved spans), replacing raw substring extraction while preserving 100% citation fidelity.
