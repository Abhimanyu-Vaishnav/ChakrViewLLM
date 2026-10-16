# Milestone Decision and Handoff

## Batches 1301–1340 -> Batches 1341+

### 1. Decision Contract and Protocol Handoff

Based on the empirical evidence gathered during Batches 1301–1340, the engineering direction is formally decided:

1. **Retain Conservative Refutation Gate as Production Default:** The attribute- and action-aligned refutation gate implemented in `chakrview/runtime/local_knowledge_assistant.py` is promoted to default behavior. It preserves 98.6% accuracy on frozen technical queries, eliminates spurious refutations on valid "why" queries, and maintains zero external heavy NLP dependencies.
2. **Cease Rule-Based Regex Expansion:** Engineering resources will not be expended on writing additional ad-hoc regular expressions for units, durations, or paraphrasing. Such heuristics have reached diminishing returns.
3. **Execute Held-Out Perplexity Benchmark on ChakrMicro (Next Sprint):** The next milestone must execute the evaluation-first experiment outlined in `NEXT_CORE_EXPERIMENT.md` to measure the true language modeling and next-token prediction capability of the 10.8M parameter neural core on unseen technical text.
4. **Preserve Checkpoint and Branch Safety:** Maintain the canonical checkpoint `training_checkpoint_step_10000.pt` intact and continue on `dev/v0.6-v0.7-validation`.

---

### 2. Checkpoint and Artifact Lineage

- **Canonical Model Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Adversarial Benchmark Dataset:** `experiments/batch1301_1340/data/adversarial_decision_100.json`  
  - SHA-256: `c87654b1ade2aeb0619f1c20e644f582e7a1afb7f4dd1eb154086ec1442308a7`
- **Evaluation Runner:** `experiments/batch1301_1340/run_adversarial_evaluation.py`
- **Machine-Readable Results:** `experiments/batch1301_1340/results/ADVERSARIAL_EVALUATION_RESULTS.json`
- **New Regression Test Suite:** `tests/test_batch1301_1340_refutation_safety.py` (8 passing tests)
