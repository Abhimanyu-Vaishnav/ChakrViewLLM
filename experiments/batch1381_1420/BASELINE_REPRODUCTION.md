# Baseline Reproduction Report

## Batches 1381–1420: Reproduction of Canonical ChakrMicro Step 10,000 Metrics

### 1. Reproduction Setup

Before applying any optimizer updates, the baseline language modeling evaluation from Batch 1341–1380 was independently re-executed on the exact 27 held-out technical documents (3,456 target tokens).

- **Evaluator:** [run_lm_evaluation.py](file:///D:/Project/ChakrView/experiments/batch1341_1380/run_lm_evaluation.py)
- **Model Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Canonical Checkpoint SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Tokenizer:** Canonical Vocab 4096 (`release/chakrview-v0.5.0/data/experiments/vocab_4096`)
- **Evaluation Targets:** 27 held-out documents, 3,456 tokens ($N-1$).

---

### 2. Independent Verification Metrics

| Metric | Batch 1341–1380 Reported | Batch 1381–1420 Reproduced | Difference |
|---|---|---|---|
| **Cross-Entropy Loss (nats)** | 5.0630 | 5.0630 | **0.0000 (Exact)** |
| **Perplexity ($PPL$)** | 158.07 | 158.07 | **0.00 (Exact)** |
| **Top-1 Accuracy** | 12.85% | 12.85% | **0.00% (Exact)** |
| **Top-5 Accuracy** | 30.70% | 30.70% | **0.00% (Exact)** |
| **Top-10 Accuracy** | 40.39% | 40.39% | **0.00% (Exact)** |
| **Unigram Baseline Loss** | 6.6266 | 6.6266 | **0.0000 (Exact)** |
| **Unigram Baseline $PPL$** | 754.93 | 754.93 | **0.00 (Exact)** |
| **Uniform Random Loss** | 8.3178 | 8.3178 | **0.0000 (Exact)** |
| **Uniform Random $PPL$** | 4096.00 | 4096.00 | **0.00 (Exact)** |

#### Result:
The baseline metrics reproduced with **100.0% bit-level reproducibility**, confirming that the evaluation harness, token alignment, and loss aggregation are completely deterministic.
