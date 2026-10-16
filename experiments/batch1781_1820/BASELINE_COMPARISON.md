# BASELINE COMPARISON REPORT
## Batches 1781–1820: Comparative Analysis on Frozen 300-Task Benchmark

### 1. Executive Summary

To determine whether ChakrView's neural representations provide a measurable advantage over simple statistical and deterministic models on intent classification, we evaluated five distinct architectures on the identical frozen **300-task semantic generalization benchmark** (`test_held_out_300.json`):

1. **Majority Class Baseline** (Constant predictor)
2. **Deterministic Keyword Baseline** (Surface token heuristic)
3. **Pure-Python TF-IDF + Multinomial Logistic Regression** (Standard unigram bag-of-words)
4. **Frozen ChakrMicro Backbone + Linear Head (Variant A)** (Linear probing on canonical weights)
5. **Fine-Tuned ChakrMicro Classifier (Variant B)** (Full backbone fine-tuning)

---

### 2. Comprehensive Baseline Comparison Table (300 Held-Out Tasks)

| Model Architecture | Parameter Count | Training Strategy | Intent Accuracy | Macro-F1 | Brier Score | Hard-Neg FP Rate | Adversarial FP Rate | Median CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0 | None (Constant) | 16.67% (50/300) | 0.0476 | 0.1389 | 0.00% (0/20) | 0.00% (0/10) | 0.002 ms |
| **Keyword Baseline** | 0 | Heuristic Regex | 55.33% (166/300) | 0.5185 | 0.2500 | 55.00% (11/20) | 80.00% (8/10) | 0.006 ms |
| **TF-IDF + Logistic Classifier** | ~3,200 | SGD on Bag-of-Words | 52.00% (156/300) | 0.4812 | 0.7548 | 65.00% (13/20) | 80.00% (8/10) | 0.024 ms |
| **Variant A: Frozen Backbone** | ~12.5M | Linear Probe Head | 55.33% (166/300) | 0.4963 | 0.5927 | 0.00% (0/20) | 100.00% (10/10) | 4.419 ms |
| **Variant B: Fine-Tuned Backbone** | ~12.5M | Full Backprop (3 Seeds) | **55.89% ± 1.84%** | **0.5280 ± 0.0195** | **0.5332** | 13.33% ± 12.58% | 100.00% (10/10) | 4.172 ms |

---

### 3. Detailed Comparative Findings

1. **Neural vs. TF-IDF (+3.89% Gain)**:
   - The fine-tuned ChakrMicro model (55.89%) outperforms the pure TF-IDF logistic baseline (52.00%) by only **3.89%**.
   - While the transformer captures word order and conversational syntax better than unigram TF-IDF (especially on conversational math like *"work out what 96 divided by 8 is"*), this modest margin confirms that the neural model functions primarily via surface lexical associations.
2. **Neural vs. Keyword Heuristic (+0.56% Gain)**:
   - The fine-tuned transformer (55.89%) barely surpasses the simple keyword baseline (55.33%) by **+0.56%**.
   - On out-of-family syntactic variations, the 12.5M parameter neural network provides virtually zero measurable advantage over a tiny 50-line deterministic regular expression rule-set.
3. **Representation Bottleneck (Frozen vs Fine-Tuned)**:
   - Variant A (frozen backbone) achieved **55.33%**, identical to the keyword baseline.
   - Variant B (fine-tuned) achieved **55.89%**, an improvement of merely **+0.56%**.
   - This proves that updating internal transformer weights did not create new compositional reasoning pathways; it merely adjusted linear projection boundaries on top of saturated pre-training representations.
