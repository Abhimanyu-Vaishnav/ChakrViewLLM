# NR-057-R1 CONTROLLED PRETRAINING EXPERIMENT DESIGN (NR-058)

**Date:** November 28, 2026  
**Mission:** NR-057-R1 Corpus Expansion Completion & Training Gate Revalidation  
**Artifact ID:** `NR057R1_NR058_TRAINING_DESIGN`  

---

## 1. Executive Summary & Scientific Hypotheses

Mission **NR-058** will execute the first controlled neural pretraining study using `Corpus V6` across both verified model families (`ChakrGenesis17M` and `ChakrGenesis38M`).

### Hypotheses to be Empirically Tested:
1. **Hypothesis 1 (Data Expansion & Perplexity Convergence):** Pretraining on Corpus V6 (4.57M clean multi-domain tokens) will achieve systematically lower validation cross-entropy loss and lower perplexity than pretraining on the starved Corpus V4 (624K tokens).
2. **Hypothesis 2 (Capacity Scaling Laws):** When exposed to identical tokens, batch size, and learning rate schedule, the 38M parameter architecture will demonstrate a steeper loss decay gradient and reach lower asymptotic cross-entropy validation loss than the 17M architecture.
3. **Hypothesis 3 (Linguistic Generalization):** Rebalancing Devanagari Hindi and Hinglish in Corpus V6 will yield measurable multi-lingual token prediction improvements without syntactic disruption of English and Python code.

---

## 2. Experimental Configuration Matrix

| Experiment ID | Architecture Family | Parameters | Context Length | Batch Size | Optimizer | Peak LR & Schedule | Token Budget | Target Epochs | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| **EXP-58-17M-BASE** | ChakrGenesis17M | 17,063,808 | 512 | 16 | AdamW | $3.0\times 10^{-4}$, Cosine decay, 500 warmup | 4.16M (1.0 epoch) | 1.0 | 42 |
| **EXP-58-17M-EXT** | ChakrGenesis17M | 17,063,808 | 512 | 16 | AdamW | $3.0\times 10^{-4}$, Cosine decay, 500 warmup | 12.49M (3.0 epochs) | 3.0 | 42, 1337, 2026 |
| **EXP-58-38M-BASE** | ChakrGenesis38M | 37,499,392 | 512 | 16 | AdamW | $2.5\times 10^{-4}$, Cosine decay, 500 warmup | 4.16M (1.0 epoch) | 1.0 | 42 |
| **EXP-58-38M-EXT** | ChakrGenesis38M | 37,499,392 | 512 | 16 | AdamW | $2.5\times 10^{-4}$, Cosine decay, 500 warmup | 12.49M (3.0 epochs) | 3.0 | 42, 1337, 2026 |

---

## 3. Evaluation Schedule & Stopping Criteria

1. **Validation Cross-Entropy Monitoring:** Evaluated every **100 optimization steps** on `data/corpus_v6/splits/corpus_v6_val.jsonl` (191,686 tokens).
2. **Domain-Specific Perplexity Tracking:** During evaluation, validation loss is tracked separately across:
   - English Wikipedia STEM/Humanities
   - Hindi Devanagari Expository
   - Python Standard Library Code
   - Formal Mathematics & Logic
3. **Sealed Benchmark Evaluation:** Post-training blind evaluation on:
   - `data/corpus_v6/splits/corpus_v6_test.jsonl` (Heldout perplexity).
   - `data/evaluation/benchmark_v3/benchmark_v3_300.json` (Uncontaminated 300-prompt capability test).
4. **Early Stopping Criteria:** Training will terminate if validation loss exhibits monotonic divergence across 3 consecutive evaluation checkpoints (>300 steps).
