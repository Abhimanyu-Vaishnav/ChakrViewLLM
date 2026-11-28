# NR-057 CONTROLLED TRAINING EXPERIMENT PLAN (FOR NR-058)

**Date:** November 28, 2026  
**Mission:** NR-057 Clean Corpus Expansion & Training Readiness Engineering  
**Artifact ID:** `NR057_CONTROLLED_TRAINING_EXPERIMENT_PLAN`  

---

## 1. Experiment Overview & Scientific Objectives

With `Corpus V5` constructed (3.565M verified clean tokens), the tokenizer validated (0.00% UNK), and the hardware profiled, the next mission (**NR-058**) will execute controlled neural pretraining to test the following scientific hypotheses:

1. **Hypothesis 1 (Data Volume):** Expanding pretraining from 0.62M tokens to 3.56M+ clean tokens will systematically lower out-of-domain validation cross-entropy loss and perplexity.
2. **Hypothesis 2 (Capacity Scaling):** The 38M parameter architecture will achieve lower validation loss and exhibit faster perplexity convergence than the 17M architecture when trained on identical clean tokens.
3. **Hypothesis 3 (Cognitive Generalization):** Uncontaminated exposure to Python algorithmic logic, STEM prose, and formal mathematics will produce measurable gains ($>0.0\%$) on Benchmark V3 reasoning tasks.

---

## 2. Experimental Design Matrix

| Experiment ID | Architecture Family | Hidden Dim / Layers | Batch Size | Seq Len | Optimizer & LR Schedule | Precision | Token Budget | Target Epochs | Random Seeds |
|---|---|---|---|---|---|---|---|---|---|
| **EXP-58-17M-A** | ChakrGenesis17M | $d=384, L=10$ | 16 | 512 | AdamW ($\text{lr}=3\times 10^{-4}$, Cosine decay, 500 warmup) | fp16 AMP | 3.18M (1 pass) | 1.0 | 42 |
| **EXP-58-17M-B** | ChakrGenesis17M | $d=384, L=10$ | 16 | 512 | AdamW ($\text{lr}=3\times 10^{-4}$, Cosine decay, 500 warmup) | fp16 AMP | 9.56M (3 passes) | 3.0 | 42, 1337 |
| **EXP-58-38M-A** | ChakrGenesis38M | $d=512, L=12$ | 16 | 512 | AdamW ($\text{lr}=2.5\times 10^{-4}$, Cosine decay, 500 warmup) | fp16 AMP | 3.18M (1 pass) | 1.0 | 42 |
| **EXP-58-38M-B** | ChakrGenesis38M | $d=512, L=12$ | 16 | 512 | AdamW ($\text{lr}=2.5\times 10^{-4}$, Cosine decay, 500 warmup) | fp16 AMP | 9.56M (3 passes) | 3.0 | 42, 1337 |

---

## 3. Checkpointing & Evaluation Schedule

1. **Intermediate Checkpoint Interval:** Checkpoints will be saved automatically every **200 steps** and at each epoch boundary.
2. **Validation Monitoring:** Validation perplexity on `data/corpus_v5/corpus_v5_val.jsonl` (221,282 tokens) will be evaluated every 100 steps.
3. **Post-Training Sealed Benchmark:** Upon training completion, models will be evaluated blindly on:
   - `data/corpus_v5/corpus_v5_test.jsonl` (heldout test perplexity).
   - `data/evaluation/benchmark_v3/benchmark_v3_300.json` (uncontaminated 300-item capability benchmark).
4. **Stopping Criteria:** If validation loss diverges or fails to improve for 3 consecutive evaluation cycles, training terminates early to preserve compute.
