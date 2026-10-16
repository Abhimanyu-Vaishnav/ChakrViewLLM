# PREVIOUS ROUTER AUDIT
## Batches 1701–1740 Detailed Audit of Batch 1661–1700 Neural Router Experiment

### 1. Scope & Objective

This audit inspects the artifacts and scripts produced in Batches 1661–1700:
- `experiments/batch1661_1700/run_router_experiment.py`
- `experiments/batch1661_1700/results/ROUTER_EVALUATION_RESULTS.json`
- `experiments/batch1661_1700/data/DATASET_MANIFEST.json`
- `experiments/batch1661_1700/data/test_held_out.json`
- `experiments/batch1661_1700/data/train.jsonl`
- `experiments/batch1661_1700/data/val.jsonl`
- `experiments/batch1661_1700/BASELINE_COMPARISON.md`
- `experiments/batch1661_1700/GENERALIZATION_AND_SAFETY_RESULTS.md`

The goal is to verify the technical accuracy of prior findings, evaluate model parameter update scopes, examine latency accounting, and identify root causes behind baseline and candidate behaviors.

---

### 2. Systematic Audit Verification Checklist

| Audit Question | Verification Status | Empirical Evidence & Findings |
| :--- | :---: | :--- |
| **1. Model & Tokenizer Loading** | **VERIFIED** | Loaded byte-level BPE tokenizer (`vocab_size=4096`) and `ChakrMicro` with exact `ModelConfig(d_model=192, L=6, n_heads=6)`. |
| **2. Train / Val / Test Separation** | **VERIFIED** | Training used 150 prompt-completion pairs. Validation used 30 distinct pairs. Held-out test used 120 frozen tasks (20 per intent class). Zero exact prompt leakage. |
| **3. Parameter Update Scope** | **VERIFIED** | During the 100-step controlled training run, all 3,443,136 unique parameters (backbone + embedding/LM head) were updated using `AdamW(lr=5e-5)`. |
| **4. Reproducibility of Results** | **VERIFIED** | The training trajectory (loss dropping from 4.05 to 1.29) and evaluation metrics (19.17% neural accuracy, 14.17% JSON validity) reproduce deterministically with seed `42`. |
| **5. Deterministic Router Low Recall** | **AUDITED & EXPLAINED** | The Governed Deterministic Router scored 23.33% intent accuracy on the 120-task benchmark. This reflects its strict, fail-closed design: it requires exact prefix formulas (e.g. `"calculate:"` or `"evaluate:"`), intentionally refusing to guess when presented with conversational paraphrases like `"I need to find the sum of 512 and 1024"`. Its low recall is a feature of conservative safety, not broken code. |
| **6. Latency Accounting Integrity** | **AUDITED & NORMALIZED** | Latency measurements in `ROUTER_EVALUATION_RESULTS.json` correctly accounted for tokenization, forward pass, and decoding: <br>• Deterministic: 0.06 ms (regex match) <br>• Keyword: <0.01 ms (substring search) <br>• Neural: ~259 ms (40 autoregressive steps on CPU) |
| **7. Autoregressive Bottleneck** | **CONFIRMED** | 85.83% of neural outputs failed JSON syntax parsing due to trailing garbage, unclosed braces, or premature truncation. The autoregressive formulation was the primary bottleneck. |

---

### 3. Conclusion for Batch 1701–1740 Experimentation

The prior audit establishes that:
1. Autoregressive text generation for intent labels is inherently brittle and slow on small models.
2. A direct classification head over the final Transformer sequence representation is mathematically motivated to solve the 85.8% syntax invalidity problem.
3. However, whether the neural representations themselves have sufficient semantic separation across the 6 intents—or whether data scaling and backbone fine-tuning are required—must now be tested via rigorous multi-variant experiments.
