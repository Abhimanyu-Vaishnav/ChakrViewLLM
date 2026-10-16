# CLASSIFIER REPRODUCTION RESULTS
## Batches 1741–1780: Independent Multi-Seed Reproduction & Representation Audit

### 1. Executive Summary

In Batches 1701–1740, an initial experiment with fine-tuning a direct intent classification head (`ChakrMicroIntentClassifier`, Variant B) reported **77.78%** accuracy on a 180-task held-out evaluation set.

In Work Package 3 of Batches 1741–1780, we conducted an independent, multi-seed reproduction across seeds **42, 123, and 999** trained on strictly partitioned family data (240 training samples, 60 validation samples) and evaluated on the new frozen **300-task semantic generalization benchmark** (`test_held_out_300.json`, 50 samples per intent across 6 canonical intents).

#### Headline Findings:
1. **Multi-Seed Variance**: On the genuinely out-of-family 300-task benchmark, Variant B achieves:
   - **Mean Intent Accuracy**: **55.89% ± 1.84%** (Seed 42: 57.67%, Seed 123: 54.00%, Seed 999: 56.00%).
   - **Mean Macro-F1**: **0.5280 ± 0.0195** (Seed 42: 0.5403, Seed 123: 0.5055, Seed 999: 0.5383).
2. **Comparison with Baseline**: The simple keyword heuristic achieves **55.33%** intent accuracy (Macro-F1: 0.5185). The fine-tuned neural classifier yields a marginal +0.56% accuracy improvement over a simple keyword baseline on unseen semantic families.
3. **Frozen Backbone (Variant A)**: Freezing the backbone and training only the MLP head achieved **55.33%** accuracy (Macro-F1: 0.4963), confirming that fine-tuning internal transformer layers provided minimal generalization benefit (+0.56% accuracy) when semantic families are strictly decoupled.

---

### 2. Multi-Seed Training Configuration & Checkpoints

All training runs utilized:
- **Base Checkpoint**: Canonical ChakrMicro Step 10,000 (`training_checkpoint_step_10000.pt`)
  - **SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified bit-exact).
- **Optimization**: AdamW, lr=1e-4 (head) / 2e-5 (backbone for Variant B), weight_decay=0.01, 100 steps, batch_size=8.
- **Hardware Platform**: Intel x86_64 CPU (PyTorch 2.4.0, zero external APIs).

| Checkpoint Name | Architecture | Random Seed | Training Steps | Training Wall Time | Checkpoint SHA-256 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `variant_a_frozen_seed_42.pt` | Frozen Backbone + MLP | 42 | 100 | 3.03s | `a9839aeaacc3eee14198acc4267fc76a7c4423473eb0bdaa672708cce187dfa1` |
| `variant_b_seed_42.pt` | Full Fine-Tuning + MLP | 42 | 100 | 5.26s | `fa5ca2ab1823c9b51577068c52aed5302575cee2a839049e0cd9c00d54e07e72` |
| `variant_b_seed_123.pt` | Full Fine-Tuning + MLP | 123 | 100 | 5.53s | `371d5e705cc0a6050d87a39af8f977a74bbb93434086e4036156a3baea2d9759` |
| `variant_b_seed_999.pt` | Full Fine-Tuning + MLP | 999 | 100 | 5.63s | `29b691c3a8b316ef178d41be35301fed14eb7f20fe3a7b9500602a5ee513e60a` |

---

### 3. Training & Validation Trajectories

| Model / Seed | Step 20 Loss (Val Acc) | Step 40 Loss (Val Acc) | Step 60 Loss (Val Acc) | Step 80 Loss (Val Acc) | Step 100 Loss (Val Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Variant A (Seed 42)** | 0.9909 (30.0%) | 0.3641 (33.3%) | 0.4686 (33.3%) | 0.4112 (33.3%) | 0.0690 (33.3%) |
| **Variant B (Seed 42)** | 1.2474 (28.3%) | 0.5339 (38.3%) | 0.4658 (40.0%) | 0.2680 (41.7%) | 0.1020 (43.3%) |
| **Variant B (Seed 123)** | 0.9800 (46.7%) | 0.5713 (38.3%) | 0.3093 (38.3%) | 0.1420 (53.3%) | 0.1210 (60.0%) |
| **Variant B (Seed 999)** | 1.0617 (50.0%) | 0.6201 (50.0%) | 0.3505 (50.0%) | 0.1954 (50.0%) | 0.0751 (50.0%) |

**Observation**: All runs rapidly overfit the 240 training samples by Step 80-100 (train loss drops below 0.12), while validation accuracy plateaus between 33% and 60% on the distinct validation families.

---

### 4. Per-Class Performance on the 300-Task Benchmark

#### Variant B (Seed 42 - Representative Best Seed):
- **Intent Accuracy**: 57.67% (173 / 300)
- **Macro-F1**: 0.5403
- **Argument Accuracy**: 52.00% (156 / 300)

| Intent Label | Precision | Recall | F1-Score | Support | Dominant Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `CALCULATE` | 89.47% | 68.00% | 0.7727 | 50 | Conversational arithmetic routed to KNOWLEDGE_QUERY |
| `VERIFY_CLAIM` | 66.00% | 66.00% | 0.6600 | 50 | Numeric statements confused with CALCULATE |
| `KNOWLEDGE_QUERY`| 55.56% | 40.00% | 0.4651 | 50 | Under-specified queries confused with OUT_OF_SCOPE |
| `CODE_TASK` | 53.19% | 100.00% | 0.6944 | 50 | Over-predicting CODE_TASK whenever technical tokens appear |
| `CLARIFY` | 80.00% | 8.00% | 0.1455 | 50 | Severe under-detection; missing arguments ignored |
| `OUT_OF_SCOPE` | 41.56% | 64.00% | 0.5039 | 50 | Adversarial injections falsely classified as CODE_TASK |

---

### 5. Why Did the Batch 1701 Model Claim 77.78% Accuracy?

Our reproduction reveals that the 77.78% accuracy in Batch 1701 was an artifact of:
1. **Semantic Family Proximity**: In Batch 1701, the held-out test set reused prompt syntaxes and entity domains present in the training set (e.g. standard arithmetic formulas like `add X and Y`).
2. **Lack of Adversarial Distractors**: Batch 1701's test set contained few hard negatives. When exposed to 20 hard negatives (conversational mentions of math/code) and 10 prompt injections in the 300-task benchmark, Variant B's intent accuracy drops to **55.89%**.
3. **Severe Vulnerability to `CLARIFY`**: In Batch 1701, incomplete requests were scarce. In our benchmark, Variant B achieves only **8% recall** on `CLARIFY` (missing arguments), greedily routing incomplete requests to tools where execution fails.

### 6. Reproduction Verdict

The reported 77.78% accuracy **does not generalize** to out-of-family, adversarial, or hard-negative distributions. On a rigorous, leakage-free 300-task benchmark, the fine-tuned classifier achieves **55.89% ± 1.84%** accuracy, which fails the predeclared 70% acceptance threshold.
