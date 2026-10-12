# Batch 78 — Generalization Audit and Next-Step Decision

## 1. Post-Hoc Generalization Audit Questions

### 1.1 Was the starting checkpoint truly identical across conditions?
**Confirmed.** Both Condition A and Condition B loaded `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`. Its SHA-256 hash was verified prior to training as `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.

### 1.2 Were token budgets and training configurations sufficiently matched?
**Confirmed.** Both conditions trained for 100 steps with batch size 4 and sequence length 64 ($100 \times 4 \times 64 = 25,600$ trained tokens). The total tokenized corpus size was 56,231 tokens for both Condition A and Condition B. Both runs used the same learning rate ($5 \times 10^{-4}$), AdamW optimizer settings, and seed 42.

### 1.3 Did the model learn a non-constant mapping?
**Negative.** The greedy outputs from Condition B collapsed entirely to the single digit `6`. 100% of all generated responses across training (30/30), validation (10/10), held-out test (10/10), and symmetry diagnostic (11/11) were `6`.

### 1.4 Did gains persist on unseen operand pairs?
**Negative.** While exact-match was recorded as 2/10 on the held-out test set, this was exclusively due to the fact that two held-out test pairs (`0 + 6` and `1 + 5`) happen to have a sum of 6. For all other pairs (e.g. `0 + 2`, `3 + 0`, `4 + 3`), the model predicted `6` instead of the correct answers.

### 1.5 Did performance change across different prompt templates?
In synthetic pretraining, the model observed 8 different templates. This was sufficient to achieve template token cross-entropy reduction (from 6.36 to 0.0016) and 100% EOS generation, but did not resolve input-operand conditioning at the answer token.

### 1.6 Did EOS completion improve independently of answer correctness?
**Confirmed.** EOS completion improved from 0.0% (in Baseline and Condition A) to 100.0% in Condition B. This confirms the finding from Batches 67–74 that template and delimiter mastery decoupled from functional mathematical reasoning.

### 1.7 Did training loss fall while validation performance stayed flat or worsened?
**Confirmed.** Training loss on synthetic pretraining dropped from 5.0390 to 0.3734. Answer-token loss dropped from 10.08 to 2.31. However, true mathematical generalization remained at 0% (constant attractor).

### 1.8 Did the result reproduce after fresh checkpoint reload?
**Confirmed.** A separate, clean evaluation from `condition_b_arithmetic_curriculum_step_100.pt` yielded exact identical answer loss (2.1251) and predictions (`6` for all test items).

### 1.9 Is apparent improvement explained by answer-frequency imbalance or leakage?
**Confirmed.** The apparent 20% validation and test exact-match score is entirely explained by the presence of the digit 6 in those test cases, combined with uniform modal prediction of `6`. Zero split leakage occurred (verified by audit of `train_pairs \cap test_pairs = \emptyset`).

---

## 2. Scientific Synthesis

| Phenomenon | Sprint 71–74 (Pure SFT) | Sprint 75–78 (Synthetic Pretraining) | Difference & Insight |
|:---|:---|:---|:---|
| **Modal Attractor** | Predicted `7` everywhere | Predicted `6` everywhere | Pretraining shifts the attractor to `6` (the median of the multi-template data). Neither setup conditioned on $(a, b)$. |
| **Answer-Token Loss** | ~2.20 | ~2.12 | Synthetic pretraining slightly improves answer cross-entropy compared to plain SFT. |
| **Template Loss** | ~0.005 | ~0.0016 | Synthetic pretraining provides even stronger delimiter / EOS structure. |
| **Held-Out Reasoning** | 0% (1/10 was modal hit on 7) | 0% (2/10 was modal hit on 6) | Scaling synthetic training tokens without architectural/task changes does not induce arithmetic operator learning. |

ChakrMicro (6 layers, $d_{model}=192$, 3.44M params) possesses the capacity to fit small datasets (as verified by 100% on the 3-example overfit control), but when trained with standard causal autoregressive loss on uniform batches, the loss landscape favors predicting the central tendency of single digits over computing the exact non-linear attention routing necessary for addition.

---

## 3. Recommended Next Steps

1. **Explicit Arithmetic Scratchpad / Chain-of-Thought Representations**: Test whether intermediate tokens (e.g. `Add 3 + 4: 3 ... 4 ... 7`) provide the autoregressive depth needed for low-parameter transformers to compute operations.
2. **Balanced-Sum Batching**: Construct a training distribution where every sum from 0 to 9 has an exact uniform frequency in each batch, neutralizing the loss advantage of a single modal attractor.
3. **Number Embedding Probing**: Evaluate whether token embeddings for digits 0–9 in ChakrMicro cluster linearly by magnitude or if they remain scattered in semantic space.
