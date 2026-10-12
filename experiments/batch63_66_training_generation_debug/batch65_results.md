# Batch 65 — Controlled Recovery Experiment Results

## 1. Context and Objective

In Batch 63, two primary issues were isolated in the training-to-generation pipeline:
1. **Delimiter Stripping Discrepancy**: Calling `.strip()` on ChatML prompts stripped the trailing newline `\n`. During training, the prompt ended at `ant`, conditioning the first target response token on `ant`. At generation time, the model received a raw prompt ending with `\n`, an untrained boundary token that defaulted to the pretraining Wikipedia table attractor `|` (`|277`).
2. **Template Dominance vs Token Underfitting**: In each response (`"23<|im_end|>\n"`), 88% of target tokens are ChatML tags and newlines. The model quickly fits the template while underfitting the arithmetic answer tokens.

Batch 65 executes controlled recovery experiments under:
- **Canonical Architecture & Checkpoint**: Immutable baseline step 10,000 checkpoint (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Exact Delimiter Alignment**: Prompt and response delimiters are preserved without stripping (`strip_prompt_delimiters=False`), guaranteeing that the training sequence and greedy generation prompt match bit-for-bit at the assistant turn boundary.
- **Data Integrity & Trust Gates**: Synthetic arithmetic dataset from Batch 60 with disjoint Train (32), Validation (8), and Held-Out Benchmark (8) splits.

---

## 2. Experiment 1: Standard 50-Step SFT Engine Recovery Run

- **Configuration**:
  - Script: `experiments/batch63_66_training_generation_debug/learning_experiment/train_recovery.py`
  - Optimizer: AdamW (`lr=1e-4`, `min_lr=1e-5`, `weight_decay=0.01`, `grad_clip=1.0`)
  - Steps: 50, Batch Size: 2 (minibatch SGD)
  - Train dataset SHA-256: `89299ff5e2d81b143c68a6b6c9aae858b69a486f699c6c90c9855600bfbbd4c5`
  - Validation dataset SHA-256: `f905c1310e2c827a2e89d931507af389e001f943fcb82816891ffe020523b354`
  - Benchmark dataset SHA-256: `8a0ccd7ba784032c612b6b0d145de7abce204b0b15772d378f0c79a534d2f030`
  - Checkpoint: `experiments/batch63_66_training_generation_debug/learning_experiment/run_50steps/sft_checkpoint_step_50.pt`

### Quantitative Results (50 Steps)

| Split | Total Records | Exact Match Count | Exact Match Accuracy | Mean Token Loss | EOS Rate | Repetition Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | 32 | 3 | 9.38% (0.0938) | 0.3504 | 100.0% | 0.0% |
| **VALIDATION** | 8 | 0 | 0.00% (0.0000) | 0.4145 | 100.0% | 0.0% |
| **HELD_OUT_BENCHMARK** | 8 | 1 | 12.50% (0.1250) | 0.3394 | 100.0% | 0.0% |

### Key Observations (50 Steps)
1. **Elimination of `|277` table attractor**:
   - Because generation prompts now end with `\n` matching the training condition, the model NO LONGER emits `|277`.
   - Instead, the model emits valid numeric text: `27<|im_end|>`.
2. **Modal Collapse to '27'**:
   - With batch size 2 and 50 steps at `lr=1e-4`, the model learns the ChatML closing tag and the mode of the training dataset's arithmetic answers (prefix `'2'`, suffix `'7'`).
   - The token loss reaches ~0.35 solely from template tokens and common prefix digits.

---

## 3. Experiment 2: Full-Batch Capacity & Memorization vs Generalization Test

To definitively determine whether ChakrMicro v0.1 has sufficient capacity to learn the 32 training examples or whether optimization/loss weighting was the barrier, we conducted a full-batch training test on all 32 examples for 100 steps (`lr=5e-4`).

### Quantitative Results (100 Steps Full-Batch)

| Metric / Split | TRAIN (32 Examples) | VALIDATION (8 Examples) |
|:---|:---:|:---:|
| Initial Loss | 6.5695 | 6.5695 |
| Step 25 Mean Loss | 0.2711 | - |
| Step 50 Mean Loss | 0.1591 | - |
| Step 75 Mean Loss | 0.0173 | - |
| Step 100 Mean Loss | **0.0029** | - |
| **Exact Match Accuracy** | **32 / 32 (100.0%)** | **0 / 8 (0.0%)** |
| EOS Completion Rate | 100.0% | 100.0% |

### Detailed Validation Outputs (Unseen Examples)

| ID | Prompt Expression | Expected Answer | Model Generated | Exact Match |
|:---|:---|:---:|:---:|:---:|
| `SYNTH-ADD-13-05` | 13 + 5 | 18 | `14` | False |
| `SYNTH-ADD-11-10` | 11 + 10 | 21 | `11` | False |
| `SYNTH-ADD-03-16` | 3 + 16 | 19 | `17` | False |
| `SYNTH-ADD-02-15` | 2 + 15 | 17 | `21` | False |
| `SYNTH-ADD-18-15` | 18 + 15 | 33 | `26` | False |
| `SYNTH-ADD-15-01` | 15 + 1 | 16 | `27` | False |
| `SYNTH-ADD-03-20` | 3 + 20 | 23 | `17` | False |
| `SYNTH-ADD-13-04` | 13 + 4 | 17 | `22` | False |

---

## 4. Root-Cause Verdict and Evidence Attribution

| Hypothesis / Phenomenon | Status | Evidence |
|:---|:---:|:---|
| **Prompt Delimiter Mismatch (`.strip()`)** | **CONFIRMED (Fixed)** | When `.strip()` dropped `\n`, the model evaluated on `\n` generated `|277`. When exact delimiters were preserved (`strip_prompt_delimiters=False`), generation cleanly outputs numbers and `<|im_end|>`. |
| **Teacher-Forced Logit Alignment** | **CONFIRMED (Correct)** | Position $|P|$ predicts target $R_0$. Harness verifies greedy decode step 0 exactly equals $\text{argmax}(\text{logits}[|P|])$. |
| **Model Capacity for Memorization** | **CONFIRMED (Sufficient)** | ChakrMicro (3.44M params) achieves 32/32 (100%) exact match on training examples when loss drops to 0.0029. |
| **Arithmetic Transformation Generalization** | **CONFIRMED (Zero)** | Despite 100% memorization on train, validation exact match is 0/8. The model memorizes string associations rather than learning the addition operator. |
| **EOS Detection vs Answer Accuracy** | **CONFIRMED (Independent)** | Model achieves 100% EOS completion even when exact match is 0%. EOS rate must never be conflated with capability. |
| **Non-Uniform Tokenizer Granularity** | **SUPPORTED** | Two-digit numbers are inconsistently tokenized (e.g. `20` is 1 token, `21` is 2 tokens), increasing difficulty for small causal models. |
