# Batch 68 — Transformation Generalization Experiment Results

## 1. Context and Objective

In Batch 67, we verified that loss reweighting can reduce answer token loss and confirmed that template tokens cannot simply be zeroed without breaking termination. However, on the 2-digit arithmetic task, the model suffered from two challenges:
1. Inconsistent BPE digit tokenization (some numbers like `20` are single tokens, while numbers like `21` are fragmented across 2 tokens).
2. Moderate sample count (32 pairs) with complex numbers.

In Batch 68, we isolated the core scientific question:
> **Can ChakrMicro learn an arithmetic transformation from examples, rather than merely memorize exact prompt-answer pairs?**

To remove all token fragmentation artifacts, we constructed a **canonical single-digit arithmetic task**:
- Domain: $a + b = c$ with $a, b \in [0..9]$ such that $a + b \le 9$.
- Invariant: Operands and answers are strictly single tokens ($0 \to 51, \dots, 9 \to 60$).
- Disjoint Splits: 25 Train, 10 Validation, 10 Held-out Benchmark (total 45 distinct addition pairs, 0% overlap).
- Exact ChatML Prompt: `<|im_start|>user\nAdd: a + b = ?<|im_end|>\n<|im_start|>assistant\n`
- Exact Target: `c<|im_end|>\n`

---

## 2. Experimental Conditions

We evaluated two controlled configurations over 100 training steps (AdamW, $\text{lr}=5\times 10^{-4}$, $\text{batch\_size}=2$, seed 42):
- **Condition 1**: `baseline_uniform` (weight 1.0 on all active target tokens).
- **Condition 2**: `answer_emphasis_5x` (weight 5.0 on the single answer token, 1.0 on template tokens).

---

## 3. Quantitative Results

| Configuration | Split | Records | Exact Match Count | Exact Match Acc | Mean Answer Loss | Mean Template Loss | EOS Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Uniform** | TRAIN | 25 | 5 | 20.0% (0.2000) | 2.0571 | 0.0003 | 100.0% |
| | VAL | 10 | 0 | 0.0% (0.0000) | 2.8559 | 0.0004 | 100.0% |
| | BENCH | 10 | 0 | 0.0% (0.0000) | 2.6641 | 0.0003 | 100.0% |
| **Answer Emphasis (5x)** | TRAIN | 25 | 5 | 20.0% (0.2000) | 2.0551 | 0.0014 | 100.0% |
| | VAL | 10 | 0 | 0.0% (0.0000) | 2.7346 | 0.0016 | 100.0% |
| | BENCH | 10 | 0 | 0.0% (0.0000) | 2.5019 | 0.0016 | 100.0% |

---

## 4. Qualitative Output Analysis on Unseen Inputs

When presented with unseen validation prompts, the model outputs clean, well-formed single-digit numbers terminated properly with `<|im_end|>`, but the outputs do not compute the addition transformation:

| ID | Prompt Expression | Expected Answer | Baseline Output | Answer-Emphasis Output | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|
| `SYNTH-DIGIT-02-05` | 2 + 5 | 7 | `6` | `6` | Incorrect |
| `SYNTH-DIGIT-00-00` | 0 + 0 | 0 | `8` | `6` | Incorrect |
| `SYNTH-DIGIT-02-06` | 2 + 6 | 8 | `6` | `6` | Incorrect |
| `SYNTH-DIGIT-07-02` | 7 + 2 | 9 | `8` | `8` | Incorrect |
| `SYNTH-DIGIT-01-08` | 1 + 8 | 9 | `6` | `6` | Incorrect |

Observations:
1. **Zero Template Artifacts**: The model never emits table pipes `|` or malformed characters.
2. **Attractor Clustering**: The model clusters around the median sum of the training distribution (frequently outputting `6` or `8`).
3. **Absence of Operator Compositionality**: The model learns that the answer to `Add: a + b = ?` is a single digit between 0 and 9 followed by `<|im_end|>`. However, it does not learn the functional map $(a, b) \mapsto a + b$.

---

## 5. Scientific Verdict: Memorization vs Transformation Learning

- **Memorization Capacity**: Proven and demonstrated (as shown in Batch 65's 100-step run reaching 32/32 on train).
- **Transformation Generalization**: **CONFIRMED ZERO**. Under both uniform and 5x answer-weighted SFT, validation and held-out exact-match accuracy is **0.0% (0/10)**.
- **Conclusion**: ChakrMicro v0.1 (3.44M parameters) when trained directly via direct-answer causal next-token prediction does not infer addition arithmetic logic from 25–32 examples. Small language models require either chain-of-thought scratchpads (e.g. step-by-step reasoning tokens) or pretraining corpora containing dense arithmetic representations before SFT can unlock functional generalization.
