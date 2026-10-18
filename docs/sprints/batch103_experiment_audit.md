# Batch 103 — Opening Audit and Experiment Validity

**Sprint**: ChakrView Master Sprint Batches 103–106  
**Module**: Dynamic Argument Learning: Slot-Weighted SFT and Generalization Audit  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Opening Repository State and Invariants Audit

Before any modifications were made to the repository, the baseline git state and canonical artifacts were audited:

- **Git Branch**: `main`
- **Initial Commit**: `b4e0189a9f1ccf4466137db11d9647ec8d473d04` (Commit verified as current HEAD).
- **Working Tree**: No staged changes. User untracked files preserved.
- **Canonical Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - **Calculated SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - **Invariant**: Strictly matched expected hash; immutable base model preserved bit-for-bit.
- **Existing Test Suite**: Full test suite across 2,583 tests ran to completion with 100% pass rate (`2583 passed, 3 skipped in 811.03s`).

---

## 2. Audit of Existing Representation Probe (Batches 99–102)

The linear probing methodology introduced in Batches 99–102 (`experiments/batch99_102_representation_audit/representation_probe.py`) was audited for experimental validity:

1. **Operand Prompt Offset Derivation**:
   - In Batches 99–102, prompt tokens were parsed dynamically by matching the prompt structure `<|im_start|>user\nRequest: add numbers\nOperand A: {a}\nOperand B: {b}\nRequired action: calculator addition<|im_end|>\n<|im_start|>assistant\n`.
   - The probe examined hidden state representations at three distinct spatial locations:
     - `pos_a`: The token position corresponding to Operand A in the prompt.
     - `pos_b`: The token position corresponding to Operand B in the prompt.
     - `pos_gen`: The final prompt token position (`<|im_start|>assistant\n`), representing the generation decision boundary.
   - *Audit finding*: Operand positions in the prompt were derived from tokenized prompt offsets rather than guessed indices.

2. **Probe Feature Target Isolation**:
   - The probe used solely the activations from `model.transformer.h[layer]` at prompt positions.
   - The target tool-call JSON, completion tokens, and answer values were strictly masked and excluded from probe inputs.

3. **Digit Encoding and Tokenization Heterogeneity**:
   - The BPE tokenizer (`vocab_4096`) exhibits significant context-dependent tokenization for single digits:
     - When preceded by `'a": '`, digit `0` tokenizes as `[1290]`, `1` tokenizes as `[628]` (`'1, '`), while `9` tokenizes as `[60, 276]` (`'9'`, `', '`).
     - When preceded by `'b": '`, digit `0` tokenizes as `[1631]`, `1` tokenizes as `[585]`, and digits 3–9 tokenize as two tokens `[X, 128]` (`'{d}'`, `'}'`).
   - The probe in Batch 99 tested single-token digit labels in the prompt, but the generation target value tokens require explicit multi-token handling.

4. **Sample Size and Generalization Limitations**:
   - The evaluation dataset consists of:
     - `train`: 64 records
     - `val`: 18 records
     - `test`: 18 records
   - *Statistical caution*: With $n=18$, a 95% Clopper-Pearson confidence interval around an observed accuracy of 5.56% (1/18) is $[0.14\%, 27.29\%]$. Distinguishing a 1/18 match from random constant attractor collision requires verifying whether the output changes across varying inputs. A single test hit due to the constant pair `(1, 8)` being present in the test set cannot be interpreted as argument binding generalization.

5. **Trivial Recovery vs. Generation Binding**:
   - In Batches 99–102, the probe achieved 100% linear probe accuracy for extracting Operand A from `pos_a` in Layer 0. This is trivial memorization: early transformer layers retain token identity embeddings.
   - In contrast, probe accuracy for extracting Operand A from `pos_gen` (the prompt end) dropped significantly, confirming that the bottleneck is **cross-attention transport from prompt to generation position**, not the absence of the token in the prompt embedding.

---

## 3. Slot-Weighting Experimental Design Validity

To test whether loss-weighting on argument tokens can force the model to bind dynamic inputs without collapsing:
1. **Target Position Resolution**: Position mapping must track the token offsets in the target sequence where the numeric argument values are emitted.
2. **Shifted-Label Causal Alignment**: In autoregressive next-token prediction, token at position $t$ predicts target at position $t+1$. The weight assigned to token $t$ in the loss must weight the loss for predicting that argument value token, preserving proper causal masking.
3. **Controlled Comparison Matrix**:
   - **Condition A**: Condition C baseline checkpoint from Batch 95 (`condition_c_slot_binding_step_100.pt`, SHA-256 `d7702580c5e586ed4ffa3a6498d4f35520886d92e8f3b581ab8df1c26d4cdaa2`).
   - **Condition B**: New uniform-loss control (weight = 1.0× across all tokens) trained from canonical step-10,000 checkpoint on the exact same token sequence and step budget.
   - **Condition C**: Slot-weighted run (weight = 5.0× strictly on target argument values) trained from the identical canonical step-10,000 checkpoint with identical optimizer seeds, step count, and batch schedule.

Batch 103 validity checks confirm all prerequisites are met.
