# Batch 44 Root-Cause Investigation: Generation Failure & Repetitive Degeneration

**Date:** 2026-10-12  
**Investigation Target:** Zero EOS termination rate & repetitive table syntax during greedy decoding.

---

## 1. Grounded Reproduction of Generation Behavior

Evaluated greedy decoding across canonical Step 10,000, Batch 40 Step 20, and Batch 41 Step 20 checkpoints.

### Empirical Output Across Checkpoints

| Prompt | Expected Output | Base Checkpoint (Step 10k) | Batch 41 Checkpoint (Step 20) | Stopping Condition |
| :--- | :--- | :--- | :--- | :--- |
| `What is 2 + 3?` | `5` | `\|-\n\|200\n\|-\n\|2005...` | `\|-\n\|200/01\|\|Premier League...` | Max token limit (40 tok) |
| `How do you check that d_model divides evenly...?` | `assert d_model % n_heads == 0` | `\|-\n\| scope=&quot;row&quot;...` | `\|-\n\| scope=&quot;row&quot;...` | Max token limit (40 tok) |
| `What is Pre-RMSNorm?` | `Pre-RMSNorm` | `\|-\n\| scope=&quot;row&quot;...` | `\|-\n\| scope=&quot;row&quot;...` | Max token limit (40 tok) |

In 100% of cases, generation terminates strictly due to reaching `max_new_tokens=40`. EOS token (`id=1`) was never emitted.

---

## 2. Hypothesis Testing & Evidence Matrix

| Hypothesis | Mechanism Investigated | Evidence / Finding | Conclusion |
| :--- | :--- | :--- | :--- |
| **H1: Tokenizer Bug / EOS Missing** | Is token ID 1 missing or unmapped? | BPE tokenizer has `<EOS>: 1`. Decodes cleanly. Emitted properly when supervised. | **REJECTED** |
| **H2: Label Shifting Bug** | Are target tokens shifted incorrectly in training? | Target at index $i$ predicts $x_{i+1}$. Next-token alignment is mathematical exact. | **REJECTED** |
| **H3: Delimiter Boundary Collapse** | Does `<|im_end|>\n` collide with trailing newline? | Tokenized cleanly as `[63, 127, 1012, 98, 275, 103, 127, 65, 13]`. | **REJECTED** |
| **H4: Model Incapable of Emitting EOS** | Is the 3.4M architecture incapable of learning EOS? | Diagnostic overfit test trained on 1 pair for 25 steps: emitted `'5<|im_end|>\n'` and terminated at `id=1` (EOS) perfectly. | **REJECTED** |
| **H5: Pretraining Attractor Dominance** | Did pretraining data imprint repetitive table priors? | Decoded Stage C shard 0: contains dense Wikipedia tables with exact syntax `{| class=&quot;wikitable&quot;\n\|- style=&quot;background-color...` | **CONFIRMED** |
| **H6: Insufficient SFT Gradient Volume** | Can 20 gradient steps on 39 examples unseat 10,000 pretraining steps? | At step 20, token `127` (`|`) probability after `assistant\n` dropped from 91.8% to 76.6%, but remains the #1 top-1 logit. Model needs sustained gradient mass on non-table text. | **CONFIRMED** |

---

## 3. Targeted Controlled Overfitting Experiment

To scientifically determine whether the neural core and decoding loop are capable of clean instruction answering and EOS termination, a controlled single-example training experiment was executed:

- **Target Record**: `prompt = "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n<|im_start|>assistant\n"`, `response = "5<|im_end|>\n"`
- **Initial Loss**: 6.8160 (Logit 127 `|` had >90% probability).
- **After 15 Steps (lr=1e-3)**: Loss = 0.0035. Output: `'5<|im_end|>\n'`
- **After 25 Steps (lr=1e-3)**: Emitted token sequence `[56, 63, 127, 1012, 98, 275, 103, 127, 65, 13, 1]`.
  - Token `1` (`<EOS>`) was emitted immediately following `<|im_end|>\n`.
  - Generation stopped deterministically on EOS (`eos_reached = True`).

---

## 4. Scientific Verdict & Path to Recovery

The generation failure is **NOT** a code bug, tokenizer defect, or architectural limitation. It is the natural consequence of:
1. Deep Wikipedia table attractors in the 10,000-step pretraining baseline.
2. Inadequate SFT sample volume (only 39 examples across 20 steps).

**Recovery Path:**
The model requires a richer dataset ($\ge 100$ records) with diverse coding, mathematics, and procedural examples trained over a sufficient step budget ($\ge 50$–$100$ steps) to overcome the pretraining table basin.
