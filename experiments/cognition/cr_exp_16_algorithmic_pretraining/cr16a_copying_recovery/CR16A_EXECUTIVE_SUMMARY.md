# CR-EXP-16A Executive Summary

## 1. Overview and Core Breakthrough
CR-EXP-16A was commissioned to independently investigate and resolve the complete failure of CR-EXP-16 Stage A, which reduced training/validation loss from $4.6302$ to $1.8828$ but achieved **0/300** exact sequence copying and **0/300** token presence accuracy.

Through forensic line-by-line tracing, label alignment inspection, and four controlled experimental arms, this mission has:
1. **Identified the True Root Cause:** The historical Stage A pilot trained with standard full-sequence causal language modeling loss over both prompt and target. Because prompt tokens comprised ~60% of sequence mass and targets followed repetitive syntax, gradient descent minimized loss by fitting static prompt tokens and collapsing into high-frequency unigram priors (`stone`, `storm`, `blue`), completely bypassing the formation of key-query token-copying induction circuits.
2. **Experimentally Validated the Solution (Target-Only Loss Masking):** Masking prompt tokens with `-100` forces 100% of gradient updates into target token prediction conditioned on prompt context.
3. **Broken the 0% Copying Barrier on Frozen Benchmarks:**
   - **Baseline (Stage B step 10k):** Exact sequence copying = **0/100 (0.0%)** (generates repetitive natural language hallucination `"The first of the world..."`).
   - **Historical Stage A Pilot:** Exact sequence copying = **0/100 (0.0%)** (generates mode-collapsed unigrams `"north stone stone stone..."`).
   - **Arm C (Multi-template target-masked):** Exact sequence copying = **63/100 (63.0%)**, token presence = **65/100 (65.0%)**.
   - **Arm D (Generalization-preserved induction):** Exact sequence copying = **66/100 (66.0%)**, token presence = **74/100 (74.0%)**, with sequence length 3 reaching **90.5% (19/21)**.
   - **First Token In-Context Copy Accuracy:** Reached **100.0% (100/100)** on exact sequences, **78.0% (78/100)** on single tokens, and **88.0% (88/100)** on multi-token unseen entities.

## 2. Capability Progression Scorecard

| Model Arm / Evaluation Condition | Loss Objective | Exact Sequence Copying (100 probes) | Token Presence (100 probes) | First Target Token Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Stage B Canonical Baseline** | Full sequence | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0.0% |
| **CR-16 Stage A Pilot** | Full sequence | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 6.0% |
| **Arm A (Direct Replication)** | Full sequence | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 10.0% |
| **Arm B (Target-only Masked)** | Target-only (`-100`) | 0 / 100 (0.0%)* | 0 / 100 (0.0%)* | 50.7% |
| **Arm C (Curriculum Multi-template)** | Target-only (`-100`) | **63 / 100 (63.0%)** | **65 / 100 (65.0%)** | **100.0%** |
| **Arm D (High-Diversity Induction)** | Target-only (`-100`) | **66 / 100 (66.0%)** | **74 / 100 (74.0%)** | **100.0%** |

*\*Note: Arm B demonstrated 100% exact zero-shot copying on held-out 2-word combinations in controlled probe tests, but failed cross-template benchmark transfer due to being trained on a single formatting string.*

## 3. Key Scientific Conclusions
1. **Architectural Sufficiency:** The canonical ChakrMicro architecture ($d_{\text{model}}=192, 6\text{ layers}, 6\text{ heads}, d_{\text{head}}=32, 3.44\text{M params}$) has sufficient capacity to execute in-context algorithmic copying. The previous 0% scores were an artifact of objective mismatch and prompt-memorization collapse, not an intrinsic parameter bottleneck.
2. **Teacher-Forced vs Autoregressive Gap:** On multi-token names, teacher-forced first-token accuracy reaches 78–88%, but autoregressive decoding drifts into subword spelling errors (`'Grace'` instead of `'Gautam'`) due to the subword BPE tokenization barrier.
3. **Strict Integrity Maintained:** No external weights, no API models, zero prompt/sequence data leakage against the frozen benchmark, and `chakrview/runtime/session.py` remains bit-for-bit untouched.

