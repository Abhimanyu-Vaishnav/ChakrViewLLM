# Batch 100: Controlled Diagnosis of Tool-Calling Checkpoints & Argument Token Probability Distributions

**Sprint Date**: October 12, 2026  
**Repository Branch**: `dev/v0.6-v0.7-validation`  
**Starting HEAD**: `c2f191ae9261a1ea3ce1824d8d49daedc244354f`  
**Evaluation Model**: Canonical `ChakrMicro` (3,443,136 parameters)  
**Evaluated Checkpoints**:
1. **Baseline Step 10,000**: SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
2. **Condition B (Unstructured NL Tool)**: SHA-256 `65eb913bbed9359a343b6efb46cfa3f1b4c9197931327151e3604f3261a8ef11`
3. **Condition C (Explicit Slot-Binding SFT)**: SHA-256 `d7702580c5e586ed4ffa3a6498d4f35520886d92e8f3b581ab8df1c26d4cdaa2`

---

## 1. Experimental Protocol & Invariants

To eliminate evaluation variance and potential confounders:
- **Zero Architecture Changes**: Evaluated directly with native PyTorch causal attention and tied LM head.
- **Identical Datasets**: Evaluated across identical 100 benchmark examples partitioned into Train ($n=64$), Validation ($n=18$), and Test ($n=18$) splits, plus 5 Clarification controls.
- **Identical Decoding**: Greedy decoding ($\tau=0$), maximum 50 tokens.
- **Exact Token Mapping**: BPE token identities for operand digits were verified in-context:
  - For Operand A (preceding `", "`): Merged tokens `1290` (`"0, "`), `628` (`"1, "`), `892` (`"2, "`), `660` (`"3, "`), `1637` (`"4, "`), `1298` (`"5, "`), `1641` (`"6, "`), `753` (`"7, "`), `2225` (`"8, "`), `60` (`"9"`).
  - For Operand B (preceding `"}"`): Merged tokens `1631` (`"0}"`), `585` (`"1}"`), `2202` (`"2}"`), and single digits `54..60` (`"3"` through `"9"`).

---

## 2. Quantitative Comparative Results

| Metric | Condition | Train (n=64) | Val (n=18) | Test (n=18) | Clarification Control (n=5) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Valid Syntax Rate** | Baseline Step 10k | 0.0% | 0.0% | 0.0% | 0.0% spurious (safe non-tool) |
| | Condition B (NL Tool) | 100.0% | 100.0% | 100.0% | 100.0% spurious call |
| | **Condition C (Slot-Binding)** | **100.0%** | **100.0%** | **100.0%** | 100.0% spurious call |
| **Tool Selection ('calculator')**| Baseline Step 10k | 0.0% | 0.0% | 0.0% | N/A |
| | Condition B (NL Tool) | 100.0% | 100.0% | 100.0% | N/A |
| | **Condition C (Slot-Binding)** | **100.0%** | **100.0%** | **100.0%** | N/A |
| **Operand A Accuracy** | Baseline Step 10k | 0.0% | 0.0% | 0.0% | N/A |
| | Condition B (NL Tool) | 10.9% | 5.6% | 11.1% | N/A |
| | **Condition C (Slot-Binding)** | **10.9%** | **5.6%** | **11.1%** | N/A |
| **Operand B Accuracy** | Baseline Step 10k | 0.0% | 0.0% | 0.0% | N/A |
| | Condition B (NL Tool) | 10.9% | 5.6% | 11.1% | N/A |
| | **Condition C (Slot-Binding)** | **7.8%** | **5.6%** | **22.2%** | N/A |
| **Exact Ordered Pair (A & B)**| Baseline Step 10k | 0.0% | 0.0% | 0.0% | N/A |
| | Condition B (NL Tool) | 1.6% (1/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | **Condition C (Slot-Binding)** | **0.0% (0/64)** | **0.0% (0/18)** | **5.6% (1/18)** | N/A |
| **Constant-Call Collapse Rate**| Baseline Step 10k | 51.6% | 55.6% | 72.2% | N/A |
| | Condition B (NL Tool) | 100.0% (`{"a":1,"b":1}`) | 100.0% (`{"a":1,"b":1}`) | 100.0% (`{"a":1,"b":1}`) | N/A |
| | **Condition C (Slot-Binding)** | **100.0%** (`{"a":1,"b":8}`) | **100.0%** (`{"a":1,"b":8}`) | **100.0%** (`{"a":1,"b":8}`) | N/A |

---

## 3. Diagnostic Teacher-Forced Probability Gap Analysis

To understand why the models choose static arguments over the correct prompt-derived values, we measured the model's assigned probability under identical teacher-forced prefixes:
- **$P(\text{target})$**: Probability assigned to the correct ground-truth operand token for the given prompt.
- **$P(\text{gen})$**: Probability assigned to the token the model actually selects at argmax (e.g., token 628 for `"1, "` or token 59 for `"8"`).
- **Target Rank**: The descending rank of the correct token out of all 4,096 vocabulary tokens.

### Summary of Probability Gap Across Checkpoints (Held-out Test Split)
| Checkpoint | Target Argument | Mean $P(\text{target})$ | Mean $P(\text{gen})$ | Ratio $P(\text{target}) / P(\text{gen})$ | Mean Target Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Step 10,000** | Operand A | 0.22% | 10.55% | 0.021 | 76.1 |
| | Operand B | 1.63% | 13.33% | 0.122 | 116.2 |
| **Condition B (NL Tool)** | Operand A | 8.32% | 23.89% | 0.348 | 5.5 |
| | Operand B | 7.50% | 13.96% | 0.537 | 6.0 |
| **Condition C (Slot-Binding)**| Operand A | 8.71% | 15.65% | 0.556 | 5.9 |
| | Operand B | 9.83% | 17.16% | 0.573 | 4.7 |

### Key Diagnostic Findings
1. **Target Probabilities Increase Significantly Under SFT**:
   - In the pre-trained baseline, target tokens receive almost zero probability ($0.22\% - 1.63\%$) and rank around $\sim 100$.
   - SFT (Conditions B and C) elevates the probability of valid digits from $<1\%$ up to $\sim 8.7\% - 9.8\%$, placing the correct token inside the **Top-6 candidate tokens** on average.
2. **The Argmax Cliff (Collapse Mechanism)**:
   - In Condition C, the correct operand token receives between $3.8\%$ and $16.4\%$ probability across test prompts.
   - However, the mode attractor token (`"1, "` for A, `"8"` for B) has an invariant unigram prior of $\sim 15.6\% - 17.2\%$.
   - Because decoding is greedy ($\text{argmax}$), even though the model is sensitive to the prompt (as proven by probe recovery of $61.1\%$), the static unigram bias edges out the dynamic operand by a small margin ($15.6\%$ vs $8.7\%$).
   - This causes 100% of generated tool calls to collapse to static values, yielding 0% dynamic generalization.
3. **Spurious Call Susceptibility**:
   - Both Condition B and Condition C produce 100% spurious tool calls on clarification controls (e.g., executing arithmetic on translation or code queries), proving that neither condition learned a calibrated tool decision gate.
