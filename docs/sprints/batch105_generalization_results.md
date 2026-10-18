# Batch 105 — Evaluation of Generalization and Failure Modes

**Sprint**: ChakrView Master Sprint Batches 103–106  
**Module**: Dynamic Argument Learning: Generalization & Failure Mode Audit  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. End-to-End Evaluation Metrics

All models were evaluated under identical greedy decoding settings (`temperature=0.0`, `max_tokens=55`) on the canonical splits:
- `train`: 64 records
- `val`: 18 records
- `test`: 18 records
- `clarification_controls`: 5 negative/out-of-domain requests

### Performance Comparison Table

| Condition | Split | Valid Syntax | Tool / Op Acc | Op A Acc | Op B Acc | Exact Pair Acc | Unique Calls | Constant Collapse | Most Common Call | Spurious Tool Calls |
|---|---|---|---|---|---|---|---|---|---|---|
| **Condition A** *(Cond C Benchmark)* | Train (64) | 100.0% | 100.0% | 10.9% | 7.8% | **0.0%** | 1 | 100.0% | `{"a": 1, "b": 8}` | 5/5 (100%) |
| | Val (18) | 100.0% | 100.0% | 5.6% | 5.6% | **0.0%** | 1 | 100.0% | `{"a": 1, "b": 8}` | — |
| | Test (18) | 100.0% | 100.0% | 11.1% | 22.2% | **5.6%** *(1/18)* | 1 | 100.0% | `{"a": 1, "b": 8}` | — |
| **Condition B** *(Uniform Control)* | Train (64) | 100.0% | 100.0% | 10.9% | 7.8% | **0.0%** | 1 | 100.0% | `{"a": 1, "b": 8}` | 5/5 (100%) |
| | Val (18) | 100.0% | 100.0% | 5.6% | 5.6% | **0.0%** | 1 | 100.0% | `{"a": 1, "b": 8}` | — |
| | Test (18) | 100.0% | 100.0% | 11.1% | 22.2% | **5.6%** *(1/18)* | 1 | 100.0% | `{"a": 1, "b": 8}` | — |
| **Condition C** *(Slot-Weighted 5×)* | Train (64) | 100.0% | 100.0% | 10.9% | 7.8% | **1.6%** *(1/64)* | 2 | 79.7% | `{"a": 3, "b": 8}` | 5/5 (100%) |
| | Val (18) | 100.0% | 100.0% | 11.1% | 5.6% | **0.0%** | 2 | 88.9% | `{"a": 3, "b": 8}` | — |
| | Test (18) | 100.0% | 100.0% | 5.6% | 22.2% | **0.0%** *(0/18)* | 2 | 77.8% | `{"a": 3, "b": 8}` | — |

---

## 2. Teacher-Forced Probability & Rank Analysis

To probe internal representations beyond greedy decoding, we measured the probabilities and vocabulary ranks of the ground-truth target tokens under teacher forcing at the exact argument positions:

| Metric | Condition A (Cond C Baseline) | Condition B (Uniform Control) | Condition C (Slot-Weighted 5×) |
|---|---|---|---|
| **Operand A Target Prob** | 8.71% $\pm$ 4.2% | 8.71% $\pm$ 4.2% | **8.88%** $\pm$ 4.6% |
| **Operand A Mean Rank** | 5.9 (median: 6.0) | 5.9 (median: 6.0) | **5.8** (median: 6.0) |
| **Operand A Modal Top Token** | `628` (`"1, "`) | `628` (`"1, "`) | `660` (`"3, "`) |
| **Operand B Target Prob** | 9.83% $\pm$ 6.3% | 9.83% $\pm$ 6.3% | **10.95%** $\pm$ 6.1% |
| **Operand B Mean Rank** | 4.7 (median: 4.0) | 4.7 (median: 4.0) | **4.8** (median: 4.0) |
| **Operand B Modal Top Token** | `59` (`"8"`) | `59` (`"8"`) | `59` (`"8"`) |

### Key Observations:
1. **Marginal Probability Uplift**: Applying a $5\times$ slot weight increased ground truth target probability marginally:
   - Operand A: from $8.71\%$ to $8.88\%$ (+0.17 percentage points).
   - Operand B: from $9.83\%$ to $10.95\%$ (+1.12 percentage points).
2. **Attractor Transition Without Input Binding**:
   - The extra loss weight destabilized the primary attractor `a: 1` (`628`) and shifted the dominant token to `a: 3` (`660`).
   - However, the generated arguments **do not track the input prompt operands**. Across 14 out of 18 test examples, Condition C generated `{"a": 3, "b": 8}`, while across the remaining 4 it generated `{"a": 3, "b": 8}` with a slightly different trailing whitespace boundary.
3. **Exact Pair Generalization on Held-Out Test Set**:
   - Condition B scored 1/18 (5.56%) on test solely because the static prediction `(1, 8)` matched test item `SLOT-TEST-03`.
   - Condition C scored 0/18 (0.0%) on test because the pair `(3, 8)` was not present in the held-out test split.
   - On the training split, Condition C scored 1/64 (1.6%) solely because `(3, 8)` is present once in train.
   - **Conclusion**: There is no dynamic input-conditioned generalization; the model replaced one static output collapse mode (`1, 8`) with another (`3, 8`).

---

## 3. Failure Mode Synthesis & Security Controls

1. **Constant-Call Collapse**:
   - Both uniform and slot-weighted SFT suffer from constant-output attractor collapse ($\ge 77.8\%$).
   - Increasing the loss weight on the target slot by $5\times$ merely alters the static attractor mode (`1` $\to$ `3`), failing to establish input-to-output attention routing.
2. **Clarification Control Security**:
   - All models generated tool calls for 100% (5/5) of out-of-domain clarification prompts (e.g., executing code or translation queries).
   - SFT on pure tool examples causes the model to unlearn the conversational gate, requiring explicit negative training or gated decoding.
3. **Deterministic Safety Invariants**:
   - The allowlisted calculator tool runtime successfully intercepted and safely executed the calls without code execution or side effects.
