# Batch 74 — Rationale-Style Data Comparison Report

## 1. Hypothesis and Motivation

In Batches 68 and 73, direct-answer causal SFT ($P(\text{answer} \mid \text{prompt})$) showed 0% generalization on unseen arithmetic inputs. Direct prediction requires the model to compute $(a, b) \mapsto a + b$ in a single feedforward step across 40 attention positions without scratchpad tokens.

Batch 74 tests the **Intermediate Rationale Hypothesis**:
> Does introducing intermediate reasoning tokens ($a + b = c.\text{ Answer: } c$) enable ChakrMicro's attention layers to route and compute the transformation rather than memorizing direct prompt-to-answer associations?

---

## 2. Experimental Setup

- **Task**: The identical single-digit addition benchmark established in Batch 72 (30 Train, 10 Val, 10 Held-Out Test).
- **Template Comparison**:
  - Direct Condition (Condition A): `response: "c<|im_end|>\n"`
  - Rationale Condition: `response: "a + b = c. Answer: c<|im_end|>\n"`
- **Evaluation Criteria**: Exact match on the extracted final answer (`Answer: X` or trailing digit) across all splits.
- **Controlled Invariants**: Step count (100 steps), optimizer (AdamW, $\text{lr}=5\times 10^{-4}$), batch size (2), seed (42), and baseline weights were identical.

---

## 3. Quantitative Results: Direct vs. Rationale SFT

| Metric | Direct-Answer SFT (Condition A) | Rationale-Style SFT (Batch 74) |
|:---|:---:|:---:|
| **Training Steps** | 100 | 100 |
| **TRAIN Exact Match Acc** | 5 / 30 (16.67%) | **9 / 30 (30.0%)** |
| **VAL Exact Match Acc** | 1 / 10 (10.0%)* | **0 / 10 (0.0%)** |
| **TEST Exact Match Acc** | 1 / 10 (10.0%)* | **1 / 10 (10.0%)** |
| **EOS Completion Rate** | 100.0% | 100.0% |
| **Malformed Output Rate** | 0.0% | 0.0% |

*\*Note: In the direct-answer condition, the 1/10 accuracy was an artifact of constant-attractor modal collapse (`7`).*

---

## 4. Qualitative Output Inspection & Structural Hallucination

Inspection of generated responses in the rationale condition reveals a fascinating failure mode:
1. **Template Adherence**: The model learns the surface rationale syntax with 100% precision:
   `"X + Y = Z. Answer: Z<|im_end|>\n"`.
2. **Intermediate Hallucination (Premise Drift)**:
   When given prompt `Add: 4 + 2 = ?`, the model outputs:
   `"4 + 0 = 7. Answer: 7<|im_end|>\n"`
   When given prompt `Add: 2 + 7 = ?`, the model outputs:
   `"2 + 6 = 7. Answer: 7<|im_end|>\n"`
   When given prompt `Add: 1 + 2 = ?`, the model outputs:
   `"2 + 3 = 7. Answer: 7<|im_end|>\n"`
3. **Verdict**:
   Rather than conditioning the intermediate explanation on the prompt's actual operands, the model hallucinated memorized training equations (`4 + 0 = 7`, `2 + 3 = 7`) from the training split. Intermediate tokens did not unlock reasoning; instead, the model treated the rationale sentence as an extended syntactic template subject to the same mode collapse.

---

## 5. Scientific Conclusion

Intermediate rationale tokens in a 3.44M parameter model trained for 100 steps on 30 examples do not induce functional operator generalization. The model mimics the explanatory format but drifts from the prompt premises. True arithmetic transformation learning in small transformers requires either dense mathematical pretraining or structured positional embeddings where operands are bound directly to computation circuits.
