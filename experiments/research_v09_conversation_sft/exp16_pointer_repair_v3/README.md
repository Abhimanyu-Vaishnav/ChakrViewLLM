# ChakrView Experiment 16 v3 — Dataset Pipeline & Contract Verification

**Status**: Dataset & Contract Verified (NO MODEL TRAINING RUN)  
**Location**: `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/`  
**Date**: 2026-10-22  

---

## 1. Overview and Core Innovations

Exp16 v3 hardens the pointer-retraining benchmark against template shortcuts and candidate mask vulnerabilities identified during prior audits:
1. **Balanced Grounding for Fields 7 and 8**:
   - `missing_context_abstention` (Field 7) and `factual_uncertainty_handling` (Field 8) now feature an 80/20 balance (24 answerable, 6 abstentions in Train; 5/1 in Val; 10/2 in Test), eliminating the 100% constant-abstention shortcut where field-ID alone determined the label.
2. **Strict Candidate Masking**:
   - Position 0 is reserved as the `<bos>` abstention sentinel (`candidate_mask[0] == True`).
   - All evidence context tokens have `candidate_mask == True`.
   - All prompt query tokens, role labels, and `Assistant:` formatting prefixes are strictly excluded (`candidate_mask == False`), preventing the model from copying tokens from the question.
3. **Fail-Closed Character-to-Token Alignment**:
   - Every answerable token slice is verified to decode cleanly via `ChakrTokenizer` to the intended target text ($2 \le \text{length} \le 12$).
4. **Enhanced Human Review Queue**:
   - [`HUMAN_REVIEW_QUEUE_V3.md`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/HUMAN_REVIEW_QUEUE_V3.md) displays full conversation context, demarcating the target with `>>>target<<<` to enable complete semantic auditing.

---

## 2. Dataset Files & Cryptographic SHA-256 Hashes

| Split | File Path | Examples (Ans / Abs) | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **Train** | `data/train_exp16_v3.jsonl` | 300 (222 / 78) | `f044bb71333cb92b15e4f4fb3ea22765275e7a3ec58eb04a29a0f44383a15233` |
| **Val** | `data/val_exp16_v3.jsonl` | 60 (46 / 14) | `45237ff4f6279f538356c5a0890bf255ee8186214fb5798c8c22934c2aeead4c` |
| **Test** | `data/test_exp16_v3.jsonl` | 120 (92 / 28) | `54d9229f6d4825bbbe9ef4c1c9ff5ccaa56b106e236314890c0f99f36da7623a` |

---

## 3. Running the Test Suite

To verify all contract tests:
```powershell
.venv\Scripts\pytest.exe experiments/research_v09_conversation_sft/exp16_pointer_repair_v3/test_exp16_v3_contract.py -v
```
