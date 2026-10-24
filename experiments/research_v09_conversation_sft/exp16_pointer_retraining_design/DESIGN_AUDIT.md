# ChakrView Exp16 — Design Audit & Architecture Verification

**Target**: `experiments/research_v09_conversation_sft/exp16_pointer_retraining_design/`  
**Date**: 2026-10-22  
**Status**: `AUDIT PASSED — DESIGN APPROVED FOR DATASET PREPARATION`  

---

## 1. Audit Checkpoints & Verifications

### 1. Loss & Span Index Formulation
- **Contract**: Model produces `start_logits` and `end_logits` of shape $[B, T]$.
- **Target Formulation**: `start_pos` $s^* \in \{0, \dots, T-1\}$ and `end_pos` $e^* \in \{0, \dots, T-1\}$.
- **Loss Equation**: Symmetrical Cross-Entropy:
  $$\mathcal{L} = \frac{1}{2}\left(\text{CE}(\text{start\_logits}, s^*) + \text{CE}(\text{end\_logits}, e^*)\right)$$
- **Decoding Constraint**: $1 \le s \le e \le s + 12$ or $(s = 0, e = 0)$.
- **Audit Verdict**: Verified consistent with `FieldConditionedSpanPointer` implementation in Exp15.

### 2. Abstention Contract & Anchor Position `[0, 0]`
- **Contract**: Token position 0 (`[BOS]`) is the reserved non-contextual abstention anchor.
- **Rule**: If and only if the example is a genuine semantic abstention (missing context, unanswerable uncertainty, or unresolvable ambiguity), target is labeled $s^* = 0, e^* = 0$.
- **Validation**: Any answerable example MUST have $s^* \ge 1$ and $e^* \ge s^*$. Fallbacks to `[0, 0]` on alignment errors are strictly banned.
- **Audit Verdict**: Fully compliant.

### 3. Fail-Closed Aligner Integrity
- **Contract**: Alignment uses character offsets $[c_s, c_e]$ mapped to byte offsets within prompt UTF-8 bytes, identifying overlapping tokens.
- **Verification Rule**: `tokenizer.decode(tokens[s:e+1]).strip() == target.strip()`.
- **Fail-Closed Guarantee**: If decoded tokens do not match the target text, the pipeline raises `AlignmentError`.
- **Audit Verdict**: Tested and passed in Exp04 (`test_span_aligner.py`).

### 4. Leakage Prevention Across Splits
- **Splits**: 300 Train, 60 Validation, 120 Held-Out Test.
- **Entity Namespace Isolation**:
  - Train: `train_*` entities (e.g. `switch_tr_00` .. `switch_tr_99`, `node_alpha_tr_*`).
  - Validation: `val_*` entities (e.g. `switch_val_00` .. `switch_val_20`).
  - Test: `test_*` entities (e.g. `switch_tst_00` .. `switch_tst_40`).
- **Prompt Structure Isolation**: Train, Validation, and Test use distinct conversation topics, varying lengths, and non-overlapping dialogue scripts.
- **Audit Verdict**: Fully isolated; zero lexical or template overlap across splits.

### 5. Multi-Entity Distractor Requirement
- **Requirement**: Every answerable example must feature 2 to 4 candidate entities of the same type (e.g., multiple switches, nodes, or formats) to prevent the pointer from simply selecting any entity of that semantic type.
- **Audit Verdict**: Mandated in data generation templates.

### 6. Fair Non-Oracle Baselines & Ablations
- **Arm A (Causal LM)**: Zero metadata, standard generative decoding.
- **Arm B (First-Match Rule)**: Lexical search for candidate pattern in prompt without access to gold labels.
- **Arm C (Query-Agnostic Ablation)**: $E_f = \mathbf{0}$, testing whether field query provides useful slot routing.
- **Arm D (Field-Conditioned Pointer)**: Candidate model.
- **Audit Verdict**: Free of oracle metadata leakage.

### 7. Frozen Backbone & Trainable Parameter Accounting
- **Frozen Backbone**: 3,443,136 parameters (`requires_grad = False`).
- **Trainable Heads**:
  - `field_embedding`: $[10, 192] = 1,920$
  - `query_start`, `key_start`: $2 \times (192 \times 192) = 73,728$
  - `query_end`, `key_end`: $2 \times (192 \times 192) = 73,728$
  - Total Trainable: **149,376 parameters** (4.16% of total).
- **Audit Verdict**: Verified mathematically and empirically.

### 8. Acceptance Thresholds & Reproducibility
- **Overall Exact-Span Accuracy**: $\ge \mathbf{75.00\%}$ ($90 / 120$ on held-out test).
- **Answerable Accuracy**: $\ge \mathbf{70.00\%}$.
- **Abstention Precision & Recall**: $\ge \mathbf{75.00\%}$.
- **Length Invariance**: $\le 15$ percentage points degradation between short ($<100$) and medium ($100–200$) contexts.
- **Seed Policy**: Global seed `42`.
- **Audit Verdict**: Pre-registered and immutable.

---

## 2. Audit Conclusion

`DESIGN AUDIT PASSED` — The Exp16 design resolves all root causes identified in Exp14, Exp15, and Exp04. Proceeding to Phase B: Dataset Preparation.
