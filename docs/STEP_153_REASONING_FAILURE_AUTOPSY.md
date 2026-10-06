# Step 153 — Reasoning Failure Autopsy & Diagnostic Protocol

## 1. Executive Summary

In Steps 145–152, neural language pretraining demonstrated real parameter learning:
- **Training loss:** $8.3650 \to 6.9691$
- **Held-out language accuracy:** $0.0000 \to 0.1220$

However, the neural reasoning benchmark produced:
- **Baseline accuracy:** $0.0000$
- **Candidate accuracy:** $0.0000$
- **Reasoning $\Delta$:** $0.0000$

Step 153 conducted a forensic diagnostic autopsy across the tokenizer, model forward pass, attention masks, sequence construction, target alignment, and optimization trajectory to determine the exact technical root causes.

---

## 2. Evidence-Backed Root Cause Hypotheses

| Hypothesis ID | Category | Confidence | Concrete Diagnostic Evidence |
| :--- | :--- | :--- | :--- |
| **H1** | `TOKENIZER_FRAGMENTATION` & `TARGET_ALIGNMENT_MISMATCH` | **0.95** | Multi-character greek symbols (`alpha`, `beta`, `gamma`) tokenize into disjoint subwords: `['al', 'ph', 'a ']`, `['bet', 'a ']`, `['ga', 'mm', 'a ']`. When evaluating the model's next-token prediction immediately following `therefore `, the continuous stream's next token was `al` (token id `288`), whereas the standalone target lookup expected `alpha` (token id `2573`). This caused an automatic target mismatch at test time. |
| **H2** | `ENTITY_GENERALIZATION_GAP` | **0.90** | Training sets evaluated symbols `alpha...zeta` while held-out sets evaluated disjoint sets `phi...theta`. In small 3.44M parameter models without prior relational binding pretraining, zero-shot entity generalization fails completely because embeddings for unseen entities share no relational structure. |
| **H3** | `INSUFFICIENT_INDUCTION_CIRCUIT` | **0.85** | 15-35 optimization steps on CPU are sufficient to fit local syntax (`:`, `>`, `and`, `therefore`) but insufficient for the attention heads to specialize into induction circuits that reliably copy transitive tokens across a 20-token context window. |
| **H4** | `UNSCAFFOLDED_CURRICULUM_JUMP` | **0.88** | Presenting 2-hop transitive syllogisms immediately without verifying direct relation recognition (Level 0) or 1-step transformations caused the model to fall back to frequency bias. |

---

## 3. Forensic Trace & Diagnostics

```text
Prompt Prefix: "fact: alpha > beta and beta > gamma . therefore "
Target Token:  "alpha" (Standalone ID: 2573)
Token Stream:  [..., 288 ('al'), 412 ('ph'), ...]
Result:        Stream token at position t+1 does not match standalone target ID.
Loss Behavior: Decreases from 8.23 to 4.22 by predicting syntax boilerplate (':', '>', 'therefore').
```

---

## 4. Remediation Implemented in Wave 153–160

1. **Single-Byte ASCII Symbol Normalization:** Entities are mapped to single-byte ASCII characters (`A`, `B`, `C`, ..., `Z`), guaranteeing 1-to-1 mapping between prompt stream and target tokens.
2. **9-Level Curriculum Ladder (Step 154):** Staged ladder from Level 0 (direct relations) to Level 8 (mixed families).
3. **Contamination Guard (Step 155):** Semantic hashing guarantees zero logical data leakage across train, val, and held-out splits.
4. **Honest Capability Reporting:** Distinguishing in-distribution memorization from held-out systematic/compositional generalization.
