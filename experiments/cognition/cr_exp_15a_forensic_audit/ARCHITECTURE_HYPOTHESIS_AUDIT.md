# CR-EXP-15A Architecture Hypothesis Audit
## Evidence-Based Classification of Structural Claims

**Mission Identifier:** CR-EXP-15A  
**Auditor:** Autonomous Scientific Auditor & Core Architecture Maintainer  
**Date:** November 2026  

---

## 1. Classification of Architectural and Mechanistic Claims

Every claim made in CR-EXP-15 regarding transformer architecture, tokenization, and capability failure was independently evaluated against empirical evidence:

### Claim 1: "Subword fragmentation is the primary cause of entity binding failure."
- **Classification:** **CONTRADICTED BY EVIDENCE**
- **Evidence:** Tested across 30 entities ranging from 1 to 9 tokens (`TOKENIZER_BINDING_DIAGNOSTICS.md`). Single-token entities (e.g. `Dev` $\to$ token ID `[1302]`) failed direct copying ($0/1, 0.0\%$) and in-context selection ($0/1, 0.0\%$) at rates identical to multi-token entities. Fragmentation is not the limiting constraint.

### Claim 2: "Doubling attention head dimension ($d_{\text{head}}=32 \to 64$) autonomously enables pointer routing."
- **Classification:** **UNSUPPORTED / EXPERIMENTALLY CONTRADICTED**
- **Evidence:** Arch Arm B ($192\text{-dim}, 3\text{ heads}, d_{\text{head}}=64$) scored $0.00\%$ across all tasks and collapsed into repeating `Assistant:` tokens. Expanding subspace rank per head without pretraining does not induce pointer routing.

### Claim 3: "Increasing model parameter capacity ($3.44\text{M} \to 5.77\text{M}$) solves in-context binding."
- **Classification:** **UNSUPPORTED**
- **Evidence:** Arch Arm C ($256\text{-dim}, 4\text{ heads}, d_{\text{head}}=64$) scored $0.00\%$ across all tasks and suffered degenerate character collapse (`AAAA...`). Capacity scaling during short fine-tuning fails to develop reasoning primitives.

### Claim 4: "Candidate D failed because of an Induction Circuit Deficit (lack of copy-routing attention heads)."
- **Classification:** **SUPPORTED BUT NOT DIRECTLY ESTABLISHED**
- **Evidence:** Direct text outputs confirm that the model substitutes memorized parametric tokens rather than copying prompt tokens. However, layer-by-layer attention head attribution matrices (measuring attention weights from query to subject) were not visualized. While theoretically consistent with mechanistic interpretability literature, direct head activation tracing is required to elevate this from a strong hypothesis to a proven theorem.

### Claim 5: "CR-14's reported 8.33% subject accuracy was an evaluator defect."
- **Classification:** **DIRECTLY DEMONSTRATED**
- **Evidence:** Proven via exact code inspection of CR-14's evaluator: `has_subj` was hardcoded to `True` for unanswerable probes ($100 / 1200 = 8.333\%$), while real answerable subject tracking was $0/1,100$.

