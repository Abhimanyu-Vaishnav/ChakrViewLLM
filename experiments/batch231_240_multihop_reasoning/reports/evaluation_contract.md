# Batch 232: Versioned Multi-Hop Evaluation Contract

**Document Version**: 1.0.0-frozen  
**Date**: 2026-10-13  
**Frozen Multi-Hop Manifest**: `experiments/batch231_240_multihop_reasoning/data/frozen_multihop_manifest_100.json`  
**Manifest SHA-256**: `b70b38a880e307be77d5eaa1df68bd86b977ad67a2f9d13a79e4b1b7e0939016`  
**Fixed Primary Denominator**: $N = 100$ (80 Answerable 2-hop chains, 20 Adversarial missing-link / unanswerable cases)

---

## 1. Multi-Hop Relational Chain Contract
Each example models a two-hop composite query across multiple documents:
- **Hop 1 (Bridge Extraction)**: Entity $E_1 \xrightarrow{P_1} E_2$ (found in Document A).
- **Hop 2 (Target Value Retrieval)**: Intermediate Entity $E_2 \xrightarrow{R_2} V^*$ (found in Document B).
- **Adversarial Distractor**: Document C reports a spurious direct attribute: $E_1 \xrightarrow{R_2} V_{\text{distractor}}$ to test whether the model falls into naive single-hop word-matching.

---

## 2. Evaluation Rules & Answer Normalization
1. **Primary Score**: Exact Match (EM) over all 100 frozen test items ($N=100$):
   $$\text{EM} = \frac{\sum_{i=1}^{100} \mathbb{I}(\text{Norm}(\text{Answer}_i) = \text{Norm}(\text{Gold}_i))}{100}$$
2. **Intermediate Link Attribution**:
   Record whether the intermediate entity $E_2$ was localized or whether the system jumped directly to a lexical match.
3. **Abstention Verification**:
   When Document B is missing (missing link), the system must produce an ungrounded abstention ("Not mentioned in context.") instead of guessing or copying the distractor from Document C.
4. **Candidate Space Constraint**:
   Candidates for TokenTrie constrained selection are extracted exclusively from the concatenated multi-document context supplied for that query. No test-label metadata or global knowledge oracle may be injected.
