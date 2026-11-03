# ChakrView Research Priority Matrix v1.0
## Objective Multi-Factor Ranking of Experimental Candidates

**Repository:** `ChakrView`  
**Location:** `docs/experiments/RESEARCH_PRIORITY_MATRIX_v1.0.md`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Version:** 1.0.0  
**Effective Date:** 2026-10-31  
**Status:** Canonical Priority Matrix  

---

## 1. Multi-Factor Evaluation Methodology

To ensure disciplined research progression and prevent wasteful trial-and-error training runs, prospective candidates are ranked across nine objective evaluation dimensions:
1. **Scientific Value (SV, 1–5):** Potential to resolve fundamental questions regarding ChakrMicro's learning capacity.
2. **Causal Isolation (CI, 1–5):** Ability to definitively isolate a single root cause without confounding variables.
3. **Relevance to Known Failures (RF, 1–5):** Direct alignment with demonstrated failure modes from M3, M4, M5, and M5.1.
4. **Implementation Simplicity (IS, 1–5):** Ease of engineering code and datasets without introducing architectural complexity.
5. **Hardware & Compute Efficiency (HE, 1–5):** Execution efficiency on local CPU threads with low RAM requirements.
6. **Dataset Governance Readiness (DR, 1–5):** Availability of clean, split-isolated datasets satisfying `docs/data/` rules.
7. **Evaluator Readiness (ER, 1–5):** Availability of verified, leak-free, word-boundary evaluation harnesses.
8. **Reproducibility Risk (RR, 1–5):** Freedom from nondeterministic operations or unverified dependency chains (higher = lower risk).
9. **Time-to-Evidence (TE, 1–5):** Rapidity with which conclusive empirical evidence can be obtained (higher = faster).

*Overall Priority Score (Composite, Out of 45) = Sum of all 9 dimensions.*

---

## 2. Priority Scoring and Ranking Table

| Rank | Candidate Identifier | SV | CI | RF | IS | HE | DR | ER | RR | TE | Composite Score | Current Status | Recommended Action |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **`CAND_01_CONTROLLED_ENTITY_COPYING`** | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 5 | 5 | **43 / 45** | `READY_FOR_REVIEW` | **Primary Next Candidate (Submit for Pre-Execution Approval)** |
| **2** | **`CAND_04_TOKENIZER_SEGMENTATION_DIAGNOSTIC`** | 4 | 5 | 4 | 5 | 5 | 4 | 4 | 5 | 5 | **41 / 45** | `PROPOSED` | Diagnostic baseline (Can execute in parallel as read-only probe) |
| **3** | **`CAND_02_SUPERVISION_WEIGHTING_FACTORIAL`** | 5 | 4 | 5 | 3 | 4 | 3 | 3 | 4 | 3 | **34 / 45** | `NEEDS_DESIGN` | Secondary follow-up (Formalize loss masking and telemetry) |
| **4** | **`CAND_03_CONTEXT_POSITION_SENSITIVITY`** | 4 | 4 | 4 | 3 | 4 | 3 | 3 | 4 | 3 | **32 / 45** | `BLOCKED` | Blocked until `CAND_01` demonstrates copying $>50\%$ |
| **5** | **`CAND_06_UNSEEN_ENTITY_GENERALIZATION`** | 5 | 4 | 4 | 3 | 4 | 2 | 3 | 4 | 3 | **32 / 45** | `BLOCKED` | Blocked until in-distribution copying is proven in Track 1 |
| **6** | **`CAND_05_MULTI_TURN_STATE_RETENTION`** | 5 | 3 | 4 | 2 | 3 | 2 | 2 | 3 | 2 | **26 / 45** | `BLOCKED` | Blocked until 1-hop associative retrieval is established |

---

## 3. In-Depth Justification of the Primary Next Candidate

### Why `CAND_01_CONTROLLED_ENTITY_COPYING` Ranks #1:
1. **Resolves the Foundational M5.1 Uncertainty:** M5 (Exp35) failed with 0/200 accuracy, and M5.1 revealed a near-zero sensitivity delta ($\Delta p = 0.00014$). However, Exp35 confounded model capacity with massive static template overhead (78.96% template tokens). `CAND_01` strips away all conversational framing to answer the decisive prerequisite question: *Can ChakrMicro copy an arbitrary token under ideal, noise-free conditions?*
2. **Minimal Compute Footprint:** A synthetic key-value copying task requires very short sequence lengths ($\le 32$ tokens), reducing CPU compute time by $>80\%$ compared to full conversational training.
3. **Decisive Causal Signal:** If `CAND_01` succeeds, template dilution is proven to be the primary blocker in Exp35. If `CAND_01` fails, it conclusively proves that the architecture itself cannot perform 1-hop associative routing, pointing directly to width/head capacity limits or rotary positional limitations.
4. **Unblocks Downstream Tracks:** Successful execution of `CAND_01` is the explicit gateway required to unblock Candidates 3, 5, and 6.

---

## 4. Operational Gating and Recommendation

- **Formal Recommendation:** Research team should focus immediate engineering on drafting the formal Section 13 proposal for **`CAND_01_CONTROLLED_ENTITY_COPYING`**.
- **Governance Barrier:** This ranking constitutes an analytical recommendation for review. It does **not** authorize training execution. Execution requires passing all 11 gates of Protocol v1.0 Section 14 with explicit maintainer sign-off.

