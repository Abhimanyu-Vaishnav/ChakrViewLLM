# UCIP-10E Limitations and Open Research Problems

**Date:** 2026-11-18  
**Mission:** UCIP-10E Intelligence Evaluation Infrastructure  

---

## 1. Identified Capability Limitations in 12.9M Checkpoints

Across all evaluated models (UCIP-09, UCIP-09R, UCIP-09S), empirical evaluation on the 100-probe sealed core reveals fundamental limitations:

1. **Exact Copying & Token Memorization:** Zero-shot exact string copying scored 0.0% across all models. Small causal transformers repeatedly truncate or substitute high-frequency bi-grams rather than propagating arbitrary character/token streams.
2. **Multi-Turn State Tracking:** In pure neural inference (Memory OFF), small models fail to maintain entity consistency across turns without an external retrieval memory module.
3. **Multi-Step & Compositional Reasoning:** Small models fail to execute multi-step logic (0% in logic deduction and generalization). Direct next-token autoregression lacks the working memory buffer needed for multi-step algorithmic derivation.
4. **Self-Correction Deficit:** When prompted to verify or correct an erroneous answer, the models lack internal error detection and either repeat the erroneous answer or generate unrelated completions.
5. **Epistemic Abstention:** The models do not possess calibrated uncertainty or abstention mechanisms; they hallucinate answers to unanswerable queries rather than declaring insufficient information.

---

## 2. Distinction: Pure Neural vs Runtime/Tool Capabilities

| Capability | Current Pure Neural Status | Recommended Runtime / Tool Approach |
|---|---|---|
| **Multi-digit Arithmetic** | Incoherent / Failure | Deterministic math tool execution via session tool-calls |
| **Exact Copying** | Fails zero-shot | Explicit scratchpad buffers or copy-pointer heads |
| **Multi-turn Memory** | Incoherent | Session memory module (`session.py` with verified memory store) |
| **Code Execution** | Fails pure synthesis | Sandboxed runner execution and syntax verification |
