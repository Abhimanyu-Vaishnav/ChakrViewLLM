# CHAKRVIEW — EXP35 LEAKAGE AUDIT REPORT
## Verification of Strict Train / Validation Separation Across Three Isolation Barriers

**Experiment:** Exp35-DPCE  
**Date:** 2026-10-30  

---

## 1. STRING-LEVEL OVERLAP CHECKS

- **Exact Prompt Overlap:**
  $$\text{Train Prompts} \cap \text{Val Prompts} = \emptyset \quad (0 \text{ matches})$$
- **Exact Response Overlap:**
  $$\text{Train Responses} \cap \text{Val Responses} = \emptyset \quad (0 \text{ matches})$$
- **Normalized Prompt Overlap:**
  Lowercase and whitespace-collapsed prompts: **0 matches**.
- **Normalized Response Overlap:**
  Lowercase and whitespace-collapsed responses: **0 matches**.

---

## 2. STRUCTURAL COMBINATION SEPARATION

1. **Entity-Slot Isolation:**
   Each validation record features distinct identifier indices and parameter pairings (e.g. specific station badges, register configurations, and shipment numbers) that never appear in the training split.
2. **Deterministic Reproducibility:**
   Split executed via deterministic PRNG `seed=42`, recording full cryptographic SHA-256 digests in `exp35_split_manifest.json`.

**Leakage Audit Ruling:** **PASS — ZERO CONTAMINATION DETECTED.**

