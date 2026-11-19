# UCIP-12R: Corpus Count Reconciliation Report

**Mission ID:** UCIP-12R — Forensic Verification  
**Audit Target:** Discrepancy between nominal figures (7.7M vs. 8.1M vs. 8.3M tokens)  
**Execution Script:** `experiments/ucip12r/scripts/verify_corpus_counts.py`  

---

## 1. Executive Summary

Phase D addressed the exact mathematical reconciliation of token counts cited in the UCIP-12 report:
- **Claim A:** Stage C contains **7,703,067 tokens**.
- **Claim B:** Combined corpus contains **8,100,563 tokens**.

### Audit Finding: **VERIFIED & RECONCILED WITH EXPLANATION**
The nominal figure of 7,703,067 tokens represents the **raw text content tokens** in `stage_c_manifest.json` *excluding document EOS boundary delimiters*.
When packed into the 32 binary shards with 1 EOS delimiter per document, the actual binary shard token count is **7,708,601 tokens**.

Furthermore, the 8,100,563 figure cited in UCIP-12 was an initial manual table estimate. An exhaustive physical audit of all actual pre-tokenized binary shards across the repository reveals **8,304,352 tokens**.

---

## 2. Complete Token Accounting Table

| Dataset / Component | Filesystem Path | Format | Raw Documents / Pairs | Binary Files | Shard Size (Bytes) | Exact Token Count | Included in Pretraining? |
|---|---|---|---|---|---|---|---|
| **Stage C (Pretraining Corpus)** | `data/tokenized/stage_c/**/*.bin` | uint16 bin | 5,534 docs | 32 shards | 15,417,202 | **7,708,601** | **YES (Primary Foundation)** |
| **Stage B (Curated Shards)** | `data/tokenized/stage_b/**/*.bin` | uint16 bin | ~7,200 snippets | 4 shards | 1,003,306 | **501,653** | Optional Alignment Mix |
| **Curriculum Step 53** | `data/tokenized/curriculum_step53/**/*.bin` | uint16 bin | 1,185 items | 6 shards | 86,048 | **43,024** | Targeted Task Shards |
| **Reasoning Step 54** | `data/tokenized/reasoning_step54/**/*.bin` | uint16 bin | ~800 items | 3 shards | 94,536 | **47,268** | Targeted Reasoning |
| **Coding Arena Step 55** | `data/tokenized/coding_arena/**/*.bin` | uint16 bin | 8 items | 2 shards | 7,612 | **3,806** | Synthetic Arena Code |
| **SUBTOTAL BINARY SHARDS** | — | — | — | **47 shards** | **16,608,704** | **8,304,352** | — |
| **Exp34 v3 Instruction Tuning** | `data/sft/exp34_candidate_v3/*.jsonl` | JSONL text | 1,000 pairs | 2 files | 544,972 | ~115,340 | **NO (Quarantined SFT)** |

---

## 3. Discrepancy Reconciliation Summary

1. **Stage C Content vs. Shards ($+5,534$ tokens):**
   $$\text{Shard Tokens } (7,708,601) = \text{Manifest Content Tokens } (7,703,067) + \text{Document EOS } (5,534)$$
   This accounts for the exact 5,534-token difference with zero unexplained variance.

2. **Combined Total ($8,100,563$ vs. $8,304,352$ tokens):**
   The UCIP-12 report included Stage C (7.70M) + rough estimates for Curriculum (120k), Reasoning (52k), Arena (43k), Stage B text (66k), and Exp34 v3 (115k), yielding 8,100,563.
   The actual verified binary inventory on disk contains **8,304,352 binary tokens** in uint16 shards, of which **7,708,601 tokens belong strictly to Stage C**.

---

## 4. Final Verdict on Data Scale

ChakrView's repository genuinely contains **7.71 million verified, pre-tokenized tokens in Stage C alone**, and **8.30 million tokens across all binary shards**. The Level 4 (>5.0M tokens) readiness classification is **VERIFIED**.
