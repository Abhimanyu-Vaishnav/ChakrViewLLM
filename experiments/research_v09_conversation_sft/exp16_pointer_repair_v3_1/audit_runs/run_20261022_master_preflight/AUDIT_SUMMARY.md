# ChakrView Exp16 v3.1 — Master Pre-Training Audit Summary

**Audit Date**: 2026-10-22  
**Audit Directory**: `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/audit_runs/run_20261022_master_preflight/`  
**Recommendation**: `CONDITIONAL NO-GO` (Awaiting Human Semantic Review)

---

## 1. Verified Machine Facts
- **Bit-for-Bit Preservation**: All 10 protected artifacts and JSONL datasets match their canonical SHA-256 digests exactly.
- Canonical Architecture Hash `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` verified as the fresh-initialized parameter digest of `ChakrMicro(ModelConfig())` seeded at 42, not a file SHA-256.
- Deterministic Tokenization & Offset Alignment: 480 / 480 records verified against the canonical vocab_4096 tokenizer with 0 re-tokenization mismatches and 0 span alignment mismatches.
- Dataset Counts:
  - `train_exp16_v3_1.jsonl`: 300 rows, 300 unique IDs, 0 duplicate IDs in raw stream.
  - `val_exp16_v3_1.jsonl`: 60 rows, 60 unique IDs, 0 duplicate IDs in raw stream.
  - `test_exp16_v3_1.jsonl`: 120 rows, 120 unique IDs, 0 duplicate IDs in raw stream.
  - Total records: 480 rows, 480 unique IDs.
- Zero Cross-Split Overlap: 0 duplicate IDs or prompt strings across any of the three split pairs (Train/Val, Train/Test, Val/Test).
- Span & Mask Contracts: 480 / 480 records pass character slice matching, candidate mask permission, context boundary isolation, and 512 context token limits (max length: 211 tokens).
- Unit & Contract Tests: 12 / 12 tests passed in `test_exp16_v3_1_contract.py`.

---

## 2. Open Issues Requiring Human Decision
1. Polar vs. Entity Target Mismatch in Category 8:
   - In `EXP16_V31_TRAIN_UNCERTAIN_000`, prompt asks “Did ... record a transient state ...?” but target is `SURGE_FAULT`. Requires human decision on question rewrite for v3.2.
2. Premise Ambiguity in Category 0 Abstention:
   - In `EXP16_V31_TRAIN_DYN_BIND_024`, supervisor states “checked sensor...” while operator says “dropped before any sensor was engaged”. Requires human evaluation of conversational naturalness.
3. Category 9 Construct Validity:
   - `constrained_code_synthesis` evaluates function identifier retrieval from API specs, not generative multi-line code synthesis. Requires lead architect decision on construct scope.
4. Human Review Sign-Off:
   - Formal human review of `SEMANTIC_REVIEW_QUEUE.csv` remains pending.

---

## 3. Audit Verdict
`CONDITIONAL NO-GO FOR TRAININGX
*Structural integrity, data pipeline contracts, deterministic tokenization, and leakage mitigations are fully verified. Training is held pending human semantic review of flagged items.*
