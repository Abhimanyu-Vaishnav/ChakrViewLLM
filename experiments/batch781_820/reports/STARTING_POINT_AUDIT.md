# Starting Point Audit: Batches 781–820
## Grammar-Constrained Autoregressive Planning, Exact Argument Copying & Generalization

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `9d6e810` (`feat(batch741-780): implement autoregressive plan DSL, variable-length neural planning, and safe execution`)  

---

### 1. Checkpoint and Working Tree Lineage

1. **Canonical Step-10,000 Checkpoint:**
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Measured SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
   - Status: **VERIFIED IMMUTABLE AND INTACT**.
2. **Batch 741–780 Autoregressive Candidate Checkpoint:**
   - Path: `experiments/batch741_780/checkpoints/candidate_step250_autoregressive_planner.pt`
   - Measured SHA-256: `38db3d09668bdb980fc72c76da17398d2267644ca698a58a1e6f10d1d392628c`
   - Status: **VERIFIED PRESENT AND INTACT**.
3. **Working Tree & User Preservation:**
   - `chakrview/runtime/session.py` remains modified by user, uncommitted, and untouched.

---

### 2. Independent Reproduction of Prior Milestone Results

The 30-item frozen evaluation from Batch 741–780 was executed independently on an NVIDIA RTX A1000 6GB Laptop GPU.
- **Single-Turn Baseline:** 7/30 (23.3%)
- **Fixed-Head Baseline (Batch 701–740):** 13/30 (43.3%)
- **Autoregressive Plan DSL Raw (Batch 741–780):** 8/30 (26.7%)
- **Autoregressive Plan DSL Hybrid (Batch 741–780):** 8/30 (26.7%)
- **Regression Suite:** 60/60 tests passing across Batches 541–780 in 5.25 seconds.

---

### 3. Forensic Analysis: Why Unconstrained Autoregressive Generation Failed on Multi-Step Plans

Inspection of generated token sequences revealed the exact failure mechanism:
1. **Punctuation and Delimiter Drift:**
   On multi-step plans (lengths $\ge 2$), greedy sampling frequently emits invalid syntax fragments such as:
   - `args={text='Ne}` (unclosed quotes or premature closing brace)
   - `args={expr=strips=[]` (missing closing brace before `deps`)
   - `STEP 0: text_transform | text_utility | args={expr=3} |`
2. **Missing Terminal `<END_PLAN>` Token:**
   When token drift occurs early in generation, subsequent token predictions lose syntactic coherence, failing to emit `<END_PLAN>` before hitting the maximum token limit (`max_new_tokens=120`).
3. **Denominators and Metric Discrepancies:**
   - In earlier reports, unsupported-tool rejection used a dedicated 4 or 5-item subset as denominator, yielding 100% rejection, whereas raw plan accuracy evaluated the full $N=30$ items. This has been formalized and disambiguated in `METRIC_DEFINITIONS.md`.

---

### 4. Technical Strategy for Batches 781–820

To overcome unconstrained drift and achieve reliable variable-length plan synthesis:
1. **Incremental Context-Free Grammar (CFG) Constrained Decoding:**
   Implement an incremental parser state machine (`GrammarConstrainedDecoder`) that tracks the Plan DSL EBNF grammar token-by-token and masks invalid next tokens in the logits vector ($-\infty$) before sampling.
2. **Exact Argument Copying & Tokenizer Vocabulary Awareness:**
   Support multi-character tokens, quoted strings, escaped characters, and exact numeric/text preservation from user prompts.
3. **Multi-Seed Curriculum Expansion:**
   Expand the dataset toward 500 diverse examples across 1 to 4+ step plans, train with prompt masking across multiple seeds, and evaluate on a frozen held-out benchmark.
