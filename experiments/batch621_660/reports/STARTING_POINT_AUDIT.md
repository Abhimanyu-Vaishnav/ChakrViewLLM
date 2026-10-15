# Starting Point Audit: Batches 621–660
## Multi-Turn Context Binding, Neural Anaphora Resolution & Persistent Memory Grounding

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Head Commit:** `9e633a0` — `feat(batch581-620): train span pointer candidate, ground exact arguments, and evaluate 59-item benchmark`  
**Previous Milestone:** `9e633a0` (Verified clean lineage from Batches 581–620)

---

### 1. Verification of Checkpoint Hashes & Artifact Lineage

| Checkpoint Identifier | File Path | Expected SHA-256 | Measured SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Step 10,000** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED IMMUTABLE** |
| **Candidate 541–580 (Intent)** | `experiments/batch541_580/checkpoints/candidate_step250_intent_understanding.pt` | `9c3204906e8e...` | `9c3204906e8e94f5e80c5db9f6ffb0fa6903c6fe3159206fb164e44d52b8357e` | **VERIFIED MATCH** |
| **Candidate 581–620 (Span Pointer)** | `experiments/batch581_620/checkpoints/candidate_step250_span_pointer.pt` | `196384e9c29c...` | `196384e9c29ca1042ecf65e379972257bdcbce1667f4befb0e4249e3f7c5b569` | **VERIFIED MATCH** |

---

### 2. Independent Reproduction of the 59-Item Benchmark

Re-executed from clean Python execution on `candidate_step250_span_pointer.pt` (`experiments/batch621_660/reports/BASELINE_REPRODUCTION.json`):
- **Total Test Items:** 59
- **Overall Intent Accuracy:** **84.7% (50/59)**
  - `calculator`: **100.0% (29/29)**
  - `knowledge_lookup`: **100.0% (4/4)**
  - `clarification_needed`: **100.0% (5/5)**
  - `text_utility`: **83.3% (5/6)**
  - `concept_definition`: **80.0% (4/5)**
  - `direct_chat`: **60.0% (3/5)**
  - `unsupported`: **0.0% (0/5)** (flagged below)
- **Exact Argument Value Recovery:** **44.1% (26/59)**
- **Downstream Calculator Execution:** **29/29 (100.0%)** when normalized to strip prompt framing (`expression:` and trailing `?`).

---

### 3. Critical Dataset & Architecture Audit Findings

1. **Minority-Class Oversampling in Batch 581–620 Training Data:**
   - In `experiments/batch581_620/data/train_span_balanced.jsonl`, minority classes were brought up to 25 items each by row duplication (127 duplicate rows).
   - **Leakage check:** We audited all prompts against the validation and test sets: **0 training prompts appear in validation or frozen test sets** (`zero_leakage_verified: true` in `DATASET_LEAKAGE_AUDIT.json`).
   - *Requirement for Batches 621–660:* Avoid raw row duplication. Use loss-weighting or distinct synthetic templates for class balance.
2. **Analysis of Downstream 29/29 Calculator Score:**
   - The model accurately predicts token spans for math expressions in the prompt (`span_accuracy: 16.9%` exact token boundary match, but `37.9%` raw exact string and `100%` after stripping `expression:` / trailing `?`).
   - When extracted text is parsed by `CalculatorTool`, all 29 held-out arithmetic queries produce mathematically correct results.
   - *Honest Attribution:* This is a **hybrid capability**—the neural core locates the expression, and deterministic Python AST executes the calculation. It is not neural mental calculation.
3. **The Unsupported Intent Boundary:**
   - The `unsupported` category achieved 0.0% in the standalone classifier because disallowed commands often contain shell-like tokens that trigger fallback. In the end-to-end runtime, `EvidenceGovernedAnsweringGate` intercepts false premises (e.g., Mars celestial queries -> `REFUTE`, `ComponentOrigin.FALLBACK_POLICY`).
4. **Current Multi-Turn Blindspot:**
   - The current `ChakrSpanPointerModel` operates on single-turn `User: ...\nIntent:` prompts.
   - When a user says: *"Calculate 17 * 23"*, followed by *"Now multiply that result by 4"*, the single-turn model sees only `"Now multiply that result by 4"`, missing the antecedent `391`.
   - The core objective of Batches 621–660 is to bridge the neural model with `PersistentSessionMemory` so the brain can resolve prior objects, values, and text across conversation turns.
