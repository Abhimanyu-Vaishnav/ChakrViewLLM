# PREREGISTRATION REQUIRED CHANGES SPECIFICATION

**Batch:** 2501–2540  
**Phase:** Phase 5 — Actionable Protocol Revisions for Sanskrit Curriculum Experiment  
**Status:** **MANDATORY REVISION CHECKLIST PRIOR TO EXPERIMENT AUTHORIZATION**  

---

## Required Revisions Checklist

Before any human researcher authorizes dataset ingestion or continued pretraining, the experiment protocol in [`experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.md`](file:///d:/Project/ChakrView/experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.md) must be amended with the following 5 requirements:

### 1. Explicit Checkpoint Lineage Invariants
- **Requirement:** State explicitly that **Arm 2 (Sanskrit-Augmented CPT)** and **Arm 3 (Volume-Matched Control)** MUST initialize their starting weights directly from the canonical step 10,000 checkpoint (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705a683...`).
- **Rationale:** Prevents compounding training steps (e.g. 500 initial steps + 500 subsequent steps = 1000 steps), ensuring that compute budget, learning rate schedules, and step counts (500 steps) are identical between Arm 1, Arm 2, and Arm 3.

### 2. Multi-Tiered Sanskrit Leakage Protocol
- **Requirement:** Replace the superficial "zero 8-gram matches" claim with a three-tier leakage protocol:
  1. **Tier 1 (Surface Screening):** Exact token-level n-gram match ($n=8$).
  2. **Tier 2 (Normalized Fuzzy Screening):** Devanagari NFKC Unicode normalization, whitespace/punctuation stripping, and virama/avagraha invariance matching.
  3. **Tier 3 (Semantic & Subword Overlap):** Tokenized containment check against the 61 documents in `wikipedia_sa_bhagavad_gita` requiring Jaccard similarity $< 0.10$. Any document exceeding 0.10 must be discarded.
- **Rationale:** Classical Sanskrit poetic meter, compounding, and sandhi variants evade simple exact string matches.

### 3. Preflight Tokenizer Fragmentation Audit
- **Requirement:** Define a preflight requirement that tokenizes the proposed Sanskrit corpus with the canonical BPE tokenizer (`vocab.json`, 4,096 tokens) and computes the **Characters-per-Token (CPT) compression ratio**:
  - If the Sanskrit compression ratio drops below $1.8 \text{ chars/token}$ (indicating severe byte fallback or character-level fragmentation), the curriculum mixture must be adjusted or token-weighting recalculated.
- **Rationale:** A tokenizer trained predominantly on English and Hindi may fragment Classical Sanskrit into isolated characters, distorting sequence lengths and effective learning rates.

### 4. Statistically Grounded Decision Boundaries
- **Requirement:** Replace point-estimate thresholds with variance-aware criteria based on the 3 pre-registered seeds (`42, 43, 44`):
  - **Verified Success:** $\text{Mean}(\text{TW-CE}_{\text{sa}}^{\text{Arm 2}}) \le \text{Mean}(\text{TW-CE}_{\text{sa}}^{\text{Arm 1}}) - 2 \cdot \sigma_{\text{sa}}$, AND no control domain degrades by more than $1 \cdot \sigma$.
  - **Inconclusive:** If the observed delta is within the seed standard deviation ($\Delta < 1 \cdot \sigma$), the hypothesis of significant curriculum transfer is rejected.

### 5. Documented Failure Modes & Abort Triggers
- **Requirement:** Predefine early stopping abort triggers during training:
  - If validation cross-entropy on English literature or Python code increases by $> 0.20$ CE at any evaluation checkpoint (step 100, 200, 300, 400), training is immediately aborted to prevent catastrophic representational damage.

---

## Conclusion
With these five amendments integrated, the protocol will satisfy the highest standards of scientific reproducibility and experimental rigor.
