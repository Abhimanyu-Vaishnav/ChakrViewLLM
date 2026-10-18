# SANSKRIT CURRICULUM PREREGISTRATION PROTOCOL (AMENDED)

**Batch:** 2541–2580  
**Phase:** Phase 3 — Scientifically Amended Preregistration Protocol  
**Document Status:** FROZEN PREREGISTERED SPECIFICATION (NO TRAINING IN THIS BATCH)  
**Machine-Readable Spec:** [`experiments/batch2541_2580/SANSKRIT_CURRICULUM_PREREGISTRATION_AMENDED.json`](file:///d:/Project/ChakrView/experiments/batch2541_2580/SANSKRIT_CURRICULUM_PREREGISTRATION_AMENDED.json)  
**Amendment Diff:** [`experiments/batch2541_2580/PREREGISTRATION_AMENDMENT_DIFF.md`](file:///d:/Project/ChakrView/experiments/batch2541_2580/PREREGISTRATION_AMENDMENT_DIFF.md)  
**Scientific Audit Review:** [`experiments/batch2541_2580/PREREGISTRATION_FINAL_AUDIT.md`](file:///d:/Project/ChakrView/experiments/batch2541_2580/PREREGISTRATION_FINAL_AUDIT.md)  

---

## 1. Core Research Question & Statistical Hypothesis

### Primary Research Question:
Does adding an independent, non-leaking Classical Sanskrit training curriculum to Continued Pretraining (CPT) resolve the observed held-out regression on Classical Sanskrit (`wikipedia_sa_bhagavad_gita`, $\Delta_{\text{sa}} = +0.0776$ CE / $+8.07\%$ PPL) without causing negative transfer, perplexity degradation, or catastrophic forgetting in existing English, Hindi, or Code domains?

### Hypothesis Testing Plan:
- **Null Hypothesis ($H_0$):** Incorporating Classical Sanskrit does not produce a statistically significant reduction in held-out Sanskrit cross-entropy ($\text{Mean}(\text{TW-CE}_{\text{sa}}^{\text{Arm 2}}) \ge \text{Mean}(\text{TW-CE}_{\text{sa}}^{\text{Arm 1}}) - 1.96 \cdot \sigma_{\text{sa}}$), or causes a statistically significant regression ($\Delta \ge +1.96 \cdot \sigma$) in any control domain.
- **Alternative Hypothesis ($H_1$):** Incorporating Classical Sanskrit achieves a statistically significant improvement on held-out Sanskrit ($\text{Mean}(\text{TW-CE}_{\text{sa}}^{\text{Arm 2}}) < \text{Mean}(\text{TW-CE}_{\text{sa}}^{\text{Arm 1}}) - 1.96 \cdot \sigma_{\text{sa}}$) while maintaining parity on English, Code, and Hindi controls within $1 \cdot \sigma$ guardrails.

---

## 2. Experimental Arms & Lineage Invariants

To guarantee that the experimental arms are strictly compute-matched and unconfounded by step count disparities, all training arms branch directly from the **canonical step 10,000 checkpoint**:

| Arm ID | Arm Name | Starting Checkpoint | Training Steps | Total Target Tokens | Sanskrit % | Purpose |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Arm 0** | **Canonical Baseline** | Step 10,000 (`96357705a683...`) | 0 (Frozen) | 0 | 0.0% | Pre-intervention anchor reference. |
| **Arm 1** | **CPT Seed 42 Reference** | Step 10,000 (`96357705a683...`) | 500 steps | 146,694 | 0.0% | Existing CPT benchmark model (no Sanskrit). |
| **Arm 2** | **Sanskrit-Augmented CPT** | Step 10,000 (`96357705a683...`) | 500 steps | 146,694 | 15.0% | Evaluates Sanskrit-specific representation recovery. |
| **Arm 3** | **Volume-Matched Control** | Step 10,000 (`96357705a683...`) | 500 steps | 146,694 | 0.0% | 15% extra English/Hindi; isolates semantic vs volume effects. |

### Rigorous Hardware & Hyperparameter Invariants:
- **Starting Weights:** Exactly `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Architecture:** ChakrMicro (3,443,136 parameters, 6 layers, $d_{\text{model}}=192$, $n_{\text{heads}}=6$, head dimension 32, tied embeddings, RoPE, SwiGLU).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$, weight decay 0.01), cosine schedule with 50 warmup steps, peak learning rate $1.5 \times 10^{-4}$, min learning rate $1.5 \times 10^{-5}$.
- **Batching & Context:** Micro-batch size 4, gradient accumulation 2 (effective batch size 8), sequence context length 256.
- **Seeds:** Pre-registered 3-seed evaluation across all active arms (`seeds = [42, 43, 44]`). All runs must be reported.

---

## 3. Multi-Tiered Sanskrit Leakage Screening & Provenance Protocol

The protocol rejects naive 8-gram matching as proof of zero leakage and institutes a multi-tiered screening pipeline:

### 1. Provenance & Public Domain Licensing:
- Proposed Classical Sanskrit text must be sourced exclusively from public domain literary collections (e.g. *Hitopadesha* or *Panchatantra*) with documented source URLs, license metadata, and immutable pre-cleaning SHA-256 digests.
- MediaWiki and Wikipedia scrapes are strictly barred from candidate training sets.

### 2. Multi-Tiered Leakage Screening Pipeline:
- **Tier 1 (Surface Token Match):** Exact 8-gram token sequence check against all 61 documents in `wikipedia_sa_bhagavad_gita` ($T_1 = 0$ matches allowed).
- **Tier 2 (Normalized Fuzzy Screening):** Unicode NFKC normalization, removal of danda/double-danda punctuation, virama invariance, and case/diacritic normalization. Jaccard 8-gram similarity must be $< 0.05$.
- **Tier 3 (Subword & Verse-Level MinHash Screening):** MinHash LSH using 13-gram shingles over subword tokens. Any candidate document exhibiting Jaccard similarity $> 0.10$ against any held-out evaluation document is unconditionally discarded.
- **Mandatory Linguistic Review:** Any candidate passage flagged by Tier 2 or Tier 3 triggers manual linguistic review. If a passage shares thematic or verse-level identity with the Bhagavad Gita test set, the entire document is purged.

---

## 4. Tokenizer Preflight & Subword Fragmentation Specification

Before any training manifest is finalized, candidate Classical Sanskrit text must undergo a preflight fragmentation analysis using the canonical BPE tokenizer (`release/chakrview-v0.5.0/data/experiments/vocab_4096`):
1. **Characters-per-Token (CPT) Metric:** $\text{CPT} = \frac{\text{Total UTF-8 Characters}}{\text{Total BPE Tokens}}$.
   - **Acceptance Boundary:** $\text{CPT} \ge 1.80$ characters/token.
   - **Rejection Boundary:** $\text{CPT} < 1.80$ indicates excessive byte-fallback or single-character tokenization, which would distort effective sequence length and degrade gradient stability.
2. **Out-of-Vocabulary (OOV) / Unknown Tokens:** 0 unknown tokens permitted.
3. **Sequence Length Distribution:** Documents must provide sufficient length to populate 256-token windows without excessive padding ($< 15\%$ padding across batches).

---

## 5. Statistical Decision Rules & Guardrails

| Outcome Classification | Statistical Decision Criteria | Project Action |
| :--- | :--- | :--- |
| **VALIDATED RECOVERY** | 1. $\text{Mean}(\text{TW-CE}_{\text{sa}}^{\text{Arm 2}}) \le \text{Mean}(\text{TW-CE}_{\text{sa}}^{\text{Arm 1}}) - 1.96 \cdot \sigma_{\text{sa}}$<br>2. Global test CE $\le 4.2900$<br>3. Hindi control CE $\le 3.9250$ (within $1\cdot\sigma$ of baseline)<br>4. Document win rate $\ge 70.0\%$ | **ACCEPT CURRICULUM AS SCIENTIFICALLY SOUND**<br>Eligible for future release review. |
| **PARTIAL RECOVERY WITH DRIFT** | Sanskrit CE improves significantly ($\Delta \le -1.96 \cdot \sigma$), but Hindi, English, or Code regresses by $> 1 \cdot \sigma$. | **REVISE MIXTURE WEIGHTING**<br>Reduce Sanskrit proportion from 15% to 7.5% and re-test. |
| **INCONCLUSIVE / NO EFFECT** | Sanskrit difference between Arm 2 and Arm 1 is within seed variance ($|\Delta| < 1.96 \cdot \sigma$). | **REJECT HYPOTHESIS**<br>Unreinforced vocabulary drift was not the primary mechanism. |
| **CATASTROPHIC REGRESSION** | Global test CE exceeds baseline ($> 4.4789$) or early stopping divergence threshold ($+0.20$ CE on validation set). | **ABORT AND REJECT**<br>Candidate retained strictly in research-only status. |

---

## 6. Preregistration Status

This amended protocol resolves all 5 methodological gaps identified in Batch 2501–2540. It is frozen and ready for human review. **Zero model training is authorized in this batch.**
