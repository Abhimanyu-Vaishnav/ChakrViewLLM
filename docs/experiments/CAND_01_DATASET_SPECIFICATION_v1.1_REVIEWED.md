# CHAKRVIEW — CAND_01 DATASET SPECIFICATION v1.1 (REVIEWED)
## Synthetic Key-Value Corpus Design, Token Disjointness, and Leakage Prevention

**Document ID:** `DOC-EXP-CAND01-DAT-002`  
**Location:** `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.1_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Target Architecture:** ChakrMicro Baseline (Vocab: 4,093 active BPE tokens)  
**Parent Governance Standards:** `docs/data/DATASET_GOVERNANCE_INDEX.md`, `docs/data/DATASET_REPRODUCIBILITY.md`  
**Status:** `SPECIFICATION_ONLY` (Implementation Plan Only — Execution Strictly Unapproved)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Scope

This document represents the formal v1.1 scientific revision of `CAND_01_DATASET_SPECIFICATION_v1.0.md`. It resolves critical and high-severity issues identified during dataset governance review:
- **Correction 1 (Disjoint vs. Recombination Distinctions):** Explicitly separated (a) disjoint unseen-token evaluation from (b) unseen key-value recombination of familiar tokens.
- **Correction 2 (Unit of Independence Defined):** Defined the exact hierarchical units of statistical independence (Unique Token ID $\to$ Key-Value Pair $\to$ Context Mapping $\to$ Synthetic Record).
- **Correction 3 (Token Frequency & Embedding Stratification):** Added mandatory frequency-band stratification across single-token entities to prevent frequency bias in embedding representations from confounding copy performance.
- **Correction 4 (Governance Classification & Release Clearances):** Clarified that synthetic dataset generation is classified as `INTERNAL_ONLY` and `SPECIFICATION_ONLY` until full generation manifest verification and license audits pass.

---

## 2. Corpus Purpose and Engineering Philosophy

The CAND_01 dataset is a purely synthetic, algorithmic sequence corpus designed to train and benchmark associative binding and copying capabilities in ChakrMicro.

### Core Design Rules
1. **Zero Natural Language Boilerplate:** Prohibits conversational markers, polite framing, filler words, and narrative syntax.
2. **Single-Token Entity Guarantee:** Every key and every value must encode to **exactly one discrete BPE token ID** in `data/experiments/vocab_4096`. This eliminates the compound error probability introduced by subword fragmentation (identified in M5.1).
3. **Multi-Level Independence Partitioning:** Formally separates seen-token recombination from true out-of-vocabulary token generalization.
4. **Permutation Invariance:** For multi-pair sequences, the order in which key-value pairs are listed in context must be randomized using fixed cryptographic seeds to prevent positional shortcut exploitation (e.g. "always copy the last value").

---

## 3. Token Pool Construction & Frequency Stratification

Under `data/experiments/vocab_4096`, the active vocabulary size is 4,093 tokens:
- Token IDs 0, 1, 2 are reserved (`<BOS>`, `<EOS>`, `<PAD>`).
- Token IDs 3 to 258 are raw byte tokens (ASCII characters, numerals, punctuation).
- Token IDs 259 to 4092 are multi-byte merged tokens (Hindi morphemes, English subwords, frequent n-grams).

### 3.1 Token Filtering Criteria
To guarantee single-token entity encoding:
1. Select candidate tokens $t \in [3, 4092]$ whose decoded byte sequence forms a standalone, non-whitespace, alphanumeric or morphemic unit.
2. Filter out tokens that represent common English syntactic words (e.g., `the`, `is`, `at`, `in`, `and`) to avoid pre-existing language model priors.
3. Establish an entity pool $\mathcal{E}$ of exactly 600 single-token vocabulary items.

### 3.2 Frequency Stratification Control
To prevent embedding norm or frequency disparities from confounding associative accuracy, the pool $\mathcal{E}$ is stratified into three frequency tiers based on base pretraining corpus frequencies:
- **Tier 1 (High Frequency):** Top 33% of candidate single tokens.
- **Tier 2 (Medium Frequency):** Middle 33% of candidate single tokens.
- **Tier 3 (Low Frequency):** Bottom 34% of candidate single tokens.

### 3.3 Partitioning Protocol
The 600-token stratified pool is partitioned into two mutually exclusive sets:
- **Training Token Pool ($\mathcal{E}_{\text{train}}$):** 450 tokens ($75\%$, balanced across tiers).
- **Test Token Pool ($\mathcal{E}_{\text{test}}$):** 150 tokens ($25\%$, balanced across tiers).
- **Strict Intersection Constraint:** $\mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{test}} = \emptyset$.

---

## 4. Distinguishing Capability Evaluation Conditions

To avoid conflating distinct cognitive tasks, test evaluation is stratified into distinct sub-benchmarks:

```
                  [ CAND_01 TEST EVALUATION REGIMES ]
                                    |
       +----------------------------+----------------------------+
       |                                                         |
[ Regime 1: Unseen Recombination ]           [ Regime 2: Token-Disjoint Generalization ]
• Keys and Values from E_train                • Keys and Values strictly from E_test
• Pairings never occurred in training         • Tokens never encountered in fine-tuning
• Tests dynamic associative routing           • Tests pure zero-shot token induction
• Isolates binding from static memory         • Isolates representation from token identity
```

1. **Unseen Key-Value Recombination (Regime 1 / TF-06):**
   Keys and values are drawn from $\mathcal{E}_{\text{train}}$, but their pairings $(K_i, V_j)$ are strictly disjoint from all pairings observed during training. Tests whether the model performs dynamic runtime lookup rather than static associative recall.
2. **Disjoint Token Generalization (Regime 2 / OOD Test):**
   Keys and values are drawn strictly from $\mathcal{E}_{\text{test}}$. Tests whether the model's associative mechanism functions over previously un-fine-tuned token IDs.

---

## 5. Dataset Schema & Standardized Record Formats

All records are serialized as standardized JSON Lines (`.jsonl`), complying with the ChakrView SFT contract.

### 5.1 JSON Field Schema Definition
```json
{
  "id": "CAND01-TF02-00042",
  "task_family": "TF-02",
  "task_name": "SINGLE_KEY_VALUE_ASSOCIATION",
  "data_role": "TRAIN",
  "prompt": "PAIR: K7P9=X3M1. QUERY: K7P9 ->",
  "response": "X3M1",
  "context_pairs": [
    {"key": "K7P9", "value": "X3M1", "is_target": true, "key_token_id": 78, "val_token_id": 91}
  ],
  "query_key": "K7P9",
  "query_key_token_id": 78,
  "target_value": "X3M1",
  "target_val_token_id": 91,
  "num_distractors": 0,
  "token_offset_distance": 6,
  "total_prompt_tokens": 12,
  "single_token_entity_verified": true,
  "token_frequency_tier": "TIER_2_MEDIUM",
  "split_deterministic_seed": 420042
}
```

---

## 6. Volume, Split Construction, and Statistical Budget

| Split | Proportion | Record Count | Task Distribution | Token Pool Source | Evaluation Regime |
|---|---|---|---|---|---|
| **Train** | **80.0%** | **4,000** | TF-01 to TF-05 (balanced 800/task) | $\mathcal{E}_{\text{train}}$ only | Supervised Training |
| **Validation** | **10.0%** | **500** | All 8 Task Families (~62/task) | $\mathcal{E}_{\text{train}}$ combinations | Hyperparameter / Checkpoint Monitoring |
| **Test (OOD)** | **10.0%** | **500** | All 8 Task Families (~62/task) | $\mathcal{E}_{\text{test}}$ strictly disjoint | Final Generalization Benchmark |
| **Total** | **100.0%** | **5,000** | Factorial Coverage | Enforced Disjointness | Verified via Pre-Flight Auditor |

---

## 7. Cryptographic Leakage Prevention & Audit Directives

A prospective generation script must enforce the following cryptographic and relational integrity checks:

1. **Exact-Match Key/Value String Intersect:**
   $$\text{Keys}(\text{Train}) \cap \text{Keys}(\text{Test}) = \emptyset, \quad \text{Values}(\text{Train}) \cap \text{Values}(\text{Test}) = \emptyset$$
2. **Cross-Split Hash Verification:**
   Compute SHA-256 digests for all serialized prompt strings. Ensure zero prompt collision between Train, Validation, and Test splits.
3. **Single-Token Encoding Assertion:**
   Assert that `len(tokenizer.encode(key)) == 1` and `len(tokenizer.encode(value)) == 1` for all entity instances. Any violation aborts generation with a `TOKENIZER_FRAGMENTATION_DEFECT`.
4. **Reproducibility Classification:**
   Must achieve Tier 1 (`GENERATION_DETERMINISTIC`) under `docs/data/DATASET_REPRODUCIBILITY.md`:
   - Single seed-based deterministic generator (`seed=42`).
   - Zero network access or external API calls.
   - Platform-independent sorting and canonical JSON formatting.

---

## 8. Dataset Release Restrictions & Governance Status

- **Dataset Identifier (Prospective):** `DS-CAND01-SYNTH-V1`
- **Governance Classification:** `SPECIFICATION_ONLY` / `INTERNAL_ONLY`
- **Release Clearances:** Synthetic datasets generated for candidate exploration are strictly restricted to internal development. In accordance with `docs/data/DATASET_RELEASE_CHECKLIST.md`, no synthetic experimental dataset may be released publicly without written licensing clearance and dataset index registration.
- **Execution Mandate:** This document specifies dataset architecture only. **No dataset generation is authorized by this document.**

