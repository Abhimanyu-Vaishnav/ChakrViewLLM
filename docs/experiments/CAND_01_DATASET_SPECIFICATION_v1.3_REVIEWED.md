# CHAKRVIEW — CAND_01 DATASET SPECIFICATION v1.3 (REVIEWED)
## Synthetic Key-Value Corpus Design, Token Disjointness, and Leakage Prevention

**Document ID:** `DOC-EXP-CAND01-DAT-004`  
**Location:** `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.3_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.2_REVIEWED.md`, `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Target Architecture:** ChakrMicro Baseline (Vocab: 4,093 active BPE tokens)  
**Parent Governance Standards:** `docs/data/DATASET_GOVERNANCE_INDEX.md`, `docs/data/DATASET_REPRODUCIBILITY.md`  
**Status:** `SPECIFICATION_ONLY` (Implementation Plan Only — Execution Strictly Unapproved)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Scope (v1.3 Targeted Redesign)

This document represents the formal v1.3 scientific revision of `CAND_01_DATASET_SPECIFICATION_v1.2_REVIEWED.md`. It incorporates specific architectural and mathematical resolutions to independent audit findings (ADV-01 and ADV-02):
- **Correction ADV-01 (TF-01 Cardinality Resolution & Mathematical Feasibility):** Resolved the combinatorial exhaustion in Task Family TF-01 (Direct Copy). Formally excluded TF-01 from Partitions B and C (Recombination Splits) because identity copying ($X \to X$) contains no relational bindings to recombine. Allocated 350 examples in Partition A (Train) and 50 examples in Partition D (Disjoint-Token), mathematically guaranteeing prompt-disjointness ($350 \le 450$ and $50 \le 150$) with zero duplicate leakage across splits.
- **Correction ADV-02 (Frequency Stratification Gating):** Marked frequency tier cutoffs as `BLOCKED_PENDING_DATA` rather than prescribing arbitrary logarithmic thresholds. Formulated an empirical percentile protocol requiring empirical histogram calculation on tokenized Stage C data prior to finalizing token pool boundaries.
- **Accurate Realism in Partition D:** Formally classified Partition D as measuring **"disjoint-token in-context copying"**, acknowledging that target tokens are physically presented in the context window.

---

## 2. Corpus Engineering Principles & Token Selection

### 2.1 Single-Token Guarantee
Every key and every value must encode to **exactly one discrete BPE token ID** in `data/experiments/vocab_4096`. This guarantees that:
$$\forall (k, v) \in \mathcal{E}, \quad \text{len}(\text{tokenize}(k)) = 1 \quad \land \quad \text{len}(\text{tokenize}(v)) = 1$$
This isolates associative retrieval from tokenizer subword fragmentation.

### 2.2 Token Selection & Frequency Stratification Policy (Resolution ADV-02)
Under `data/experiments/vocab_4096` (4,093 active tokens):
- Tokens 0, 1, 2 are reserved (`<BOS>`, `<EOS>`, `<PAD>`).
- Tokens 3 to 258 are raw byte tokens.
- Tokens 259 to 4092 are multi-byte merged tokens.

To construct the candidate single-token entity pool $\mathcal{E}$ ($|\mathcal{E}| = 600$ tokens):
1. **Filtering:** Exclude whitespace tokens, control tokens, punctuation delimiters, and high-frequency English syntax morphemes (`the`, `is`, `at`, `in`).
2. **Frequency Measurement Protocol:** When authorized, extract empirical token frequencies $C(t)$ across `data/tokenized/stage_c/validation/shard_*.bin`.
3. **Percentile Stratification (Replacing Fabricated Log Thresholds):**  
   Once empirical counts are extracted, rank candidate tokens by count and partition into three equal-count percentile tiers:
   - **Tier 1 (High Frequency):** Top 33% of filtered candidate single tokens.
   - **Tier 2 (Medium Frequency):** Middle 33% of filtered candidate single tokens.
   - **Tier 3 (Low Frequency):** Bottom 34% of filtered candidate single tokens.
4. **Governance Status:** Because raw token counts have not yet been extracted, frequency cutoffs are classified as **`BLOCKED_PENDING_DATA`** until script execution is formally authorized.

---

## 3. Mathematical Cardinality Analysis & Partition Allocation (Resolution ADV-01)

### 3.1 Combinatorial Cardinality Analysis
- Available Training Tokens: $|\mathcal{E}_{\text{train}}| = 450$ tokens.
- Available Test Tokens: $|\mathcal{E}_{\text{test}}| = 150$ tokens (disjoint from $\mathcal{E}_{\text{train}}$).

| Task Family | Prompt Structure | Maximum Combinatorial Capacity ($\mathcal{E}_{\text{train}}$) | Required Records (Train / Val / Test Recomb) | Feasibility Result |
|---|---|---|---|:---:|
| **TF-01** (Direct Copy) | `COPY: X -> X` | Exactly 450 unique prompts | Train: 350 / Val: 0 / Test Recomb: 0 | **FEASIBLE ($350 \le 450$)** |
| **TF-02** (Single Pair) | `PAIR: K=V. QUERY: K ->` | $450 \times 449 = \mathbf{202,050}$ unique pairs | Train: 913 / Val: 71 / Test Recomb: 71 | **FEASIBLE ($1,055 \ll 202,050$)** |
| **TF-03** (Multi-Pair) | `PAIRS: K1=V1, K2=V2...` | $> 10^7$ unique combinations | Train: 913 / Val: 71 / Test Recomb: 71 | **FEASIBLE** |
| **TF-04** (Distractor) | Context with 1, 3, 7 distractors | $> 10^8$ unique combinations | Train: 912 / Val: 71 / Test Recomb: 71 | **FEASIBLE** |
| **TF-05** (Similar-Key) | Prefix collision disambiguation | $> 1,000$ prefix pairs | Train: 912 / Val: 71 / Test Recomb: 71 | **FEASIBLE** |
| **TF-06** (Recombination) | Novel combination of seen pairs | $> 10^5$ unique combinations | Train: 0 / Val: 72 / Test Recomb: 72 | **FEASIBLE** |
| **TF-07** (Offset Scaling) | Distance padded 10 to 200 tokens | $> 10^5$ unique combinations | Train: 0 / Val: 72 / Test Recomb: 72 | **FEASIBLE** |
| **TF-08** (Context Scaling)| Context packed 64 to 400 tokens | $> 10^5$ unique combinations | Train: 0 / Val: 72 / Test Recomb: 72 | **FEASIBLE** |

### 3.2 Four-Partition Allocation Architecture (Total: 5,500 Records)

```
                              [ TOTAL CORPUS: 5,500 RECORDS ]
                                             |
    +------------------------+---------------+------------------------+------------------------+
    |                        |                                        |                        |
[ PARTITION A: TRAIN ]  [ PARTITION B: VAL RECOMB ]      [ PARTITION C: TEST RECOMB ] [ PARTITION D: TEST DISJOINT ]
  4,000 records            500 records                              500 records              500 records
  Keys/Vals: E_train       Keys/Vals: E_train                       Keys/Vals: E_train       Keys/Vals: E_test (Disjoint)
  Tuples: S_train          Tuples: S_val (Held out)                 Tuples: S_test (Held out) Tuples: S_disjoint (Disjoint)
  TF-01: 350 records       TF-01: EXCLUDED                          TF-01: EXCLUDED          TF-01: 50 records
  TF-02-05: 912-913/task   TF-02-08: 71-72/task                     TF-02-08: 71-72/task     TF-01-08: 62-63/task
  Supervised Training      Model Selection / Diagnostics            In-Distribution Test     Disjoint-Token Test
```

---

## 4. Mathematical Exposure Rules & Leakage Prevention

A generation script must strictly verify the following mathematical invariants prior to serializing any dataset partition:

1. **Token Pool Disjointness Invariant:**
   $$\mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{test}} = \emptyset$$
   No token ID in $\mathcal{E}_{\text{test}}$ may appear anywhere in Partitions A, B, or C.
2. **Tuple / Association Disjointness Invariant:**
   Let $\mathcal{P}(R) = \{(k, v) \mid (k, v) \in R.\text{context\_pairs}\}$ be the set of key-value pairs defined in record $R$. Then:
   $$\left( \bigcup_{R \in \text{Partition A}} \mathcal{P}(R) \right) \cap \left( \bigcup_{R \in \text{Partition B}} \mathcal{P}(R) \right) = \emptyset$$
   $$\left( \bigcup_{R \in \text{Partition A}} \mathcal{P}(R) \right) \cap \left( \bigcup_{R \in \text{Partition C}} \mathcal{P}(R) \right) = \emptyset$$
   No key-value binding observed during training may ever appear in validation or test recombination partitions.
3. **Prompt String Cryptographic Hash Invariant:**
   $$\text{SHA256}(\text{Prompt}_i) \neq \text{SHA256}(\text{Prompt}_j) \quad \forall i \in \text{Train}, j \in \{\text{Val, Test}\}$$
   Zero exact prompt collisions across splits.
4. **Permutation Randomization Rule:**
   For multi-pair sequences (TF-03, TF-04, TF-05, TF-08), the positional index of the target key-value pair within the prompt context must be uniformly distributed across context slots ($p = 1/K$), preventing ordinal recency shortcuts.

---

## 5. Standardized Record Schema (JSON Lines)

```json
{
  "id": "CAND01-PARTC-TF04-00128",
  "partition": "PARTITION_C_TEST_RECOMB",
  "task_family": "TF-04",
  "task_name": "DISTRACTOR_RESISTANCE",
  "data_role": "TEST",
  "prompt": "PAIRS: D1=N4, K7=X3, D2=R9. QUERY: K7 ->",
  "response": "X3",
  "context_pairs": [
    {"key": "D1", "value": "N4", "key_id": 78, "val_id": 92, "is_target": false},
    {"key": "K7", "value": "X3", "key_id": 85, "val_id": 114, "is_target": true},
    {"key": "D2", "value": "R9", "key_id": 90, "val_id": 105, "is_target": false}
  ],
  "query_key": "K7",
  "query_key_token_id": 85,
  "target_value": "X3",
  "target_val_token_id": 114,
  "target_pair_index": 1,
  "num_distractors": 2,
  "token_offset_distance": 14,
  "total_prompt_tokens": 28,
  "single_token_entity_verified": true,
  "token_frequency_tier": "TIER_2_MEDIUM",
  "split_deterministic_seed": 420128
}
```

---

## 6. Dataset Governance, Provenance, and Licensing

- **Dataset Identifier (Prospective):** `DS-CAND01-SYNTH-V1`
- **Reproducibility Classification:** Tier 1 (`GENERATION_DETERMINISTIC`) under `docs/data/DATASET_REPRODUCIBILITY.md`:
  - 100% synthetically generated via deterministic pseudorandom number generator (`seed=42`).
  - Zero external HTTP dependencies, API calls, or web scraping.
  - Generates identical file SHA-256 digests across all platforms.
- **License & Release Clearances:**
  - Classified strictly as **`INTERNAL_ONLY`** and **`SPECIFICATION_ONLY`**.
  - In accordance with `docs/data/DATASET_RELEASE_CHECKLIST.md`, this synthetic experimental dataset is restricted to internal research and cannot be distributed externally.
- **Execution Mandate:** This document specifies dataset architecture only. **No dataset generation is authorized by this document.**

