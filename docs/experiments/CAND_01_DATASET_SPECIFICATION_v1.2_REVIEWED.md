# CHAKRVIEW — CAND_01 DATASET SPECIFICATION v1.2 (REVIEWED)
## Synthetic Key-Value Corpus Design, Token Disjointness, and Leakage Prevention

**Document ID:** `DOC-EXP-CAND01-DAT-003`  
**Location:** `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.2_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Target Architecture:** ChakrMicro Baseline (Vocab: 4,093 active BPE tokens)  
**Parent Governance Standards:** `docs/data/DATASET_GOVERNANCE_INDEX.md`, `docs/data/DATASET_REPRODUCIBILITY.md`  
**Status:** `SPECIFICATION_ONLY` (Implementation Plan Only — Execution Strictly Unapproved)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Scope (v1.2)

This document represents the formal v1.2 scientific revision of `CAND_01_DATASET_SPECIFICATION_v1.1_REVIEWED.md`. It incorporates specific architectural and statistical resolutions to independent audit findings (F-01 and F-06):
- **Resolution F-01 (Four-Partition Split Architecture):** Formally replaced the three-way split with an explicit four-partition design: Train (4,000), Validation Recombination (500), Test Recombination (500), and Test Disjoint OOD (500). Total dataset volume expanded to 5,500 records to eliminate validation/test distributional asymmetry.
- **Resolution F-06 (Log-Frequency Binning & Stratification):** Defined token pool stratification using log-frequency cutoffs derived from base pretraining corpus token counts rather than arbitrary uniform quantiles, preventing frequency-norm bias from confounding associative accuracy.
- **Exposure Rules Formally Specified:** Formulated exact mathematical rules preventing token, association, and template leakage across partitions.

---

## 2. Corpus Engineering Principles & Token Selection

### 2.1 Single-Token Guarantee
Every key and every value must encode to **exactly one discrete BPE token ID** in `data/experiments/vocab_4096`. This guarantees that:
$$\forall (k, v) \in \mathcal{E}, \quad \text{len}(\text{tokenize}(k)) = 1 \quad \land \quad \text{len}(\text{tokenize}(v)) = 1$$
This isolates associative retrieval from the multi-step autoregressive error accumulation identified in Milestone 5.1.

### 2.2 Token Selection & Log-Frequency Stratification (Resolution F-06)
Under `data/experiments/vocab_4096` (4,093 active tokens):
- Tokens 0, 1, 2 are reserved (`<BOS>`, `<EOS>`, `<PAD>`).
- Tokens 3 to 258 are raw byte tokens.
- Tokens 259 to 4092 are multi-byte merged tokens.

To construct the candidate single-token entity pool $\mathcal{E}$ ($|\mathcal{E}| = 600$ tokens):
1. **Filtering:** Exclude whitespace-only tokens, control tokens, and high-frequency English syntax morphemes (`the`, `is`, `at`, `in`).
2. **Frequency Measurement:** Obtain empirical token counts $C(t)$ from the base Stage C pretraining validation stream.
3. **Log-Frequency Stratification:** Stratify $\mathcal{E}$ into three log-frequency tiers:
   - **Tier 1 (High Frequency):** $\log_{10} C(t) \ge 4.0$ (Frequent base vocabulary).
   - **Tier 2 (Medium Frequency):** $2.5 \le \log_{10} C(t) < 4.0$ (Moderate base vocabulary).
   - **Tier 3 (Low Frequency):** $\log_{10} C(t) < 2.5$ (Rare / byte-level vocabulary).
4. **Partition Balancing:** Training and test entity pools are constructed by stratified random sampling across these three log-frequency tiers in equal proportions.

---

## 3. The Four-Partition Architecture & Split Allocation (Resolution F-01)

To prevent distributional mismatch between validation and testing, the dataset implements four disjoint partitions totaling 5,500 records:

```
                              [ TOTAL CORPUS: 5,500 RECORDS ]
                                             |
    +------------------------+---------------+------------------------+------------------------+
    |                        |                                        |                        |
[ PARTITION A: TRAIN ]  [ PARTITION B: VAL RECOMB ]      [ PARTITION C: TEST RECOMB ] [ PARTITION D: TEST OOD ]
  4,000 records            500 records                              500 records              500 records
  Keys/Vals: E_train       Keys/Vals: E_train                       Keys/Vals: E_train       Keys/Vals: E_test (Disjoint)
  Tuples: S_train          Tuples: S_val (Held out)                 Tuples: S_test (Held out) Tuples: S_ood (Disjoint)
  Task Families 1-5        All 8 Task Families                      All 8 Task Families      All 8 Task Families
  Supervised Training      Model Selection / Diagnostics            In-Distribution Test     Out-of-Distribution Test
```

### Partition Specifications

| Partition | Split Role | Record Count | Token Pool Source | Association / Tuple Constraints | Target Evaluation Purpose |
|---|---|---|---|---|---|
| **A** | **Train** | **4,000** | $\mathcal{E}_{\text{train}}$ (450 tokens) | Tuples $S_{\text{train}} = \{(K_i, V_j)\}$ | Supervised gradient optimization |
| **B** | **Validation Recomb** | **500** | $\mathcal{E}_{\text{train}}$ (450 tokens) | $S_{\text{val}} \cap S_{\text{train}} = \emptyset$ | Checkpoint monitoring & early stopping |
| **C** | **Test Recomb** | **500** | $\mathcal{E}_{\text{train}}$ (450 tokens) | $S_{\text{test}} \cap (S_{\text{train}} \cup S_{\text{val}}) = \emptyset$ | In-distribution recombination benchmark ($A_{\text{test,recomb}}$) |
| **D** | **Test OOD** | **500** | $\mathcal{E}_{\text{test}}$ (150 tokens) | $\mathcal{E}_{\text{test}} \cap \mathcal{E}_{\text{train}} = \emptyset$ | Disjoint token generalization benchmark ($A_{\text{test,ood}}$) |
| **Total** | **Corpus** | **5,500** | **600 Tokens** | **Strict Cryptographic Disjointness** | **Factorial Evaluation Coverage** |

---

## 4. Mathematical Exposure Rules & Leakage Prevention

A generation script must strictly verify the following mathematical invariants prior to serializing any dataset partition:

1. **Token Pool Disjointness Invariant:**
   $$\mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{test}} = \emptyset$$
   No token ID in $\mathcal{E}_{\text{test}}$ may appear anywhere in Partitions A, B, or C (neither as key, value, nor distractor).
2. **Tuple / Association Disjointness Invariant:**
   Let $\mathcal{P}(R) = \{(k, v) \mid (k, v) \in R.\text{context\_pairs}\}$ be the set of key-value pairs defined in record $R$. Then:
   $$\left( \bigcup_{R \in \text{Partition A}} \mathcal{P}(R) \right) \cap \left( \bigcup_{R \in \text{Partition B}} \mathcal{P}(R) \right) = \emptyset$$
   $$\left( \bigcup_{R \in \text{Partition A}} \mathcal{P}(R) \right) \cap \left( \bigcup_{R \in \text{Partition C}} \mathcal{P}(R) \right) = \emptyset$$
   No key-value binding observed during training may ever appear in validation or test recombination partitions.
3. **Prompt String Cryptographic Hash Invariant:**
   $$\text{SHA256}(\text{Prompt}_i) \neq \text{SHA256}(\text{Prompt}_j) \quad \forall i \in \text{Train}, j \in \{\text{Val, Test}\}$$
   Zero exact prompt collisions across splits.
4. **Permutation Randomization Rule:**
   For multi-pair sequences (TF-03, TF-04, TF-05, TF-08), the positional index of the target key-value pair within the prompt context must be uniformly distributed across all available context slots ($p = 1/K$), preventing ordinal recency shortcuts.

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

