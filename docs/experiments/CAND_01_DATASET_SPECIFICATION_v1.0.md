# CHAKRVIEW — CAND_01 DATASET SPECIFICATION v1.0
## Synthetic Key-Value Corpus Design, Token Disjointness, and Leakage Prevention

**Document ID:** `DOC-EXP-CAND01-DAT-001`  
**Location:** `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Target Architecture:** ChakrMicro Baseline (Vocab: 4,093 active BPE tokens)  
**Parent Governance Standards:** `docs/data/DATASET_GOVERNANCE_INDEX.md`, `docs/data/DATASET_REPRODUCIBILITY.md`  
**Status:** `SPECIFICATION_ONLY` (Implementation Plan Only — No Dataset Generation Authorized)  
**Date:** 2026-10-31  

---

## 1. Corpus Purpose and Engineering Philosophy

The CAND_01 dataset is a purely synthetic, algorithmic sequence corpus designed to train and benchmark associative binding and copying capabilities in ChakrMicro. 

### Core Design Rules
1. **Zero Natural Language Boilerplate:** Prohibits conversational markers, politeness phrases, introductory sentences, and filler words.
2. **Single-Token Entity Guarantee:** Every key and every value must encode to **exactly one discrete BPE token ID** in `data/experiments/vocab_4096`. This eliminates the compound error probability introduced by subword fragmentation (identified in M5.1).
3. **Disjoint Partitioning:** The key and value token IDs used in the test split must be **strictly disjoint** from those used in the training split.
4. **Permutation Invariance:** For multi-pair sequences, the order in which key-value pairs are listed in context must be randomized to prevent ordinal shortcut exploitation (e.g., "always copy the last value").

---

## 2. Token Pool Construction & Single-Token Guarantee

Under `data/experiments/vocab_4096`, the active vocabulary size is 4,093 tokens:
- Token IDs 0, 1, 2 are reserved (`<BOS>`, `<EOS>`, `<PAD>`).
- Token IDs 3 to 258 are raw byte tokens (e.g., individual ASCII characters, numerals, punctuation).
- Token IDs 259 to 4092 are multi-byte merged tokens (Hindi morphemes, English subwords, frequent n-grams).

### 2.1 Token Selection Policy
To guarantee single-token entity encoding:
1. Select an entity pool $\mathcal{E}$ consisting exclusively of tokens $t \in [3, 4092]$ whose decoded byte sequence forms a standalone, non-whitespace, alphanumeric or morphemic unit.
2. Filter out tokens that represent common English syntactic words (e.g., `the`, `is`, `at`, `in`) to prevent pre-existing language model priors from skewing association strength.
3. Partition the verified single-token pool $\mathcal{E}$ ($|\mathcal{E}| = 600$ tokens) into two mutually exclusive subsets:
   - **Training Token Pool ($\mathcal{E}_{\text{train}}$):** 450 tokens ($75\%$).
   - **Test Token Pool ($\mathcal{E}_{\text{test}}$):** 150 tokens ($25\%$).
   - **Intersection Constraint:** $\mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{test}} = \emptyset$.

---

## 3. Dataset Schemas and Record Formats

All records are serialized as standardized JSON Lines (`.jsonl`), complying with the ChakrView SFT contract.

### 3.1 Field Schema Definition
```json
{
  "id": "CAND01-TF02-00042",
  "task_family": "TF-02",
  "task_name": "SINGLE_KEY_VALUE_ASSOCIATION",
  "data_role": "TRAIN",
  "prompt": "PAIR: K7P9=X3M1. QUERY: K7P9 ->",
  "response": "X3M1",
  "context_pairs": [
    {"key": "K7P9", "value": "X3M1", "is_target": true}
  ],
  "query_key": "K7P9",
  "target_value": "X3M1",
  "num_distractors": 0,
  "token_offset_distance": 6,
  "total_prompt_tokens": 12,
  "single_token_entity_verified": true,
  "split_deterministic_seed": 420042
}
```

### 3.2 Canonical Syntax Specifications Across Task Families

1. **TF-01: Direct Token Copy**
   - Format: `COPY: [VAL] ->`
   - Response: `[VAL]`
   - Example: `COPY: M8 ->` / `M8`

2. **TF-02: Single Key-Value Association**
   - Format: `PAIR: [KEY]=[VAL]. QUERY: [KEY] ->`
   - Response: `[VAL]`
   - Example: `PAIR: K1=V1. QUERY: K1 ->` / `V1`

3. **TF-03: Multi-Pair Association (2 to 4 pairs)**
   - Format: `PAIRS: [K1]=[V1], [K2]=[V2], [K3]=[V3]. QUERY: [K2] ->`
   - Response: `[V2]`

4. **TF-04: Distractor Resistance (1, 3, 7 distractors)**
   - Format: `PAIRS: [D1]=[W1], [K_tgt]=[V_tgt], [D2]=[W2]. QUERY: [K_tgt] ->`
   - Response: `[V_tgt]`

5. **TF-05: Similar-Key Discrimination**
   - Format: `PAIRS: [KEY_A]=[V1], [KEY_B]=[V2]. QUERY: [KEY_A] ->`
   - Constraint: `Levenshtein(KEY_A, KEY_B) == 1`
   - Response: `[V1]`

6. **TF-06: Unseen Combination Rebinding**
   - Prompt draws keys and values from training pool $\mathcal{E}_{\text{train}}$, but pairs them in combinations that never occurred during training.

7. **TF-07: Context-Position Variation**
   - Injects controlled neutral padding tokens between the definition pair and the query to shift token offset distance $D \in \{10, 50, 100, 200\}$.

8. **TF-08: Context-Length Variation**
   - Populates context with increasing numbers of distractor pairs to scale total prompt length from 64 to 128, 256, and 400 tokens.

---

## 4. Volume, Split Construction, and Statistical Budget

To prevent overfitting while providing sufficient gradient steps, the dataset volume is scaled to match ChakrMicro's capacity:

| Split | Proportion | Record Count | Task Distribution | Token Disjointness |
|---|---|---|---|---|
| **Train** | **80.0%** | **4,000** | TF-01 through TF-05 (balanced 800/task) | $\mathcal{E}_{\text{train}}$ only |
| **Validation** | **10.0%** | **500** | All 8 Task Families (~62/task) | $\mathcal{E}_{\text{train}}$ combinations |
| **Test (OOD)** | **10.0%** | **500** | All 8 Task Families (~62/task) | $\mathcal{E}_{\text{test}}$ strictly disjoint |
| **Total** | **100.0%** | **5,000** | Complete Factorial Coverage | Enforced via Validator |

---

## 5. Leakage Prevention and Verification Protocol

A prospective generation script must enforce the following cryptographic and relational integrity checks before any file is saved:

1. **Exact-Match Key/Value String Intersect:**
   $$\text{Keys}(\text{Train}) \cap \text{Keys}(\text{Test}) = \emptyset, \quad \text{Values}(\text{Train}) \cap \text{Values}(\text{Test}) = \emptyset$$
2. **N-gram Fingerprinting:**
   Extract 3-gram and 5-gram token hashes across all records. Ensure zero cross-split collisions on prompt sequences.
3. **Target Token Isolation:**
   Validate that `target_value` appears exactly once in the response, and that the prompt contains the target value exactly once (in the defining key-value pair) and never in distractor values.
4. **Reproducibility Tier:**
   Must achieve Tier 1 (`GENERATION_DETERMINISTIC`) under `docs/data/DATASET_REPRODUCIBILITY.md`:
   - Single seed-based deterministic generator (`seed=42`).
   - Zero external HTTP network calls.
   - Platform-independent sorting and serialization.

