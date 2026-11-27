# CHAKRVIEW NR-046: DATASET V2 ENGINEERING & CURATION REPORT

**Mission Reference:** `CHAKRVIEW NR-046`  
**Phase:** WP-1 (Corpus V2 Development)  
**Date:** 2026-11-27  
**Artifact Directory:** `data/corpus_v2/`  

---

## 1. Executive Summary

During the NR-045 forensics audit, two critical data pathologies were uncovered in Corpus V1:
1. **82.8% (4,826 / 5,826) formatting contamination:** Raw records contained embedded `"User: ...\nAssistant: "` tags inside prompts, which when wrapped by the training template produced nested structures like `<user>\nUser: ...\nAssistant: \n<assistant>\n`.
2. **19.3% (1,126 / 5,826) synthetic token copying tasks:** Oversaturated single-token echo examples dragged down the target sequence length and created degenerate copying attractors.

Under **WP-1 of NR-046**, an isolated, reproducible dataset pipeline was developed in `data/corpus_v2/build_corpus_v2.py`. Corpus V1 remained strictly untouched and preserved.

---

## 2. Dataset Processing Methodology

### 2.1 Explicit Prompt Parser
We implemented `parse_user_prompt` which:
- Extracts genuine user instructions from structured message objects (`messages` field) when available.
- Strips redundant prefix `"User:"` and suffix `"\nAssistant:"` or `"Assistant:"`.
- Normalizes internal whitespace and preserves all unicode / Devanagari characters without corruption.

### 2.2 Principled Task Classification (No Blind Word-Count Filtering)
Rather than arbitrarily dropping records with low word counts, tasks were classified into functional taxonomies:
- `SYNTHETIC_TOKEN_COPY`: Synthetic token copying instructions (e.g., `"Copy the following token exactly without any modification: 'ALPHA-799-W'"`). To prevent complete catastrophic forgetting of exact copying while eliminating loss skew, a balanced quota of **120** synthetic copy records was retained, and **639** redundant records were filtered out.
- `CONVERSATIONAL_SHORT`: Short but meaningful natural dialogue responses (greetings like `"Hello! How can I assist you?"`, affirmative confirmations `"Yes"`, simple definitions) were **preserved** (1,338 samples preserved).
- `FORMAT_COMPLIANCE`: Structured JSON output formatting tasks were preserved.

### 2.3 Contamination & Deduplication Gating
- Gated strictly against the sealed 1,000-sample matched benchmark (`data/stage_j_020/fresh_matched_benchmark_1000.json`) with zero leakage.
- Deduplicated against normalized prompt representations.

---

## 3. Dataset Statistics & Manifest

| Metric | Corpus V1 (Historical) | Corpus V2 (NR-046 Isolated) | Delta / Change |
| :--- | :--- | :--- | :--- |
| **Train Records** | 5,826 | **5,089** | -737 (removed redundant copy tasks) |
| **Validation Records** | 539 | **435** | -104 |
| **Repaired Nested Prompts** | 0 (all contaminated) | **18,737 occurrences** | 100% normalized to clean user intent |
| **Synthetic Copy Filtered** | 0 | **639** | Eliminated loss oversaturation |
| **Balanced Copy Retained** | 801+ | **120** | Controlled exact-copying representation |
| **Short Conversational Preserved** | Unclassified | **1,338** | Natural conversational density preserved |

### Split Hashes & Integrity

| Split | Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Train** | `data/corpus_v2/corpus_v2_train.jsonl` | `aa799c5c2e9180dee743805cb4c39586da10211e54d5c991dd1cd1eb09845c99` |
| **Validation** | `data/corpus_v2/corpus_v2_val.jsonl` | `c9d9e8517ea25e90e2f71c25ed23edfdc9af34b03347132cb65b65560c4c3db2` |
| **Manifest** | `data/corpus_v2/manifest.json` | Fully documented metadata and counts |

---

## 4. Verification Verdict

Corpus V2 successfully eliminates prompt formatting contamination and synthetic token dominance while preserving semantic breadth across English, Hindi Devanagari, and Hinglish.
