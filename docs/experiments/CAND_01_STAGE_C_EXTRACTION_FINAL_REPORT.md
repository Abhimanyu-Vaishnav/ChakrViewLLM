# CHAKRVIEW — CAND_01 STAGE C FREQUENCY EXTRACTION FINAL REPORT

**Document ID:** `REP-EXP-CAND01-EXT-001`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Execution Timestamp:** 2026-11-03T23:15:10+05:30  
**Run ID:** `run_1793727910`  
**Output Directory:** `experiments/cand01/production_extraction/run_1793727910`  
**Execution Mode:** Authorized Physical Production Extraction (`synthetic_mode=False`)  
**Status:** **`PASS`** (All integrity, reconciliation, and security boundaries verified)  

---

## 1. Executive Summary

Physical Stage C frequency extraction for CAND_01 was executed under explicit project owner authorization and verified through the canonical authorization gate.

All 27 production binary shards in `data/tokenized/stage_c/train/` were validated cryptographically against their registered SHA-256 digests prior to decoding. Frequency accumulation reconciled exactly to 6,651,093 uint16 tokens. A total of 754 candidate entity tokens survived the v1.1 lexical and duplicate-string mutual disqualification filter, of which 707 exhibited non-zero frequency in the corpus.

Stratification into high-, medium-, and low-frequency tiers and the allocation of the 450 training candidates ($E_{train}$) and 150 held-out test candidates ($E_{test}$) were completed deterministically and saved atomically to isolated outputs.

---

## 2. Environment and Hardware Telemetry

- **Host Architecture**: Intel i9 13th Gen (14 physical cores, 20 logical threads)
- **Host Physical RAM**: 31.64 GB total, ~12.2 GB free headroom
- **Execution Engine**: Native Python 3.14.7 on Windows x86_64, Single-threaded
- **Elapsed Wall-Clock Time**: 0.2169 seconds
- **Process CPU Time**: 0.0781 seconds
- **Peak RSS Memory**: 34.81 MB (vastly below the 512 MB safety ceiling)

---

## 3. Shard and Token Count Reconciliation

| Metric | Declared in Specification / Metadata | Observed / Counted in Physical Run | Reconciliation Status |
| :--- | :--- | :--- | :---: |
| **Shard Count** | 27 shards | 27 shards processed | **EXACT MATCH** |
| **Total Tokens** | 6,651,093 tokens | 6,651,093 tokens counted | **EXACT MATCH** |
| **Token Dtype** | `uint16` little-endian (`<u2`) | `uint16` little-endian | **VERIFIED** |
| **Total Shard Bytes**| 13,302,186 bytes | 13,302,186 bytes ($6,651,093 \times 2$) | **EXACT MATCH** |
| **Shard Checksums** | 27 SHA-256 digests in metadata | 27 SHA-256 pre-read checks passed | **100% MATCH** |
| **Source Metadata Hash**| `36034d8c756fceb98a8de4dc0896da31f5775efe6d2a175f53bb1d2b454acfb3` | Matched on disk | **MATCH** |

---

## 4. Vocabulary Filtering & Stratification Results

- **Input Vocabulary**: `data/experiments/vocab_4096/vocab.json` (SHA-256 `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)
- **Total Entries Evaluated**: 4,093
- **Rejections by Contract**:
  - Reserved IDs (< 259): 256
  - Unicode Replacement (`\ufffd`): 294
  - Length Under 3 Characters: 732
  - Non-Alphanumeric: 2,057
  - Whitespace / Control Characters: 0
  - Duplicate Decoded Strings: 0
- **Surviving Eligible Tokens**: 754 (margin: +154 over required 600)
- **Non-Zero Eligible Tokens**: 707
- **Zero-Frequency Excluded Tokens**: 47
- **Tier Allocations**:
  - Tier 1 (High Frequency): 235 tokens (counts range from 44,980 down to 876)
  - Tier 2 (Medium Frequency): 235 tokens (counts range from 872 down to 184)
  - Tier 3 (Low Frequency): 237 tokens (counts range from 182 down to 1)
- **Experimental Pools**:
  - $E_{train}$ Pool: 450 tokens (150 from each tier)
  - $E_{test}$ Pool: 150 tokens (50 from each tier)
  - **Disjointness**: $E_{train} \cap E_{test} = \emptyset$ (Strictly 0 intersecting token IDs)

---

## 5. Output Artifact Verification

All generated outputs are isolated under `experiments/cand01/production_extraction/run_1793727910/`:

| Filename | Byte Size | SHA-256 Digest |
| :--- | :---: | :--- |
| `token_frequencies_stage_c_train.json` | 62,392 | `d3caff1855749d230e382501bf8df8331f0af7b9081af39658b9268787d49fc6` |
| `candidate_entity_pools.json` | 109,452 | `628873f424d4f72cf57eb06f99771b488f667b820dfcdfe3d6375affb7fbc340` |
| `extraction_manifest.json` | 1,554 | `d22cd8a38aa535aee3c2f134e77b9f987d131013af4053370e7eb73b062314ba` |

---

## 6. Protected-State and System Boundary Invariants

- **Source Shards**: Untouched and unmodified (`data/tokenized/stage_c/train/*.bin`).
- **Canonical Model Hashes**:
  - `chakrview/brain/model.py`: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` (MATCH)
  - `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (MATCH)
  - Baseline Step 10k: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (MATCH)
  - User `.gitignore`: `700f3d8b6d8c5c216884f1808f5d5c7da17282ce4d69f6e5d563f2ee49e4fda6` (MATCH)
- **Zero Model Operations**: Zero training steps, zero backward passes, zero inference.
- **Excluded Directories**: `experiments/overnight_runs/` was strictly excluded and unaccessed.

---

## 7. Final Verdict

**EXTRACTION VERDICT: `PASS`**  
The Stage C production candidate entity pools and corpus frequency mapping are validated, cryptographically bound, and ready for CAND_01 benchmark dataset generation.
