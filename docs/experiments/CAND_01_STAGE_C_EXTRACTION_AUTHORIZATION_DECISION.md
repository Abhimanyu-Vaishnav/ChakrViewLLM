# CHAKRVIEW — CAND_01 STAGE C FREQUENCY EXTRACTION
## FORMAL AUTHORIZATION DECISION PACKAGE

**Document ID:** `DEC-EXP-CAND01-EXT-001`  
**Location:** `docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Repository:** `D:\Project\ChakrView` | **Branch:** `main` | **Target HEAD:** `5aeaa2734de5a98802b96afe36324765f3244738`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Target Architecture:** Frozen ChakrMicro Baseline (3.44M params, 6L/192D/6H, RoPE, Tied Embeddings, Step 10,000 Checkpoint)  
**Review Type:** Formal Governance and Readiness Decision Review (Strictly Non-Executable)  
**Date:** 2026-11-03  

---

## 1. Executive Decision

### Formal Recommendation
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      FORMAL GOVERNANCE RECOMMENDATION                         ║
║                                CONDITIONAL GO                                 ║
║                                                                               ║
║  Status: READY FOR SCRIPT IMPLEMENTATION & REVIEW                             ║
║  Execution: UNAUTHORIZED_PENDING_SEPARATE_EXPLICIT_APPROVAL                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Executive Determination Summary
The proposed CAND_01 Stage C Frequency Extraction is recommended for **`CONDITIONAL GO`**. 

The scientific rationale, input data boundaries, mathematical token-count reconciliation, and vocabulary decoding algorithms are fully substantiated by repository code and manifest evidence:
1. **Token Discrepancy Formally Resolved:** The $+4{,}755$ token difference between manifest content tokens (`6,646,338`) and physical shard metadata (`6,651,093`) is definitively traced to `scripts/stage_c/acquire_and_ingest_stage_c.py` line 936, which appends exactly one `<EOS>` token (`token_ids.append(EOS_ID)`) per document across all 4,755 training documents.
2. **Vocabulary Eligibility Validated:** The v1.1 decoding and filtering contract successfully qualifies **754 candidate tokens** from `data/experiments/vocab_4096/vocab.json`, providing a viable $+154$ token margin over the required 600 tokens ($450$ train $+ 150$ test).
3. **Execution Prohibited:** This decision authorizes the **authoring, code review, and isolated synthetic testing** of the dedicated extraction runner (`scripts/extract_stage_c_frequencies.py`). It **DOES NOT authorize live execution** against physical Stage C binary shards, which remains strictly `UNAUTHORIZED_PENDING_SEPARATE_EXPLICIT_APPROVAL`.

---

## 2. Scope of Review

This review evaluated the technical and procedural requirements for extracting pretraining token counts $C(t)$ across Stage C training data:
- **Included in Scope:** Verification of repository manifests, tokenization pipelines, vocabularies, allocation formulas, and security boundaries.
- **Excluded from Scope:** Execution of extraction, opening binary shards, inspecting canonical checkpoints, evaluating model weights, generating production datasets, and model training.

---

## 3. Evidence Inventory & Epistemic Classification

Every fact, document, and parameter was traced and classified according to strict evidentiary standards:

| Evidence Element | Source / Artifact Reference | Classification | Evidentiary Basis |
| :--- | :--- | :---: | :--- |
| **Stage C Manifest Train Tokens** | `data/manifests/stage_c_manifest.json` line 12 | **VERIFIED_DIRECTLY** | Explicit JSON entry: `"train": 6646338`. |
| **Stage C Metadata Train Tokens** | `data/tokenized/stage_c/train/metadata.json` line 3 | **VERIFIED_DIRECTLY** | Explicit JSON entry: `"total_tokens": 6651093`. |
| **Stage C Document Count (4,755)**| `data/tokenized/stage_c/train/metadata.json` line 4 | **VERIFIED_DIRECTLY** | Explicit JSON entry: `"total_documents": 4755`. |
| **Shard Count & File Inventory** | `data/tokenized/stage_c/train/metadata.json` line 5 | **VERIFIED_DIRECTLY** | `"shard_count": 27`; shards `shard_00000.bin` to `shard_00026.bin` verified on disk. |
| **Document Delimiter Addition** | `scripts/stage_c/acquire_and_ingest_stage_c.py` line 936 | **VERIFIED_DIRECTLY** | Pipeline appends `EOS_ID` (1) to every document prior to sharding. |
| **Tokenizer Checksum Match** | `stage_c_manifest.json` & `metadata.json` | **VERIFIED_DIRECTLY** | Hash `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f` matches. |
| **Candidate Vocabulary Pool** | `data/experiments/vocab_4096/vocab.json` | **VERIFIED_DIRECTLY** | 4,093 active entries (IDs 3..4095); 754 pass v1.1 filters with zero duplicates. |
| **Synthetic Unit Tests (29/29)** | `tests/test_cand01_*.py` | **VERIFIED_DIRECTLY** | 18 infrastructure tests + 11 extraction spec tests passing (100%). |
| **Resource Limits (RAM $\le 512$MB)** | Package v1.1 §4.4 | **REPORTED_NOT_RECHECKED** | Plausible based on 0.5 MB shard size; unbenchmarked on cold disk. |
| **Timeout Ceiling ($\le 60$s)** | Package v1.1 §4.4 | **REPORTED_NOT_RECHECKED** | Conservative safe ceiling for Windows I/O latency. |
| **Two-Way Estimator Degeneracy** | `chakrview/eval/statistical_bootstrap.py` | **VERIFIED_DIRECTLY** | 35.2% mean degeneracy in crossed simulations; trips to `INCONCLUSIVE`. |

---

## 4. Definitive Resolution of the Stage C Token-Count Discrepancy

### 4.1 Underlying Source Analysis
The discrepancy was investigated by inspecting the authoritative ingestion pipeline [`scripts/stage_c/acquire_and_ingest_stage_c.py`](file:///d:/Project/ChakrView/scripts/stage_c/acquire_and_ingest_stage_c.py):
1. **Manifest Generation (Phase 6, lines 895–918):**
   - Content tokens are accumulated directly from raw document encodings:
     ```python
     for r in manifest_records:
         manifest_payload["split_distribution"][sp] += r["token_count"]
     ```
   - In `manifest_records`, `r["token_count"]` records pure document content tokens without boundary tokens. For the training split, this sums to exactly **6,646,338 tokens**.
2. **Production Sharding (Phase 7, lines 920–945):**
   - Shards are emitted via `ShardWriter`:
     ```python
     for d in split_docs[split_name]:
         token_ids = tokenizer.encode(d.text, add_bos=False, add_eos=False)
         token_ids.append(EOS_ID)  # EOS boundary contract (line 936)
         writer.add_document(token_ids)
     ```
   - Exactly one `<EOS>` token (`EOS_ID = 1`) is appended to every document in the stream.
3. **Exact Mathematical Proof:**
   $$\text{Shard Stream Tokens} = \text{Content Tokens} + (\text{Documents} \times 1)$$
   $$\text{Shard Stream Tokens} = 6{,}646{,}338 + 4{,}755 = \mathbf{6{,}651{,}093\ \text{tokens}}$$
   $$\Delta = 6{,}651{,}093 - 6{,}646{,}338 = \mathbf{4{,}755\ \text{tokens}}$$

### 4.2 Reconciliation Verdict
The explanation is **proved directly from pipeline implementation code**. The manifest records content tokens ($6{,}646{,}338$), while the physical binary shards record content tokens plus document boundary tokens ($6{,}651{,}093$). The physical extraction stream will count exactly **6,651,093 `uint16` tokens**.

---

## 5. Vocabulary Eligibility Contract Audit

The v1.1 decoding and filtering contract was tested directly against `data/experiments/vocab_4096/vocab.json`:
- **Total Vocabulary Entries:** 4,093 active tokens (IDs 3..4095; IDs 0..2 reserved).
- **Hex Decoding:** Successfully decodes 100% of entries using `bytes.fromhex(val).decode("utf-8", errors="replace")`.
- **Lexical Filters:**
  - Reserved range exclusion: IDs 0..258 (259 tokens removed).
  - Multi-byte tokens evaluated: IDs 259..4095 (3,834 tokens).
  - Length $\ge 3$ characters: 1,829 tokens survive.
  - Alphanumeric (`isalnum()`) without whitespace or delimiters: **754 tokens survive**.
  - Duplicate string tokens: Exactly 0 duplicates found among candidates.
- **Feasibility Margin:**
  $$\text{Eligible Tokens} = 754 \ge 600 \implies \text{Margin} = +154\ \text{tokens}$$
- **Finding:** The contract is mathematically feasible and provides sufficient candidate headroom.

---

## 6. Frequency-Tier Allocation Audit

The frozen integer tier allocation formula:
$$N_1 = \lfloor N / 3 \rfloor, \quad N_2 = \lfloor N / 3 \rfloor, \quad N_3 = N - N_1 - N_2$$
was evaluated for determinism and edge cases:
- **Descending Count Ordering:** Guaranteed.
- **Tie-Breaking:** Strictly resolved by ascending Token ID ($t_i < t_j$).
- **Disjoint & Exhaustive:** Verified across unit tests for divisible ($N=600 \to 200/200/200$) and non-divisible ($N=754 \to 251/251/252$) populations.
- **Zero-Frequency Entries:** Any candidate token with $C(t) = 0$ in the Stage C training corpus must be excluded prior to tier partitioning.

---

## 7. Extraction Script & Safety Controls Audit

A code-level safety audit confirms the required architecture for the prospective script (`scripts/extract_stage_c_frequencies.py`):
1. **Physical Shard Inventory:** Exactly 27 files (`shard_00000.bin` to `shard_00026.bin`) located in `data/tokenized/stage_c/train/`.
2. **Read-Only Handle Contract:** Must enforce `open(path, "rb")` with `assert f.readable() and not f.writable()`.
3. **Strict Path Restriction:** Must use `os.path.normpath(os.path.abspath(path))` to reject any path outside `data/tokenized/stage_c/train/`.
4. **Validation/Test Shard Prohibition:** Attempting to read `data/tokenized/stage_c/validation/` or `test/` must raise an immediate fatal `PermissionError`.
5. **Fail-Closed Verification:** Must verify each shard's SHA-256 against `metadata.json` prior to reading. Any mismatch terminates execution immediately.

---

## 8. Output Isolation & Failure Handling Review

The extraction output must be strictly isolated to prevent contamination:
- **Proposed Output Directory:** `data/experiments/cand_01/frequency/` (new, non-canonical directory).
- **Prohibited Directories:** Cannot write to `data/tokenizer/`, `data/tokenized/`, `data/manifests/`, or `models/`.
- **Atomic Serialization:** Output files must be written to temporary files (`.tmp`) and renamed atomically only upon complete verification.
- **Output Artifacts:**
  1. `token_frequencies_stage_c_train.json` (Full 4,096-array frequency mapping).
  2. `candidate_entity_pools.json` (Eligible tokens, tier allocations, and cryptographic digests).
  3. `extraction_manifest.json` (Input shard hashes, runtime telemetry, environment metadata).

---

## 9. Scientific & Statistical Limitations

The governance posture regarding statistical analysis remains immutable:
1. **Provisional Classification:** The two-way cluster bootstrap ($w_m = c_K \times c_V$) is classified as **`PROVISIONAL_DIAGNOSTIC_ONLY`**.
2. **Confirmatory Prohibition:** It cannot be used to declare experimental success or confirm capability hypotheses ($H_1: >75\%$) until formal simulation criteria are accredited.
3. **Degeneracy Tripping:** Under sparse crossed evaluation ($N=500$), degeneracy remains $\approx 35.2\%$, tripping uncertainty reporting strictly to **`INCONCLUSIVE`**.
4. **Secondary Status:** One-way clustering on values remains a secondary sensitivity check.
5. **Independence from Extraction:** Frequency extraction is a deterministic corpus accounting task and does not validate model capability.

---

## 10. Protected-State Verification

All protected baseline files were verified bit-exact before and during this review:

| File Path | Verified SHA-256 Digest | Status |
| :--- | :--- | :--- |
| [`chakrview/brain/model.py`](file:///d:/Project/ChakrView/chakrview/brain/model.py) | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **UNMODIFIED / INTACT** |
| [`chakrview/runtime/session.py`](file:///d:/Project/ChakrView/chakrview/runtime/session.py) | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **UNMODIFIED / INTACT** |
| Canonical Step 10k Checkpoint | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **UNMODIFIED / INTACT** |
| [`.gitignore`](file:///d:/Project/ChakrView/.gitignore) | `700f3d8b6d8c5c216884f1808f5d5c7da17282ce4d69f6e5d563f2ee49e4fda6` | **PRESERVED EXACTLY** |

- **Git Working Tree:** Branch `main`, HEAD commit `5aeaa2734de5a98802b96afe36324765f3244738`. Exactly 1 tracked modification (`.gitignore`), zero staged changes.
- **Excluded Directory:** `experiments/overnight_runs/` was **NEVER** accessed, opened, or scanned. Zero binary shards were read.

---

## 11. Unresolved Blockers & Explicit Conditions for Execution

The following conditions are mandatory prerequisites before live extraction can occur:
1. **Blocker 1 (Dedicated Runner Implementation):** The extraction script (`scripts/extract_stage_c_frequencies.py`) must be authored implementing all Section 7 and Section 8 controls and verified in a dry-run mode.
2. **Blocker 2 (Pre-Execution Shard Checksum Audit):** The runner must programmatically confirm the SHA-256 hash of each of the 27 binary shards against `data/tokenized/stage_c/train/metadata.json` before counting tokens.
3. **Blocker 3 (Candidate Margin Gate):** The runner must execute `assert len(eligible_tokens) >= 600` prior to writing candidate pools.

---

## 12. Proposed Extraction Scope

When authorized under a future gate, the extraction operation will be strictly bounded as follows:
- **Command:** `python scripts/extract_stage_c_frequencies.py --split train --input-dir data/tokenized/stage_c/train --output-dir data/experiments/cand_01/frequency`
- **Inputs:** 27 binary shards in `data/tokenized/stage_c/train/` (6,651,093 `uint16` tokens).
- **Outputs:** Frequency array and candidate entity pools written to `data/experiments/cand_01/frequency/`.
- **Resource Limits:** Timeout $\le 60$ seconds, RAM $\le 512$ MB, single thread.

---

## 13. Explicit Authorization Boundaries

The boundaries of this decision package are strictly enforced:
- **Authorizes:** Formal documentation of the Stage C token-count reconciliation, authorization decision criteria, and implementation plan for the future extraction runner.
- **DOES NOT Authorize:**
  - Reading Stage C binary shards.
  - Executing frequency extraction.
  - Generating production candidate entity pools.
  - Generating CAND_01 experiment datasets.
  - Executing 60-step calibration.
  - Performing model inference against canonical checkpoints.
  - Initiating model training.

---

## 14. Final Governance Recommendation

### Final Recommendation: **`CONDITIONAL GO`**

The technical, mathematical, and procedural foundations for Stage C Frequency Extraction are verified and sound. Advancing to the preparation and isolated review of the dedicated extraction runner is recommended, while physical shard access remains strictly prohibited until formal execution authorization is granted.

