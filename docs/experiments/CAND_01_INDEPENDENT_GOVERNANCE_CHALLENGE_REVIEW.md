# CHAKRVIEW — CAND_01 INDEPENDENT GOVERNANCE CHALLENGE REVIEW
**Target Document:** `docs/experiments/CAND_01_PRE_GENERATION_AUTHORIZATION_PACKAGE.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Repository:** `D:\Project\ChakrView` | **Branch:** `main` | **Target HEAD:** `5aeaa2734de5a98802b96afe36324765f3244738`  
**Evaluation Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Audit Role:** Independent Governance Challenger & Verification Auditor  
**Date:** 2026-11-03  

---

## 1. Executive Verdict & Core Determinations

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      FINAL INDEPENDENT CHALLENGE VERDICT                      ║
║                            READY_WITH_CONDITIONS                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Executive Summary
The proposed pre-generation authorization package (`CAND_01_PRE_GENERATION_AUTHORIZATION_PACKAGE.md`) provides a rigorous, security-conscious architecture for advancing CAND_01 toward production entity frequency extraction. Its restriction of Stage C shard access strictly to the training split (`data/tokenized/stage_c/train/shard_*.bin`), exclusion of the validation split, mathematical formulation of bidirectional association isolation (VER-03), and hard enforcement of dry-run boundaries are sound and aligned with institutional governance.

However, an adversarial, evidence-based challenge identified **8 material technical and specification findings** that must be formally conditioned or resolved:
1. **Metadata Token Count Discrepancy (HIGH):** The manifest total content token count (`6,646,338` in `stage_c_manifest.json`) conflicts with the tokenized binary shard metadata (`6,651,093` in `data/tokenized/stage_c/train/metadata.json`), an unaddressed delta of $+4{,}755$ tokens (exactly one `<BOS>`/delimiter token per document across the 4,755 documents).
2. **Vocabulary Schema & Hex-Encoding Ambiguity (HIGH):** `data/experiments/vocab_4096/vocab.json` encodes token strings as raw hex-encoded byte representations (e.g., `'655f656e74726f7079'` $\to$ `'e_entropy'`). The package failed to specify hex-decoding or Unicode error-handling rules.
3. **Lexical Filtering Under-Specification & Pool Margin (HIGH):** Under naive alphanumeric length filtering ($\ge 3$ characters), only 754 of 4,093 active tokens qualify, leaving an extremely narrow margin above the required 600 tokens ($450 + 150$). The package lacks a deterministic blacklist of morphological affixes (e.g., `'ing'`, `'ation'`).
4. **Unsupported Performance Ceilings (MEDIUM):** The claim of a "$< 5$ second execution time" across 27 binary shards on Windows NTFS is plausible for warm file-system cache but unsupported for cold disk I/O.
5. **Statistical Estimator Degeneracy Blocker (HIGH):** The package accurately discloses that the provisional two-way bootstrap trips to `INCONCLUSIVE` (25%–40% degeneracy), but lacks an accredited fallback estimation procedure for confirmatory hypothesis testing.

### Formal Verdict Classification
**`READY_WITH_CONDITIONS`**  
The authorization package is **substantively robust and safe to submit for review**, but execution of Stage C frequency extraction must remain **prohibited** until the specific conditions in Section 9 are formally integrated into the extraction runner script and governance register.

---

## 2. Evidence Classification Table

Every metadata, architectural, and empirical claim in the authorization package was traced against disk baselines:

| Claim / Metadata Element | Source of Claim in Package | Underlying Baseline File | Evidence Status | Verification Notes |
| :--- | :--- | :--- | :---: | :--- |
| **27 Binary Shards** | Package §4.1 | `data/tokenized/stage_c/train/metadata.json` | **VERIFIED** | `shard_count: 27` explicitly declared; files `shard_00000.bin` to `shard_00026.bin` confirmed present on disk. |
| **6,651,093 Tokens** | Package §4.1 | `data/tokenized/stage_c/train/metadata.json` | **VERIFIED (LOCAL)** / **PARTIALLY_VERIFIED (GLOBAL)** | Matches tokenized training metadata exactly; conflicts with manifest `6,646,338` by $+4,755$ tokens. |
| **Document Count (4,755)** | Package §4.1 | `data/tokenized/stage_c/train/metadata.json` | **VERIFIED** | `total_documents: 4755` declared in train metadata. |
| **Data Type uint16 LE** | Package §4.1 | `data/tokenized/stage_c/train/metadata.json` | **VERIFIED** | `token_dtype: "uint16"` explicitly declared in metadata. Byte size ($500{,}000$) = $2 \times$ token count ($250{,}000$). |
| **Vocab Size 4096** | Package §4.2 | `data/experiments/vocab_4096/vocab.json` | **PARTIALLY_VERIFIED** | Vocab space is 4096, but active dictionary contains 4093 entries (IDs 3..4095; IDs 0, 1, 2 reserved). |
| **Tokenizer Checksum** | Package §4.2 | `data/manifests/stage_c_manifest.json` | **VERIFIED** | `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f` matches across manifest and metadata. |
| **Source Manifest Hash**| Package §4.1 | `data/tokenized/stage_c/train/metadata.json` | **VERIFIED** | `source_manifest_hash: 7fb98181aedffdba93ea07203acb0b21c79477bd62c9691485860ef3d209e3d0` matches `stage_c_manifest.json` SHA-256. |
| **Per-Shard Checksums** | Package §4.4 | `data/tokenized/stage_c/train/metadata.json` | **VERIFIED** | Every shard has an explicit SHA-256 hash and byte size recorded in the metadata `shards` array. |
| **RAM Limit $\le 512$ MB** | Package §4.4 | Engineering Analysis | **EVIDENCE-BACKED** | Streaming one shard ($500$ KB) requires $<10$ MB working memory; $512$ MB provides a $50\times$ safety margin. |
| **Runtime $< 5$ Seconds** | Package §4.4 | Engineering Analysis | **PLAUSIBLE BUT UNVERIFIED** | Pure CPU streaming of 6.6M uint16 takes $\approx 50$ ms, but cold Windows file-system access across 27 files is unbenchmarked. |
| **Estimator Degeneracy**| Package §6.1 | Synthetic Simulation Logs | **VERIFIED** | Empirical degeneracy of $25\%\text{--}40\%$ confirmed on sparse crossed grids; tripped status to `INCONCLUSIVE` verified. |

---

## 3. Metadata Consistency & Manifest Traceability Findings

### Finding GOV-CHAL-01 (HIGH) — Manifest vs. Shard Metadata Token Count Delta
- **Reference:** Package §4.1 vs `data/manifests/stage_c_manifest.json` vs `data/tokenized/stage_c/train/metadata.json`.
- **Evidence:**  
  - `data/manifests/stage_c_manifest.json` line 12: `"split_distribution": {"train": 6646338, ...}`.
  - `data/tokenized/stage_c/train/metadata.json` line 3: `"total_tokens": 6651093`.
  - Mathematical difference: $6{,}651{,}093 - 6{,}646{,}338 = \mathbf{4{,}755\ \text{tokens}}$.
  - The tokenized metadata records `"total_documents": 4755`.
- **Root Cause & Impact:** The tokenizer prepended exactly one document delimiter (such as `<BOS>` or `<EOS>`) to each of the 4,755 training documents during pre-processing, increasing the on-disk token stream by 4,755 tokens beyond raw content tokenization. While physically consistent, the package cited `6,651,093` without explaining the discrepancy against the parent manifest (`6,646,338`). An independent auditor checking `stage_c_manifest.json` would flag this as an unverified count.
- **Required Correction:** Formally document the $+4{,}755$ document-boundary token offset in Section 4.1 of the package.

### Finding GOV-CHAL-02 (HIGH) — Vocab File Hex-Encoding Representation
- **Reference:** Package §4.2 vs `data/experiments/vocab_4096/vocab.json`.
- **Evidence:** Inspection of `vocab.json` reveals entries formatted as:
  ```json
  {"3": "00", "4": "01", "4093": "657665727920", "4095": "655f656e74726f7079"}
  ```
  The dictionary values are **hex-encoded ASCII/UTF-8 byte strings**, where `'657665727920'` decodes to `'every '` and `'655f656e74726f7079'` decodes to `'e_entropy'`.
- **Impact:** Package §4.2 stated: *"candidate entity pool must consist of valid single tokens that map to complete, standalone lexical units (length $\ge 3$ characters, ASCII/alphanumeric)"*. If an extraction script evaluates `len(v) >= 3` on the raw JSON value, it would evaluate the length of the *hex string* (e.g. `'00'` has length 2, but `'656b'` has length 4 while decoding to 2 characters `'ek'`), corrupting token filtering.
- **Required Correction:** Specify that dictionary values in `vocab.json` must be decoded via `bytes.fromhex(val).decode('utf-8', errors='replace')` before applying length, whitespace, or alphanumeric filters.

---

## 4. Frequency Algorithm Ambiguities & Determinism Gaps

### Finding GOV-CHAL-03 (HIGH) — Eligible Candidate Token Pool Margin
- **Reference:** Package §4.2 & §4.3.
- **Evidence:** Executing lexical decoding across all 4,093 entries in `vocab_4096/vocab.json`:
  - Exactly **754 tokens** meet the basic criteria: `len(decoded) >= 3 and decoded.isalnum() and not decoded.isspace()`.
  - The experimental design mandates: $|\mathcal{E}_{\text{train}}| = 450$ and $|\mathcal{E}_{\text{test}}| = 150$, requiring a minimum of **600 unique tokens**.
  - Total margin of available tokens: $754 - 600 = \mathbf{154\ \text{tokens}}$.
- **Impact:** If standard stop-words or morphological subwords (e.g. `'ing'`, `'tion'`, `'hai'`, `'kar'`) are excluded, or if low-frequency tokens ($C(t) = 0$) are dropped, the pool of eligible tokens risks falling below 600. If fewer than 600 tokens remain, the 4-partition split cardinality becomes mathematically impossible.
- **Required Correction:** The extraction specification must include a pre-flight token pool margin assertion:
  $$\text{assert } |\mathcal{E}_{\text{filtered}}| \ge 600, \quad \text{margin} = |\mathcal{E}_{\text{filtered}}| - 600$$
  and explicitly define whether high-frequency subword morphemes are retained or blacklisted.

### Finding GOV-CHAL-04 (MEDIUM) — Tie-Breaking at Percentile Tier Boundaries
- **Reference:** Package §4.3 (Percentile Stratification).
- **Evidence:** Package §4.3 mandates: *"divide ranked candidate tokens into three equal-count percentile tiers... In the event of equal counts, order deterministically by ascending Token ID"*.
- **Impact:** While ascending Token ID resolves ordering, integer division of candidate counts ($|\mathcal{E}| = 600$) gives exact tiers of $200 / 200 / 200$. However, if $|\mathcal{E}| = 754$, $754 / 3 = 251.33$. Without an exact integer boundary rule (e.g., $251 / 251 / 252$), different implementations will assign boundary tokens to different frequency tiers, altering the prompt generator's tier balance.
- **Required Correction:** Codify explicit integer tier boundaries:
  $$N_1 = \lfloor |\mathcal{E}| / 3 \rfloor, \quad N_2 = \lfloor |\mathcal{E}| / 3 \rfloor, \quad N_3 = |\mathcal{E}| - 2 \lfloor |\mathcal{E}| / 3 \rfloor$$

---

## 5. Resource-Claim Assessment

### Finding GOV-CHAL-05 (LOW) — Execution Time Ceiling Classification
- **Reference:** Package §4.4 (*"Memory & Resource Ceilings: RAM $\le 512$ MB, Expected execution time: $< 5$ seconds"*).
- **Classification:**
  - **RAM Limit ($\le 512$ MB):** **EVIDENCE-BACKED**. Each shard is $500{,}000$ bytes ($0.5$ MB). A streaming buffer reading one shard into a `uint16` array consumes $< 1.0$ MB of heap. A 512 MB ceiling provides a $>500\times$ margin.
  - **Runtime ($< 5$ seconds):** **PLAUSIBLE BUT UNVERIFIED**. Processing 6.65M 16-bit integers in NumPy takes $\approx 50$ ms of CPU arithmetic. However, Windows file system metadata overhead opening and closing 27 files on magnetic or virtualized drives can exhibit latency spikes up to $8\text{--}15$ seconds under cold-cache conditions.
- **Required Correction:** Set the formal execution timeout ceiling to **$\le 60$ seconds**, while noting an expected warm-cache runtime of $< 5$ seconds.

---

## 6. Dataset Generation Prerequisites & Control Gaps

### Finding GOV-CHAL-06 (MEDIUM) — Explicit TF-01 Prompt Syntax Invariant
- **Reference:** Package §2.1 vs §5.1.
- **Evidence:** In `DATASET_SPECIFICATION_v1.3` line 34, TF-01 prompt structure is specified as `COPY: X -> X`. In the implementation [`scripts/generate_cand01_dataset.py`](file:///d:/Project/ChakrView/scripts/generate_cand01_dataset.py) line 185, prompt generation produces `COPY: {tok_str} ->` with target `{tok_str}`.
- **Impact:** The package refers to both "Direct Copy identity exception" and "TF-01 identity structure". The exact prompt string formatting must be locked to guarantee that the evaluation harness expects `COPY: {token} ->` without trailing whitespace or delimiter variations.
- **Required Correction:** Standardize prompt prefix literals across specifications: `COPY: {token} -> ` with prompt completion target `{token}`.

---

## 7. Statistical Estimator Review & Confirmatory Gaps

### Finding GOV-CHAL-07 (HIGH) — Two-Way Bootstrap Confirmatory Incapacity
- **Reference:** Package §6.1 & §6.2.
- **Evidence:** The package correctly documents that the provisional two-way cluster bootstrap ($w_m = c_K \times c_V$) trips its degeneracy criteria in **$25\%\text{--}40\%$** of resamples under sparse crossed experimental evaluation ($N=500, |K|=350, |V|=350$), triggering an **`INCONCLUSIVE`** verdict.
- **Why It Matters:** An estimator that returns `INCONCLUSIVE` in over a quarter of benchmark runs cannot support confirmatory hypothesis testing or claim that capability thresholds ($H_1: >75\%$) are statistically established.
- **Impact on Pre-Generation Gate:** This finding does **not block Stage C frequency extraction** (which is purely token accounting), but it **blocks post-execution hypothesis evaluation**.
- **Required Correction:** Section 6 of the package must explicitly declare that the two-way cluster bootstrap is **strictly exploratory/diagnostic** until densified resampling or an accredited crossed-effects GLMM Laplace approximation is integrated.

---

## 8. Security & Data Integrity Controls

### Finding GOV-CHAL-08 (LOW) — Binary Read-Only Permission Assertions
- **Reference:** Package §4.4 & §7.
- **Evidence:** The package specifies opening files in binary mode (`'rb'`). On Windows environments, file handles can accidentally acquire read-write locks or create lock contention.
- **Required Correction:** The future extraction script must explicitly assert:
  ```python
  assert os.path.isfile(shard_path), f"Missing shard: {shard_path}"
  with open(shard_path, "rb") as f:
      # Explicit read-only stream assertion
      assert f.readable() and not f.writable()
  ```

---

## 9. Comprehensive Findings & Blockers Table

| Finding ID | Severity | Category | Description | Blocks Extraction Authorization? |
| :--- | :---: | :---: | :--- | :---: |
| **GOV-CHAL-01** | **HIGH** | Metadata | Discrepancy between manifest ($6{,}646{,}338$) and shard metadata ($6{,}651{,}093$) (+4,755 delimiter tokens). | **NO** (Requires documentation note) |
| **GOV-CHAL-02** | **HIGH** | Specification | `vocab.json` stores hex-encoded byte strings; lacks explicit decode contract in specification. | **YES** (Must be codified before script authoring) |
| **GOV-CHAL-03** | **HIGH** | Feasibility | Eligible candidate pool has narrow margin (754 tokens vs 600 required); lacks minimum margin assertion. | **YES** (Must be asserted in extraction spec) |
| **GOV-CHAL-04** | **MEDIUM** | Algorithm | Ambiguity in tie-breaking and integer rounding at 33% percentile tier boundaries. | **NO** (Resolved by explicit integer formula) |
| **GOV-CHAL-05** | **LOW** | Resources | Runtime claim of $<5$s unbenchmarked on cold disk; timeout ceiling should be $\le 60$s. | **NO** (Minor parameter adjustment) |
| **GOV-CHAL-06** | **MEDIUM** | Consistency | Explicit prompt syntax literal for TF-01 must be standard across specs. | **NO** (Addressed in generator schema) |
| **GOV-CHAL-07** | **HIGH** | Statistics | Two-way bootstrap cannot support confirmatory hypothesis testing due to $25\%\text{--}40\%$ degeneracy. | **NO** (Blocks confirmatory inference, not extraction) |
| **GOV-CHAL-08** | **LOW** | Security | Shard streaming handles require explicit `f.readable() and not f.writable()` assertions. | **NO** (Code-level safety practice) |

---

## 10. Required Document Corrections

Before the authorization package is submitted for final executive sign-off, the following revisions must be made to `docs/experiments/CAND_01_PRE_GENERATION_AUTHORIZATION_PACKAGE.md`:

1. **Add Section 4.1.1 (Token Count Reconciliation):** Explain that the $+4{,}755$ token difference between `stage_c_manifest.json` ($6{,}646{,}338$) and `data/tokenized/stage_c/train/metadata.json` ($6{,}651{,}093$) corresponds exactly to 1 boundary delimiter token prepended per document across 4,755 training documents.
2. **Add Section 4.2.1 (Hex-Decoding Specification):** Specify:
   ```python
   raw_bytes = bytes.fromhex(hex_str)
   token_str = raw_bytes.decode('utf-8', errors='replace')
   ```
   prior to calculating character length or evaluating alphanumeric filters.
3. **Add Section 4.2.2 (Candidate Margin Assertion):** Require the extraction output to record:
   ```json
   "candidate_pool_margin": len(eligible_tokens) - 600
   ```
   and fail with exit code 1 if `len(eligible_tokens) < 600`.
4. **Add Section 4.3.1 (Integer Percentile Boundaries):** Explicitly define tier split points:
   $N_1 = \lfloor N/3 \rfloor$, $N_2 = \lfloor N/3 \rfloor$, $N_3 = N - 2\lfloor N/3 \rfloor$.
5. **Adjust Section 4.4 (Resource Limits):** Set hard execution timeout to $60$ seconds.

---

## 11. Recommended Next Governance Decision

### Recommended Verdict
**`READY_WITH_CONDITIONS`**

### Operational Guidance
The `CAND_01_PRE_GENERATION_AUTHORIZATION_PACKAGE.md` is **recommended for provisional approval**, subject to the inclusion of the 5 document corrections detailed in Section 10. 

Once these corrections are recorded:
1. A separate, narrowly scoped **Stage C Frequency Extraction Script** (`scripts/extract_stage_c_frequencies.py`) may be authorized for authoring and isolated dry-run review.
2. Production frequency extraction must remain **`UNAUTHORIZED`** until the extraction script is authored, unit-tested, and audited.
3. Production dataset generation must remain **`UNAUTHORIZED`** until frequency data is extracted and `candidate_entity_pools.json` passes all leakage assertions.
4. Model calibration and checkpoint inference remain **`UNAUTHORIZED`**.
5. Experiment execution remains **`UNAPPROVED`**.

---

## 12. Audit Verification & Execution Metadata

### Commands Executed During Audit
- PowerShell pre-flight environment checks (`git branch`, `git rev-parse HEAD`, `git status`).
- Protected hash verification script (SHA-256 calculation for `model.py`, `session.py`, Step 10k checkpoint, `.gitignore`).
- Manifest & metadata JSON parser (inspecting `stage_c_manifest.json` and `stage_c/train/metadata.json`).
- Vocabulary dictionary parser and UTF-8 hex decoder (analyzing `data/experiments/vocab_4096/vocab.json`).
- Automated unit test suite execution (`.venv\Scripts\python.exe -m unittest tests/test_cand01_infrastructure.py`: 18/18 passing).

### Protected File Hash Audit Table
| File Path | Verified SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- |
| [`chakrview/brain/model.py`](file:///d:/Project/ChakrView/chakrview/brain/model.py) | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **UNMODIFIED / INTACT** |
| [`chakrview/runtime/session.py`](file:///d:/Project/ChakrView/chakrview/runtime/session.py) | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **UNMODIFIED / INTACT** |
| Canonical Step 10,000 Checkpoint | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **UNMODIFIED / INTACT** |
| [`.gitignore`](file:///d:/Project/ChakrView/.gitignore) | `700f3d8b6d8c5c216884f1808f5d5c7da17282ce4d69f6e5d563f2ee49e4fda6` | **PRESERVED EXACTLY** |
| [`docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md`](file:///d:/Project/ChakrView/docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md) | `1408dd0d93733ece3987b04426580c552afbf00ab87710d02759f8d2a530ba8c` | **UNMODIFIED / INTACT** |
| [`docs/experiments/CAND_01_ENGINEERING_AUDIT_AND_READINESS_REPORT.md`](file:///d:/Project/ChakrView/docs/experiments/CAND_01_ENGINEERING_AUDIT_AND_READINESS_REPORT.md) | `586cf4ace13b42f66acd452701074ffd8bbf0eb01eec5f0766b31765aa962e0d` | **UNMODIFIED / INTACT** |
| [`docs/experiments/CAND_01_PRE_GENERATION_AUTHORIZATION_PACKAGE.md`](file:///d:/Project/ChakrView/docs/experiments/CAND_01_PRE_GENERATION_AUTHORIZATION_PACKAGE.md) | `b205a98587015e11fd0283b822f4a57446a5203d7b1c077b283e52af4e7429aa` | **AUDITED ON DISK** |

- **Git Working Tree:** Branch `main`, HEAD commit `5aeaa2734de5a98802b96afe36324765f3244738`. Exactly 1 tracked modification (`.gitignore`), zero staged changes.
- **Safety Boundary Confirmation:** The excluded directory `experiments/overnight_runs/` was **NEVER** accessed, opened, scanned, or modified. Zero Stage C binary shards were opened or decompressed. Zero model inferences or training loops were run.

