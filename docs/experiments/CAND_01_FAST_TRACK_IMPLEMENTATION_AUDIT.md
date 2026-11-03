# CAND_01 — Fast Track Implementation Audit Report

**Document ID:** `docs/experiments/CAND_01_FAST_TRACK_IMPLEMENTATION_AUDIT.md`  
**Candidate Name:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Phase:** Fast Track Software Implementation & Security Audit  
**Audit Date:** 2026-11-03  
**Auditor:** Independent Engineering Auditor & Repository Maintainer  
**Audit Verdict:** `FAST_TRACK_AUDIT_PASS_WITH_CONDITIONS`  
**Physical Stage C Shard Access:** **Strictly UNAUTHORIZED (Zero physical shard access verified)**  

---

## 1. Executive Summary & Audit Verdict

This audit conducted an independent, focused engineering inspection of the standalone frequency extraction runner (`scripts/extract_stage_c_frequencies.py`) and its associated test suites (`tests/test_cand01_extraction_runner.py`, `tests/test_cand01_extraction_spec.py`, `tests/test_cand01_infrastructure.py`).

### Audit Verdict
```text
FAST_TRACK_AUDIT_PASS_WITH_CONDITIONS
```

### Mandatory Non-Negotiable Affirmations
- **Zero Production Stage C Shard Access:** Confirmed. Neither `extract_stage_c_frequencies.py` nor any test opened, memory-mapped, hashed, read, or decompressed any physical shard in `data/tokenized/stage_c/train/*.bin`.
- **Zero Production Extraction:** Confirmed. No production frequency extraction was performed; no production entity pools were generated.
- **Zero Access to `experiments/overnight_runs/`:** Confirmed. Directory was neither scanned, read, nor modified.
- **Zero Production Authorization Granted:** Confirmed. Production extraction remains strictly **UNAUTHORIZED**.

---

## 2. Critical Findings & Identified Discrepancies

During initial audit execution across the three relevant test files, **14 failures/errors** were discovered in `test_cand01_extraction_runner.py` (12 errors, 2 failures). A rigorous root-cause analysis revealed that while the underlying runner logic was sound, several localized interface mismatches and synchronization gaps existed between the newly created test suite and the runner script:

1. **Metadata Filename Contract:**
   - *Finding:* Shard extraction expected `input_dir / "metadata.json"`, but the test harness generated `stage_c_metadata.json` (reflecting Stage C convention).
   - *Impact:* Triggered `FileNotFoundError` across 11 test cases during synthetic shard ingestion.
2. **Path Traversal & Output Path Protection Assertion Failure:**
   - *Finding:* In `normalize_and_validate_path()`, path traversal (`..`) was being resolved by `Path.resolve()` before checking `raw_p.parts`, allowing relative normalized paths to evade the traversal exception check. Additionally, `validate_output_directory()` was only called conditionally inside `execute_frequency_extraction()` rather than directly within `normalize_and_validate_path()`.
   - *Impact:* `test_22_path_traversal_rejection` and `test_24_output_path_protection` failed.
3. **Zero-Frequency Accounting Discrepancy:**
   - *Finding:* `allocate_frequency_tiers()` excluded zero-frequency candidates correctly from tier assignments, but omitted the explicit key `"zero_frequency_count"` in the returned dictionary.
   - *Impact:* `test_08_zero_frequency_exclusion` failed with `KeyError: 'zero_frequency_count'`.
4. **Production Lock on Direct Function Calls:**
   - *Finding:* The CLI interface in `main()` strictly gated execution on `--synthetic`, but direct programmatic calls to `execute_frequency_extraction()` did not assert `synthetic_mode == True` at the top of the function.
   - *Impact:* An imported script could hypothetically invoke `execute_frequency_extraction()` without explicitly asserting synthetic mode.
5. **Resource Ceilings (Telemetry vs. Enforcement):**
   - *Finding:* Timeout limits were enforced via loop checking (`time.perf_counter() > max_timeout_seconds`), but memory limit checks were purely recorded as telemetry rather than actively enforced against a ceiling.
   - *Impact:* Runaway memory consumption on corrupt inputs would not fail closed before hitting system limits.

---

## 3. Engineering Fixes Performed

All identified defects were strictly localized to `scripts/extract_stage_c_frequencies.py` and its dedicated test suite `tests/test_cand01_extraction_runner.py`. No canonical architecture, model files, or runtime engines were touched.

### Fix 1: Metadata File Resolution
- **File:** `scripts/extract_stage_c_frequencies.py`
- **Correction:** Updated metadata resolution to inspect both `stage_c_metadata.json` and `metadata.json`:
  ```python
  meta_path = input_dir / "stage_c_metadata.json"
  if not meta_path.exists():
      meta_path = input_dir / "metadata.json"
  if not meta_path.exists():
      raise FileNotFoundError(f"Missing required metadata file: {input_dir / 'metadata.json'}")
  ```

### Fix 2: Path Traversal & Output Directory Protection Hardening
- **File:** `scripts/extract_stage_c_frequencies.py`
- **Correction:** Enforced pre-resolution traversal checks on `Path(path_str).parts` and integrated `validate_output_directory(p)` directly into `normalize_and_validate_path()` when validating output parameters:
  ```python
  raw_p = Path(path_str)
  if ".." in raw_p.parts:
      raise ValueError(f"FATAL: Path traversal forbidden in '{param_name}': {path_str}")
  p = raw_p.resolve()
  norm_str = str(p).replace("\\", "/").lower()
  if "experiments/overnight_runs" in norm_str:
      raise PermissionError(f"FATAL GOVERNANCE BREACH: '{param_name}' attempts access to excluded overnight_runs.")
  if "output" in param_name.lower():
      validate_output_directory(p)
  ```

### Fix 3: Zero-Frequency Count Return Value
- **File:** `scripts/extract_stage_c_frequencies.py`
- **Correction:** Explicitly calculated and returned `"zero_frequency_count": len(candidate_counts) - n` in `allocate_frequency_tiers()`.

### Fix 4: Programmatic Production Execution Lock
- **File:** `scripts/extract_stage_c_frequencies.py`
- **Correction:** Added immediate fail-closed gate at the entry point of `execute_frequency_extraction()`:
  ```python
  if not synthetic_mode:
      raise PermissionError(
          "FATAL GOVERNANCE BREACH: Physical Stage C Frequency Extraction is strictly UNAUTHORIZED.\n"
          "execute_frequency_extraction() requires synthetic_mode=True until formal governance authorization is signed."
      )
  ```

### Fix 5: Active RAM Ceiling Enforcement
- **File:** `scripts/extract_stage_c_frequencies.py`
- **Correction:** Integrated `psutil.Process().memory_info().rss` checks inside the per-shard loop alongside elapsed wall-clock timeout checks, raising `MemoryError` if RSS exceeds `max_ram_mb` (default 512 MB). Also enforced explicit little-endian uint16 dtype `"<u2"`.

### Fix 6: Test Suite Expansion
- **File:** `tests/test_cand01_extraction_runner.py`
- **Correction:** Expanded `test_28_resource_limit_failure_behavior` to verify both timeout ceiling and RAM ceiling enforcement, and added `test_28b_production_lock_function_call` to verify the programmatic function call lock.

---

## 4. Test Results

Following the fixes, all three authorized CAND_01 test files were executed using `.venv\Scripts\python.exe -m unittest`:

```text
Ran 60 tests in 5.796s
OK
```

### Breakdown by Test Module:
1. `tests/test_cand01_extraction_runner.py`: **31/31 passed** (100% PASS)
   - Validates all 30 specification requirements plus function-level production locking and active RAM ceiling enforcement.
2. `tests/test_cand01_extraction_spec.py`: **11/11 passed** (100% PASS)
   - Validates hex-to-UTF8 decoding, replacement char rejection, reserved ID exclusion, duplicate mutual disqualification, integer tier formulas, and read-only file handle assertions.
3. `tests/test_cand01_infrastructure.py`: **18/18 passed** (100% PASS)
   - Validates 8 Task Families (TF-01..TF-08), bidirectional association leakage prevention (VER-03), deterministic generation, metric decoupling ($A_{FT}$ vs $A_{FC}$), bootstrap reproducibility, and Kish ESS bounds.

**Cumulative Authorized Test Suite:** **60/60 tests passed** with zero errors and zero regressions.

---

## 5. Security & Verification Assessment

### 5.1 Production Execution Lock Assessment
- **CLI Invocations:** If `--execute-production` is supplied or `--synthetic` is absent, `main()` prints a fatal governance error and exits with code 1 before any file operations.
- **Imported / Programmatic Invocations:** If `execute_frequency_extraction()` is called without `synthetic_mode=True`, a `PermissionError` is raised immediately.
- **Default Arguments:** The default for `synthetic_mode` is `False`, ensuring that any call without explicit opt-in fails closed.
- **Bypass Risk:** `NONE IDENTIFIED`.

### 5.2 Checksum & Path Safety Assessment
- **Pre-Execution Cryptographic Integrity:** All shard files are pre-hashed using SHA-256 and verified against the metadata specification before tokens are read. Any mismatch immediately aborts execution.
- **Extraneous Shard Detection:** Rejects any uncataloged `.bin` file in the directory.
- **Read-Only Handle Contract:** Asserts `f.readable() and not f.writable()`.
- **Path Traversal & Symlinks:** `normalize_and_validate_path()` explicitly blocks `..` traversal and prevents escaping repository boundaries. Output paths cannot target protected directories (`chakrview`, `models`, `checkpoints`, `experiments/overnight_runs`, `data/tokenized`, etc.).

### 5.3 Output Integrity Assessment
- **Overwrite Protection:** Output directory cannot contain existing `.json` files.
- **Atomic Commit:** All outputs (`token_frequencies_stage_c_train.json`, `candidate_entity_pools.json`, `extraction_manifest.json`) are written into `.extraction_tmp` and moved via atomic `os.replace`. Mid-execution aborts leave no partial artifacts in the target directory.
- **Manifest Accuracy:** Manifest records source metadata SHA-256, processed shard hashes, vocabulary audit rejections, and execution telemetry.

### 5.4 Resource Controls Assessment
- **Timeout Limit:** **Real enforcement.** Checked per shard iteration; raises `TimeoutError` if wall-clock exceeds ceiling (default 60s).
- **RAM Limit:** **Real enforcement.** Checked per shard iteration via `psutil`; raises `MemoryError` if resident memory exceeds ceiling (default 512 MB).
- **Telemetry:** Wall-clock seconds, CPU seconds, and RSS memory usage are permanently recorded in `extraction_manifest.json`.

---

## 6. Conditions for Production Extraction Gate

While the runner software and synthetic test suites are fully verified and hardened, advancing to physical Stage C frequency extraction requires compliance with the following mandatory conditions:

1. **Formal Human Governance Sign-Off:** Execution against `data/tokenized/stage_c/train/*.bin` requires explicit human authorization approving the transition from synthetic testing to production execution.
2. **Metadata Path Binding:** Production execution must explicitly bind to `data/tokenized/stage_c/train/metadata.json` (or `stage_c_metadata.json`), ensuring exact shard count (27) and token count ($6{,}651{,}093$) reconciliation.
3. **Dedicated Output Isolation:** Output artifacts must be written to a dedicated non-canonical directory (e.g. `artifacts/cand01_frequencies/`), completely isolated from source and canonical checkpoint paths.

---

## 7. Repository Invariants & Protected File Hashes

| Target File | Authoritative SHA-256 Digest | Audit Verification |
| :--- | :--- | :--- |
| `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **Bit-Exact Match** |
| `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **Bit-Exact Match** |
| `.gitignore` | `700f3d8b6d8c5c216884f1808f5d5c7da17282ce4d69f6e5d563f2ee49e4fda6` | **Preserved Modification** |
| `experiments/overnight_runs/` | *Directory Excluded / Ignored* | **Zero Access / Untouched** |
| `data/tokenized/stage_c/train/*.bin` | *Production Binary Shards* | **Zero Access / Untouched** |

### Git State
- **Branch:** `main` (commit `5aeaa2734de5a98802b96afe36324765f3244738`)
- **Tracked Modifications:** Exactly 1 (`modified: .gitignore`, preserving `experiments/overnight_runs/`).
- **No Git staging, commits, pushes, resets, or history modifications performed.**

