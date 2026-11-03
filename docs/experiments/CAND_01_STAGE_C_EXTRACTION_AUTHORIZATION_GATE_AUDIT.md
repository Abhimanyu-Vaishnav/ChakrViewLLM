# CAND_01 — Stage C Frequency Extraction Authorization Gate Audit

**Document ID:** `AUD-EXP-CAND01-AUTHGATE-001`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Target Runner:** `scripts/extract_stage_c_frequencies.py`  
**Test Suite:** `tests/test_cand01_extraction_runner.py`  
**Governing Decision Document:** `docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md`  
**Date:** 2026-11-03  

---

## 1. Executive Summary

This engineering audit documents the implementation of an explicit, cryptographically auditable production authorization gate in `scripts/extract_stage_c_frequencies.py`.

Previously, `scripts/extract_stage_c_frequencies.py` contained a hard-coded synthetic-only lock that terminated with exit code 1 or raised `PermissionError` unconditionally whenever production mode was requested. This prevented physical execution even after explicit user authorization.

The new gate provides a tamper-evident mechanism that unlocks production extraction only when explicit operator intent and a valid reference to the authoritative authorization decision document are supplied.

---

## 2. Authorization Mechanism Architecture

The authorization gate is implemented at both the Command-Line Interface (CLI) and programmatic Python function interfaces:

### 2.1 CLI Interface Requirements
Production extraction via CLI mandates three conditions:
1. `--execute-production`: An explicit flag declaring intent to run physical extraction.
2. `--authorization-decision-ref`: A path reference pointing to `CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md`.
3. Mutual Exclusivity: `--synthetic` and `--execute-production` cannot both be specified.

If either requirement is omitted, or if no mode is specified, the CLI immediately prints a fatal governance error to `sys.stderr` and terminates with exit code 1 without reading any shards.

### 2.2 Programmatic Function Interface Requirements
The extraction function signature is:
```python
def execute_frequency_extraction(
    input_dir: Path,
    vocab_path: Path,
    output_dir: Path,
    expected_shards: Optional[int] = None,
    expected_tokens: Optional[int] = None,
    vocab_size: int = REGISTERED_VOCAB_SIZE,
    max_timeout_seconds: float = PROPOSED_TIMEOUT_SECONDS,
    max_ram_mb: float = PROPOSED_RAM_CEILING_MB,
    synthetic_mode: bool = False,
    authorization_decision_ref: Optional[str] = None,
) -> Dict[str, Any]:
```
When `synthetic_mode=False`:
1. `authorization_decision_ref` must be non-None and non-empty (raises `PermissionError` if missing).
2. Path traversal (`..`) is strictly prohibited (raises `ValueError`).
3. The reference must resolve to a valid file whose name is `CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md` (raises `PermissionError` if not found).
4. The resolved file's cryptographic hash must bit-exact match the registered SHA-256 digest:
   `e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00` (raises `PermissionError` on mismatch).

---

## 3. Test Coverage and Verification

Five new unit and CLI integration tests were added to `tests/test_cand01_extraction_runner.py`:
- `test_28b_production_lock_function_call`: Verifies `synthetic_mode=False` without ref raises `PermissionError`.
- `test_28c_production_missing_ref_rejection`: Verifies explicit `authorization_decision_ref=None` raises `PermissionError`.
- `test_28d_production_invalid_ref_rejection`: Verifies non-matching filename raises `PermissionError`.
- `test_28e_production_tampered_ref_rejection`: Verifies tampered content raises `PermissionError`.
- `test_28f_production_authorized_invocation_synthetic_fixtures`: Verifies authorized invocation succeeds against synthetic test fixtures.
- `test_28g_cli_production_authorization_gate`: Verifies CLI rejection on missing mode, mutual exclusivity violation, and missing authorization reference.

All 65 CAND_01 tests passed with 100% success rate.

---

## 4. Preservation of Governance Boundaries

- **Zero Physical Shard Access:** Physical Stage C shards (`data/tokenized/stage_c/train/*.bin`) were not read, mapped, or scanned.
- **Zero Model Operations:** No training, calibration, fine-tuning, or inference was performed.
- **Protected Hashes Preserved:** All protected runtime and baseline model hashes remain bit-exact.

