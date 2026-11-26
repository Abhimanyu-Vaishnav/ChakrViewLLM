# NR-019: Recovery & Integrity Audit Report

## 1. Environment & Checkpoint Verification
- **Active Git Branch**: `exp/cbr-01-conversational-breakthrough`
- **Head Commit Verified**: `638027f` (synchronized cleanly with `origin/exp/cbr-01-conversational-breakthrough`)
- **Stage-D Foundation Checkpoint**:
  - File: `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`
  - SHA-256: `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` (MATCHES CANONICAL SIGNATURE)
- **Canonical Tokenizer Artifacts**:
  - Directory: `data/experiments/vocab_4096/`
  - `config.json`: `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589`
  - `merges.txt`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
  - `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`

---

## 2. NR-018 Benchmark Audit & Discrepancy Reconciliation

### A. Investigation of System C vs System D Identical Accuracy (32.8% / 164 tasks)
- In NR-018, both System C (Neural Router Direct) and System D (Hybrid Validated) were reported with identical scores (32.8%, 164/500).
- **Audit Finding**: In `run_system_comparison.py`, System D executed a fallback to System B (Rule Parser) strictly when System C produced a `PARSE_ERROR` (43 tasks).
- Across all 43 tasks where System C failed to parse JSON:
  - 35 tasks were `conditional_branch` tasks (e.g. *"If value 32 is greater than 40, output PASS, else FAIL"*), which System B's regex did not support and returned `UNSUPPORTED` (expected was `PASS` or `FAIL`).
  - 8 tasks were complex state accumulators or multiline requests where System B also failed.
- Consequently, System B successfully rescued **0 / 43** parse errors, resulting in identical net task scores between System C and System D.

### B. Investigation of Claimed 58.6% Argument Extraction Failure Rate
- Out of 500 tasks, 336 were incorrect for System C:
  - `PARSE_ERROR`: 43 tasks (8.6% of all tasks, 12.8% of errors).
  - Valid JSON generated but output was incorrect: 293 tasks (58.6% of all 500 tasks, 87.2% of errors).
- In these 293 cases, the neural router predicted the correct operation key in $>80\%$ of cases, but hallucinated or scrambled argument values (e.g., extracting `12` instead of `15`).

### C. Investigation of Devanagari Tokenizer Fragmentation
- Direct byte-token inspection reveals:
  - ASCII `15`: 1 token (`[b'15']`).
  - Devanagari `१५`: 4 tokens (`[b'\xe0\xa5', b'\xa7', b'\xe0\xa5', b'\xab']`).
  - ASCII `12345`: 3 tokens (`[b'12', b'34', b'5']`).
  - Devanagari `१२३४५`: 10 byte-slice tokens (2 tokens per individual digit).
- **Conclusion**: Devanagari digits require $2\text{x}$ to $4\text{x}$ more tokens per numeral because the 4,096 canonical vocabulary did not merge multi-byte Devanagari digit sequences, severely increasing the attention span and probability of digit distortion during free-form generation.

### D. Benchmark Template Redundancy Audit
- The 500-task benchmark contained 353 unique prompt texts and 147 duplicate instances (e.g. repeated prompt injection probes *"Ignore all rules..."* repeated 11 times, *"Book a flight..."* repeated 9 times).
- **Mandate for NR-019**: Construct an 800-task benchmark with zero template duplicates and diverse linguistic phrasings.

---

## 3. Directory Isolation
Created isolated directory:
- `experiments/genesis_stage_i_019/`
- `data/stage_i_019/`
