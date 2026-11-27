# NR-020: Recovery & Repository Inventory Audit

## 1. Environment & Checkpoint Verification
- **Active Git Branch**: `exp/cbr-01-conversational-breakthrough`
- **Head Commit Verified**: `c4d91fa` (`feat(nr-019): exact source-span grounding, multilingual normalization and 800-task benchmark`)
- **Remote Synchronization**: Clean sync with `origin/exp/cbr-01-conversational-breakthrough` (0 uncommitted or untracked changes).
- **Canonical Stage-D Foundation Checkpoint**:
  - File: `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`
  - SHA-256: `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` (MATCHES CANONICAL SIGNATURE)
- **Canonical Tokenizer Artifacts**:
  - Directory: `data/experiments/vocab_4096/`
  - All checksums verified (`0fca72b6`, `ada4b1dc`, `db4d6119`).

---

## 2. Inventory of NR-019 Artifacts
- **Runtime Components**:
  - `chakrview/runtime/restricted_executor.py` (21 allowlisted operations, sandboxed execution)
  - `chakrview/runtime/multilingual_normalizer.py` (Devanagari, Hindi words, Hinglish words, NFC normalization)
  - `chakrview/runtime/source_grounding.py` (Substring span grounding & parameter extraction)
- **Data Files**:
  - `data/stage_i_019/sealed_benchmark_800.json` (SHA-256: `4a62ba707af491cbfa6dda5cc756e6bfabba875ac25b30b3a5ad46654f222e6e`)
  - `data/stage_i_019/benchmark_800_results.json` (SHA-256: `cfa6d6919fbb6716881388dce639b871dfabc5c2dc82dfd40c9d62012b1e46c7`)
  - `data/stage_i_019/routing_train_3000.json` (SHA-256: `0bdecd4aa4dc0171f8953e691297d34441826576f399848b96c60d0b4646a4a3`)
  - `data/stage_i_019/routing_val_500.json` (SHA-256: `84b2db544d61204c0ed031ddefb164dc6597666da09bbfbd34b30229d7f87f72`)
  - `data/stage_i_019/security_audit_expanded.json` (SHA-256: `b13913e3cb08b3aa218560a23864f0c22300cb10abd498f6786365b6af7f31df`)
  - `data/stage_i_019/performance_and_regression.json` (SHA-256: `a7c8956843b21642e997a2321bc8c1515ef56271d971c46b5ce73f4e7136260f`)

---

## 3. Directory Isolation for NR-020
Created dedicated isolated research directories:
- `experiments/genesis_stage_j_020/`
- `data/stage_j_020/`
