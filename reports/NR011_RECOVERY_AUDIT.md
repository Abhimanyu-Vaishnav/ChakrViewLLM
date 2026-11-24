# CV-GENESIS-NR-011 Repository Recovery and Safety Audit

## 1. Executive Summary

Mission **CV-GENESIS-NR-011 Phase 0** certifies the clean state of the repository following the successful execution and remote synchronization of Mission CV-GENESIS-NR-010.

All foundation artifacts—spanning Canonical 5.18M Pretraining, NR-005 Conversational SFT, NR-008 Balanced SFT, and NR-010 17M Pretraining/SFT—have been cryptographically verified against their reported SHA-256 hashes.

---

## 2. Git & Remote Status

- **Repository Root:** `D:\Project\ChakrView`
- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Starting Commit:** `f164026d5a78b962b55505a05188835b42812ca6`
- **Remote URL:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
- **Synchronization State:** In sync with `origin/exp/cbr-01-conversational-breakthrough`.
- **Working Tree:** Clean; zero uncommitted or untracked changes.

---

## 3. Cryptographic Verification of Base Artifacts

| Artifact Name | Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Canonical Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| **Canonical Pretrain Best** | `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt` | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` | **MATCH** |
| **Canonical Pretrain Final**| `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_final.pt`| `06ffb32722c9a39879ec8b83ae1f4a03ab5ca952253444d06eb2b8297c69aa95` | **MATCH** |
| **NR-005 SFT Best** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | **MATCH** |
| **NR-005 SFT Final** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt` | `cbde4b3249cd2f38fa72f64240fa2976a1adb048853a5e7c92c115d03c4662e0` | **MATCH** |
| **NR-008 Balanced Best** | `experiments/genesis_balanced_sft_008/checkpoints/balanced_sft_best.pt` | `776a572501323967c7dd26d11a9f236d4535b6d3b71a1ad8ba5aa1b973890da0` | **MATCH** |
| **NR-008 Balanced Final** | `experiments/genesis_balanced_sft_008/checkpoints/balanced_sft_final.pt` | `a6cae04cfd58ee5ea20c234253c205a7b3ecc7dbd08f3c2305b954aa05360cac` | **MATCH** |
| **NR-010 17M Pretrain Best** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` | **MATCH** |
| **NR-010 17M Pretrain Final**| `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_final.pt`| `c018e3221a09dab67e97ec2383def345652e3949893bab606f67e7d46cea0a3b` | **MATCH** |
| **NR-010 17M SFT Best** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt` | `ac62c43fda2e33acb3a24cbdad1038d104d8f2a8653ac39bd6c081606aac56dd` | **MATCH** |
| **NR-010 17M SFT Final** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_final.pt` | `c92c97265767a662936607d2342f513ad1a7a467427d6200de49cd2da86134aa` | **MATCH** |

---

## 4. Hardware and Storage Boundaries

- **Drive D Available Space:** **80.21 GB free** of 203.29 GB total.
- **Accelerator:** NVIDIA RTX A1000 Laptop GPU (6.0 GB VRAM, compute capability 8.6).
- **Safety Policy:**
  - Strictly no overwriting of existing checkpoints or Stage-C binary shards.
  - Zero concurrent GPU processes.
  - All Stage-D datasets and models stored in new dedicated experiment paths.
