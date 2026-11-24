# CV-GENESIS-NR-009 Artifact Inventory

## 1. Checkpoints

| Checkpoint Name | Relative Path | Parameters | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Canonical Pretrain Best** | `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt` | 5.18M | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` |
| **Canonical Pretrain Final**| `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_final.pt`| 5.18M | `06ffb32722c9a39879ec8b83ae1f4a03ab5ca952253444d06eb2b8297c69aa95` |
| **NR-005 SFT Best** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | 5.18M | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` |
| **NR-005 SFT Final** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt` | 5.18M | `cbde4b3249cd2f38fa72f64240fa2976a1adb048853a5e7c92c115d03c4662e0` |
| **NR-008 Balanced Best** | `experiments/genesis_balanced_sft_008/checkpoints/balanced_sft_best.pt` | 5.18M | `776a572501323967c7dd26d11a9f236d4535b6d3b71a1ad8ba5aa1b973890da0` |
| **NR-008 Balanced Final** | `experiments/genesis_balanced_sft_008/checkpoints/balanced_sft_final.pt` | 5.18M | `a6cae04cfd58ee5ea20c234253c205a7b3ecc7dbd08f3c2305b954aa05360cac` |

---

## 2. Tokenizer & Architectures

- **Canonical BPE Vocabulary (4,096 tokens):** `data/experiments/vocab_4096/vocab.json` (`db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)
- **ChakrGenesis-5M Module:** `chakrview/brain/genesis/chakr_genesis.py`
- **ChakrGenesis-17M Module:** `chakrview/brain/genesis/chakr_genesis_17m.py`
- **Cognitive Dual-Path Module:** `chakrview/brain/cognitive_architecture.py`

---

## 3. Evaluation Benchmarks

- **300-Prompt Blind Comparative Results:** `reports/NR009_300_BENCHMARK_RESULTS.json`
- **Cognitive Architecture Unit Suite:** `tests/test_cognitive_architecture_nr009.py` (13/13 passing)
