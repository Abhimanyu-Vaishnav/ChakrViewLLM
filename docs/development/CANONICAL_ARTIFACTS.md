# ChakrView Canonical Artifacts & Verification Invariants

**Document Status**: Official Architectural Invariant Specification
**Applies To**: All Releases, Research Banches, and CI Environments

---

## 1. Canonical Reference Specification

The canonical baseline model for ChakrView is the **ChakrMicro Step 10,000 checkpoint**, trained from first principles without external pretrained weights.

| Canonical Artifact | Canonical File Path | Size (Bytes) | Cryptographic SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Model Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 42,212,787 | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | 112,627 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | 76,299 | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` |
| **Protected Model** | `chakrview/brain/model.py` | 7,162 | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` |
| **Protected Session**| `chakrview/runtime/session.py` | 12,511 | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` |

> [!NOTE]
> **Tokenizer Invariant & Serialization History**:
> The canonical release checkpoint (`96357705...`) and runtime validation suite (`tests/test_v060_release_smoke.py`) strictly require the 112,627-byte `vocab.json` (`db4d6119...`) and 76,299-byte `merges.json` (`ada4b1dc...`) assets registered above in `EXPECTED_TOKENIZER_FILES_HASHES`. Historical commits on `main` tracked earlier whitespace/formatting serializations (108,533 bytes `e0f11c07...` and 72,461 bytes `7498d92a...`) containing the identical 4,093 token vocabulary mappings and 3,837 merges. Release verification gates enforce the canonical formatted files to guarantee bit-exact determinism across all release artifacts.

---

## 2. Parameter Accounting & Size Discrepancy Reconciliation

- **Unique Float32 Parameters**: `3,443,136` parameters across 6 decoder layers (hidden width 192, MLP 512, 6 heads, head dim 32, RoPE).
- **Tied Embedding Accounting**: The token embedding layer and output language modeling head share identical weights (`4096 * 192 = 786,432` parameters counted once).
- **Theoretical Float32 Memory Footprint**: `3,443,136 * 4 bytes = 13,772,544 bytes` (~13.77 MB).
- **Full Serialized Checkpoint File Size**: `42,212,787 bytes` (~42.2 MB). The full training checkpoint contains the model state dict, AdamW first and second moment optimizer states for all weights, learning rate scheduler state, RNG seeds, and complete training loss history (10,000 steps).

---

## 3. Demarcation Between Canonical and Experimental Assets

1. **Frozen Baseline ($\Delta W = 0$)**:
   - The canonical checkpoint weights are frozen and immutable. All cognitive modules, routers, memory bridges, and tool execution engines operate strictly around this baseline without mutating its parameters.
2. **Quarantine of Experimental Checkpoints**:
   - Experimental checkpoints generated across research runs (e.g. SFT pilots, pointer mechanisms, 15M scaling prototypes) MUST remain labeled with their experiment ID and stored in `experiments/`.
   - Never rename or substitute an experimental checkpoint as `training_checkpoint_step_10000.pt`.
3. **Public Distribution Mechanism**:
   - Because ordinary Git history must not be bloated with 42.2 MB binary weights, the canonical checkpoint is distributed exclusively as an authenticated release asset attached to the official GitHub Release (`v0.8.0`).
