# CANONICAL AND CORPUS HASH RECONCILIATION

**Batch:** 2301–2340  
**Focus:** Verification and Audit of Invariant Cryptographic Hashes  
**Status:** ALL CANONICAL INVARIANTS VERIFIED AND PASSING  

---

## 1. Summary of Protected Invariants

The repository enforces strict architectural and data-integrity invariants across all experimentation batches:

| Invariant Artifact | Expected / Protected Hash | Actual Verified SHA-256 | Match | Description & Provenance |
| :--- | :--- | :--- | :---: | :--- |
| **Canonical Baseline Step-10,000 Checkpoint** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **YES** | 42,212,787 bytes at `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`. |
| **Canonical Architecture Initialization Digest** | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **YES** | Deterministic weight initialization under `torch.manual_seed(42)` for `ChakrMicro(ModelConfig())` (3,443,136 parameters). |
| **Cleaned Corpus Manifest** | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` | **YES** | 1,345,799 bytes at `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` (2,985 deduplicated passages). |
| **Canonical Tokenizer Vocabulary** | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **YES** | 112,627 bytes at `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json`. |
| **Canonical Tokenizer Merges** | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **YES** | Merges file at `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json`. |
| **Protected File Integrity** | `chakrview/runtime/session.py` | Untouched | **YES** | Pre-existing user modifications preserved without conflict. |

---

## 2. Manifest Hash Reconciliation Details

### Original Corpus v2 vs Cleaned Corpus Manifest
- **Original Manifest:** `experiments/batch2021_2060/CORPUS_V2_MANIFEST.json` (SHA-256: `4c7fc6ed32790bbec183ee670eb094c64672bb35b93f22bdb3c003c07de02519`). This manifest contained 3,000 raw documents prior to data cleansing.
- **Cleaned Manifest:** `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` (SHA-256: `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380`). Cleansed in Batch 2101–2140, resolving 15 duplicate/degenerate passages to establish a 2,985-passage frozen corpus.
- **Audit Conclusion:** The difference between `4c7fc6ed...` and `e3db033c...` reflects genuine data cleaning in Batch 2101–2140, not data corruption. All subsequent evaluation runs strictly employ `CORPUS_V2_CLEANED_MANIFEST.json`.

---

## 3. Architecture Hash Mechanism

The protected hash `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` is calculated as follows:
```python
import hashlib, torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

torch.manual_seed(42)
model = ChakrMicro(ModelConfig())
hasher = hashlib.sha256()
with torch.no_grad():
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode('utf-8'))
        hasher.update(param.detach().cpu().numpy().tobytes())
digest = hasher.hexdigest()
assert digest == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
```
Direct re-execution confirms that the model architecture definition, layer ordering, and initialization parameters are 100% bit-exact with the codebase invariants.
