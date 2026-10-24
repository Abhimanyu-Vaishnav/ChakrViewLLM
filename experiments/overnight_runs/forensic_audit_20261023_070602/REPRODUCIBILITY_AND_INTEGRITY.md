# ChakrView Post-Overnight Forensic Audit — Reproducibility & Integrity

**Auditor**: Independent ML Experiment Auditor  
**Audit Directory**: `experiments/overnight_runs/forensic_audit_20261023_070602/`  
**Evaluation Target**: `experiments/overnight_runs/mission_20261023_003512/`  
**Execution Timestamp**: 2026-10-23T07:10:45+05:30  

---

## 1. Canonical Protected Asset Verification

We independently computed the SHA-256 cryptographic digests of all five protected canonical ChakrView assets using standard Python `hashlib`.

| File Role | Workspace Relative Path | Frozen Reference SHA-256 | Current Workspace SHA-256 | Integrity Status |
| :--- | :--- | :---: | :---: | :---: |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED (MATCH)** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED (MATCH)** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED (MATCH)** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED (MATCH)** |
| **Protected Model** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **VERIFIED (MATCH)** |

- **Integrity Status**: **100% BIT-EXACT MATCH**. Zero canonical files were modified, replaced, or mutated during the overnight mission.

---

## 2. Git State & Working Tree Audit

- **Active Branch**: `dev/v0.6-v0.7-validation`
- **Current HEAD**: `f30136a5dc02934180a503fdf0a4c287469bf106`
- **Tracked Modifications**: **ZERO**. `git status --short` shows no modifications to tracked files.
- **Untracked Additions**: Strictly isolated under experiment and document folders:
  - `experiments/overnight_runs/mission_20261023_003512/` (Completed overnight run)
  - `experiments/overnight_runs/forensic_audit_20261023_070602/` (Current forensic audit directory)
  - `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/` (Official dataset split files)
  - Pre-existing research and documentation drafts in `docs/` and `experiments/`.

Zero commits, resets, cleans, or destructive operations occurred.

---

## 3. Checkpoint Loadability & Model Architecture Verification

We verified the experimental checkpoints saved in `mission_20261023_003512/sft_run/` using the repository's native runtime loader [`NeuralInferenceContract.from_checkpoint`](file:///D:/Project/ChakrView/chakrview/runtime/neural_inference_contract.py#L178):
- `sft_best_checkpoint.pt`: Successfully loaded. Model weights are finite (no NaNs or Infs), state dict conforms to `ChakrMicro` architecture (`max_seq_len=512`, `d_model=256`, `n_layers=6`, `n_heads=8`).
- `sft_checkpoint_step_400.pt`: Successfully loaded. Finite weights verified.

---

## 4. Reproducibility Commands

The findings of this audit can be independently reproduced using the following read-only commands:

```powershell
# 1. Canonical SHA-256 Hash Verification
.venv\Scripts\python.exe -c "
import hashlib
files = [
  'experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt',
  'data/experiments/vocab_4096/vocab.json',
  'data/experiments/vocab_4096/merges.json',
  'chakrview/runtime/session.py',
  'chakrview/brain/model.py'
]
for f in files:
  print(f, hashlib.sha256(open(f, 'rb').read()).hexdigest())
"

# 2. Checkpoint Loading Test
.venv\Scripts\python.exe -c "
from chakrview.runtime.neural_inference_contract import NeuralInferenceContract
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
tok, _ = load_tokenizer_artifacts('data/experiments/vocab_4096')
contract = NeuralInferenceContract.from_checkpoint('experiments/overnight_runs/mission_20261023_003512/sft_run/sft_best_checkpoint.pt', tok)
print('Successfully loaded sft_best_checkpoint.pt! Weight hash:', contract.expected_weight_hash)
"
```
