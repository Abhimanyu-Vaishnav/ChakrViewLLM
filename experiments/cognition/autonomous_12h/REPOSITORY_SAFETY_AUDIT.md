# CHAKRVIEW — STAGE 0 REPOSITORY AND ARTIFACT SAFETY AUDIT

## 1. Environment & Repository Status
- **Date & Timestamp**: 2026-11-05T23:13:30+05:30
- **Repository Root**: `D:\Project\ChakrView`
- **Active Git Branch**: `exp/cbr-01-conversational-breakthrough`
- **HEAD Commit**: `8e866b495c9693566b39c6a715701e220a028227`
- **Python Environment**: `Python 3.14.7` (venv at `D:\Project\ChakrView\.venv\Scripts\python.exe`)
- **PyTorch**: `2.14.0+cpu`
- **Hardware Profile**: 14 physical cores / 20 logical threads, 31.64 GB RAM, CPU-first execution.

---

## 2. Protected Checkpoint & Artifact Hash Verification

| Artifact / Target | Expected SHA-256 | Actual SHA-256 | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Canonical Baseline** (`seed=42`, ChakrMicro v0.1) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **EXACT MATCH** | Model parameter weight digest via `compute_canonical_model_hash` |
| **Parent Checkpoint** (`cbr_sft_04_step_200.pt`) | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **EXACT MATCH** | File size: 41,417,634 bytes |
| **Stage A Candidate** (`stage_a_candidate_step_100.pt`) | `1ac891fbb21af0945751e85f946d3020deb3f01d21969ed298205ba176b9facf` | `1ac891fbb21af0945751e85f946d3020deb3f01d21969ed298205ba176b9facf` | **EXACT MATCH** | File size: 41,398,929 bytes |
| **Session Runtime** (`chakrview/runtime/session.py`) | *Protected Invariant* | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **UNMODIFIED** | Size: 12,511 bytes |
| **Tokenizer Vocab** (`data/experiments/vocab_4096/vocab.json`) | *Protected Invariant* | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **UNMODIFIED** | Vocab size: 4,096 |
| **Tokenizer Merges** (`data/experiments/vocab_4096/merges.json`)| *Protected Invariant* | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **UNMODIFIED** | 3,840 merges |

---

## 3. Audit of Existing CR-EXP-03 Deliverables
All required deliverables for **CR-EXP-03** (`experiments/cognition/cr_exp_03/`) were completed and verified:
- `RESIDUAL_SIGNAL_CAUSALITY_STUDY.md`: Present, complete, and contains quantitative data.
- `checkpoint_identity_audit.json`: Verified with transcription error resolution.
- `implementation_audit.md`: Audited forward sequence and training configuration.
- `experiment_a_results.json`: Residual vector geometry across layers.
- `experiment_b_results.json`: FFN causal ablation across layers.
- `experiment_c_results.json`: Attention causal ablation across layers.
- `experiment_d_results.json`: Counterfactual activation replacement.
- `experiment_e_results.json`: Multi-prompt consistency (Rhodium, Platinum, Phoenix, Silver).
- `reproducibility_manifest.json`: System and git metadata recorded.

---

## 4. Model Loading & Inference Interfaces
- **Model Definition**: `chakrview/brain/model.py` (`ChakrMicro` class).
- **Configuration**: `chakrview/brain/config.py` (`ModelConfig`: `vocab_size=4096, d_model=192, n_layers=6, n_heads=6, hidden_dim=512, tie_embeddings=True`).
- **Tokenizer**: `chakrview/tokenizer/serialization.py` (`load_tokenizer_artifacts`).
- **Inference Pipeline**: `chakrview/runtime/pipeline.py` (`InferenceEngine`, `InferenceRequest`, `GenerationConfig`).

---

## 5. Available Conversational Datasets in Workspace
- `data/sft/cbr_sft_04/dataset.jsonl` (and `train.jsonl`, `validation.jsonl`): Clean bilingual dialogue and conversational exchanges used in CBR-SFT-04.
- `data/sft/cbr_sft_02/splits/cbr_sft_02_train_540.jsonl`: 540 train samples of dialogue.
- `data/sft/cbr_sft_03b/splits/cbr_sft_03b_train_45.jsonl`: Targeted repair samples.
- `data/sft/exp34_instruction_mixture_train.jsonl`: Instruction-following dataset.

---

## 6. Safety Audit Decision
All protected files, canonical hashes, and parent checkpoints are intact.
No destructive operations have taken place.
**Safety Gate PASSED.** Proceeding to Stage 1 verification review and Stage 2 Conversational Capability Audit.

