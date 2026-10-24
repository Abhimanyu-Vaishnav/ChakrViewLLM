# ChakrView Exp17 Instruction SFT v1 — Preflight Verification Checklist

**Experiment ID**: `exp17_instruction_sft_v1`  
**Parent Directory**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/`  
**Status**: PREFLIGHT GATE SPECIFICATION (Mandatory Verification Before Training)

---

## 1. Absolute Preservation & Security Invariants

All checks in Section 1 must evaluate to **VERIFIED** before any training script or generator is executed:

- [ ] **Canonical Base Checkpoint Immutability**:
  - File: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Expected SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Protection: File must be opened strictly in `"rb"` mode; never overwritten or passed as an output path.
- [ ] **Tokenizer Vocabulary & Merges Immutability**:
  - `data/experiments/vocab_4096/vocab.json` (Expected SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)
  - `data/experiments/vocab_4096/merges.json` (Expected SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`)
  - Verification: Tokenizer files must remain unaltered; vocab size must remain exactly 4096.
- [ ] **Protected Runtime Code Immutability**:
  - `chakrview/runtime/session.py` (Expected SHA-256: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`)
- [ ] **Exp16 v3 & v3.1 Datasets Immutability**:
  - All six Exp16 JSONL dataset files must match their established SHA-256 digests.
- [ ] **Git Working Tree Protection**:
  - No destructive Git commands (`git clean`, `git reset`, `git checkout`, `git stash`).
  - No commits performed during preflight or training phases.

---

## 2. Dataset Preflight & Integrity Verification Gates

Before fine-tuning can proceed, the generated dataset files (`train_exp17_v1.jsonl`, `val_exp17_v1.jsonl`, `test_exp17_v1.jsonl`) must pass the following automated checks:

- [ ] **Record Count & Split Sizing**:
  - Train split: exactly 320 records.
  - Val split: exactly 40 records.
  - Test split: exactly 40 records.
  - Total records: exactly 400.
- [ ] **JSON Schema Validation**:
  - 100% of records validate cleanly against the schema defined in [DATASET_SPEC.md](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/DATASET_SPEC.md).
- [ ] **ID Uniqueness & Bijection**:
  - Zero duplicate IDs within or across splits: $|ID_{\text{total}}| == 400$, $|ID_{\text{unique}}| == 400$.
- [ ] **Normal Chat Gate Benchmark Contamination Gate**:
  - Pre-run Levenshtein & n-gram overlap check confirms 0% exact match and $< 0.80$ similarity against the 20 prompts in `experiments/normal_chat_gate_v01/prompts.jsonl`.
- [ ] **Cross-Split Leakage Gate**:
  - $Prompt_{\text{train}} \cap Prompt_{\text{val}} == \emptyset$.
  - $Prompt_{\text{train}} \cap Prompt_{\text{test}} == \emptyset$.
- [ ] **Tokenization Bounds & Roundtrip Fidelity**:
  - For every record, `decode(encode(prompt)) == prompt` and `decode(encode(response)) == response`.
  - Input sequence length $1 + |p| + |r| \le 512$ tokens.
  - Zero out-of-bounds token IDs ($0 \le t < 4096$).

---

## 3. Training Runtime Preflight Gates

- [ ] **Loss Masking Parity Check**:
  - Synthetic test sequence confirms prompt positions are assigned target $-100$.
  - Final response token is correctly assigned target $\text{EOS}$ (`1`).
  - Active tokens count $> 0$ for all training batches.
- [ ] **CPU Resource Availability**:
  - System memory available $\ge 1.0\text{ GB}$.
  - PyTorch CPU multithreading configured cleanly.
- [ ] **Atomic Checkpointing Verification**:
  - Verification that the checkpoint directory exists and is writable.

---

## 4. Execution Sign-Off Gate

| Gate Level | Verification Requirement | Status | Authorized By |
| :--- | :--- | :---: | :--- |
| **Level 1: Design Review** | Architecture, dataset spec, and protocols complete. | **COMPLETE** | Antigravity AI |
| **Level 2: Dataset Generation**| Synthetic generation script created and validated. | **GATED** | Awaiting Human Approval |
| **Level 3: Preflight Hashes** | Checkpoint, tokenizer, and dataset hashes match. | **GATED** | Awaiting Level 2 |
| **Level 4: Training Execution**| Optimizer steps, loss logging, checkpoint creation. | **GATED** | Awaiting Level 3 & Human Approval |

> [!CAUTION]
> Training, optimizer steps, or dataset writing must NOT occur until explicit human approval is granted.
