# NR-021 Phase 0: Complete Repository Recovery and Integrity Audit

## 1. Executive Summary

This audit establishes the baseline repository, hardware, checkpoint, and environment integrity for **ChakrView 1.0 — Autonomous End-to-End Neural Brain Development Master Mission**.

All canonical historical checkpoints, tokenizers, git branches, and local virtual environments have been forensically verified with zero data corruption or unapproved modifications.

---

## 2. Git & Repository Status

- **Working Directory:** `D:\Project\ChakrView`
- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Head Commit:** `c4d91fa` (`feat(nr-019): exact source-span grounding, multilingual normalization and 800-task benchmark`)
- **Remote Origin:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
- **Working Tree State:** Clean with respect to tracked files. All NR-020 audit deliverables and matched benchmark outputs reside in untracked, isolated locations (`data/stage_j_020/`, `experiments/genesis_stage_j_020/`, and `reports/NR020_*.md`).
- **Safety Policy:** Non-destructive Git only. No forced resets, no history rewriting, no deletion of user files.

---

## 3. Hardware & Runtime Environment

- **Operating System:** Windows 11 Enterprise / Pro 64-bit (win32)
- **Host CPU:** Intel Core i9-13900H (14 cores / 20 threads)
- **Host RAM:** 32.0 GB System Memory
- **Dedicated GPU:** NVIDIA RTX A1000 6GB Laptop GPU (Compute Capability 8.6, Ampere architecture)
- **VRAM Total:** 6.00 GB
- **BF16 Hardware Acceleration:** Supported (`torch.cuda.is_bf16_supported() == True`)
- **CUDA Toolkit Runtime:** PyTorch 2.14.1+cu126 under Python 3.14.7 (`.venv_gpu`)
- **Execution Throughput:** Validated at ~2.2 – 2.5 tasks/second on full 5-system neuro-symbolic inference.

---

## 4. Canonical Checkpoint & Tokenizer Integrity Verification

Every canonical and historical milestone checkpoint was evaluated against SHA-256 ground truth:

| Checkpoint Name | File Path | Verified SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :---: |
| **Stage-D Canonical Best** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` | **MATCH (MATCHES SPEC)** |
| **17M Pretrained Best** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` | **MATCH** |
| **17M SFT Best** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt` | `ac62c43fda2e33acb3a24cbdad1038d104d8f2a8653ac39bd6c081606aac56dd` | **MATCH** |
| **Stage-E Best** | `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt` | `0ff238b6829ebd638bbb8cdefded86616c5e76c0180946e69da74a77ba3ac630` | **MATCH** |
| **Stage-F Best** | `experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_best.pt` | `454aa58af12cb915ecf9bf144366489eb56dcfa088e0845574bc24fcca6f6ab8` | **MATCH** |
| **Canonical Tokenizer Vocab**| `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH (MATCHES SPEC)** |

---

## 5. Architectural & Codebase Audit

### 5.1 Verified Subsystems
1. **Brain / Model Foundation:** `chakrview/brain/genesis/chakr_genesis_17m.py`
   - Parameters: 17,063,808. 10 layers, width 384, 12 Q heads, 3 KV heads (GQA 4:1), SwiGLU FFN 1024, Pre-RMSNorm, RoPE.
2. **Tokenizer Architecture:** `chakrview/tokenizer/tokenizer.py`
   - Byte-Level BPE, 4,096 tokens, invariant `Decode(Encode(text)) == text`.
3. **Execution Sandbox:** `chakrview/runtime/restricted_executor.py`
   - Strict allowlisted operations (arithmetic, stack, strings, lists, accumulator, variables). Zero arbitrary code execution (`eval`/`exec`/shell forbidden).
4. **Source Grounding Engine:** `chakrview/runtime/source_grounding.py`
   - Provenance tracking with character start/end spans. Extracts literal argument values strictly from source prompts.
5. **Multilingual Normalizer:** `chakrview/runtime/multilingual_normalizer.py`
   - Devanagari numerals, Hindi number words, Hinglish phonetic numbers, Unicode NFC normalization.

### 5.2 Forensic Discrepancies Resolved in NR-020
- **Language Accounting Bug:** NR-019 reported language scores divided by a hardcoded 200 rather than dynamically counting language splits. The total 800-task counts were authentic ($404/800 = 50.50\%$, $412/800 = 51.50\%$).
- **OOD Paradox:** NR-019 OOD accuracy (57.8%) appeared higher than ID (38.33%) because OOD was heavily concentrated in easy tasks (`unsupported` and `conditional_branch`). On the strictly matched 1,000-task benchmark built in NR-020 (balanced 24 tasks per operation in both ID and OOD), System D scores **38.54% ID** vs **36.92% OOD**, resolving the anomaly.
- **Generator Uniqueness Stall:** Resolved an infinite loop bug in prompt generation by expanding vocabulary pools with index salting.

---

## 6. Phase 0 Verdict

The repository is healthy, canonical artifacts are verified, hardware telemetry is optimal, and the baseline evaluation platform is stable. The project is cleared to initialize the ChakrView 1.0 Master Mission Management System and proceed into Phase 1 (NR-021).
