# CHAKRVIEW 1.0 RC1 — ARTIFACT INTEGRITY & VERIFICATION REPORT

## 1. Executive Summary

This independent forensic audit verified all frozen release artifacts, tokenizer files, release copies, and canonical historical baselines for **ChakrView 1.0 Release Candidate 1 (RC1)**.

### Verdict: **PASS — 100% BIT-FOR-BIT EQUIVALENCE VERIFIED**

---

## 2. Checkpoint & Artifact Checksums

| Target File | Verified SHA-256 Checksum | File Size (Bytes) | Verification Status |
| :--- | :--- | :---: | :---: |
| `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` | `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E` | 68,292,101 | **MATCH** |
| `dist/huggingface_release/pytorch_model.pt` | `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E` | 68,292,101 | **IDENTICAL TO RELEASE** |
| `experiments/genesis_stage_l_028/checkpoints/chakr_1_0_rc1_best.pt` | `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E` | 68,292,101 | **SOURCE MATCH** |
| `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | `2C21632EBAE4F0B05B9FF7091797ECC956AC0AD00EF6EC934381223BC273B269` | 68,292,655 | **UNTOUCHED CANONICAL** |
| `data/experiments/vocab_4096/vocab.json` | `DB4D6119FEDDBCC08D0CA8C580965BFC2B6D9729AED741E83044181307E7218A` | 112,627 | **CANONICAL VOCAB** |
| `data/experiments/vocab_4096/merges.json` | `ADA4B1DCFBB3D962A2E60203089881872A5063DE73B2FEF9EEFF81CE0016EFA7` | 76,299 | **CANONICAL MERGES** |
| `data/experiments/vocab_4096/config.json` | `0FCA72B6A70E9AE812589199C661EF6BE20CB9973F06353DB125A1EC7264E589` | 1,130 | **CANONICAL CONFIG** |
| `dist/huggingface_release/vocab.json` | `DB4D6119FEDDBCC08D0CA8C580965BFC2B6D9729AED741E83044181307E7218A` | 112,627 | **IDENTICAL TO CANONICAL** |
| `dist/huggingface_release/merges.json` | `ADA4B1DCFBB3D962A2E60203089881872A5063DE73B2FEF9EEFF81CE0016EFA7` | 76,299 | **IDENTICAL TO CANONICAL** |
| `dist/huggingface_release/tokenizer_config.json`| `0FCA72B6A70E9AE812589199C661EF6BE20CB9973F06353DB125A1EC7264E589` | 1,130 | **IDENTICAL TO CANONICAL** |
| `dist/huggingface_release/config.json` | `895D66D3F59BE8C3B69D64421B5AB65C36AFF54E437C12A8907E56DF1FEFC0DD` | 463 | **HF MODEL CONFIG** |
| `dist/huggingface_release/README.md` | `41F13C68AF189AF11DF3F5C9121EB6C2D8D2018C2F2A4BB4D2C6DC7DE810BA48` | 2,209 | **MODEL CARD** |

---

## 3. Tokenizer Path & Merge File Investigation
- **Discrepancy Clarification:** In early project scripts and some historical text files, the BPE merges were referred to as `merges.txt`. In the actual canonical tokenizer implementation (`chakrview/tokenizer/serialization.py`), the merges artifact is serialized strictly as JSON (`merges.json`).
- **Verification:** Both `vocab.json` and `merges.json` in `dist/huggingface_release/` match `data/experiments/vocab_4096/` bit-for-bit. A clean-process encoding test confirmed identical token ID generation (`test_sentence` produces identical token vectors across both directory paths).

---

## 4. Parameter & Architecture Verification
- **Total Parameters:** `17,063,808`
- **Layers:** 10 Transformer Blocks
- **Width ($d_{model}$):** 384
- **Attention Heads:** 12 Query Heads, 3 Key-Value Heads (GQA 4:1)
- **Non-Linearity:** SwiGLU ($d_{ffn}=1024$)
- **Positional Encoding:** RoPE ($\theta=10,000.0$)
- **Clean Process Loading:** Successfully instantiated directly from `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` with zero dependencies on local scratch files.
- **Historical Canonical Preservation:** Canonical Stage-D (`2c21632ebae4f0b...`) remained strictly unwritten and untouched.
