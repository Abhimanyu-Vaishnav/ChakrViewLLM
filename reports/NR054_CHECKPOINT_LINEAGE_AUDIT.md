# NR-054: CHECKPOINT IDENTITY & PROVENANCE LINEAGE AUDIT

**Mission Identifier:** NR-054  
**Date:** November 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Exhaustive SHA-256 identification, lineage verification, and reconciliation of prefix discrepancies across canonical, historical, and experimental checkpoints.

---

## 1. Executive Summary

During prior missions (NR-051 through NR-053), various reports cited shortened SHA-256 prefixes or ambiguous names when referring to model checkpoints. This audit conducted an exhaustive byte-level hash recalculation of all primary and secondary model checkpoints stored in the repository.

Every model file has been located, verified for integrity, mapped to its exact pretraining/SFT lineage, and recorded with its full 64-character SHA-256 hash. **No checkpoint was modified, overwritten, or retrained.**

---

## 2. Reconciled Primary Checkpoint Registry

Below is the definitive reference table of the primary checkpoints across all generations of ChakrView:

| Checkpoint Identifier | Canonical Path | Parameter Count | Architecture Spec | Full SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| **Stage-D Canonical** | `checkpoints/genesis_17m_staged_best.pt` | 17.5M | d=512, L=6, H=8, seq=512 | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` |
| **RC1 Release** | `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` | 17.5M | d=512, L=6, H=8, seq=512 | `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e` |
| **RC2 Experimental** | `release/v1.0.0-rc2/chakr_1_0_rc2_best.pt` | 17.5M | d=512, L=6, H=8, seq=512 | `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30` |
| **RC3 Baseline** | `release/v1.0.0-rc3/chakr_1_0_rc3_best.pt` | 17.5M | d=512, L=6, H=8, seq=512 | `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c` |
| **NR051 17M Base** | `checkpoints/nr051/chakr_17m_pretrained_base.pt` | 17.5M | d=512, L=6, H=8, seq=512 | `8d871ca0aff6e23398bdb17f86521eb26dfb4b778d227035d77e2fe44803c25f` |
| **NR051 38M Base** | `checkpoints/nr051/chakr_38m_pretrained_base.pt` | 38.3M | d=768, L=6, H=12, seq=512 | `22306c9a479c80cd9ce3e937e3145fb1520241d96c3b17b3ff8aed2af4d7f304` |
| **NR051 17M SFT** | `checkpoints/nr051/chakr_17m_sft_candidate.pt` | 17.5M | d=512, L=6, H=8, seq=512 | `9f63e097108f4c3ef2fdc882826d570845a101d02f036bc1c055ad66183600b4` |
| **NR051 38M SFT** | `checkpoints/nr051/chakr_38m_sft_candidate.pt` | 38.3M | d=768, L=6, H=12, seq=512 | `a0639a3fba90352c47b151fb70e60209a1889cd3265b80c3cadcb751e3c5b31f` |
| **NR052 17M Base** | `checkpoints/nr052/chakr_17m_v4_base.pt` | 17.5M | d=512, L=6, H=8, seq=512 | `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062` |
| **NR052 38M Base** | `checkpoints/nr052/chakr_38m_v4_base.pt` | 38.3M | d=768, L=6, H=12, seq=512 | `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb` |
| **NR052 17M SFT** | `checkpoints/nr052/chakr_17m_nr052_sft.pt` | 17.5M | d=512, L=6, H=8, seq=512 | `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb` |
| **NR052 38M SFT** | `checkpoints/nr052/chakr_38m_nr052_sft.pt` | 38.3M | d=768, L=6, H=12, seq=512 | `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed` |

---

## 3. Discrepancy Resolution & Audit Findings

### Discrepancy 1: SFT Candidate Naming in NR-053
- **Observed Text in NR-053 Reports:** In some sections of NR-053 narrative text, the candidate was cited as `chakr_17m_sft_v4.pt`.
- **Finding:** The physical file produced by the NR-052 training pipeline is named `chakr_17m_nr052_sft.pt` (`c13211bea8...`). The hash reported in NR-053 tables (`c13211...`) matches the physical file `chakr_17m_nr052_sft.pt`. No file named `chakr_17m_sft_v4.pt` exists; it was a cosmetic typographical shorthand.

### Discrepancy 2: Hash Truncation Inconsistencies
- Different scripts truncated hashes to 7, 8, 10, or 12 characters (e.g., `8d871ca0`, `22306c9a`, `c13211be`).
- **Standard Established:** All subsequent reports and machine-readable JSON artifacts must maintain full 64-character SHA-256 strings with an explicit 8-character prefix reference.

---

## 4. Lineage Trees

```
[Corpus V1 / Synthetic Staged]
               │
               ▼
     genesis_17m_staged_best.pt (2c21632e...) 
               │
               ▼
     chakr_1_0_rc1_best.pt (cf585ebf...) [Severe formatting bug]
               │
               ▼ (Corpus V2 Remediation)
     chakr_1_0_rc2_best.pt (33f93c75...)
               │
               ▼ (Corpus V3 Remediation)
     chakr_1_0_rc3_best.pt (61b39ae2...)

[Corpus V3 Pretraining]
  ├── chakr_17m_pretrained_base.pt (8d871ca0...) ──(SFT V3)──► chakr_17m_sft_candidate.pt (9f63e097...)
  └── chakr_38m_pretrained_base.pt (22306c9a...) ──(SFT V3)──► chakr_38m_sft_candidate.pt (a0639a3f...)

[Corpus V4 Pretraining]
  ├── chakr_17m_v4_base.pt (58781710...) ──────────(SFT V4)──► chakr_17m_nr052_sft.pt (c13211be...)
  └── chakr_38m_v4_base.pt (7fb29b35...) ──────────(SFT V4)──► chakr_38m_nr052_sft.pt (e08d363e...)
```

---

## 5. Verification Conclusion

All 12 checkpoints exist on disk, are completely intact, load cleanly without tensor corruption, and match their recorded training configurations. No checkpoint has been altered during NR-054.
