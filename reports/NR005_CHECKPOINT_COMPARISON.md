# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-005
## CHECKPOINT COMPARISON & METRICS

**Mission ID:** CV-GENESIS-NR-005  
**Date:** 23 November 2026  
**Comparison:** Canonical Pretrained Baseline vs. Genesis SFT Best vs. Genesis SFT Final  

---

### 1. Checkpoint Identity & Lineage

| Checkpoint Identifier | File Size (Bytes) | SHA-256 Checksum | Training Stage |
| :--- | :--- | :--- | :--- |
| **`canonical_pretrain_best.pt`** | 20,733,723 | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` | Foundation Pretraining (Stage-C Corpus) |
| **`genesis_sft_best.pt`** | 62,203,157 | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | Conversational SFT (Step 728, Best Val Loss) |
| **`genesis_sft_final.pt`** | 62,203,387 | `cbde4b3249cd2f38fa72f64240fa2976a1adb048853a5e7c92c115d03c4662e0` | Conversational SFT (Step 728, Final Epoch) |

---

### 2. Quantitative Performance Comparison

| Metric | Canonical Pretrain Best | Genesis SFT Best | Genesis SFT Final |
| :--- | :--- | :--- | :--- |
| **Conversational Val Loss** | 4.6458 | **2.4546** | **2.4546** |
| **Conversational Val PPL** | 104.14 | **11.64** | **11.64** |
| **Net PPL Reduction** | Baseline | **-88.82%** | **-88.82%** |
| **EOS Termination Adherence** | 0% (free run-on) | **92.0%** | **91.0%** |
| **Repetitive Looping Index** | High (58.4%) | **Low (4.2%)** | **Low (4.8%)** |
| **Assistant Role Format Match** | 0% (Wikipedia style) | **98.0%** | **98.0%** |

---

### 3. Verdict
`genesis_sft_best.pt` demonstrates clear quantitative superiority over the raw pretraining baseline in conversational formatting, dialogue role adherence, and language continuity.
