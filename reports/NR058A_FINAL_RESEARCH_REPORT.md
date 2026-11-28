# MISSION NR-058A FINAL RESEARCH & PILOT REPORT
## ChakrView | Controlled Training Pilot (Corpus V6 | 17M vs 38M)

**Date:** November 28, 2026  
**Mission:** NR-058A Controlled Training Pilot  
**Artifact ID:** `NR058A_FINAL_RESEARCH_REPORT`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Pilot Workspace:** [experiments/pilot_nr058a/](file:///d:/Project/ChakrView/experiments/pilot_nr058a/)  
**Comparative JSON:** [pilot_comparison_analysis.json](file:///d:/Project/ChakrView/experiments/pilot_nr058a/pilot_comparison_analysis.json)

---

## 1. Executive Summary

Mission **NR-058A** was executed under strict, bounded authorization as a controlled training pilot comparing the physical `ChakrGenesis17M` and `ChakrGenesis38M` model families on the newly expanded, decontaminated `Corpus V6` dataset.

### Strict Mission Boundary Enforcement:
- **Step Cap Enforced:** Exactly **100 optimizer steps per model** were executed.
- **Zero Full-Scale Pretraining:** Multi-epoch, multi-seed, and unbounded training were strictly avoided.
- **Data Split Integrity:** Optimization occurred exclusively on `Corpus V6 Train` (4,163,555 tokens); evaluation occurred exclusively on `Corpus V6 Val` (191,686 tokens). The `Test Split` (217,990 tokens) and sealed benchmark suites were **100% untouched and isolated**.
- **No Artifact Overwriting:** Canonical checkpoints (`release/v1.0.0-rc1`, `rc2`, `rc3`) and canonical tokenizer (`vocab_4096`) remained frozen and unmodified.

---

## 2. Key Quantitative Pilot Results

Both models completed their authorized 100-step pilots smoothly and stably, processing matched token exposures with finite gradients, zero NaNs, and controlled VRAM consumption.

| Performance Metric | ChakrGenesis17M | ChakrGenesis38M | Comparative Delta (38M vs 17M) |
|---|---|---|---|
| **Parameters** | 17,063,808 | 37,499,392 | +20,435,584 (+120% capacity) |
| **Architecture Specification** | $d=384, L=10, Q=12, KV=3, FFN=1024$ | $d=512, L=12, Q=8, KV=4, FFN=1408$ | Scaled dimension & layers |
| **Optimizer Steps** | 100 | 100 | Matched step budget |
| **Tokens Processed** | 319,036 | 317,660 | Matched exposure (~318k tokens) |
| **Initial Train Loss (Step 1)** | 8.3907 | 8.4312 | +0.0405 |
| **Final Train Loss (Step 100)** | 6.2932 | **6.0986** | **-0.1946 (Lower Loss)** |
| **Initial Val Loss (Pre-train)** | 8.3855 | 8.4235 | +0.0380 |
| **Final Val Loss (Post-100)** | 6.1671 | **6.0940** | **-0.0731 (Lower Loss)** |
| **Initial Val Perplexity** | 4,382.96 | 4,552.76 | +169.80 |
| **Final Val Perplexity** | 476.80 | **443.19** | **-33.61 (Lower PPL)** |
| **Next-Token Val Accuracy** | 3.77% | **3.79%** | +0.02% |
| **Validation Loss Decay** | -2.2184 | **-2.3295** | **Steeper loss reduction** |
| **Throughput (Tokens/s)** | **23,491.8 tok/s** | 16,093.6 tok/s | -7,398.2 tok/s (17M is 46% faster) |
| **Average Step Latency** | **126.5 ms** | 176.6 ms | +50.1 ms |
| **Peak GPU VRAM** | **1.491 GB** | 2.057 GB | +0.566 GB (Both < 35% of 6.0 GB) |
| **Host System RAM** | 1.60 GB | 1.66 GB | Negligible footprint |
| **Total Pilot Runtime** | **13.58 seconds** | 19.74 seconds | Both < 20 seconds |
| **Checkpoint Size** | 65.13 MB | 143.09 MB | Matches float32 state dicts |

---

## 3. Loss Dynamics and Convergence Behavior

1. **Faster Loss Decay in 38M:** Despite starting at a slightly higher initial cross-entropy validation loss ($8.4235$ vs $8.3855$), the 38M parameter model experienced a faster loss decay rate ($\Delta = -2.3295$ vs $-2.2184$), ending at a lower final validation loss ($6.0940$ vs $6.1671$) and lower validation perplexity ($443.19$ vs $476.80$).
2. **Gradient Stability:**
   - Both models exhibited well-behaved gradient norms. Initial warmup gradient norms peaked at 2.6–3.8 and settled smoothly into the $0.55 - 0.95$ range between steps 20 and 100.
   - Zero gradient clipping anomalies or unstable divergence occurred.
3. **Hardware Headroom:**
   - At sequence length 256 and batch size 16, the 38M model peaked at **2.057 GB VRAM** (only 34.3% of the 6.00 GB GPU capacity), confirming ample headroom for longer sequence lengths (512 tokens) or larger batch sizes during future full-scale runs.

---

## 4. Checkpoint Lineage & Cryptographic Artifact Inventory

All pilot checkpoints were saved into the isolated workspace `experiments/pilot_nr058a/` without modifying any historical checkpoint:

| Artifact Description | File Path | File Size | SHA-256 Hash |
|---|---|---|---|
| **ChakrGenesis17M Pilot Checkpoint** | `experiments/pilot_nr058a/chakr_chakrgenesis17m_pilot_100step.pt` | 65.13 MB | `81acd27c7992c15c15d29d17107db03ecdbaef334a61c0d3f1f0f59f6701c6f3` |
| **ChakrGenesis38M Pilot Checkpoint** | `experiments/pilot_nr058a/chakr_chakrgenesis38m_pilot_100step.pt` | 143.09 MB | `bf37d46f926a8337c77c2308e4b0a6e761fe20f1f2c2fa61e2d6626db256a743` |
| **17M Pilot Summary Record** | [chakrgenesis17m_pilot_summary.json](file:///d:/Project/ChakrView/experiments/pilot_nr058a/chakrgenesis17m_pilot_summary.json) | ~20 KB | Verified JSON |
| **38M Pilot Summary Record** | [chakrgenesis38m_pilot_summary.json](file:///d:/Project/ChakrView/experiments/pilot_nr058a/chakrgenesis38m_pilot_summary.json) | ~20 KB | Verified JSON |
| **Comparative Analysis Record** | [pilot_comparison_analysis.json](file:///d:/Project/ChakrView/experiments/pilot_nr058a/pilot_comparison_analysis.json) | ~3 KB | Verified JSON |
| **Preflight Audit Record** | [preflight_audit.json](file:///d:/Project/ChakrView/experiments/pilot_nr058a/preflight_audit.json) | ~2 KB | Verified JSON |

---

## 5. Scientific Limitations & Boundaries

In accordance with strict research integrity:
1. **No Semantic Claims:** A 100-step reduction in cross-entropy loss from ~8.4 to ~6.1 demonstrates mathematical convergence and optimization stability, but **does NOT constitute proof of linguistic intelligence, reasoning, or understanding**.
2. **No Premature Winner Declaration:** While 38M achieved lower loss in 100 steps, a 100-step window (~318k tokens) is too brief to confirm asymptotic superiority or long-term generalization.
3. **No Benchmark Overfitting:** Sealed Benchmark V3 was intentionally unexecuted to protect it against pilot overfitting or evaluation leakage.

---

## 6. Mission Outcome & Verdict

### Model Status Verdict:
- **ChakrGenesis17M:** **PASS** (100 steps completed, stable loss decay, 100% finite gradients, zero anomalies).
- **ChakrGenesis38M:** **PASS** (100 steps completed, lower final validation loss, stable gradients, ample VRAM headroom).

### Overall Mission Verdict: **PASS**

The controlled training pilot successfully validated that:
1. The `Corpus V6` training and validation pipeline functions flawlessly in PyTorch with AMP fp16 and AdamW.
2. The canonical Byte-Level BPE 4096 tokenizer drives language modeling smoothly with zero runtime errors.
3. Both physical architectures train stably on the local RTX A1000 GPU with zero memory overflow risks.
4. The system is verified and ready for a staged multi-epoch pretraining campaign when authorized.
