# NR-055: FINAL RESEARCH REPORT & SCIENTIFIC SYNTHESIS

**Mission Identifier:** NR-055  
**Date:** November 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Architecture identity reconciliation, complete lineage verification, loader validation, robust benchmark development, and definitive capability evaluation across all historical candidates.

---

## 1. Executive Summary

Mission **NR-055** was commissioned to resolve the architectural discrepancies exposed during NR-054, establish an authoritative architecture registry, validate model loaders, eliminate evaluation keyword vulnerabilities, and measure genuine model capability using an independently authored 95-prompt benchmark suite.

### Key Conclusions:
1. **Authoritative Architecture Identity Established:**
   - There are exactly two foundational model architectures: `ChakrGenesis17M` (17,063,808 unique parameters) and `ChakrGenesis38M` (37,499,392 unique parameters).
   - The reported NR-054 discrepancy citing 38M as `d_model 768, 6 layers` was a documentation error; all four 38M checkpoints are verified at the tensor level to be `d_model 512, 12 layers, 8 Q heads, 4 KV heads, hidden_dim 1408`.
2. **Model Loaders Strictly Verified:**
   - All 12 primary checkpoints load with 0 missing and 0 unexpected keys under strict state dictionary loading.
   - Deterministic greedy generation succeeds across all checkpoints.
3. **Benchmark V2 Eliminates False Positives:**
   - The new 95-prompt evaluation suite and boundary-enforcing scoring engine successfully detected and rejected adversarial substring tricks (e.g. `ice` in `notice`, random numbers in prose).
4. **Empirical Capability Hierarchy:**
   - **`NR052_38M_SFT`** demonstrates the strongest overall performance (Raw Mean 0.179 / 3.0, 6.0%), with statistically significant superiority over NR-051 candidates ($p < 0.05$).
   - Factual recall on prominent entities (Tokyo, Paris, Canberra, Pacific Ocean, British Pound) is consistently reproducible.
   - Arithmetic, multi-step logical reasoning, and reading comprehension extraction remain near zero across all 17M and 38M models.

---

## 2. Definitive Answers to Core Mission Questions

### 1. What architectures actually exist in the repository?
- Exactly two:
  - **`ChakrGenesis17M`:** $d_{\text{model}} = 384, L = 10, n_{\text{q}} = 12, n_{\text{kv}} = 3, \text{hidden\_dim} = 1024, V = 4096$, RoPE, RMSNorm, tied embeddings. Unique parameters: **17,063,808**.
  - **`ChakrGenesis38M`:** $d_{\text{model}} = 512, L = 12, n_{\text{q}} = 8, n_{\text{kv}} = 4, \text{hidden\_dim} = 1408, V = 4096$, RoPE, RMSNorm, tied embeddings. Unique parameters: **37,499,392**.

### 2. Which architecture produced each historical checkpoint?
- Stage-D Canonical, RC1, RC2, RC3, NR-051 17M Base/SFT, and NR-052 17M Base/SFT were all produced by `ChakrGenesis17M`.
- NR-051 38M Base/SFT and NR-052 38M Base/SFT were all produced by `ChakrGenesis38M`.

### 3. Whether reported parameter counts are correct?
- **Yes.** The parameter counts are verified down to individual tensor elements:
  - 17M models: 17,063,808 unique trainable parameters (18,636,672 total tensor elements including duplicate tied LM head export).
  - 38M models: 37,499,392 unique trainable parameters (39,596,544 total tensor elements including duplicate tied LM head export).

### 4. Whether training and evaluation use the intended model configurations?
- **Yes.** Training scripts in NR-051 and NR-052 instantiated the exact matching configuration classes (`Genesis17MConfig` and `Genesis38MConfig`). Forward passes in evaluation scripts matched the checkpoint dimensions identically.

### 5. Which candidate demonstrates the strongest reproducible capability on independently authored questions?
- **`NR052_38M_SFT`** (`chakr_38m_nr052_sft.pt`) demonstrated the highest performance (score 17/285, mean 0.179/3.0, 6.0%), with `RC3_Baseline` performing second (13/285, mean 0.137/3.0, 4.6%).

### 6. Is any candidate capable of reliable reasoning?
- **No.** Across 15 arithmetic prompts and 10 logical reasoning prompts, no candidate demonstrated reliable step-by-step logic. The few points scored were from isolated factual associations (e.g. recognizing that fish live in water).

### 7. Does 38M outperform 17M under controlled conditions?
- **In pretraining validation loss:** Undeniably yes ($1.1684$ vs $1.4763$, a $26.5\%$ perplexity drop).
- **In Benchmark V2 capability:** 38M achieved 17 points vs 17M's 8 points. While this shows an upward capability trend, on a 95-prompt sample size the paired difference has $p = 0.1244$, falling short of statistical significance at $\alpha = 0.05$.

### 8. What is the most important next engineering bottleneck?
- **Token Budget and Pretraining Scale:** 10.5M tokens is insufficient for a 38M parameter model to develop general syntax and broad entity representation. Chinchilla optimal scaling requires $\sim 700\text{M}$ tokens for 38M parameters. Scaling pretraining data volume from 10M to 100M+ tokens is the primary engineering bottleneck.

---

## 3. Mission Status & Ledger Update

- Mission NR-055 is marked as **COMPLETED & SCIENTIFICALLY VERIFIED**.
- All historical checkpoints remain intact and unaltered.
- Machine-readable files:
  - `experiments/forensics_nr055/architecture_registry.json`
  - `experiments/forensics_nr055/checkpoint_lineage.json`
  - `experiments/forensics_nr055/benchmark_v2_results.json`
  - `experiments/forensics_nr055/blind_evaluation_records.json`
