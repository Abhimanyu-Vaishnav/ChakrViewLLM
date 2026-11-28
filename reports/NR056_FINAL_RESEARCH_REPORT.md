# NR-056: FINAL RESEARCH REPORT & SCIENTIFIC SYNTHESIS

**Mission Identifier:** NR-056  
**Date:** November 28, 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Independent scientific audit of NR-055 findings, architecture lineage, token budgets, benchmark independence, evaluator reliability, statistical replication, and clean capability measurement on Benchmark V3.

---

## 1. Executive Summary

Mission **NR-056** was conducted as a rigorous, audit-only investigation into the empirical findings and architectural claims of ChakrView. In strict accordance with mission directives, **no model training, fine-tuning, or checkpoint modification was executed**.

### Definitive Answers to the 12 Core Mission Questions:

1. **Which architecture configurations actually exist?**
   - Exactly two:
     - `ChakrGenesis17M` ($d_{\text{model}} = 384, L = 10, n_{\text{q}} = 12, n_{\text{kv}} = 3, \text{hidden\_dim} = 1024, V = 4096$, RoPE, RMSNorm, tied weights). Unique trainable parameters: **17,063,808**.
     - `ChakrGenesis38M` ($d_{\text{model}} = 512, L = 12, n_{\text{q}} = 8, n_{\text{kv}} = 4, \text{hidden\_dim} = 1408, V = 4096$, RoPE, RMSNorm, tied weights). Unique trainable parameters: **37,499,392**.
   - The historical citation of `d_model=512, 6 layers` for 17M was an uninstantiated proposal, and `d_model=768, 6 layers` for 38M was a documentation transcription error.
2. **Which checkpoint lineages are confirmed?**
   - All 12 primary checkpoints have unbroken, mathematically verified provenance trails back to root seeds and documented driver scripts (`experiments/forensics_nr056/checkpoint_lineage.json`).
3. **Which historical token-budget claims are verified?**
   - NR-051 Pretrain split: **1,026,382** tokens (1,048,576 tokens processed across 256 steps).
   - NR-052 Corpus V4: **624,068** verified unique tokens (1,181,206 tokens processed across 358 steps).
   - The citation of "10.5M tokens" in NR-055 was a **typographical error** confusing the initial target with the actual assembled dataset.
4. **Is Benchmark V2 sufficiently independent?**
   - **No.** While authored independently, 40% of its prompts had confirmed factual exposure in Corpus V1/V3, and all non-zero hits (Tokyo, Paris, Canberra, Pacific Ocean, British Pound) matched training facts.
5. **How reliable is the evaluator?**
   - High precision on English and arithmetic (0% false positives on adversarial tests), but suffers a **Devanagari word-boundary defect** where ASCII regex `\b` fails to recognize Hindi unicode word characters.
6. **Are NR-055 statistical results reproducible?**
   - The descriptive numbers are 100% reproducible. However, under **Holm-Bonferroni multiple-comparison correction**, the statistical advantage of 38M over 17M fails ($p_{\text{adj}} = 0.7464$), and 38M vs RC3 Baseline fails ($p_{\text{adj}} = 1.0000$).
7. **Which candidate performs best on clean evaluation?**
   - On the sealed, 300-prompt Benchmark V3, **NR-052 38M SFT** achieved the highest score (**1.22%**, 11/900 pts), followed by **RC3 Baseline** (**0.89%**, 8/900 pts).
8. **Is any candidate demonstrably superior after statistical correction?**
   - **No.** The confidence intervals between NR-052 38M, RC3 Baseline, and NR-052 17M overlap heavily. No candidate possesses a defensible, statistically significant capability advantage on clean held-out tasks.
9. **What can the models reliably do?**
   - They retrieve high-frequency trained factual pairs (Tokyo-Japan, Paris-France, Pacific-Ocean), generate short coherent English sentences, and exhibit 0.0% prompt echo.
10. **What capabilities remain unproven?**
    - Arithmetic, multi-step reasoning, logic syllogisms, reading comprehension extraction, and strict single-word instruction adherence remain completely unproven (0% performance).
11. **What is the most important engineering bottleneck?**
    - **Extreme Pretraining Under-Exposure:** 624k unique tokens is orders of magnitude below the Chinchilla scaling threshold for 17M–38M models. Models are capacity-starved and memorization-bound.
12. **What should NR-057 investigate next?**
    - Pretraining corpus expansion to $\ge 50\text{M}$ clean tokens and architectural optimization for long-horizon pretraining convergence.

---

## 2. Reconciled Science: Fact vs Claim

| Claim Category | Prior Understanding | Reconciled Forensic Truth |
| :--- | :--- | :--- |
| **Pretraining Token Volume** | "10.5M Tokens" | **624,068 unique tokens**; 1.18M tokens seen over ~1.9 epochs. |
| **38M Benchmark Superiority** | "Statistically significant improvement" | **Not statistically significant** under Holm correction ($p_{\text{adj}} = 0.7464$). |
| **Benchmark V2 Independence** | "Clean held-out evaluation" | **40% factual overlap** with Corpus V1/V3 training data. |
| **Factual Capability** | "General knowledge acquisition" | **Memorization of exposed training triples**; drops to 0% on unseen entities. |
| **Reasoning Ability** | "Emerging capability" | **Non-existent (0%)** on arithmetic, logic, and extraction. |

---

## 3. Mission Status

Mission **NR-056 is COMPLETED & FORENSICALLY RECONCILED**. All historical artifacts remain unmodified, and all findings are substantiated by raw, reproducible JSON records.
