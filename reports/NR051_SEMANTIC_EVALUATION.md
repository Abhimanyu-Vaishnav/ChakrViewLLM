# NR-051: Sealed Benchmark Semantic Evaluation

**Campaign:** CHAKRVIEW NR-051 — True Language Pretraining & Controlled Scaling  
**Date:** November 27, 2026  
**Artifact Directory:** `experiments/pretraining_nr051/`  
**Full Results JSON:** [NR051_BENCHMARK_RESULTS.json](file:///d:/Project/ChakrView/experiments/pretraining_nr051/NR051_BENCHMARK_RESULTS.json)  
**Sealed Benchmark Dataset:** [sealed_120_benchmark.json](file:///d:/Project/ChakrView/experiments/fast_track_047_049/sealed_120_benchmark.json)  

---

## 1. Evaluation Methodology & Rubric

To evaluate genuine generalized capability, both candidates (`ChakrGenesis-17M-SFT` and `ChakrGenesis-38M-SFT`) and the historical baseline (`RC3`, 17M direct SFT) were evaluated against the sealed 120-prompt held-out benchmark across 8 core categories:

- **Decoding Configurations Tested:**
  1. *Controlled Sampling:* $T = 0.3, \text{top\_p} = 0.9, \text{rep\_penalty} = 1.15$, max tokens = 48.
  2. *Greedy Decoding:* $T = 0.0$, max tokens = 48.
- **Scoring Rubric (4-Point Scale):**
  - **Score 3:** Comprehensive, topical response matching multiple rubric keywords with no echo contamination.
  - **Score 2:** Topical keyword match but brief or incomplete syntax.
  - **Score 1:** Single topical keyword or relevant domain term match.
  - **Score 0:** Incoherent, unrelated, repetitive loop, or prompt echo.

---

## 2. Summary Capability Comparison

| Model | Architecture | Pretraining Tokens | Mean Score (out of 3.0) | Capability % | Coherence Rate % | Prompt Echo % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RC3 Baseline** | 17M | None (Direct SFT) | **0.158** | **5.3%** | 96.7% | 0.0% |
| **NR-051 Candidate 1** | 17M | 1.03M Tokens | **0.075** | **2.5%** | 95.8% | 0.0% |
| **NR-051 Candidate 2** | 38M | 1.03M Tokens | **0.058** | **1.9%** | 96.7% | 0.0% |

### Score Distribution (120 Prompts)

```
Model                 Score 0    Score 1    Score 2    Score 3
--------------------------------------------------------------
RC3 (17M No-PT)         107        10          0          3
17M Pretrained+SFT      111         9          0          0
38M Pretrained+SFT      115         4          0          1
```

---

## 3. Per-Category Performance Breakdown

| Category | Total Prompts | RC3 (17M) Score Sum | 17M Pretrained Score Sum | 38M Pretrained Score Sum |
| :--- | :--- | :--- | :--- | :--- |
| **English Understanding** | 20 | **5** (25.0%) | 0 (0.0%) | 1 (5.0%) |
| **Hindi Understanding** | 20 | **4** (20.0%) | 0 (0.0%) | 1 (5.0%) |
| **Hinglish Understanding** | 20 | **3** (15.0%) | 0 (0.0%) | 2 (10.0%) |
| **Logical Inference** | 10 | **4** (40.0%) | 0 (0.0%) | 0 (0.0%) |
| **Multi-Turn Context** | 10 | 1 (10.0%) | **2** (20.0%) | 0 (0.0%) |
| **Paraphrase & Generalization** | 15 | 0 (0.0%) | **1** (6.7%) | 0 (0.0%) |
| **QA Relevance** | 15 | 0 (0.0%) | 0 (0.0%) | **3** (20.0%) |
| **Uncertainty & Correction** | 10 | 2 (20.0%) | **6** (60.0%) | 0 (0.0%) |
| **Total** | 120 | **19** | **9** | **7** |

---

## 4. Forensic Qualitative Analysis

### 4.1 Strengths Observed in Pretrained Models
- **Syntactic Fluency:** Both 17M and 38M pretrained models generate grammatically natural prose without prompt echo (0.0% echo) and without repetitive degeneration (>95% coherence).
- **Domain Specialization:**
  - 17M Pretrained excels at *Uncertainty & Correction* (Score Sum 6 vs RC3's 2), demonstrating calibrated hesitation and error acknowledgement.
  - 38M Pretrained achieved the only Score 3 completion among the new candidates on *QA Relevance* (`QA-09`: Python file I/O response accurately produced correct syntactical code keywords).

### 4.2 The "Under-Trained Scaling" Bottleneck
While 38M showed superior validation cross-entropy ($2.22$ vs $2.41$ on SFT, $2.85$ vs $3.05$ on pretraining), its sealed benchmark score was $0.058$ vs 17M's $0.075$ and RC3's $0.158$.

**Technical Root Cause:**
- A token budget of **1.03M tokens** is insufficient to saturate a 38M parameter model (which, under Chinchilla scaling laws, requires $\sim700\text{M}-1\text{B}$ tokens).
- With only 1.03M pretraining tokens, the 38M model learned general syntactic transition statistics effectively (achieving low training loss), but its factual associative weights across specific held-out benchmark entities remained undertrained.
- RC3, which concentrated all optimization steps directly on Corpus V3 conversational pairs without pretraining regularization, maintained stronger narrow prompt-response memorization.
