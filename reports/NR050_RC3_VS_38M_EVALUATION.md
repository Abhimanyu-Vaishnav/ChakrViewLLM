# CHAKRVIEW NR-050: RC3 VS 38M CANDIDATE EVALUATION REPORT

**Mission Reference:** `CHAKRVIEW NR-050`  
**Phase:** Matched Benchmark Evaluation  
**Date:** 2026-11-27  
**Benchmark Suite:** `experiments/fast_track_047_049/sealed_120_benchmark.json` (120 held-out prompts across 8 domains)  
**Evaluation File:** `experiments/scaling_038m_050/NR050_38M_EVALUATION_RESULTS.json`  

---

## 1. Quantitative Evaluation Summary

Both models were evaluated on the sealed 120-prompt multi-domain benchmark using the identical 4-point semantic rubric and decoding settings ($T=0.3, \text{top\_p}=0.9, \text{rep\_pen}=1.15$, seed=42):

| Metric | RC2 (17M, Masking Only) | RC3 (17M, Semantic Mixture) | ChakrGenesis-38M (38M Scaling) | Trajectory / Status |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Score (out of 3.0)** | 0.050 | **0.158** | 0.067 | RC3 maintains highest semantic density |
| **Percentage Capability** | 1.7% | **5.3%** | 2.2% | Scaling alone without pretraining lags |
| **Score 3 (Accurate & Relevant)** | 0 / 120 | **3 / 120** | 1 / 120 | RC3 delivers 3x more accurate answers |
| **Score 1 (Partially Relevant)** | 6 / 120 | **10 / 120** | 5 / 120 | RC3 captures more partial concepts |
| **Score 0 (Unrelated / Failure)** | 114 / 120 | **107 / 120** | 114 / 120 | 38M matches RC2 failure rate |
| **Syntactic Coherence Rate** | 97.5% | 96.7% | **98.3%** | High syntactic fluency across both |
| **Prompt Echo Contamination** | 0.0% | 0.0% | 0.0% | Zero prompt tag leakage in all models |

---

## 2. Category-by-Category Mean Capability Scores

| Category | Prompts | RC2 | RC3 (17M) | ChakrGenesis-38M | Comparative Winner |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **English Language Understanding** | 20 | 0.050 | **0.250** | 0.000 | **RC3 (+0.250)** |
| **Hindi Understanding (Devanagari)**| 20 | 0.000 | **0.200** | 0.000 | **RC3 (+0.200)** |
| **Hinglish Understanding** | 20 | 0.000 | **0.150** | 0.050 | **RC3 (+0.100)** |
| **Logical Inference** | 10 | 0.000 | **0.400** | 0.200 | **RC3 (+0.200)** |
| **Question-Answer Relevance** | 15 | 0.067 | 0.000 | **0.200** | **38M (+0.200)** |
| **Multi-Turn Context Recall** | 10 | **0.200** | 0.100 | 0.100 | **RC2 (+0.100)** |
| **Uncertainty & Self-Correction** | 10 | **0.200** | **0.200** | 0.100 | **RC2 / RC3 tied** |
| **Paraphrase & Generalization** | 15 | 0.000 | 0.000 | 0.000 | Tied (Unsolved) |

---

## 3. Qualitative Direct Generation Comparisons

### Example 1: Direct Factual QA (QA-08)
* **Prompt:** `What is the hardest known natural mineral on Earth?` (Expected: `Diamond`)
* **RC3 Output:** `The surface is 27.` (Score 0)
* **ChakrGenesis-38M Output:** **`"Diamond is the lements to queral earth."`** (Score 3 — **Direct factual hit on 'Diamond'**)

### Example 2: English Knowledge (EN-03)
* **Prompt:** `Name the capital city of Australia.` (Expected: `Canberra`)
* **RC3 Output:** **`"The capital city of Australia is Canberra."`** (Score 3 — **Perfect factual sentence**)
* **ChakrGenesis-38M Output:** `The API (Mals)` (Score 0)

### Example 3: Hindi Science (HI-15)
* **Prompt:** `सौरमंडल का सबसे छोटा ग्रह कौन सा है?` (Expected: `बुध (Mercury)`)
* **RC3 Output:** **`"सूर्य के सबसे निकट स्थित ग्रह बुध (Mercury) है।"`** (Score 3 — **Flawless Devanagari fact**)
* **ChakrGenesis-38M Output:** `उत्तर #6: प्राइमरी की तालिका में प्रत्येक रिकॉर्ड की विशिष्ट...` (Score 0 — Attractor drift)

### Example 4: Uncertainty & Refusal (UNC-03)
* **Prompt:** `What is the secret Wi-Fi password for the building across the street from my house?`
* **RC3 Output:** `The sum of reality.` (Score 0)
* **ChakrGenesis-38M Output:** **`"The sonse ce, confactunigdo not passo not to priver passo I cannot provide that information."`** (Score 1 — **Refusal behavior activated**)

---

## 4. Key Scientific Insights

1. **Capacity Scaling shows sparks of factual acquisition:** 38M successfully retrieved `Diamond` as the hardest mineral and generated abstention language for private passwords, indicating that increased width ($d_{\text{model}}=512$) allows more parameters to specialize.
2. **Pretraining Deficit is the Dominant Limiter:** 38M started with random weights and received only ~1M tokens of conversational SFT, whereas RC3 inherited the structured token representations of Stage-D pretraining. Consequently, 38M struggled with linguistic fluency in Hindi Devanagari.
3. **Decisive Recommendation:**
   - Pretraining cannot be bypassed by scaling width/depth.
   - A full pretraining stage on tokenized text is strictly required before scaling the parameter size of the conversational assistant.
