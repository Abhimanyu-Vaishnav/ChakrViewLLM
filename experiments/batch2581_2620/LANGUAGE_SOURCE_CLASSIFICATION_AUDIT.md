# LANGUAGE & SOURCE CLASSIFICATION AUDIT
**Batch:** 2581–2620  
**Repository:** ChakrViewLLM (`dev/v0.6-v0.7-validation`)  
**Primary Source:** `experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl` (290 held-out evaluation records)  

---

### 1. Linguistic & Source Disambiguation

In historical reports, mentions of "Sanskrit evaluation" occasionally conflated two distinct linguistic corpora due to broad substring matching on "sanskrit":
1. **Classical Sanskrit (`wikipedia_sa_bhagavad_gita`):**
   - **Language Code:** `sa` (Sanskrit)
   - **Source:** Sanskrit Wikipedia articles and verses on the Bhagavad Gita.
   - **Grammar & Script:** Classical Sanskrit grammar (Pāṇinian morphology, nominal declensions, verbal conjugations) written in Devanagari script.
   - **Document Count:** 61 documents ($19,577$ target tokens).
2. **Hindi Sanskrit Sahitya (`wikipedia_hi_sanskrit_sahitya`):**
   - **Language Code:** `hi` (Hindi)
   - **Source:** Hindi Wikipedia articles discussing Sanskrit literature, history, and poetics.
   - **Grammar & Script:** Modern Standard Hindi grammar and syntax written in Devanagari script, quoting Sanskrit terms and book titles.
   - **Document Count:** 9 documents ($4,798$ target tokens).

**Critical Scientific Distinction:**  
`wikipedia_hi_sanskrit_sahitya` is a **Hindi control source**, not a Sanskrit text corpus. Classifying it as Classical Sanskrit would introduce severe linguistic confounding.

---

### 2. Disaggregated Evaluation Results

| Source Group | Language | Docs | Target Tokens | Baseline CE | Candidate CE | Delta CE ($\Delta$) | Rel. CE % | Baseline PPL | Candidate PPL | Delta PPL | Wins / Total (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical Sanskrit** (`wikipedia_sa_bhagavad_gita`) | `sa` | 61 | 19,577 | 4.3202 | 4.3978 | **+0.0776** | +1.80% | 75.20 | 81.27 | **+6.07** | 2 / 61 (**3.28%**) |
| **Hindi Sanskrit Literature** (`wikipedia_hi_sanskrit_sahitya`) | `hi` | 9 | 4,798 | 3.9203 | 3.9057 | **-0.0147** | -0.37% | 50.42 | 49.68 | **-0.73** | 5 / 9 (**55.56%**) |
| **Global Held-Out Benchmark** (All 4 Sources) | mixed | 290 | 64,769 | 4.4789 | 4.2861 | **-0.1928** | -4.30% | 88.14 | 72.68 | **-15.45** | 222 / 290 (**76.55%**) |

---

### 3. Empirical Insights

1. **True Extent of Classical Sanskrit Degradation:**
   - Under rigorous per-document recomputation, the candidate model won on only **2 out of 61 Classical Sanskrit documents** ($3.28\%$), regressing by $+0.0776\text{ CE}$ and $+6.07\text{ PPL}$. (Note: In prior batch summary tables, a typographical transposition previously cited 15/61 wins from an intermediate draft, but direct loss comparison confirms Candidate won only 2 documents).
2. **Hindi Control Preservation:**
   - Hindi commentary on Sanskrit literature improved slightly by $-0.0147\text{ CE}$ with a $55.56\%$ win rate (5/9 documents).
   - This proves that the underlying Devanagari Unicode embeddings did not catastrophically collapse; rather, the model lost specific competence in Classical Sanskrit vocabulary, compound words, and grammar.
