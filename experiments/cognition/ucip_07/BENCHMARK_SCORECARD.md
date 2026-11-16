# UCIP-07: Sealed 800-Probe Benchmark Scorecard

## 1. Benchmark Identification & Integrity

- **Benchmark File**: `experiments/cognition/ucip_07/sealed_800_benchmark.json`
- **Authoritative SHA-256**: `31e786802f5dc1d61d17c8a10c1525f6d476c2d907898fd76f9ccf3a1f4800c3`
- **Total Probes**: 800 across 10 evaluation categories
- **Evaluation Engine**: `experiments/cognition/ucip_07/evaluator_800.py` (Calibrated with 7/7 unit tests passing)

---

## 2. Comprehensive Comparative Results

| Category | Probe Count | UCIP-06 Medium (Parent) | UCIP-07 Medium (Trained) | UCIP-07 Core (Scratch) | Delta (UCIP-07 vs 06 Medium) |
|---|---|---|---|---|---|
| **English Conversation** | 100 | 35 / 100 (35.00%) | **41 / 100 (41.00%)** | 0 / 100 (0.00%) | **+6.00%** |
| **English Explanations** | 80 | 8 / 80 (10.00%) | **12 / 80 (15.00%)** | 0 / 80 (0.00%) | **+5.00%** |
| **English Sentence Generation** | 70 | 16 / 70 (22.86%) | **19 / 70 (27.14%)** | 0 / 70 (0.00%) | **+4.28%** |
| **Devanagari Hindi** | 100 | 36 / 100 (36.00%) | **42 / 100 (42.00%)** | 0 / 100 (0.00%) | **+6.00%** |
| **Roman Hindi** | 80 | 43 / 80 (53.75%) | **47 / 80 (58.75%)** | 1 / 80 (1.25%) | **+5.00%** |
| **Hinglish** | 100 | 48 / 100 (48.00%) | **55 / 100 (55.00%)** | 0 / 100 (0.00%) | **+7.00%** |
| **Exact Copying** | 80 | 0 / 80 (0.00%) | 0 / 80 (0.00%) | 0 / 80 (0.00%) | 0.00% |
| **Elementary Arithmetic** | 60 | 3 / 60 (5.00%) | **6 / 60 (10.00%)** | 1 / 60 (1.67%) | **+5.00%** |
| **Instruction Following** | 60 | 16 / 60 (26.67%) | **19 / 60 (31.67%)** | 0 / 60 (0.00%) | **+5.00%** |
| **Semantic Consistency** | 70 | 6 / 70 (8.57%) | **10 / 70 (14.29%)** | 0 / 70 (0.00%) | **+5.72%** |
| **OVERALL TOTAL** | **800** | **212 / 800 (26.50%)** | **251 / 800 (31.37%)** | **2 / 800 (0.25%)** | **+4.87% (+18.4% rel)** |

---

## 3. Repetition & Failure Mode Breakdown

| Metric | UCIP-06 Medium | UCIP-07 Medium | UCIP-07 Core |
|---|---|---|---|
| **Repetitive Outputs** | 14 probes | **6 probes** (57% reduction) | 277 probes |
| **Empty Responses** | 0 probes | 0 probes | 0 probes |
| **Attractor Trapping** | High (`"A compiler transforms..."`) | Moderately Reduced | N/A (Sub-word babble) |

---

## 4. Key Category Insights

1. **Conversational Recovery**: English Conversation and Hinglish saw substantial gains (+6.0% and +7.0%), generating accurate greeting completions and natural responses.
2. **Multilingual Stability**: Roman Hindi (58.75%) and Devanagari Hindi (42.00%) are the strongest non-English domains.
3. **Exact Copying Hard Wall**: Exact copying scored 0.00% across all evaluated models. Models attempt to conversationalize the response (e.g., replying `"Sure! The sky is blue"` instead of strictly returning `"The sky is blue"`), failing the exact match criterion.
4. **Arithmetic Frontier**: Arithmetic improved from 5.0% to 10.0%, correctly calculating single-digit sums and subtractions, but multi-step operations remain beyond current capacity.
