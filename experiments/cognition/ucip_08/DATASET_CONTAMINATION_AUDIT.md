# UCIP-08: Dataset Contamination & Template Overlap Audit

## 1. Audit Methodology

We performed exact, normalized, and template-level overlap scans between the frozen 800-probe benchmark (`sealed_800_benchmark.json`) and all 296 `.jsonl` training corpora across the ChakrView repository.

---

## 2. Quantitative Overlap Findings

| Dataset Category | Dataset Path | Exact Benchmark Matches | Partial / Template Overlap |
|---|---|---|---|
| **UCIP-06 SFT Training Set** | `experiments/cognition/ucip_06/data/ucip06_train.jsonl` | **47 prompts** | **140 prompts** |
| **UCIP-06 SFT Validation Set** | `experiments/cognition/ucip_06/data/ucip06_validation.jsonl` | **5 prompts** | **25 prompts** |
| **UCIP-02 Arithmetic & SFT** | `experiments/cognition/ucip_02/data/ucip02_train.jsonl` | **8 prompts** | **32 prompts** |
| **Canonical SFT Conversational** | `data/sft/cbr_sft_07/dataset.jsonl` | **1 prompt** | **18 prompts** |
| **All Other Pretraining Corpora** | Stage B, Stage C, algorithmic | **0 prompts** | Low |

---

## 3. The Synthetic Template Expansion Phenomenon

The sealed 800-probe benchmark was created by programmatically taking **80 base concepts** (e.g., `"Hello ChakrView. How are you?"`, `"What does a CPU do?"`, `"If I have 5 apples and give 2 to my friend..."`) and multiplying each concept across 10 prefix variants:
1. Base prompt
2. `Hey! {prompt}`
3. `Kindly tell me: {prompt}`
4. `Please respond: {prompt}`
5. `Greetings! {prompt}`
6. `Hello there, {prompt}`
7. `Could you answer: {prompt}`
8. `Quick question: {prompt}`
9. `ChakrView, {prompt}`
10. `Dear ChakrView, {prompt}`

### Root Cause of Generalization Failure:
Because 47 of the 80 base concepts were present in the training set of UCIP-06/UCIP-07, the benchmark was evaluating **template permutation recall** rather than open-ended conversational understanding. When the owner tested fresh, un-templated questions or nuanced variants, the model failed completely because it had only memorized the local prefix-target pairings.
