# NR-020: Benchmark Accounting Audit

## 1. Forensic Reconciliation of the 800 Tasks
A line-by-line machine audit was conducted on `data/stage_i_019/benchmark_800_results.json` containing 800 detailed evaluation records.

### A. The Denominator and Language Accounting Defect
In NR-019, the summary reporting script calculated:
```python
"lang_acc": {k: round((v / (200 if k != 'mixed' else 200)) * 100, 2) for k, v in m['lang'].items()}
```
**Forensic Audit Findings**:
1. In `generate_nr019_data.py`, the final 200 tasks were generated with random language choices:
   ```python
   lang = random.choice(["en", "hi", "hinglish"])
   ```
   As a consequence, the benchmark task records in `benchmark_800_results.json` did **not** have a `"mixed"` language label. Instead, every task was labeled as `"en"` (267 tasks), `"hi"` (269 tasks), or `"hinglish"` (264 tasks), summing to exactly 800 tasks.
2. The benchmark reporting script aggregated correct tasks under keys `"en"`, `"hi"`, and `"hinglish"`, but divided each by a fixed hardcoded denominator of `200` rather than the true count of tasks for that language!
3. For example, for System D:
   - English: 145 correct / 267 tasks = **54.3%** (incorrectly reported as $145 / 200 = 72.5\%$).
   - Hindi: 121 correct / 269 tasks = **45.0%** (incorrectly reported as $121 / 200 = 60.5\%$).
   - Hinglish: 138 correct / 264 tasks = **52.3%** (incorrectly reported as $138 / 200 = 69.0\%$).
   - Sum of correct tasks: $145 + 121 + 138 = \mathbf{404}$. Overall accuracy: $404 / 800 = \mathbf{50.50\%}$.

### B. Were all 800 tasks evaluated?
**YES**. Every system evaluated all 800 tasks without omissions or skipping.
The overall task counts ($404/800 = 50.50\%$ for System D, $412/800 = 51.50\%$ for System E) are **100% numerically genuine**. The discrepancy was strictly a reporting bug in the language sub-category percentage denominator.

---

## 2. Corrected Scorecard Across All 800 Tasks

| System | Overall Strict Acc | In-Distribution (300) | Out-of-Distribution (500) | English (267) | Hindi (269) | Hinglish (264) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Pure Neural** | **2.62%** (21/800) | 2.33% (7/300) | 2.80% (14/500) | 3.0% (8/267) | 3.7% (10/269) | 1.1% (3/264) |
| **System B: Deterministic Parser** | **17.62%** (141/800) | 5.67% (17/300) | 24.80% (124/500) | 28.1% (75/267) | 14.1% (38/269) | 10.6% (28/264) |
| **System C: Neural Router Free** | **7.38%** (59/800) | 3.33% (10/300) | 9.80% (49/500) | 7.9% (21/267) | 7.4% (20/269) | 6.8% (18/264) |
| **System D: Neural Grounded** | **50.50%** (404/800) | **38.33%** (115/300) | **57.80%** (289/500) | **54.3%** (145/267) | **45.0%** (121/269) | **52.3%** (138/264) |
| **System E: Cascaded Hybrid** | **51.50%** (412/800) | **39.67%** (119/300) | **58.60%** (293/500) | **57.3%** (153/267) | **45.0%** (121/269) | **52.3%** (138/264) |
