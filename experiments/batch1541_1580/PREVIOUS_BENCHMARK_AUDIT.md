# Audit of Previous Benchmark Integrity (Batches 1501–1540)

## Phase 2 Independent Audit Report

### 1. Purpose of this Audit
To maintain absolute scientific integrity, this audit scrutinizes the experimental claims, metrics, baselines, task counts, and denominators reported in Batches 1501–1540 ([BASELINE_VS_NEW_ROUTER_RESULTS.md](file:///d:/Project/ChakrView/experiments/batch1501_1540/BASELINE_VS_NEW_ROUTER_RESULTS.md) and [FINAL_MISSION_REPORT.md](file:///d:/Project/ChakrView/experiments/batch1501_1540/FINAL_MISSION_REPORT.md)).

---

### 2. Discrepancy & Denominator Analysis

In [BASELINE_VS_NEW_ROUTER_RESULTS.md](file:///d:/Project/ChakrView/experiments/batch1501_1540/BASELINE_VS_NEW_ROUTER_RESULTS.md), the comparison table reported:
- **Track 1 Arithmetic:** Baseline `0.00%` (0/40) vs New `100.00%` (100/100)
- **Track 2 Verification:** Baseline `50.00%` (10/20) vs New `91.00%` (91/100)
- **Track 3 Routing:** Baseline `72.00%` (36/50) vs New `98.00%` (49/50)
- **Track 4 End-to-End:** Baseline `75.00%` (30/40) vs New `92.00%` (46/50)
- **Total Aggregate:** Baseline `48.00%` (72/150) vs New `95.33%` (286/300)

#### Key Audit Findings:
1. **Different Test Suites / Incomparable Denominators:**
   - The baseline numbers (0/40, 10/20, 30/40) were drawn from the **Batch 1461–1500 capability audit suite** ([capability_audit_200.json](file:///d:/Project/ChakrView/experiments/batch1461_1500/data/capability_audit_200.json)), whereas the new system was evaluated on the **Batch 1501–1540 suite** ([evaluation_suite_300.json](file:///d:/Project/ChakrView/experiments/batch1501_1540/data/evaluation_suite_300.json)).
   - **Crucial Limitation:** While the tasks tested the same conceptual capabilities, they were **not evaluated on identical task IDs**. Pooling them into an aggregate baseline delta ($72/150 \to 286/300$, $+47.33\%$) represents a comparison between two different task distributions rather than a true head-to-head matched benchmark.
2. **Post-Hoc Dataset Generation Adjustments:**
   - In Batch 1501–1540, during development of `generate_suite_300.py`, the initial entity names in `supported_claims` (e.g. `"session management"`, `"token bucket limiter"`) were adjusted to align with actual filenames and markdown section titles present in the corpus (e.g. `"Security Audit Trail"`, `"Token Lifecycle"`, `"Gateway API"`).
   - This updated the dataset hash from `8d0339a7...` to `40b1b96e...`.
   - The final manifest ([EVALUATION_SUITE_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch1501_1540/data/EVALUATION_SUITE_MANIFEST.json)) matches the final dataset hash `40b1b96e...` exactly (300 tasks, 143,537 bytes).
3. **Scoring Function Validity:**
   - In Track 1, the 100 tasks include 80 valid arithmetic expressions and 20 security/domain limit tests. The `SafeCalculator` achieved 100/100 because it correctly evaluated all 80 expressions and blocked all 20 security attacks.
   - In Track 2, the 91/100 score reflects 18/25 `SUPPORTED`, 23/25 `CONTRADICTED`, 25/25 `INSUFFICIENT`, 15/15 `CONFLICTING`, and 10/10 `INVALID`.

---

### 3. Matched Head-to-Head Re-Evaluation Policy for Batches 1541–1580
To prevent any ambiguity:
1. **No Mixed Denominators:** Every baseline comparison in Batches 1541–1580 will be executed on **identical task IDs** under identical scoring conditions.
2. **True Matched Baselines:** Where an old system (e.g. regex refutation or unchained calculator) is compared to the new system, both must be evaluated on the frozen Batch 1541–1580 evaluation suite.
3. **Descriptive Scores vs Head-to-Head:** Descriptive historical baselines must be clearly labeled as historical cross-references rather than matched trials.
