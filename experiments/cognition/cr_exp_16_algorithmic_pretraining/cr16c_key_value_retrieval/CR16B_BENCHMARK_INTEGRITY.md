# CR-16B Benchmark Integrity Report

## 1. Structural Verification
The sealed 1,500-probe benchmark (`cr16b_sealed_benchmark_1500.json`, SHA-256: `44aba606...`) was analyzed for structural integrity:
- **Total Probe Count:** Exactly 1,500.
- **Probe ID Uniqueness:** Exactly 1,500 unique IDs (`CR16B_CAT_A_0001` through `CR16B_CAT_N_1500`).
- **Category Balancing:** Sum of 14 categories equals exactly 1,500.
- **Answer Specifications:** Every probe contains a non-empty `expected_target` string.

## 2. Leakage and Duplication Audit
1. **Train/Val Leakage:**
   Zero prompt collisions and zero target sequence collisions were found between the benchmark and the factorial training/validation sets.
2. **Prompt Multiplicity within Benchmark:**
   The audit revealed that 97 prompts appear twice across different benchmark categories (e.g. `Echo sequence: alpha\nOutput: ` in `CAT_D` short sequence and `Copy: word -> ` in `CAT_D` and `CAT_I`).
   - This occurs because short single-word probes were generated independently for length stratification (`CAT_D`) and lexical dictionary copying (`CAT_I`).
   - While each probe has a unique probe ID, this redundancy slightly over-weights single-token copying in the aggregate score.
   - Future benchmarks should enforce strict prompt uniqueness across all categories.

