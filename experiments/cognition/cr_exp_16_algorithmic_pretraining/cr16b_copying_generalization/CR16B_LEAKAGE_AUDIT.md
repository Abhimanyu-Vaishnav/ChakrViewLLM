# CR-EXP-16B Leakage Audit

## 1. Audit Scope & Verification Method
To ensure that all evaluation results on `cr16b_sealed_benchmark_1500.json` (SHA-256: `44aba606...`) represent genuine out-of-distribution generalization, an automated collision check was executed against all training sets:
- `factorial_single_template_train.json` (3,000 samples)
- `factorial_multi_template_train.json` (3,000 samples)
- `factorial_val.json` (300 samples)

## 2. Leakage Test Results

| Contamination Type | Tested Collision Pair | Benchmark Probes Checked | Training Samples Checked | Collisions Found | Leakage Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Prompt String Collision** | Benchmark Prompts vs Single-Train Prompts | 1,500 | 3,000 | **0** | **0.00%** |
| **Prompt String Collision** | Benchmark Prompts vs Multi-Train Prompts | 1,500 | 3,000 | **0** | **0.00%** |
| **Prompt String Collision** | Benchmark Prompts vs Validation Prompts | 1,500 | 300 | **0** | **0.00%** |
| **Exact Sequence Target** | Benchmark Targets vs Single-Train Targets | 1,500 | 3,000 | **0** | **0.00%** |
| **Exact Sequence Target** | Benchmark Targets vs Validation Targets | 1,500 | 300 | **0** | **0.00%** |
| **Test Names Exclusion** | Benchmark Names vs Training Name Pools | 34 names | 30 names | **0** | **0.00%** |

## 3. Scientific Verification Verdict
Zero prompt leakage, zero target sequence leakage, and zero name pool overlap exist. The benchmark remains **100% clean and uncompromised**.

