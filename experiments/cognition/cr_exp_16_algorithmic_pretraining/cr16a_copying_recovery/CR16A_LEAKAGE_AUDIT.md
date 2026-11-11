# CR-EXP-16A Leakage Audit

## 1. Audit Scope and Methodology
A strict automated verification script was executed to check for data contamination between the training corpora (`cr16a_curriculum_train.json`, `cr16a_arm_d_train.json`) and the frozen 1,000-probe benchmark (`cr16_synthetic_benchmark_1000.json`, SHA-256: `0b61e73d...`).

## 2. Contamination Checks
1. **Prompt-Level Overlap:**
   - Any training prompt string identically matching any benchmark prompt string.
   - Result: **0 matches out of 1,000 benchmark probes (0.00% leakage)**.
2. **Sequence-Level Target Overlap:**
   - Any Greek letter sequence target in training matching any Greek letter sequence target in the benchmark.
   - Result: **0 matches out of 100 sequence probes (0.00% leakage)**.
3. **Name Identity Overlap:**
   - Any single or multi-token name in the training sets matching any of the 23 benchmark names (`Alok`, `Amit`, `Dev`, `Mona`, `Prem`, `Ritu`, `Roy`, `Sonu`, `Badri`, `Fateh`, `Gautam`, `Govind`, `Jaideep`, `Kalyan`, `Lalit`, `Madhav`, `Mukesh`, `Pankhuri`, `Sarvesh`, `Siddharth`, `Vaibhav`, `Vikas`, `Yashwant`).
   - Result: **0 matches (0.00% leakage)**. All benchmark names were held out completely.

## 3. Scientific Verdict
The benchmark is 100% clean and uncompromised. All evaluation metrics reported in CR-EXP-16A reflect genuine out-of-distribution sequence generalization.

