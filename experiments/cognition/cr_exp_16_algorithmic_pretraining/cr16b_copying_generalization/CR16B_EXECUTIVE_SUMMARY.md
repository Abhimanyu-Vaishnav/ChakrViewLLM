# CR-EXP-16B Executive Summary

## 1. Mission Intent and High-Level Verdict
**CR-EXP-16B** was commissioned to independently test, attribute, and stress-test the sequence copying improvements reported in CR-EXP-16A. 

### Final Scientific Verdict:
**OPTION B: COPYING CAPABILITY PARTIALLY GENERALIZED.**

The capability is **real, reproducible, and mathematically attributable**, but it is **length-bounded, vocabulary-bounded, and tokenization-constrained**:
1. **CR-16A Results Successfully Re-verified (Gate 1 Passed):** Re-evaluation of historical checkpoints on the original 300 copying probes verified Arm D at **66/100 (66.0%)** exact sequence copying, **74/100 (74.0%)** token presence, and **100/100 (100.0%)** first target token accuracy bit-for-bit.
2. **Objective Factorial Attribution Established (Gate 4 Passed):** In a strictly controlled $2\times 2$ factorial experiment (300 steps, batch 16, $lr=3\times 10^{-4}$, 3,000 samples, 2 seeds):
   - **Target-Only Loss Masking is the dominant driver of copying capability**, lifting mean exact accuracy from **2.10% $\to$ 11.15%** ($+5.3\times$).
   - **Multi-Template Training is the primary driver of prompt transfer and first-token routing**, lifting mean first-token accuracy from **35.95% $\to$ 70.12%** ($+1.95\times$) and unseen template copying from 0% to 19.1%.
   - **Interaction Effect:** The combination (Arm D) achieves highest generalization (**12.13% mean EM**, **75.27% first token accuracy**).
3. **Generalization Across Sealed 1,500-Probe Benchmark (Gate 2 Partially Passed):**
   - On the newly generated, frozen 1,500-probe benchmark (`cr16b_sealed_benchmark_1500.json`):
     - Stage B Canonical Baseline: **0 / 1500 (0.00%)**
     - Factorial Arm A (Full + Single): **25 / 1500 (1.67%)**
     - Factorial Arm B (Target + Single): **158 / 1500 (10.53%)**
     - Factorial Arm C (Full + Multi): **26 / 1500 (1.73%)**
     - Factorial Arm D (Target + Multi): **200 / 1500 (13.33%)**
     - CR-16A Arm D (Extended 5k Pretraining): **508 / 1500 (33.87%)** EM, **628 / 1500 (41.87%)** Token Presence, **1238 / 1500 (82.53%)** First Token Accuracy.
4. **Name and Subword Boundaries Uncovered (Gate 3 Partially Passed):**
   - CR-16A Arm D successfully copies single-word English names (`'John'`: 6/6 exact matches, `'Roy'`: 1/1) and single-token Latin names (`'Victoria'`: 5/5, `'Elizabeth'`: 2/2).
   - However, Indian names that fragment into 3–5 subword characters (`'Gautam'`, `'Pankhuri'`) fail due to **pretraining bigram language prior competition** at the second subword position.

## 2. Core Scorecard Summary

| Capability / Benchmark Dimension | Stage B Baseline | Factorial Arm A (Full+Single) | Factorial Arm B (Target+Single) | Factorial Arm C (Full+Multi) | Factorial Arm D (Target+Multi) | CR-16A Arm D (5k Budget) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sealed 1,500-Probe Exact Match** | 0.00% | 1.67% | 10.53% | 1.73% | **13.33%** | **33.87% (508/1500)** |
| **Sealed 1,500-Probe Token Presence** | 0.00% | 11.33% | 15.80% | 12.33% | **29.27%** | **41.87% (628/1500)** |
| **First Target Token Accuracy** | 1.73% | 42.80% | 30.07% | 65.33% | **75.20%** | **82.53% (1238/1500)** |
| *— Familiar Greek Combos (110 probes)* | 0.0% | 5.5% | 28.2% | 3.6% | 27.3% | **62.7% (69/110)** |
| *— Short Sequences L=1..3 (110 probes)* | 0.0% | 5.5% | 31.8% | 10.0% | 33.6% | **79.1% (87/110)** |
| *— Medium Sequences L=4..6 (110 probes)* | 0.0% | 3.6% | 16.4% | 0.9% | 10.9% | **34.5% (38/110)** |
| *— Long Sequences L=7..10 (110 probes)* | 0.0% | 0.0% | 0.9% | 0.0% | 0.0% | **0.0% (0/110)** |
| *— Ordinary Words (110 probes)* | 0.0% | 3.6% | 21.8% | 6.4% | 39.1% | **91.8% (101/110)** |
| *— Repeated Sequences (100 probes)* | 0.0% | 4.0% | 12.0% | 1.0% | 13.0% | **47.0% (47/100)** |
| *— Distractor Resistant (100 probes)* | 0.0% | 0.0% | 2.0% | 0.0% | 15.0% | **29.0% (29/100)** |
| *— Unseen Prompt Templates (110 probes)* | 0.0% | 0.0% | 9.1% | 0.0% | **19.1%** | **16.4% (18/110)** |

