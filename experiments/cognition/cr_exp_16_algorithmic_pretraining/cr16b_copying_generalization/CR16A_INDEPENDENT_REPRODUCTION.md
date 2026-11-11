# CR-16A Independent Reproduction & Verification Audit

## 1. Executive Summary of Reproduction Audit
As mandated by CR-EXP-16B Part 2, an independent, automated re-evaluation was conducted across all six historical checkpoints on the frozen 300 copying probes from `cr16_synthetic_benchmark_1000.json` (SHA-256: `0b61e73d...`).

Every single metric reported in CR-EXP-16A was reproduced **bit-for-bit**:
- **Stage B Baseline:** 0/100 exact sequence copying, 0/100 single name, 0/100 multi name (0/300 overall).
- **Stage A Pilot:** 0/100 exact sequence copying, 0/100 single name, 0/100 multi name (0/300 overall).
- **Arm A (Full-sequence reproduction):** 0/100 exact sequence copying (0/300 overall).
- **Arm B (Single-template target masked):** 0/100 exact sequence copying on benchmark format (0/300 overall).
- **Arm C (Multi-template curriculum):** **63/100 (63.0%)** exact sequence copying, **65/100 (65.0%)** token presence, **100/100 (100.0%)** first target token.
- **Arm D (Generalization pretrained):** **66/100 (66.0%)** exact sequence copying, **74/100 (74.0%)** token presence, **100/100 (100.0%)** first target token.

Gate 1 (**Reproducibility**) is **CONFIRMED AND VERIFIED**.

## 2. Independent Reproduction Scorecard Table

| Model Checkpoint | Exact Sequence Copying (100 probes) | Token Presence (100 probes) | First Target Token (100 probes) | Single Name Exact (100 probes) | Single Name 1st Token | Multi Name Exact (100 probes) | Multi Name 1st Token | Overall Copying Accuracy (300 probes) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage B Baseline** | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 16 / 100 (16.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | **0 / 300 (0.00%)** |
| **Stage A Pilot** | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | **0 / 300 (0.00%)** |
| **CR-16A Arm A** | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 16 / 100 (16.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | **0 / 300 (0.00%)** |
| **CR-16A Arm B** | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | **0 / 300 (0.00%)** |
| **CR-16A Arm C** | **63 / 100 (63.0%)** | **65 / 100 (65.0%)** | **100 / 100 (100.0%)** | 0 / 100 (0.0%) | 78 / 100 (78.0%) | 0 / 100 (0.0%) | 72 / 100 (72.0%) | **63 / 300 (21.00%)** |
| **CR-16A Arm D** | **66 / 100 (66.0%)** | **74 / 100 (74.0%)** | **100 / 100 (100.0%)** | 0 / 100 (0.0%) | 78 / 100 (78.0%) | 0 / 100 (0.0%) | 88 / 100 (88.0%) | **66 / 300 (22.00%)** |

## 3. Discrepancy & Confounding Variable Analysis
The audit confirmed three critical aspects of the CR-16A experimental setup:
1. **Benchmark Consistency:** All models were evaluated against identical 300 probe inputs with identical greedy decoding stopping parameters (`\n` or EOS, max 20 tokens).
2. **Evaluator Identity:** The evaluator used strict string matching (`pred.strip() == target.strip()`). No manual intervention or post-processing occurred.
3. **Training Confounders in CR-16A:**
   - Arm A used 1,200 samples (single template, full loss).
   - Arm B used 1,200 samples (single template, masked loss).
   - Arm C used 3,000 samples (multi-template, masked loss, 400 steps).
   - Arm D used 5,000 samples (multi-template, masked loss, 500 steps).
   - Because Arm C and Arm D differed from Arm A and Arm B in **both** token budget and template count, the improvement could not be exclusively attributed to target loss versus template diversity versus sample count.
   - This necessitates the rigorous 2×2 factorial design in Part 4 of CR-EXP-16B.

