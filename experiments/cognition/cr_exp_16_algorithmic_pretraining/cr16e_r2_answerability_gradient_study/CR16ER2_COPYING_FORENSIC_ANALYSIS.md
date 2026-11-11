# CR-16E-R2 Phase D: Sequence Copying Regression Forensics

### 1. The Copying Regression Phenomenon
In CR-16A Arm D, ChakrMicro demonstrated strong sequence copying capability:
- **Parent Baseline (CR-16A Arm D):** **66 / 100 (66.0%)** exact sequence match ($74/100$ token presence).
- **CR-16E Performance ($N=100$ probes across 12 checkpoints):**
  - Arm A (Fixed, Std): Seed 2026 = $20\%$, Seed 3030 = $3\%$, Seed 4040 = $2\%$ (Mean: $8.33\%$).
  - Arm B (Shuff, Std): Seed 2026 = $1\%$, Seed 3030 = $6\%$, Seed 4040 = $0\%$ (Mean: $2.33\%$).
  - Arm C (Fixed, Bal): Seed 2026 = $0\%$, Seed 3030 = $1\%$, Seed 4040 = $1\%$ (Mean: $0.67\%$).
  - Arm D (Shuff, Bal): Seed 2026 = $0\%$, Seed 3030 = $1\%$, Seed 4040 = $0\%$ (Mean: $0.33\%$).

---

### 2. Forensic Dissection of Raw Model Predictions

To determine *why* copying broke down, we examined all $1,200$ individual copy predictions across all 12 checkpoints:

#### Table D1: Failure Mode Breakdown on CR-16 Copying Benchmark ($N=100$ Probes)

| Model Condition | Exact Matches | `[UNKNOWN]` Emissions | First Word Matches | Premature EOS (Shorter) | Exact Word Count | Longer Output | Sample Prediction vs Target |
|---|---|---|---|---|---|---|---|
| **Arm A s2026** | **20 / 100** | **0 / 100 (0%)** | **87 / 100 (87%)** | 32 / 100 (32%) | 64 / 100 (64%) | 4 / 100 (4%) | `delta beta omican` vs `delta beta omicron zeta` |
| **Arm A s3030** | 3 / 100 | 0 / 100 (0%) | 55 / 100 (55%) | 53 / 100 (53%) | 46 / 100 (46%) | 1 / 100 (1%) | `delta beta omicron zulu` vs `delta beta omicron zeta` |
| **Arm A s4040** | 2 / 100 | 0 / 100 (0%) | 50 / 100 (50%) | 50 / 100 (50%) | 48 / 100 (48%) | 2 / 100 (2%) | `delta echo opha zulu` vs `delta beta omicron zeta` |
| **Arm B s2026** | 1 / 100 | 0 / 100 (0%) | 45 / 100 (45%) | 53 / 100 (53%) | 41 / 100 (41%) | 6 / 100 (6%) | `delta echo oscar` vs `delta beta omicron zeta` |
| **Arm B s3030** | 6 / 100 | 0 / 100 (0%) | 68 / 100 (68%) | 50 / 100 (50%) | 48 / 100 (48%) | 2 / 100 (2%) | `delta beta omicron zulu` vs `delta beta omicron zeta` |
| **Arm B s4040** | 0 / 100 | 0 / 100 (0%) | 42 / 100 (42%) | 77 / 100 (77%) | 21 / 100 (21%) | 2 / 100 (2%) | `delta echo` vs `delta beta omicron zeta` |
| **Arm C s2026** | 0 / 100 | 0 / 100 (0%) | 24 / 100 (24%) | 63 / 100 (63%) | 33 / 100 (33%) | 4 / 100 (4%) | `delta echo oscar` vs `delta beta omicron zeta` |
| **Arm C s3030** | 1 / 100 | 0 / 100 (0%) | 25 / 100 (25%) | 40 / 100 (40%) | 49 / 100 (49%) | 11 / 100 (11%) | `delta echo oscar` vs `delta beta omicron zeta` |
| **Arm C s4040** | 1 / 100 | 0 / 100 (0%) | 26 / 100 (26%) | 51 / 100 (51%) | 43 / 100 (43%) | 6 / 100 (6%) | `delta echo oscar` vs `delta beta omicron zeta` |
| **Arm D s2026** | 0 / 100 | 0 / 100 (0%) | 23 / 100 (23%) | 60 / 100 (60%) | 38 / 100 (38%) | 2 / 100 (2%) | `delta echo oscar` vs `delta beta omicron zeta` |
| **Arm D s3030** | 1 / 100 | 0 / 100 (0%) | 28 / 100 (28%) | 66 / 100 (66%) | 31 / 100 (31%) | 3 / 100 (3%) | `delta echolf` vs `delta beta omicron zeta` |
| **Arm D s4040** | 0 / 100 | 0 / 100 (0%) | 21 / 100 (21%) | 45 / 100 (45%) | 47 / 100 (47%) | 8 / 100 (8%) | `delta echo oscar` vs `delta beta omicron zeta` |

---

### 3. Key Forensic Diagnoses

1. **`[UNKNOWN]` Emission is NOT the Direct Mechanism:**
   - Out of $1,200$ test evaluations on the copying benchmark, **strictly 0 / 1,200 predicted `[UNKNOWN]`**.
   - The model does not mistakenly output the literal abstention token `[UNKNOWN]` when given a copying prompt.
2. **Subword Phonetic Drift & Premature EOS:**
   - In Arm A s2026, the model matched the first word in **$87\%$** of probes, but drifted on later words (e.g. `omicron` $\to$ `omican`, `zeta` truncated).
   - In balanced-span arms (Arms C and D), first-word accuracy collapsed to $21\% - 28\%$, and the model frequently emitted familiar vocabulary words from the retrieval dictionary (e.g. `echo`, `oscar`) instead of copying Greek alphabet tokens verbatim.
3. **Hypothesis for Phase E Controlled Study:**
   - While the model does not emit literal `[UNKNOWN]` strings, did the gradient updates on the $500$ negative abstention examples perturb the attention routing heads or output projection weights needed for arbitrary sequence copying?
   - Phase E directly tests this causal hypothesis via a controlled loss-masking experiment.

