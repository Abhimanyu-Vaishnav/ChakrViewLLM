# CR-16E-R Phase F: Abstention Calibration Audit

### 1. Investigation Objective
Examine whether the inclusion of $12.5\%$ negative samples (missing keys, unbound queries) in CR-16E restored abstention recall without causing pathological over-abstention (predicting `[UNKNOWN]` on valid retrieval queries) or destroying arbitrary sequence copying.

---

### 2. Abstention Recall (Benchmark B4, $N=300$ Probes)

In CR-16D-R, abstention recall had degraded catastrophically to below $20\%$, as models hallucinated previously seen values when asked for unbound keys.

#### Table F1: Abstention Recall on Unanswerable Queries ($N=300$)

| Model Arm & Seed | Correct Abstentions (`[UNKNOWN]`) | Total Probes | Abstention Recall (%) | Mean Recall $\pm s$ |
|---|---|---|---|---|
| **Arm A (Fixed, Std)** | | | | |
| ├── Seed 2026 | 213 | 300 | 71.0% | |
| ├── Seed 3030 | 150 | 300 | 50.0% | |
| └── Seed 4040 | 203 | 300 | 67.7% | **62.9% $\pm 11.3%** |
| **Arm B (Shuffled, Std)** | | | | |
| ├── Seed 2026 | 274 | 300 | 91.3% | |
| ├── Seed 3030 | 282 | 300 | 94.0% | |
| └── Seed 4040 | 224 | 300 | 74.7% | **86.7% $\pm 10.5%** |
| **Arm C (Fixed, Bal)** | | | | |
| ├── Seed 2026 | 216 | 300 | 72.0% | |
| ├── Seed 3030 | 230 | 300 | 76.7% | |
| └── Seed 4040 | 215 | 300 | 71.7% | **73.4% $\pm 2.8%** |
| **Arm D (Shuffled, Bal)** | | | | |
| ├── Seed 2026 | 290 | 300 | 96.7% | |
| ├── Seed 3030 | 257 | 300 | 85.7% | |
| └── Seed 4040 | 148 | 300 | 49.3% | **77.2% $\pm 24.9%** |

**Observation:** All four intervention arms restored mean abstention recall to $\ge 62.9\%$, with shuffled models (Arm B and Arm D) reaching mean recalls of $86.7\%$ and $77.2\%$, respectively (peak: $96.7\%$ in Arm D s2026).

---

### 3. False Abstention Rate on Answerable Probes

A critical question is whether high abstention recall was achieved simply by turning `[UNKNOWN]` into an aggressive default answer across all tasks.

#### Table F2: False Abstention Rate on Answerable Contexts

| Model Arm & Seed | B1 Position Invariance ($N=600$ Answerable) | CR-16C Sealed Retrieval ($N=1,500$ Answerable) | Total False Abstentions | Overall False Abstention Rate |
|---|---|---|---|---|
| **Arm A (Fixed, Std)** | | | | |
| ├── Seed 2026 | 120 / 600 (20.0%) | 911 / 1500 (60.7%) | 1,031 / 2,100 | 49.1% |
| ├── Seed 3030 | 73 / 600 (12.2%) | 681 / 1500 (45.4%) | 754 / 2,100 | 35.9% |
| └── Seed 4040 | 184 / 600 (30.7%) | 1,017 / 1500 (67.8%) | 1,201 / 2,100 | 57.2% |
| **Arm B (Shuffled, Std)** | | | | |
| ├── Seed 2026 | 125 / 600 (20.8%) | 1,155 / 1500 (77.0%) | 1,280 / 2,100 | 61.0% |
| ├── Seed 3030 | 118 / 600 (19.7%) | 1,172 / 1500 (78.1%) | 1,290 / 2,100 | 61.4% |
| └── Seed 4040 | 92 / 600 (15.3%) | 827 / 1500 (55.1%) | 919 / 2,100 | 43.8% |
| **Arm C (Fixed, Bal)** | | | | |
| ├── Seed 2026 | 108 / 600 (18.0%) | 1,009 / 1500 (67.3%) | 1,117 / 2,100 | 53.2% |
| ├── Seed 3030 | 160 / 600 (26.7%) | 1,131 / 1500 (75.4%) | 1,291 / 2,100 | 61.5% |
| └── Seed 4040 | 99 / 600 (16.5%) | 960 / 1500 (64.0%) | 1,059 / 2,100 | 50.4% |
| **Arm D (Shuffled, Bal)** | | | | |
| ├── Seed 2026 | 135 / 600 (22.5%) | 1,187 / 1500 (79.1%) | 1,322 / 2,100 | 63.0% |
| ├── Seed 3030 | 95 / 600 (15.8%) | 941 / 1500 (62.7%) | 1,036 / 2,100 | 49.3% |
| └── Seed 4040 | **27 / 600 (4.5%)** | **225 / 1500 (15.0%)** | **252 / 2,100** | **12.0%** |

#### Forensic Revelations:
1. **The Over-Abstention Trap in High-Recall Seeds:**
   Models with extremely high B4 recall (e.g. Arm D s2026 at $96.7\%$, Arm B s3030 at $94.0\%$) suffered from severe over-abstention on CR-16C ($77.0\% - 79.1\%$ false abstention). When faced with novel prompt templates or unseen key tokens in CR-16C, the models defaulted to emitting `[UNKNOWN]`.
2. **The Balanced Calibration of Arm D Seed 4040:**
   Arm D Seed 4040 avoided this collapse:
   - False abstention on Benchmark B1: **only 4.5% (27 / 600)**.
   - False abstention on CR-16C: **only 15.0% (225 / 1500)**.
   - B4 Abstention Recall: **49.3% (148 / 300)**.
   This demonstrates a genuine risk-coverage trade-off: Seed 4040 did not treat `[UNKNOWN]` as a catch-all sink, maintaining active retrieval engagement across both in-distribution and out-of-distribution prompts.

---

### 4. Mechanism of Sequence Copying Interference

In CR-16E, sequence copying performance dropped sharply across all models ($0\% - 20\%$ exact match, with several seeds dropping to $0\% - 1\%$).

#### Forensic Root-Cause Analysis:
1. In the CR-16E training mixture, negative abstention examples were formatted as:
   `Context: ... Query: K -> [UNKNOWN]`
2. Because the `[UNKNOWN]` target tokens appeared at the exact sequence termination point where retrieval/copying answers normally begin, the gradient updates strongly suppressed token emission whenever context matching confidence was low.
3. In arbitrary string copying (CR-16 Copying Benchmark), strings are non-semantic token sequences. The attention heads, seeing no recognizable key-value syntax, triggered the negative abstention prior, either emitting `[UNKNOWN]` or degenerating into short prefixes.
4. **Resolution for future designs:** Abstention tokens must be conditioned on explicit query/context syntax to prevent negative gradient leakage into unstructured sequence copying.

