# CR-EXP-16A Capability Scorecard

## 1. Frozen Benchmark Results (300 Copying Probes)
Evaluated strictly against the sealed `cr16_synthetic_benchmark_1000.json` (SHA-256: `0b61e73d...`).

| Capability / Category | Stage B Baseline (Parent) | CR-16 Stage A Pilot | Arm C (Curriculum) | Arm D (Generalization) |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Sequence Copying** (100 probes) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | **63 / 100 (63.0%)** | **66 / 100 (66.0%)** |
| *— Sequence Length 3 (21 probes)* | 0 / 21 (0.0%) | 0 / 21 (0.0%) | 19 / 21 (90.5%) | **19 / 21 (90.5%)** |
| *— Sequence Length 4 (21 probes)* | 0 / 21 (0.0%) | 0 / 21 (0.0%) | 16 / 21 (76.2%) | **17 / 21 (81.0%)** |
| *— Sequence Length 5 (31 probes)* | 0 / 31 (0.0%) | 0 / 31 (0.0%) | 19 / 31 (61.3%) | **20 / 31 (64.5%)** |
| *— Sequence Length 6 (27 probes)* | 0 / 27 (0.0%) | 0 / 27 (0.0%) | 9 / 27 (33.3%) | **10 / 27 (37.0%)** |
| **Token Presence (Sequence Copying)** | 0 / 100 (0.0%) | 0 / 100 (0.0%) | **65 / 100 (65.0%)** | **74 / 100 (74.0%)** |
| **Single-Token Name Copying** (100 probes) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%)* |
| **Multi-Token Name Copying** (100 probes) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%)* |
| **Overall Copying Benchmark Accuracy** | **0 / 300 (0.0%)** | **0 / 300 (0.0%)** | **63 / 300 (21.0%)** | **66 / 300 (22.0%)** |

*\*Note: Autoregressive EM is 0% on unseen fragmented names, but first-token copy induction accuracy reaches **78.0%** on single-token names and **88.0%** on multi-token names.*

## 2. In-Context Induction Head Diagnostics
| Diagnostic Metric | Stage B Baseline | CR-16 Stage A Pilot | Arm D |
| :--- | :---: | :---: | :---: |
| Sequence First Token Accuracy | 0.0% | 6.0% | **100.0%** |
| Single Name First Token Accuracy | 0.0% | 0.0% | **78.0%** |
| Multi Name First Token Accuracy | 0.0% | 0.0% | **88.0%** |
| Common Words Copying (14 held-out words) | 0 / 14 (0.0%) | 0 / 14 (0.0%) | **14 / 14 (100.0%)** |

