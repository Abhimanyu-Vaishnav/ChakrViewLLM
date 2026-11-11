# CR-16E Dataset Specification & Contamination Audit

### 1. Dataset Partition Specifications
All four factorial arms were constructed with a total sample size of $N = 4,000$ examples, composed of:
1. **Sequence Copying Replay:** 1,500 examples (balanced template mixture: `Copy:`, `Repeat sequence:`, `Echo:`, `[INPUT] [COPY]`).
2. **Key-Value Retrieval:** 2,000 examples (structured by position and span curricula).
3. **Calibrated Abstention:** 500 unanswerable examples (`Store: ... Query: <unbound>` $\to$ `[UNKNOWN]`).

| Dataset Partition | Fact Positions | Target Value Lengths | File Name | SHA-256 Hash |
|---|---|---|---|---|
| **Arm A** | Fixed (Pos 0) | Standard (Len 1–2) | `arm_a_fixed_stdspan_train.json` | Recorded in `ARTIFACT_HASHES.txt` |
| **Arm B** | Shuffled Uniformly | Standard (Len 1–2) | `arm_b_shuffled_stdspan_train.json` | Recorded in `ARTIFACT_HASHES.txt` |
| **Arm C** | Fixed (Pos 0) | Balanced (Len 1–4) | `arm_c_fixed_balancedspan_train.json` | Recorded in `ARTIFACT_HASHES.txt` |
| **Arm D** | Shuffled Uniformly | Balanced (Len 1–4) | `arm_d_shuffled_balancedspan_train.json` | Recorded in `ARTIFACT_HASHES.txt` |
| **Validation** | Shuffled Uniformly | Balanced (Len 1–4) | `cr16e_multitask_val.json` | Recorded in `ARTIFACT_HASHES.txt` |

---

### 2. Contamination & Leakage Audit
Before synthesizing any training data, a strict contamination filter loaded 4,364 sealed benchmark prompts across:
- `CR-16 Synthetic Copying Benchmark` (1,000 probes)
- `CR-16C Sealed Retrieval Benchmark` (1,500 probes)
- `CR-16D Compositional Generalization Benchmark` (1,000 probes)
- `CR-16E B1 Position Invariance Benchmark` (600 probes)
- `CR-16E B3 Multi-Token Span Benchmark` (500 probes)
- `CR-16E B4 Calibrated Abstention Benchmark` (300 probes)

**Result:** Zero string or prompt collisions across all partitions ($0.00\%$ overlap rate). Every benchmark evaluation probe is strictly zero-shot out-of-distribution relative to the training sets.

