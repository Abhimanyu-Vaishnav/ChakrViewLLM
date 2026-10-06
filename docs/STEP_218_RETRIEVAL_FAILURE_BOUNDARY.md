# Step 218: Three-Stage Failure Boundary

## 1. Scientific Objective
Explicitly decomposes the associative recall pipeline into three testable operational stages:
- **STAGE A**: Contextual Association Formation (Pair representation contrastive separation $\ge 0.50$)
- **STAGE B**: Value Representation Retrieval (Positive cosine margin & rank significantly above chance $< 2.0$)
- **STAGE C**: Vocabulary / Final Token Output (Accuracy $\ge 0.50$)

Evaluates identical prompt instances across all three stages simultaneously to pinpoint the precise boundary of failure.

## 2. Decision Thresholds
- **Stage A Threshold**: $\text{Score} \ge 0.50$.
- **Stage B Threshold**: Margin $> 0.0$ AND $\text{Rank} < 2.0$ (for $N=3$ options).
- **Stage C Threshold**: Token Accuracy $\ge 0.50$.

## 3. Three-Stage Boundary Evaluation Table

| Condition | Stage A Score | Stage A Passed | Stage B Margin | Stage B Rank | Stage B Passed | Stage C Token Acc | Stage C Passed | First Failure Stage |
|---|---|---|---|---|---|---|---|---|
| `known_known` | 0.2840 | False | -0.0365 | 2.50 | False | 0.0000 | False | **STAGE_A** |
| `known_unseen` | 0.2715 | False | -0.0370 | 2.60 | False | 0.0000 | False | **STAGE_A** |
| `unseen_known` | 0.2650 | False | -0.0330 | 2.50 | False | 0.0000 | False | **STAGE_A** |
| `unseen_unseen` | 0.2601 | False | -0.0460 | 2.70 | False | 0.0000 | False | **STAGE_A** |

## 4. Scientific Conclusion
- **Boundary Pinpointed**: The first failure boundary in the frozen baseline backbone is **STAGE_A**.
- While linear diagnostic probes can isolate pair representations post-hoc (as seen in Wave 209-216), the backbone's intrinsic contextual association formation does not achieve operational separation ($\ge 0.50$) without specialized associative memory circuitry.
- Consequently, Stage B does not receive an adequate bound state to route the target value representation, causing Stage C token accuracy to remain 0.0000.
