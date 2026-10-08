# Wave 532: Self-Generated Curriculum Report

## 1. Executive Summary
Wave 532 implemented `SelfGeneratedCurriculumEngine` to identify emerging capability weaknesses and generate targeted practice tasks.

## 2. Epistemic Separation Guardrail
A core anti-shortcut rule of ChakrView is that a model cannot generate its own test questions, grade them itself, and claim verified improvement.
To enforce this:
1. Generated tasks are explicitly flagged with `role: "practice"`.
2. Practice tasks are routed exclusively into the isolated candidate improvement loop.
3. Verification and promotion decisions are strictly reserved for independently authored, held-out evaluation datasets.

## 3. Empirical Verification
- **Failure Clustering**: Aggregated runtime failures across execution steps, identifying a recurring `hop_cutoff` bottleneck on 4-hop chain tasks.
- **Priority Attribution**: Assigned `HIGH` priority to gaps exhibiting $\ge 3$ failures.
- **Practice Curriculum**: Successfully synthesized 4 synthetic practice tasks matching the structural relation topology of the gap.

**Wave 532 Status: PASS**
