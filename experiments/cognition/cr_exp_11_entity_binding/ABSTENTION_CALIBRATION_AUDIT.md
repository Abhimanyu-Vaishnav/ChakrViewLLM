# MANDATORY ABSTENTION CALIBRATION & EPISTEMIC DISCRIMINATION AUDIT
**Investigation:** CR-EXP-11 Phase 2 Mandated Audit  
**Auditor:** Autonomous Scientific Auditor & Neural Research Scientist  
**Date:** 2026-11-07T22:03:00+05:30  
**Test Suite:** 200 Fresh Disjoint Calibration Probes (100 Answerable vs. 100 Truly Unanswerable)

---

## 1. EXECUTIVE SUMMARY & VERDICT

> [!CAUTION]
> **CR-10 Arm B Pathological Abstention Confirmed**:
> The reported "100% missing-information abstention" of CR-10 Arm B is **NOT a cognitive capability**. It is a complete collapse into degenerate universal abstention on 1-entity contexts ($100\%$ false abstention on answerable queries, $0\%$ answer coverage).
>
> CR-10 Arm B simply predicts the phrase *"The context does not state..."* whenever given a single-entity context, whether the question is answerable or unanswerable.
>
> | Metric | Target | CR-09 Cand 3 | CR-10 Arm A | CR-10 Arm B |
> |---|---|---|---|---|
> | Answerable Accuracy | $\ge 75\%$ | 10.00% | 12.00% | **0.00%** |
> | Answer Coverage | $\ge 70\%$ | 81.00% | 29.00% | **0.00%** |
> | Selective Accuracy | $\ge 75\%$ | 12.35% | 41.38% | **0.00%** |
> | Correct Abstention (Recall) | $\ge 80\%$ | 11.00% | 74.00% | **99.00%** |
> | Abstention Precision | $\ge 80\%$ | 36.67% | 51.03% | **49.75%** |
> | False Abstention Rate | $\le 10\%$ | 19.00% | 71.00% | **100.00%** |
> | Hallucination on Unanswerable | $\le 10\%$ | 89.00% | 26.00% | **1.00%** |
> | Overall Epistemic Discrimination | $\ge 80\%$ | 10.50% | 43.00% | **49.50%** |

---

## 2. EMPIRICAL CONFUSION MATRICES ($N=200$)

### A. CR-09 Candidate 3
```
                  Model Answered     Model Abstained
Answerable (100)      10 (Correct)         19 (False Abstain)  [+71 Wrong Answers]
Unanswerable (100)    89 (Hallucinated)    11 (Correct Abstain)
```
- **Diagnosis:** Chronic overconfidence. Suffers from severe hallucination on missing information ($89\%$) and low answerable accuracy ($10\%$).

### B. CR-10 Arm A
```
                  Model Answered     Model Abstained
Answerable (100)      12 (Correct)         71 (False Abstain)  [+17 Wrong Answers]
Unanswerable (100)    26 (Hallucinated)    74 (Correct Abstain)
```
- **Diagnosis:** Moderate abstention regularization, but heavily skewed toward false abstentions ($71\%$).

### C. CR-10 Arm B
```
                  Model Answered     Model Abstained
Answerable (100)       0 (Correct)        100 (False Abstain)  [+0 Wrong Answers]
Unanswerable (100)     1 (Hallucinated)    99 (Correct Abstain)
```
- **Diagnosis:** Complete degenerate collapse. The model abstains on 199 out of 200 questions. Abstention precision is $49.75\%$ (pure chance on a 50/50 prior).

---

## 3. ROOT CAUSE IN CR-10 DATASET DESIGN

In CR-EXP-10's training file `train_arm_b.json`:
1. There were **zero** single-hop factual retrieval examples.
2. Every prompt containing a single entity and single attribute had an abstention target string (`"The context does not state..."`).
3. Multi-entity prompts were only trained with 3-entity distractor questions.
4. Hence, the loss gradient pushed the network to associate the token pattern `Context: [Name] [verb] [value]. Question: [Query]?` directly to the abstention response prefix without attending to whether the queried entity or attribute matched the context.

---

## 4. ARCHITECTURAL & TRAINING IMPLICATIONS FOR CR-EXP-11
- Epistemic abstention cannot be trained with unbalanced negative data.
- The training corpus for CR-EXP-11 **must pair every unanswerable query with an answerable query sharing identical structural syntax**.
- Evaluation must monitor **both** Answer Coverage ($\ge 70\%$) and Abstention Recall ($\ge 80\%$) simultaneously.

