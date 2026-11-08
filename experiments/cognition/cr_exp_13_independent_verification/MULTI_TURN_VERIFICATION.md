# MULTI-TURN WORKING MEMORY & ENTITY BINDING VERIFICATION REPORT
**Investigation:** CR-EXP-13 Section 5 & 6  
**Auditor:** Autonomous Cognitive Benchmark Engineer  
**Date:** 2026-11-08T10:06:30+05:30  

---

## 1. THE EVALUATOR LOOSE-MATCHING ILLUSION IN WORKING MEMORY

In CR-EXP-12, Candidate A was reported with **43.00% (129/300)** on multi-turn working memory, and Candidate B with **27.33% (82/300)**.

Our audit of the evaluation script revealed that the benchmark evaluator scored a "Hit" if:
$$\text{expected\_target} \in \text{generated.lower()}$$
regardless of whether the model identified the correct subject!

### Inspection of the Generated Text:
- **Prompt**:
  `"User: Please track this update: Aalok is staying in Shimla... Update: Aalok has now moved over to Amritsar... Where is Aalok now?"`
  - Expected target: `Amritsar`. Subject: `Aalok`.
  - **Candidate A Generated**:
    `'Ananya is currently in Amritsar.. Patna. a is curr'`
  - **Candidate B Generated**:
    `'Shreya is currently in Surat. a.'`

Because `'amritsar'` was found in Candidate A's output, the evaluator marked this as a **HIT (43.0%)**, even though Candidate A attributed the location to **Ananya** instead of **Aalok**!

---

## 2. STRICT SUBJECT + VALUE RE-SCORING ON 1,500 BENCHMARK ($N=300$)

| Evaluation Standard | Candidate A | Candidate B | Parent (CR-11 Cand A) |
|---|---|---|---|
| **Loose Matching (Target City in Output)** | 43.00% (129/300) | 27.33% (82/300) | 10.00% (30/300) |
| **Strict Matching (Subject AND Target City in Output)** | **0.00% (0/300)** | **0.00% (0/300)** | **0.00% (0/300)** |

> [!CAUTION]
> **Definitive Scientific Finding**:
> When evaluated under the strict standard mandated by Section 5 ("A response is correct only if both the subject and its associated value are correct"), **both models score 0.00% (0/300)** on multi-turn memory on unseen entities.
>
> What the models learned was **recency value routing** (locating the most recently mentioned city in dialogue and emitting it), while completely decoupling the city from the subject token. When asked about unfamiliar entities, the model substitutes familiar training names (`Ananya`, `Shreya`, `Kabir`, `Manish`) into the response.

---

## 3. INDEPENDENT 100-PROBE WORKING MEMORY TEST SUITE RESULTS
Testing on 100 held-out multi-turn and interleaved dialogues with fresh names (`Girish`, `Hema`, `Indu`):
- **Correct Subject and Correct Value**: **0/100 (0.0%)** across all models.
- **Subject Substitution (Right City, Wrong Name)**: 0/100 (in new unseen cities, city generation also reverted to training cities `Surat`, `Gwalior`, `Indore`).
- **Wrong Subject and Wrong Value**: **100/100 (100.0%)**.

### Conclusion
Multi-turn working memory has **NOT** been solved. While the network learned the syntactic sentence frame `"[Name] is currently in [City]"`, it has not acquired dynamic variable binding for open-vocabulary names.
