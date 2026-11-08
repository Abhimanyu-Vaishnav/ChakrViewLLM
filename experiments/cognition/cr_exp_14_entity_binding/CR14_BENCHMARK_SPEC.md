# CR-EXP-14 SEALED BENCHMARK SPECIFICATION
**Benchmark Identifier:** `CR-EXP-14-SEALED-1200`  
**File Location:** `experiments/cognition/cr_exp_14_entity_binding/benchmark_data/benchmark_1200_sealed.json`  
**Manifest Location:** `experiments/cognition/cr_exp_14_entity_binding/benchmark_data/CR14_BENCHMARK_MANIFEST.json`  
**Cryptographic SHA-256:** `1958e94571f83c66c26f390634ffa563d0a2ff57e58943e835d7825245b964b9`  
**Total Probes:** Exactly 1,200 probes  
**Generation Seed:** 202614  

---

## 1. PROBE DISTRIBUTION TABLE

| Category # | Category Name | Probe Count | Core Competence Tested | Strict Scoring Requirement |
|---|---|---:|---|---|
| **Cat 1** | Single-Entity State Tracking | 200 | Temporal state tracking on single entity | Both Subject AND Updated City must be present; initial stale city absent |
| **Cat 2** | Two-Entity State Tracking | 200 | Interleaved 2-entity updates | Target Subject AND Target City present; peer entity and stale values absent |
| **Cat 3** | Three-Entity State Tracking | 200 | Complex multi-entity group tracking | Target Subject AND Target City present; all 2 distractors & stale absent |
| **Cat 4** | Repeated Updates & Recency | 200 | 3-step state updates (A $\to$ B $\to$ C) | Target Subject AND Destination C present; initial A and intermediate B absent |
| **Cat 5** | Interleaved Conversational Distractors | 150 | Distractor conversation resistance | Subject AND Updated City present across intervening conversational noise |
| **Cat 6** | Counterfactual Reassignment | 150 | Context-form state updates | Target City present; initial stale city absent |
| **Cat 7** | Missing Information Abstention | 100 | Unmentioned entity & attribute abstention | Correct epistemic abstention cue; non-abstention = hallucination |
| **Total** | **Sealed Benchmark Total** | **1,200** | **Comprehensive Dynamic Entity Binding Benchmark** | Strict Evaluator Standard |

---

## 2. STRICT EVALUATOR STANDARD
A response is scored as a **Strict Hit** if and only if:
1. `subject.lower() in generated.lower()` (Correct subject identity preserved)
2. `expected_target.lower() in generated.lower()` (Correct latest value identified)
3. `earlier_value.lower() not in generated.lower()` (No stale value contamination)
4. Distractor values (if present) are excluded from the output.

If the model predicts the correct city but associates it with a substituted name (e.g. `'Ananya is in Amritsar'` when the query was `'Aalok'`), it is scored as **STRICT ZERO (Subject Substitution Failure)**.
