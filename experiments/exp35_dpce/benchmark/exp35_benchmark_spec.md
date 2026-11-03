# CHAKRVIEW — EXP35 BENCHMARK SPECIFICATION
## Independent 200-Task Grounded Extraction Benchmark

**Benchmark ID:** `EXP35_INDEPENDENT_BENCHMARK_200`  
**File:** `exp35_benchmark_200.json` (SHA-256 `29c2739c...`)  
**Manifest:** `exp35_benchmark_manifest.json`  
**Date:** 2026-10-30  

---

## 1. TASK TAXONOMY & DISTRIBUTION

The benchmark contains exactly 200 tasks distributed across 9 test families:
1. `single_entity_extraction`: 30 tasks
2. `multi_attribute_extraction`: 30 tasks
3. `multi_entity_disambiguation`: 30 tasks
4. `unseen_entity_combinations`: 30 tasks
5. `slot_filling`: 20 tasks
6. `context_dependent_qa`: 20 tasks
7. `multi_turn_retention`: 20 tasks
8. `changed_fact_tracking`: 10 tasks
9. `distractor_resistance`: 10 tasks

---

## 2. EVALUATION & SCORING PROTOCOL

- **Decoding:** Greedy decoding ($T=0.0$), maximum output length 32 tokens.
- **Scoring Heuristic:** Strict word-boundary match:
  $$\text{Regex: } r"\b" + \text{re.escape}(\text{target}) + r"\b"$$
  Case-insensitive matching is permitted only where casing does not alter entity semantics (e.g. capitalized names vs lowercase words).
- **Zero False-Positive Guarantee:**
  Unlike legacy substring scoring, matching `"no"` inside `"non-positive"` evaluates to **FAIL**.

