# Batch 155 — Robust Evidence Extraction & Abstention Evaluation
## Deterministic Extraction, Conflict Detection & Safe Abstention

### 1. Contract & Supported Extraction Types
The `EvidenceExtractor` (`chakrview/runtime/evidence_extractor.py`) supports structured extraction without hallucination:
1. **Explicit Key-Value Records**: Pattern `<key> is <val>` / `<key>: <val>` / `<key> = <val>`.
2. **Multilingual Key-Value Records**: Hindi/Devanagari patterns (`<key>: <val>`, `<key> <val> है`).
3. **Domain Entity Attributes**: Specific scientific and geographic queries (atomic numbers, locations, crater names).
4. **Conflicting Evidence Detection**: Detection of multiple distinct values for the same key within context.
5. **Safe Abstention**: Clean detection of missing attributes returning `"Not mentioned in context."` with `is_abstention=True`.

### 2. Empirical Verification Results

| Test Scenario | Query | Supplied Context | Extracted Value | Supporting Span | Status |
|---|---|---|---|---|---|
| Direct Key | "What is material?" | `material is diamond.` | `diamond` | `material is diamond` | **PASS (Exact Match)** |
| Distractor Resistance | "What is language?" | `vehicle is bicycle. language is Sanskrit.` | `Sanskrit` | `language is Sanskrit` | **PASS (Exact Match)** |
| Missing Key Abstention | "What is president?" | `mineral is quartz. continent is Asia.` | `Not mentioned in context.` | `""` | **PASS (Abstention)** |
| Multilingual (Hindi) | "धातु क्या है?" | `धातु: तांबा। तांबा एक विद्युत सुचालक धातु है।` | `तांबा` | `धातु: तांबा` | **PASS (Exact Match)** |
| Conflicting Evidence | "What is color?" | `color is blue. color is red.` | Flagged Conflicts: `['blue', 'red']` | `color is blue` | **PASS (Conflict Detected)** |
| Context Truncation | "What is fruit?" | `fruit is apple.` $\times 50$ | `apple` | `fruit is apple` | **PASS (Exact Match)** |
| Negative Scientific QA | "At what temperature does gold boil..." | `Water boils at 100 degrees Celsius.` | `Not mentioned in context.` | `""` | **PASS (Abstention)** |

### 3. Error Rates & Metrics
- **Extraction Exact Match**: 100% (on all present attributes).
- **Evidence-Span Correctness**: 100% (every extracted answer is linked to its exact span).
- **Unsupported-Answer Rate**: 0.0% (zero hallucinated values).
- **False Abstention Rate**: 0.0% (no known attributes skipped).
- **Correct Abstention Rate**: 100% (clean rejection on missing keys).
- **Conflict Discovery Rate**: 100% (both values identified and surfaced).
