# Batch 222 — Versioned Evaluation Contract & Test Manifest Specification

**Document Version**: 1.0.0-frozen  
**Date**: 2026-10-13  
**Frozen Test File**: `experiments/batch221_230_pointer_generalization/data/frozen_test_manifest_100.json`  
**Manifest SHA-256**: `f345670a7b49ed1fd17b9c700fbcf468e597ee3035e50278dc439dbfdc0fd7aa`  
**Fixed Denominator**: $N = 100$

---

## 1. Schema Contract
Each evaluation record adheres strictly to the following specification:

```json
{
  "example_id": "string (unique prefixed ID)",
  "record_id": "string (traceable back to corpus generator)",
  "domain": "string (mathematics | coding | science | general)",
  "language": "string (en | hi)",
  "entity": "string (subject entity)",
  "relation": "string (queried relation attribute)",
  "context": "string (raw context text supplied to model)",
  "query": "string (user question)",
  "gold_value": "string (canonical expected value span in context)",
  "normalized_gold": "string (lowercase, stripped trailing punctuation)",
  "distractor_entity": "string (competing entity name)",
  "distractor_value": "string (competing numeric or attribute value)",
  "is_held_out": true,
  "diagnostic_tags": ["string (slice categories)"]
}
```

---

## 2. Evaluation Rules & Answer Normalization
1. **Primary Score**: Exact Match (EM) over all 100 frozen test items ($N=100$).
   $$\text{EM} = \frac{\sum_{i=1}^{100} \mathbb{I}(\text{Norm}(\text{Answer}_i) = \text{Norm}(\text{Gold}_i))}{100}$$
2. **Normalization Function**:
   - `strip()` leading/trailing whitespace.
   - `lower()` casing for case-insensitive verification.
   - Strip terminal punctuation: `.,;:?!"'`.
3. **Candidate Space Constraint**:
   - For all TokenTrie candidate-constrained arms, candidate token sequences MUST be extracted purely from the provided context $C_i$ (words and n-grams $n \in \{1, 2, 3\}$).
   - No external oracle vocabulary, gold answer leakage, or dataset metadata may be injected into candidate generation.
4. **Diagnostic Slice Tags**:
   Instead of fragmented disjoint splits, each of the 100 test items carries non-exclusive diagnostic tags:
   - `held_out_entity`: 100/100 (all test entities are zero-leakage held-out).
   - `numeric_distractors`: items with explicitly designated competing numbers.
   - `multilingual`: items featuring Devanagari or Hindi text (language `hi`).
   - `has_digits`: items where the target value is numeric.
   - `multi_token`: items where the target value consists of $\ge 2$ words/tokens.
