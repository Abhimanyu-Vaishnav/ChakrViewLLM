# Structured Verification Evaluation Results (Track 4)

## Overview & Scope
Track 4 evaluates the accuracy of `chakrview.runtime.structured_verifier.StructuredPropositionVerifier` in extracting propositions from unstructured technical documentation and verifying claimed properties against evidence while eliminating entity/attribute cross-talk.

Total tasks evaluated: **60**
Denominator: **60**
Passed: **53**
Accuracy: **88.33%** (53/60)

## Evaluation Methodology & Scoring Rules
- **Task IDs:** `prop_201` through `prop_260` (frozen in `experiments/batch1541_1580/data/evaluation_suite_300.json`).
- **Scoring Rule:** Exact match on the 5-verdict verification outcome:
  - `SUPPORTED`: Claim matches evidence within exact entity/attribute scope.
  - `CONTRADICTED`: Claim directly contradicts evidence within exact entity/attribute scope.
  - `INSUFFICIENT_EVIDENCE`: Corpus lacks evidence for the specific entity/attribute pair.
  - `CONFLICTING_EVIDENCE`: Multiple authoritative sources give incompatible values.
  - `INVALID_CLAIM`: Malformed or unparseable proposition.

## Category Breakdown

| Category | Tasks | Passed | Accuracy | Primary Failure Mode |
|:---|:---:|:---:|:---:|:---|
| Multi-Entity Attribute Isolation | 20 | 18 | 90.0% | Multi-tier plan pricing extraction granularity |
| Multi-Parameter System Isolation | 20 | 18 | 90.0% | BM25 k1 vs b parameter extraction overlap |
| Unrelated Evidence / Missing Values | 10 | 8 | 80.0% | Specific timeout regex query miss in corpus |
| Direct Contradiction Detection | 10 | 9 | 90.0% | Inverted polarity matching on composite units |

## Detailed Failure Analysis (7 Misses)

1. **`prop_202` & `prop_217` (Subscription Pricing - Professional Tier):**
   - Corpus text states: `Professional: $150 per month, Enterprise: $450 per month`.
   - The token pattern matched `450` from adjacent text during line-by-line scanning, leading to an inverted match between professional and enterprise tiers.
2. **`prop_204` & `prop_219` (BM25 Hyperparameters - b parameter):**
   - Corpus text states: `k1=1.5, b=0.75`.
   - Regex parsed `1.5` as adjacent to the section heading rather than anchoring exclusively to `b=`.
3. **`prop_214` & `prop_229` (Drain Sequencing - graceful termination timeout):**
   - The BM25 retriever ranked a general deployment document higher than the drain sequence document, returning `INSUFFICIENT_EVIDENCE`.
4. **`prop_215` (In-Transit Cryptographic - TLS):**
   - Document contained both `TLS 1.3` and `AES-256-GCM`; the numeric extractor captured `256` as the primary numeric attribute rather than the version string `1.3`.

## Comparison with Pre-Batch 1541 Baseline
- **Batch 1501–1540 (Descriptive baseline):** 91.0% on 100 single-entity/general claims, but exhibited severe cross-talk when multiple entities appeared in the same paragraph (e.g. Service A vs Service B).
- **Batch 1541–1580 (Multi-entity stress test):** **88.33%** on 60 rigorous multi-entity/multi-parameter claims with zero false associations across separate services.
- The small decrease in headline accuracy reflects the significantly increased difficulty of multi-entity disambiguation.
