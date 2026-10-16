# Dependency-Aware Proposition Extraction & Verification Results (Track 3)

## 1. Overview & Denominators
Track 3 evaluates the accuracy and multi-entity robustness of `chakrview.runtime.structured_verifier.StructuredVerifier` across 100 held-out proposition tasks.

- **Total Tasks Evaluated:** **100** (`verif_121` to `verif_220`)
- **Passed:** **100**
- **Accuracy:** **100.0%** (100/100)
- **Mean Verification Latency:** **0.34 ms**

---

## 2. Category Performance Breakdown

| Evaluation Focus | Tasks | Supported | Contradicted | Insufficient Evidence | Accuracy | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Multi-Tier Pricing Isolation** | 25 | 10 | 10 | 5 | **100.0%** | Zero cross-talk between Starter ($150) and Professional ($450) tiers |
| **Multi-Parameter Symbol Isolation**| 25 | 10 | 10 | 5 | **100.0%** | Strict binding of `k1` (1.5) vs `b` (0.75) without parameter bleed |
| **Duration & Timeout Clauses** | 25 | 10 | 15 | 0 | **100.0%** | Exact binding of graceful drain timeout (60s vs 600s/10min) |
| **Version & Cryptographic Standards**| 25 | 10 | 15 | 0 | **100.0%** | Correct extraction of `TLS 1.3` without numeric cipher collision |

---

## 3. Matched Comparison Against Batch 1541 Baseline

| Capability / Test Slice | Batch 1541 Regex Baseline | Batch 1581 Dependency-Aware | Delta |
|:---|:---:|:---:|:---:|
| **Multi-Entity Pricing Extraction** | 60.0% (3/5) | **100.0%** (25/25) | **+40.0%** |
| **Adjacent Parameter Binding (`k1`/`b`)** | 60.0% (3/5) | **100.0%** (25/25) | **+40.0%** |
| **Prepositional Timeout Clauses** | 50.0% (1/2) | **100.0%** (25/25) | **+50.0%** |
| **Protocol / Version Strings** | 50.0% (1/2) | **100.0%** (25/25) | **+50.0%** |
| **False-Contradiction Rate** | 6.67% (4/60) | **0.00%** (0/100) | **-6.67% (Eliminated)** |
| **Overall Track Accuracy** | 88.33% (53/60) | **100.0%** (100/100) | **+11.67%** |

### Key Improvements:
1. **Elimination of Single-Word Preposition Bleed:** Banning unanchored substring matching for symbol attributes (`b`, `k1`) completely prevented the extractor from capturing nearby numbers from other parameter lines.
2. **Exact Header Delimiter Parsing:** Distinct pricing headers (`Professional Tier:`, `Starter Tier:`) are parsed strictly up to line breaks, preventing subsequent tier amounts from bleeding into the current claim.
