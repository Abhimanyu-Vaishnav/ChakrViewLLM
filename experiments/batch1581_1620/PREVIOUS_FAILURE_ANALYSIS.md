# Batches 1541–1580 Evaluation Failure Analysis

## 1. Executive Summary
In Batches 1541–1580, a frozen 300-task evaluation benchmark was executed across 5 functional tracks.
- Total tasks: **300**
- Passed: **292** (97.33%)
- Failed: **8** (2.67%)

This document reviews every failure to determine root cause, classification (implementation defect, extraction ambiguity, retriever miss, or evaluator nuance), security/isolation impact, and remediation strategy for Batches 1581–1620.

---

## 2. Itemized Failure Inventory

| Task ID | Track & Category | Claim / Query | Expected Verdict | Actual Verdict | Root Cause Classification | Security Risk? |
|:---|:---|:---|:---:|:---:|:---|:---:|
| `prop_202` | Track 4: Proposition Verification | Subscription Pricing: Professional Tier = $450 | `SUPPORTED` | `CONTRADICTED` | **Implementation Defect (Proximity Overlap)** | No |
| `prop_204` | Track 4: Proposition Verification | BM25 Hyperparameters: b parameter = 0.75 | `SUPPORTED` | `CONTRADICTED` | **Implementation Defect (Proximity Overlap)** | No |
| `prop_214` | Track 4: Proposition Verification | Drain Sequencing: graceful termination timeout = 60s | `SUPPORTED` | `INSUFFICIENT_EVIDENCE` | **Retriever Ranking / Scope Defect** | No |
| `prop_215` | Track 4: Proposition Verification | In-Transit Cryptographic: TLS = TLS 1.3 | `SUPPORTED` | `CONTRADICTED` | **Implementation Defect (Numeric Unit Bias)** | No |
| `prop_217` | Track 4: Proposition Verification | Subscription Pricing: Professional Tier = $150 | `CONTRADICTED` | `SUPPORTED` | **Implementation Defect (Proximity Overlap)** | No |
| `prop_219` | Track 4: Proposition Verification | BM25 Hyperparameters: b parameter = 1.5 | `CONTRADICTED` | `SUPPORTED` | **Implementation Defect (Proximity Overlap)** | No |
| `prop_229` | Track 4: Proposition Verification | Drain Sequencing: graceful termination timeout = 600s | `CONTRADICTED` | `INSUFFICIENT_EVIDENCE` | **Retriever Ranking / Scope Defect** | No |
| `route_279` | Track 5: Routing & Recovery | "Check if mesh network data port is 80" | `STRUCTURED_VERIFICATION` | `KNOWLEDGE_RETRIEVAL` | **Evaluator / Intent Boundary Nuance** | No |

---

## 3. Detailed Root Cause Analysis

### A. Proposition Extraction Proximity Overlap (`prop_202`, `prop_204`, `prop_217`, `prop_219`)
- **Corpus Context:**
  In `billing_subscription_tiers.md`:
  `- Professional Tier: $150 per month, includes up to 4 worker instances.`
  `- Enterprise Tier: $450 per month, includes dedicated orchestrator.`
  In `storage_bm25_index.md`:
  `- Term saturation parameter k1: 1.5.`
  `- Document length normalization parameter b: 0.75.`
- **Defect Mechanism:**
  The Batch 1541 regex implementation matched entity keywords and attribute keywords within a coarse line or sentence boundary. When a single line or adjacent lines listed multiple tiers or parameters (`k1: 1.5` followed closely by `b: 0.75`), the regex pattern `.*?[:=-]\s*(\d+)` eagerly captured the first number encountered in the block or bound `b` to `1.5` because `k1` and `b` both appeared in the document chunk.
- **Remedy for Batches 1581–1620:**
  Implement **Dependency-Aware Proposition Verification** (`DependencyAwareStructuredVerifier`). The extractor must construct an explicit dependency graph where values are strictly bound to their grammatical head or tabular column/row header, preventing adjacent parameter values from bleeding into unrelated attributes.

### B. Unit & Value Type Bias (`prop_215`)
- **Corpus Context:**
  In `network_encryption_transit.md`:
  `- In-Transit Cryptographic standard: TLS 1.3 with AES-256-GCM cipher suites.`
- **Defect Mechanism:**
  The numeric extraction pass normalized `AES-256-GCM` into integer `256` rather than capturing `TLS 1.3` as the target value for attribute `TLS`. It then compared claimed value `TLS 1.3` against evidence value `256`, triggering a false contradiction.
- **Remedy for Batches 1581–1620:**
  Support typed attribute values (string version tokens, cryptographic names, and numeric quantities) rather than forcibly collapsing all attributes to float/int.

### C. Corpus Chunk Scope & Retrieval Missing Context (`prop_214`, `prop_229`)
- **Corpus Context:**
  The corpus document `runtime_lifecycle_management.md` describes node drain sequencing and sets timeout to 60s. However, the retrieval chunk fed to the verifier prioritized general deployment overview sections where the specific drain timeout attribute was absent.
- **Defect Mechanism:**
  The verifier returned `INSUFFICIENT_EVIDENCE`. Note that for `prop_229`, where the claimed value was `600s` (contradictory), returning `INSUFFICIENT_EVIDENCE` is logically correct when the document is missing, because **absence of evidence is never contradiction**. The failure was that the retriever failed to surface the drain sequencing document.
- **Remedy for Batches 1581–1620:**
  Improve chunk passage retrieval by boosting entity-attribute exact phrase matches when routing claims to the verifier.

### D. Routing Intent Boundary Nuance (`route_279`)
- **Query:** `"Check if mesh network data port is 80"`
- **Defect Mechanism:**
  The query starts with `"Check if"`, which resembles both factual Q&A and verification. The router checked for keywords `["verify", "is it true", "confirm", "claim", "contradicts"]`. `"Check if"` was not in the regex pattern for verification and defaulted to `KNOWLEDGE_RETRIEVAL`.
- **Remedy for Batches 1581–1620:**
  Include `"check if ... is ..."` in `CanonicalRoute.STRUCTURED_VERIFICATION` intent patterns.

---

## 4. Regression Test & Dataset Separation Policy
- The 8 failure cases above are incorporated into the development regression test suite (`tests/test_batch1581_1620_container_and_verifier.py`).
- For final evaluation of Batches 1581–1620, an entirely new, held-out 300-task evaluation suite (`evaluation_suite_300_v2.json`) will be constructed, frozen, and hashed prior to evaluation.
