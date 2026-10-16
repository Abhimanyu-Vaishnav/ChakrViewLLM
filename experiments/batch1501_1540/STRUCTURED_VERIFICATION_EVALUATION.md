# Structured Verification Evaluation Report

## Batches 1501–1540: Entity-Attribute Claim Verification Benchmark Across 100 Claims

### 1. Benchmark Setup & Metrics

Track 2 evaluated the `StructuredVerifier` across 100 structured claims (`verif_001` to `verif_100`) covering five distinct verification verdicts:
1. **`SUPPORTED` Claims (25 tasks):** True factual statements supported by the indexed documentation (e.g. BM25 $k_1=1.5$, Raft heartbeat $50\text{ ms}$, Starter Tier $\$150$).
2. **`CONTRADICTED` Claims (25 tasks):** False-premise statements contradicting indexed documentation (e.g. BM25 $k_1=2.5$, Starter Tier $\$250$, mesh port $80$).
3. **`INSUFFICIENT_EVIDENCE` Claims (25 tasks):** Technical entities absent from the local index (e.g. Quantum Cryptography, Mars Satellite, Kafka, Docker). Absence of evidence must be classified as insufficient evidence, not contradiction.
4. **`CONFLICTING_EVIDENCE` Claims (15 tasks):** Statements regarding entities where two indexed sources present differing specifications (e.g. ChakrMicro V0.6 384 units vs V0.7 512 units).
5. **`INVALID_CLAIM` Claims (10 tasks):** Malformed claims missing subject entities, attribute names, or values.

---

### 2. Empirical Benchmark Outcomes

| Verification Verdict Category | Evaluated Tasks | Correctly Classified | Classification Accuracy (%) | Primary System Behavior |
|---|---|---|---|---|
| **`SUPPORTED`** | 25 | 18 | **72.00%** (18/25) | Extracted and normalized true values; validated citations. |
| **`CONTRADICTED`** | 25 | 23 | **92.00%** (23/25) | Detected value contradictions; surfaced contradicting text. |
| **`INSUFFICIENT_EVIDENCE`** | 25 | 25 | **100.00%** (25/25) | Zero false-refutations; cleanly abstained. |
| **`CONFLICTING_EVIDENCE`** | 15 | 15 | **100.00%** (15/15) | Surfaced multi-document specification discrepancies. |
| **`INVALID_CLAIM`** | 10 | 10 | **100.00%** (10/10) | Trapped missing entity/attribute fields. |
| **Total Aggregate** | **100** | **91** | **91.00% (91/100)** | **Pass Gate Reached ($\ge 90\%$)** |

---

### 3. Confusion Matrix Across Verification Verdicts

```text
Actual \ Predicted       SUPPORTED   CONTRADICTED   INSUFFICIENT   CONFLICTING   INVALID   Total
SUPPORTED                   18             5              2             0           0        25
CONTRADICTED                 0            23              2             0           0        25
INSUFFICIENT_EVIDENCE        0             0             25             0           0        25
CONFLICTING_EVIDENCE         0             0              0            15           0        15
INVALID_CLAIM                0             0              0             0          10        10
Total Predicted             18            28             29            15          10       100
```

#### Key Diagnostic Findings:
1. **Zero False Refutations on Missing Evidence:**
   In 25 out of 25 unindexed entity claims (`verif_051` to `verif_075`), the verifier produced `INSUFFICIENT_EVIDENCE`. At no point did the system treat missing evidence as proof of contradiction.
2. **High Contradiction Sensitivity (92.0%):**
   23 of 25 contradicted claims were accurately detected and refuted with exact evidence citations, compared to only 50.0% in the previous regex-based baseline.
3. **Multi-Source Conflict Detection (100.0%):**
   When multiple documents contained differing specifications (e.g. ChakrMicro V0.6 vs V0.7 specifications), the verifier successfully surfaced both sources under `CONFLICTING_EVIDENCE` rather than arbitrarily declaring one true.
