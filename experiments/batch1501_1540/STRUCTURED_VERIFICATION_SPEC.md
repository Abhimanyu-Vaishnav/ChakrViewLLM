# Structured Verification Specification

## Batches 1501–1540: Entity-Attribute Claim Verification Protocol

### 1. Specification Overview

The `StructuredVerifier` replaces brittle keyword and regex checks with a typed entity-attribute comparison engine. It extracts explicit assertions from natural language queries or structured propositions and evaluates them against reference documents indexed in ChakrView.

---

### 2. Structured Claim Representation

Every claim is represented as a typed dataclass:

```python
@dataclass
class StructuredClaim:
    claim_id: str
    subject_entity: str           # e.g., "BM25 index", "Starter Tier", "Raft consensus"
    attribute_name: str           # e.g., "k1 parameter", "monthly price", "heartbeat interval"
    claimed_value: str            # e.g., "2.5", "$150", "50 milliseconds"
    expected_unit: Optional[str]  # e.g., "ms", "$", "bytes", "nodes"
    comparison_mode: str          # "exact", "numeric_tolerance", "normalized_string"
```

---

### 3. Verification Outcomes

The verifier outputs exactly one of five mutually exclusive states:

| Verification Verdict | Semantic Meaning | Required Preconditions |
|---|---|---|
| **`SUPPORTED`** | The claimed attribute value matches the evidence-backed value. | Evidence contains the entity and attribute, and the values are equivalent under normalization. |
| **`CONTRADICTED`** | The claimed attribute value conflicts with the evidence-backed value. | Evidence contains the entity and attribute with a conflicting, incompatible value (e.g. claimed `$250`, actual `$150`). |
| **`INSUFFICIENT_EVIDENCE`** | The indexed corpus does not contain the entity or attribute. | Neither the entity nor the attribute is documented in the index. **Absence of evidence is never treated as contradiction.** |
| **`CONFLICTING_EVIDENCE`** | Multiple indexed sources disagree on the attribute value. | Multiple distinct chunks report different values for the same attribute. Surfaces both sources. |
| **`INVALID_CLAIM`** | The claim itself is structurally malformed or nonsensical. | Missing entity, missing attribute, or incompatible unit comparison. |

---

### 4. Normalization and Comparison Rules

1. **Numeric Normalization:**
   - Strips unit prefixes and suffixes (e.g. `"50 ms"` vs `"50 milliseconds"` $\to$ `50` with unit `ms`).
   - Currency: `"$150"` vs `"150 dollars"` vs `"150 USD"` $\to$ `150`.
   - Data sizes: `"10 MB"` vs `"10485760 bytes"` $\to$ converted to canonical byte counts for comparison.
2. **Provenance Preservation:**
   - Every `SUPPORTED` and `CONTRADICTED` verdict must link to:
     - `source_doc_id`: Filename of the supporting or contradicting document.
     - `chunk_index`: Specific chunk index.
     - `exact_text`: Verbatim sentence confirming or refuting the claim.
