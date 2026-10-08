# CHAKRVIEW EVIDENCE-TO-ANSWER PIPELINE SPECIFICATION
## Boundary Between External Evidence, Validation, Cognitive State, Reasoning, Claim Formation, and Answer Attribution
**Document Version**: 1.0  
**Status**: Standard Specification  
**Canonical Baseline Target**: 3,443,136 Parameters | SHA-256 `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`

---

## 1. End-to-End Grounded Pipeline

Evidence must affect cognitive state **before** final answer generation. The system avoids moving evidence verification into a purely cosmetic response layer.

```
[1. EXTERNAL EVIDENCE / TOOL OBSERVATIONS]
  - Raw payloads from Web, Documents, APIs, Sandboxes
  - Cryptographic content hashing (SHA-256)
  - Provenance: URI, timestamps, publisher, source type
                   │
                   ▼
[2. COGNITIVE VALIDATION & ATTRIBUTION GATE]
  - Exact Entity Check: Does evidence match queried entity?
  - Predicate Match: Does evidence address the target relation?
  - Multi-Source Corroboration & Anti-Syndication Clustered Proofs
  - Temporal Freshness: Validates [valid_from, valid_until]
                   │
                   ▼
[3. COGNITIVE STATE UPDATE & CONFLICT EVALUATION]
  - If Contradicted: Preserves UNRESOLVED_CONTRADICTION with dual lineage
  - If Insufficient / Missing: Enforces honest ABSTENTION ("I do not have enough evidence")
  - If Verified: Integrates candidate into active reasoning context (Delta W = 0)
                   │
                   ▼
[4. NEURAL & COMPOSITIONAL REASONING]
  - Compositional Binding (I4) & Multi-Hop Recurrence (I5)
  - Evaluates sound deductive steps across intermediate evidence links
                   │
                   ▼
[5. CLAIM FORMATION & BOUNDARY ENFORCEMENT]
  - Subclaim generation with strict overclaim guard
  - Prevents broad inductive leaps from narrow metrics
  - Categorizes claims: DIRECTLY_SUPPORTED vs INFERRED_FROM_SUPPORTED_FACTS
                   │
                   ▼
[6. FINAL ANSWER GENERATION & ATTRIBUTION GRAPH]
  - Constructs verified claim-evidence attribution graph
  - Traces 100% of substantive assertions to supporting evidence IDs
  - Inferences explicitly tagged as inferences; zero hallucinated facts
```

---

## 2. Immutable Boundary Invariants
1. **Evidence Causality**: Intervening on evidence values ($X=10 \rightarrow X=20$) causally updates the cognitive conclusion.
2. **Entity Specificity**: Real evidence about Entity A can NEVER be cited to support a conclusion about Entity B.
3. **No Silent Upgrading**: Partial evidence supports partial subclaims only; broad inductive leaps are blocked.
4. **Zero Weight Mutation**: External evidence retrieval and reasoning operates with $\Delta W = 0$.
