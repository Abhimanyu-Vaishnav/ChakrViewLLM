# Dependency-Aware Structured Verification Specification

## 1. Motivation & Defect Prevention
In previous milestones (Batches 1501–1580), proposition extraction relied on proximity matching inside text windows or whole chunks. This caused three failure modes:
1. **Multi-Entity Cross-Talk:** When multiple entities appeared in the same document or sentence (e.g. `Professional: $150`, `Enterprise: $450`), adjacent numbers were mistakenly bound to the wrong tier.
2. **Multi-Parameter Proximity:** In parameter definitions (e.g. `k1=1.5, b=0.75`), the extractor captured the first numeric token rather than the exact value governed by the specific parameter symbol.
3. **Loss of Typographic and Grammatical Structure:** Tables, bulleted lists, and parenthesized attributes lost structural parent-child associations.

---

## 2. Dependency-Aware Extraction Pipeline

The enhanced verifier in `chakrview.runtime.structured_verifier.StructuredVerifier` implements a 6-stage verification pipeline:

```
[ Unstructured Evidence Document Chunks ]
                   |
                   v
 Stage 1: Structural Segment Decomposition (Tables, Headers, Lists, Sentences)
                   |
                   v
 Stage 2: Entity Scope Identification & Anchoring
                   |
                   v
 Stage 3: Dependency Graph & Relation Extraction
                   * Explicit Symbol Binding: "k1" -> "1.5", "b" -> "0.75"
                   * Delimiter Binding: "Professional Tier: $150" -> ("Professional Tier", "$150")
                   * Grammatical Head Binding: [Attribute] [Copula/Preposition] [Value]
                   |
                   v
 Stage 4: Typed Value Normalization (Numeric, Version String, Storage Bytes, Duration)
                   |
                   v
 Stage 5: Rigorous Proposition Grounding & Verification
                   * Exact Match -> SUPPORTED
                   * Distinct Contradictory Evidence -> CONTRADICTED
                   * Conflicting Multilateral Evidence -> CONFLICTING_EVIDENCE
                   * Missing Entity/Attribute Relationship -> INSUFFICIENT_EVIDENCE
                   * Ambiguous Unlinked Proposition -> INSUFFICIENT_EVIDENCE
                   |
                   v
 Stage 6: Audit Record Generation (with Source Doc, Evidence Line, Confidence)
```

---

## 3. Strict Association Rules

1. **Rule 1 (Grammatical & Punctuation Scope):**
   A value is associated with an attribute ONLY IF they are linked by:
   - A colon delimiter (`:`) or equals sign (`=`) within the same line/bullet item.
   - An explicit copula (`is`, `was`, `equals`, `set to`) in the same grammatical clause.
   - A tabular row-column intersection.
2. **Rule 2 (No Cross-Entity Value Bleed):**
   If a sentence or bullet contains Entity $E_1$ and Entity $E_2$, values appearing after $E_2$'s definition header cannot be bound to $E_1$.
3. **Rule 3 (Safe Abstention Over Guessing):**
   If an entity is present but its claimed attribute cannot be unambiguously isolated from neighboring parameters, the verifier returns `INSUFFICIENT_EVIDENCE` with an explicit reason, rather than guessing a nearby number.
