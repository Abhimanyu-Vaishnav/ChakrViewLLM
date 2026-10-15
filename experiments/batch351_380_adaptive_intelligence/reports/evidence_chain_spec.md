# Evidence-Chain Specification: Multi-Hop Reasoning & Provenance Grounding
**Location:** `experiments/batch351_380_adaptive_intelligence/reports/evidence_chain_spec.md`  
**Milestone:** Batches 351–380  
**Scope:** ChakrVerify Local Verification & Neural Pointer Evidence Resolution  

---

## 1. Multi-Hop Chain Definition

A multi-hop verification chain represents an explicit, source-grounded sequence of relational claims:
$$\text{Source Document A} \xrightarrow{\text{Relation 1}} \text{Bridge Entity/Claim } E \xrightarrow{\text{Relation 2}} \text{Source Document B} \xrightarrow{\text{Final Relation}} \text{Target Claim}$$

### Core Invariants:
1. **No Broken Links:** A chain cannot be marked `SUPPORTED` if any intermediate bridge entity $E$ is unsupported or missing.
2. **Honest Abstention:** If Hop 1 succeeds but Hop 2 fails, the verdict must be `INSUFFICIENT_EVIDENCE` (or `CONTRADICTED` if a contradiction is encountered), explicitly stating which hop failed.
3. **Exact Span Citations:** Each hop citation must retain:
   - `doc_id`: Source file name.
   - `start_char` and `end_char`: 0-indexed byte/character offsets verified against the raw text.
   - `exact_text`: Exact verbatim passage from the document.
   - `hop_index`: Integer index (0, 1, 2, ...).
   - `bridge_entity`: Entity connecting this hop to the subsequent claim.
4. **Adversarial Distractor Rejection:** If Document A contains keywords from the target claim but the semantic bridge to Document B is nonexistent or negated, the engine must not falsely produce `SUPPORTED`.

---

## 2. Multi-Hop Data Contracts

```python
@dataclass
class EvidenceHop:
    hop_index: int
    source_doc_id: str
    target_entity: str
    relation: str
    start_char: int
    end_char: int
    passage: str
    confidence: float
    is_supported: bool

@dataclass
class MultiHopVerificationResult:
    claim: str
    verdict: VerificationVerdict  # SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, PROCESSING_ERROR
    hops: List[EvidenceHop]
    confidence: float
    explanation: str
    has_conflicts: bool
    broken_hop_index: Optional[int]
    latency_ms: float
```

---

## 3. Pointer & TokenTrie Integration Contract

1. **Neural Cross-Attention Pointer:**
   - Used for extracting candidate bridge spans from retrieved document chunks when candidate entities are ambiguous.
   - Must output an explicit attention distribution and copy gate probability ($p_{\text{copy}}$).
   - If $p_{\text{copy}} < 0.45$, the bridge is marked low-confidence / uncertain.
2. **TokenTrie Constraints:**
   - When candidate bridge entities are known or constrained by the document index, TokenTrie forces autoregressive or pointer decoding to only emit valid entity sequences existing in the corpus.
   - Prevents generation of fictitious intermediate entities.

---

## 4. Verdict & Abstention Rules

| Scenario | Hop 1 Status | Hop 2 Status | Outcome Verdict | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| Valid 2-Hop Chain | Supported | Supported | `SUPPORTED` | Full chain verified across Doc A and Doc B. |
| Missing Bridge Entity | Supported | Absent | `INSUFFICIENT_EVIDENCE` | Broken chain at Hop 2: Intermediate entity has no link to target. |
| Contradictory Bridge | Supported | Negated | `CONTRADICTED` | Hop 2 directly contradicts the expected relational link. |
| Negated Hop 1 | Negated | Any | `CONTRADICTED` | Direct contradiction encountered at initial source document. |
| Absent Hop 1 | Absent | Any | `INSUFFICIENT_EVIDENCE` | No initial grounding found in indexed documents. |
