# ChakrVerify Local Alpha: Developer & User Manual

**Product:** ChakrVerify Local  
**Version:** 0.1.0-alpha  
**Architecture:** Indigenous Local Verification Engine (CPU-First, Air-Gapped)  
**Location:** `chakrverify/`

---

## 1. Overview & Core Guarantees

ChakrVerify Local is an indigenous, self-contained document claim verification tool built inside the ChakrView neural and cognitive runtime.

### Key Capabilities:
- **Local & Air-Gapped:** Zero cloud API dependencies. Runs on local CPU without sending private documents across the internet.
- **Traceable Exact Offsets:** Every verdict provides character start/end offsets pointing directly to source sentences in the original document.
- **Four Distinct Verdicts:**
  - `SUPPORTED`: Claim matches verified document assertions.
  - `CONTRADICTED`: Claim contradicts verified document assertions (e.g. negated assertions, failed specifications).
  - `INSUFFICIENT_EVIDENCE`: Documents do not contain sufficient evidence regarding the entities in the claim (honest abstention).
  - `PROCESSING_ERROR`: File missing or unreadable format.
- **Multi-Document Conflict Detection:** If Document A supports a claim but Document B contradicts it, ChakrVerify detects the contradiction and cites both sources.

---

## 2. CLI Usage

### Verify a Single Claim
```bash
python -m chakrverify.cli --docs path/to/spec.txt --claim "Operating clock frequency is strictly verified at 450 MHz"
```

### Verify Multiple Documents Against a Batch of Claims
```bash
### Multi-Hop Relational Chain Verification
```bash
python -m chakrverify.cli --docs doc1.txt doc2.txt --multihop \
  --start-entity "ArcticPro" \
  --relation "fabricated by" \
  --target-claim "operates in Dresden Germany" \
  --bridges "Semiconductor Foundry Seven" \
  --output multihop_report.json
```

---

## 3. Python API Integration

### Single-Hop Verification
```python
from chakrverify.core import ClaimVerificationEngine, VerificationVerdict

engine = ClaimVerificationEngine()
engine.add_document("specifications.txt")

result = engine.verify_claim("Power consumption must not exceed 12 Watts")
print(result.verdict)  # VerificationVerdict.SUPPORTED
print(result.citations[0].exact_text)
print(result.citations[0].start_char, result.citations[0].end_char)
```

### Multi-Hop Chain Verification
```python
from chakrverify.core import MultiHopVerificationEngine, VerificationVerdict

engine = MultiHopVerificationEngine()
engine.add_document("chip_spec.txt")
engine.add_document("foundry_audit.txt")

result = engine.verify_multi_hop_claim(
    start_entity="ArcticPro",
    intermediate_relation="fabricated by",
    target_claim="operates in Dresden Germany",
    candidate_bridges=["Semiconductor Foundry Seven"]
)
print(result.verdict)  # VerificationVerdict.SUPPORTED
for hop in result.hops:
    print(f"Hop {hop.hop_index}: {hop.target_entity} ({hop.source_doc_id}) -> {hop.passage}")
```

---

## 4. Limitations & Truthful Disclaimers

1. **Alpha Status:** This is a testable prototype intended for enterprise verification workflows and legal/compliance diligence.
2. **Multi-Hop Chains:** Currently evaluates explicit 2-hop entity/relation chains. N-hop graph cycles and ambiguous coreferences require candidate bridge constraints or pointer scoring.
3. **Not Hallucination-Proof:** While grounded in source text, complex semantic transformations or implied logic may require additional neural pointer supervision.
4. **Supported Formats:** UTF-8 text files (`.txt`, `.md`). PDF ingestion requires pre-extraction to plain text.
