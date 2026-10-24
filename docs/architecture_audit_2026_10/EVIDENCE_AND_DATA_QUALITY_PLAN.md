# Evidence Grounding and Data Quality Plan
**Document Date**: October 21, 2026  
**Auditor**: Antigravity AI Engine  
**Project**: ChakrView Indigenous Neural-Cognitive Architecture

---

## 1. Principles of Truthfulness and Evidence Ingestion

ChakrView rejects the notion that next-token likelihood equals factual truth. In sovereign AI systems, factual claims must either be:
1. **Directly grounded** in an authenticated, cited context source span, OR
2. **Explicitly labeled** with calibrated uncertainty (`UNKNOWN`, `UNVERIFIED`, or `INSUFFICIENT_EVIDENCE`).

Unfounded hallucination or unverified assertion presented as factual truth is classified as a critical system failure.

---

## 2. Ingestion & Pre-Training Data Quality Governance

To prevent data contamination, synthetic repetition loops, and low-quality tokens, the training data pipeline (`chakrview/corpus/`) adheres to strict quarantine and acceptance filters:

```
[RAW INPUT CORPUS]
         |
         v
+------------------------+
| Exact & MinHash Dedup  | ---> [Duplicate Quarantine] (Logged with SHA-256)
+------------------------+
         |
         v
+------------------------+
| UTF-8 & Byte Sanitize  | ---> [Malformed Byte Drop]
+------------------------+
         |
         v
+------------------------+
| Perplexity & Artifact  | ---> [High-Entropy / Synthetic Loop Quarantine]
| Gating Filter          |
+------------------------+
         |
         v
[APPROVED TRAINING SHARDS (uint16 binary, Seed 42, Validated Manifest)]
```

### Ingestion Acceptance Rules:
- **Exact Hash Quarantine**: Identical text records across training and evaluation splits are rejected with zero tolerance (anti-leakage guarantee).
- **Synthetic Loop Rejection**: Documents containing $>3$ consecutive repetitive n-grams ($N \ge 4$) are automatically pruned.
- **Bi-Directional Boundary Verification**: Source span boundaries in SFT tasks must strictly align with BPE token boundaries (verified in Exp 10 and 11 audits).

---

## 3. Runtime Evidence Attribution and Answer Verification

When handling user questions and reasoning tasks, the Epistemic Engine executes a four-stage pipeline:

### Stage 1: Retrieval & Source Identity
- Document chunks are indexed with immutable metadata:
  - `document_path`: Local workspace or reference path.
  - `sha256`: Checksum of the source document at time of indexing.
  - `timestamp`: Date and version of ingestion.
  - `chunk_id`: Exact byte and line range.

### Stage 2: Independent Corroboration & Contradiction Detection
- When retrieving multiple evidence passages:
  - If passage $A$ states `Status: Active` and passage $B$ states `Status: Degraded`, the system detects a **contradiction**.
  - Rather than guessing or silently picking one, the cognitive engine halts generation and generates an epistemic clarification request or reports the contradiction with both source citations.

### Stage 3: Span-Level Citation & Answer Extraction
- Answers are synthesized either via:
  - **Span Pointer Mechanism**: Directly copying the exact character slice from the context, eliminating lexical hallucination.
  - **Constrained Schema Emission**: Emitting structured JSON verified by the finite-state controller.

### Stage 4: Post-Generation Calibrated Uncertainty
- Every emitted claim is classified into one of four epistemic states:
  - `KNOWN`: Grounded directly in an authentic source span with confidence $> 0.95$.
  - `CORROBORATED`: Supported by two or more independent passages.
  - `UNCERTAIN`: Single ambiguous mention or conflicting dates.
  - `UNKNOWN / ABSTAIN`: Value not found in source text; output explicitly set to `"none"`.

---

## 4. Benchmark & Metrics for Grounding Accuracy

The Epistemic Engine is evaluated across standard grounding test splits:
- **Citation Precision**: $\frac{\text{Correctly Cited Spans}}{\text{Total Emitted Citations}}$
- **Hallucination Rate**: Percentage of emitted entity tokens absent from the context text (target: $0.0\%$).
- **Abstention Accuracy**: Correctly emitting `"none"` when the requested field is absent from the input context (target: $100.0\%$).
- **Contradiction Detection F1**: Ability to detect conflicting claims in multi-document prompts.
