# Evidence-Grounded Answer Verification Contract: Batch 196
## Traceable Claim-Level Provenance & Safe Neural Pointer Attribution

### 1. Verification Guarantees
1. **Explicit Grounding:** An answer is marked `verified = True` if and only if the candidate value is found verbatim or via strictly defined unit normalization inside the cited supporting span.
2. **Entity-Relation Binding Check:** The cited evidence span must contain the requested entity and relation markers. If a pointer outputs a plausible number or name from an unrelated sentence in context, the verifier rejects the answer.
3. **Contradiction Rejection:** If multiple documents assert disparate values for the same requested relation, the verifier flags `has_conflicts = True` and forces explicit abstention (`"Conflicting facts detected across evidence: [...]"`).
4. **Honest Component Attribution:**
   - `neural_pointer`: Output selected by `ChakrPointerHead` ($g > 0.5$) from context and verified against evidence.
   - `neural_core`: Output generated from base model vocabulary.
   - `deterministic_extractor`: Extracted by rule-based slot extractor.
   - `fallback_policy`: Emitted when evidence is missing, conflicting, or unverified.
