# Step 75: Evidence Verification Specification

## Objective
Establish an authoritative verification layer that acts as a rigorous filter between raw external knowledge and the cognitive neural context.

## State Abstraction (`KnowledgeAcquisitionStatus`)
The verification layer classifies evidence into the following explicit states:
- `VERIFIED`: High reliability source, properly corroborated, fresh.
- `SUPPORTED`: Valid source, but lacks strong cross-corroboration.
- `CONTESTED`: Conflicting claims found among reliable sources.
- `STALE`: Information exceeds the required freshness bounds.
- `INSUFFICIENT`: Not enough quality evidence to make a determination.
- `UNKNOWN`: Unable to process or verify the source.
- `ABSTAIN`: Explicit decision to halt reasoning due to lack of safe/verified data.

## Verification Pipeline
1. **Freshness Check**: Discards or downgrades evidence older than the request's `freshness_requirement_seconds`.
2. **Reliability Check**: Rejects evidence from sources with a `reliability_score` below the safety threshold.
3. **Cross-Check (Corroboration/Contradiction)**: Evaluates multiple evidence items against one another. If items conflict, their status degrades to `CONTESTED`. If they agree, they elevate to `VERIFIED`.

## Security & Principles
- The verifier operates *outside* the neural core, ensuring deterministic rule application.
- It is designed to say "I don't know" or "Evidence is insufficient" rather than fabricating a likely-sounding synthesis.
