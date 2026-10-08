# Wave 531: Conflict Resolution and Knowledge Revalidation Report

## 1. Executive Summary
Wave 531 implemented the `ConflictResolver` engine to govern contradictions arising between competing experiences or conflicting relational assertions.
Rather than falling back on naive majority voting, the resolver evaluates provenance integrity, context conditions, evidence quorum disparities, and confidence margins.

## 2. Epistemic Safety & UNKNOWN State
A critical requirement of scientific cognitive governance is the ability to recognize uncertainty. When two contradictory claims have weak or insufficient supporting evidence ($\le 1$ observation each), the engine explicitly returns **`UNKNOWN`** rather than guessing or picking an arbitrary winner. This prevents false knowledge accumulation.

## 3. Decision Matrix
- **ACCEPT_A / ACCEPT_B**: Triggered when one claim possesses dominant evidence quorum ($\Delta \ge 3$) and valid provenance over the other.
- **CONTEXTUALIZE**: Triggered when both assertions are valid under distinct operational scopes (e.g. `single_hop` vs `multi_hop`).
- **REVALIDATE**: Triggered when competing claims possess equal corroborating evidence, scheduling an isolated probe benchmark.
- **UNKNOWN**: Triggered when evidence is insufficient on both sides.
- **REJECT_BOTH**: Triggered when both claims possess corrupted or missing provenance.

**Wave 531 Status: PASS**
