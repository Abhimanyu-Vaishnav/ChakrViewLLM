# Step 74: Knowledge Acquisition Specification

## Objective
Establish a formal, typed framework for the ChakrView cognitive brain to request and ingest external information without automatically treating it as established truth.

## Core Abstractions

### 1. `InvestigationRequest`
- Represents a formalized need for external information.
- Parameters: `user_query`, `required_information`, `scope`, `freshness_requirement`, `allowed_source_types`, `max_evidence_budget`.
- The request describes the *intent* to acquire knowledge, separate from its execution.

### 2. `InvestigationSource` & `SourceMetadata`
- Encapsulates the identity and reliability of where the information came from.
- Tracks `source_identity` (e.g., Wikipedia, specific forum) and calculates an initial `reliability_score`.

### 3. `EvidenceItem`
- Represents a specific piece of retrieved information tied back to the `InvestigationRequest`.
- Contains `content`, `confidence`, and `status`.

## Design Philosophy
- **No Automatic Truth**: The knowledge acquisition layer only retrieves *candidate* evidence.
- **Provenance Preservation**: Every piece of acquired knowledge must carry an unbreakable link back to its `source_id` and timestamp.
- **Fail-Closed Security**: External sources only provide text data and are never granted filesystem, memory, or execution authority.

## Interactions
1. `CognitiveController` or `BoundedPlanner` generates an `InvestigationRequest` based on context gaps.
2. The `GovernedToolGate` executes the retrieval, yielding `EvidenceItem` candidates.
3. The items pass to the `EvidenceVerifier` (Step 75) before touching the neural reasoning core.
