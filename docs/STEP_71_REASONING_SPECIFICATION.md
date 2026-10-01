# ChakrView Step 71 Specification: Structured Reasoning Subsystem

## 1. Architectural Purpose
Step 71 implements a deterministic, inspectable reasoning layer that operates over [CognitiveContextBundle](file:///d:/Project/ChakrView/chakrview/cognition/repository/cognitive_context.py) and [ProposalContract](file:///d:/Project/ChakrView/chakrview/cognition/repository/neural_proposal.py) without exposing private chain-of-thought tokens.

Instead of ungrounded internal free-form generations, reasoning is structured into strongly typed epistemic elements with complete provenance linkage back to repository static facts and recalled episodic memories.

## 2. Core Epistemic Partitions

Model-generated text is never treated as ground truth. The system strictly separates categories:
- `FACT`: Direct observations grounded in repository AST, dependency topology, or test assertions.
- `MEMORY`: Recalled historical solution or failure patterns from `RepositoryMemoryIndex`.
- `INFERENCE`: Deductive or inductive conclusions logically derived from known facts and memories.
- `HYPOTHESIS`: Tentative explanations under active evaluation.
- `PROPOSAL`: Concrete code modifications or refactoring directives.
- `UNCERTAINTY`: Explicitly recognized parameter boundaries, edge cases, or unresolved constraints.
- `UNKNOWN`: Complete absence of required information; requires investigation.

## 3. Data Structures

### 3.1 ReasoningClaim
- `claim_id`: Unique identifier
- `category`: `EpistemicCategory`
- `statement`: Plaintext description of the assertion
- `confidence_state`: `EpistemicConfidenceState` (`KNOWN`, `SUPPORTED`, `INFERRED`, `CONTESTED`, `UNCERTAIN`, `UNKNOWN`, `ABSTAINED`)
- `supporting_evidence_ids`: Grounded repository AST/evidence IDs
- `supporting_memory_ids`: Verified recalled memory IDs
- `contradicting_evidence_ids`: Identified counter-evidence IDs
- `target_file` / `target_symbol`: Optional targeted repository references
- `fingerprint`: Canonical SHA-256 digest

### 3.2 AssumptionRecord
- `assumption_id`: Identifier
- `description`: Underlying assumption required for validity
- `criticality`: Severity rating (`HIGH`, `MEDIUM`, `LOW`)
- `is_empirically_verified`: Boolean indicator
- `verification_requirement`: Specific check needed to prove assumption

### 3.3 InvestigationRequirement
- Interface preparing for future knowledge acquisition / investigation (Steps 74+).
- Formulates specific knowledge gaps without manufacturing fake web search results.

### 3.4 StructuredReasoningArtifact
- Complete, inspectable artifact carrying claims, assumptions, observations, candidate conclusions, unresolved questions, and overall epistemic confidence.
