# Step 68: Grounded Neural Proposal Specification

## 1. Overview & Architecture

Step 68 operationalizes the connection between Step 67 unified cognitive context and the frozen `ChakrMicro` neural core via `InferenceEngine`, producing structured, provenance-linked `ProposalContract` instances validated by `ProposalValidator`.

```
               ┌─────────────────────────────────────────┐
               │         CognitiveContextBundle          │
               │  - positive_evidence                    │
               │  - positive_memories                    │
               │  - negative_boundaries                  │
               │  - context_fingerprint                  │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
               ┌─────────────────────────────────────────┐
               │      NeuralProposalInputEncoder         │
               │  (Bounded Prompt / Token Translation)   │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
               ┌─────────────────────────────────────────┐
               │    InferenceEngine / ChakrMicro Core    │
               │  (Frozen, ΔW = 0, Greedy Sampling)      │
               └────────────────────┬────────────────────┘
                                    │ Raw Model Output
                                    ▼
               ┌─────────────────────────────────────────┐
               │       Proposal Contract Decoder         │
               │  - Structured Claims                    │
               │  - Epistemic Partitions                 │
               │  - Provenance Linkage                   │
               └────────────────────┬────────────────────┘
                                    │ Raw ProposalContract
                                    ▼
               ┌─────────────────────────────────────────┐
               │           ProposalValidator             │
               │  - File & Symbol Grounding              │
               │  - Negative Boundary Checking           │
               │  - Task Constraint Collision            │
               │  - Context Fingerprint Assertion        │
               │  - Prohibited Execution Scan            │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
                       Gated ProposalContract
               (ACCEPTED_FOR_EXECUTION_REVIEW /
                REJECTED / ABSTAIN / CONFLICTED)
```

## 2. Core Typed Structures

### `EpistemicPartition`
- `FACT_EVIDENCE`: Direct repository AST / dependency / test fact.
- `MEMORY`: Verified recalled historical solution or failure pattern.
- `INFERENCE`: Deductive conclusion derived from facts and memories.
- `PROPOSAL`: Concrete code patch or modification suggestion.
- `UNCERTAINTY`: Explicitly acknowledged boundary or unknown parameter.

### `StructuredEpistemicClaim`
- `partition: EpistemicPartition`
- `statement: str`
- `supporting_id: Optional[str]`

### `ProposalValidationStatus`
- `ACCEPTED_FOR_EXECUTION_REVIEW`: Fully grounded, unviolated, ready for deterministic branch execution.
- `REJECTED`: Violated negative boundary, hallucinated symbol/file, or invalid syntax.
- `ABSTAIN`: Insufficient evidence, missing prerequisite context, or ungrounded ambiguity.
- `CONFLICTED`: Conflicting positive recommendations or collision with task constraint.

### `ProposalContract`
- `proposal_id: str`
- `task_id: str`
- `proposal_type: str`
- `summary: str`
- `proposed_changes: Dict[str, str]` (file_path -> replacement or patch content)
- `target_files: Tuple[str, ...]`
- `target_symbols: Tuple[str, ...]`
- `reasoning_trace_summary: str` (Structured rationale, zero private chain-of-thought)
- `supporting_evidence_ids: Tuple[str, ...]`
- `supporting_memory_ids: Tuple[str, ...]`
- `negative_boundary_ids: Tuple[str, ...]`
- `confidence: float`
- `epistemic_state: EpistemicState`
- `context_fingerprint: str`
- `neural_generation_metadata: Dict[str, Any]`
- `claims: Tuple[StructuredEpistemicClaim, ...]`
- `validation_status: ProposalValidationStatus`
- `validation_reasons: Tuple[str, ...]`

## 3. Grounding & Validation Invariants

1. **ΔW = 0 Neural Invariant**: Parameter weights are strictly immutable across all inference passes. Pre- and post-inference parameter SHA-256 digests are asserted against `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.
2. **Authoritative Validator Gate**: The neural model never validates its own proposal. Only `ProposalValidator` assigns `ACCEPTED_FOR_EXECUTION_REVIEW`.
3. **Negative Boundary Containment**: Any proposal whose patch content matches a prohibited phrase in `negative_boundaries` is immediately `REJECTED`.
4. **Task Constraint Conflict Isolation**: Any proposal targeting a file explicitly forbidden by task constraints is marked `CONFLICTED`.
5. **No Context Escape**: Proposals cannot modify files outside `allowed_files` or touch ungrounded symbols.
6. **Zero Authority**: Proposals contain data only; execution handles (`os.system`, `subprocess`, `eval`, etc.) trigger immediate `REJECTED` status.
