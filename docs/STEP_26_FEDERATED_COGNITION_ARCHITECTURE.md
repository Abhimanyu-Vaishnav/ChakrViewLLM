# ChakrView — Step 26: Multi-Agent Federated Cognition & Cooperative Intelligence Foundation

**Ratification Status**: COMPLETED & VERIFIED  
**Baseline Commit**: `ea772d5` (Step 25)  
**Model Core**: ChakrMicro v0.1 (FROZEN & IMMUTABLE)  
**Parameter Count**: 3,443,136  
**Context Ceiling**: 512 tokens  
**Vocabulary Size**: 4,096 tokens  
**Special Tokens**: BOS=0, EOS=1, PAD=2  
**Weight Fingerprint**: SHA-256 Verified Pre & Post Cycle  

---

## 1. Objective

Step 26 establishes the foundational architecture for multiple cooperative ChakrView cognitive agents cooperating on bounded, complex cognitive tasks while strictly maintaining:
- **Central Authority Invariant**: `AGENT != AUTHORITY`. Agents are specialized cognitive roles, not independent sovereign decision-makers.
- **Frozen Neural Core**: A single `ChakrMicro v0.1` neural brain powers all generative/reasoning candidates. No new neural models, no multi-model sprawl, no weight modifications.
- **Tenant & Session Isolation**: Strict isolation across agent registration, message exchange, memory recall, and task decomposition.
- **Deterministic Execution**: Predictable, reproducible federated reasoning under fixed inputs, configurations, seeds, and memory states.
- **Fail-Closed Governance**: Strict integration with Step 17 `CapabilityGate`, Step 18 Epistemic Uncertainty, Step 21 Deliberation, Step 23 Critical Thinking, Step 24 Governed Memory, and Step 25 Unified Cognitive Architecture.

---

## 2. Architectural Principles

1. **Agent != Authority**: No agent can directly execute system capabilities, modify weights, promote memory to semantic status, or initiate destructive actions.
2. **Single Shared Brain**: All cognitive agents invoke the identical, frozen `ChakrMicro v0.1` neural core (3,443,136 parameters, 512 context).
3. **Typed Explicit Contracts**: Agents communicate strictly through validated, cryptographically hashed messages (`AgentMessage`, `MessageEnvelope`). No raw unvalidated dictionaries.
4. **Outputs as Evidence, Not Truth**: Agent outputs are treated as hypotheses, claims, or analytical candidates. Truth is never established by majority vote ("3 agents said it, therefore it is true" is strictly rejected).
5. **Minority Evidence Preservation**: When agents disagree, minority claims and counter-evidence are preserved in conflict records rather than discarded.
6. **Bounded Resource Ceiling**: Hard upper bounds are enforced universally ($\le 8$ agents, $\le 8$ rounds, $\le 128$ messages, $\le 4$ delegation depth) regardless of hardware tier.
7. **Fault Isolation**: A crashing, timing out, or contract-violating agent is isolated; the federation degrades gracefully without compromising system integrity.
8. **Tamper-Evident Auditability**: Every message envelope and trace has deterministic SHA-256 fingerprints and optional hash-chain links.

---

## 3. Agent Model

Agents represent logical cognitive perspectives operating over bounded subtasks:

```
                          +------------------------+
                          |   ChakrMicro v0.1 Core |
                          |  (FROZEN & IMMUTABLE)  |
                          +-----------+------------+
                                      | (Read-Only Neural Core)
        +-------------+---------------+-------------+-------------+
        |             |               |             |             |
        v             v               v             v             v
   +---------+  +------------+   +---------+  +-----------+  +-----------+
   | ANALYST |  | RESEARCHER |   | CRITIC  |  |  PLANNER  |  |SYNTHESIZER|
   +---------+  +------------+   +---------+  +-----------+  +-----------+
        |             |               |             |             |
        +-------------+---------------+-------------+-------------+
                                      |
                                      v
                    +------------------------------------+
                    | Federated Cognitive Coordinator    |
                    | (Task Decomp, Protocol, Conflict)  |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    | Unified Cognitive Decision Layer   |
                    | Capability Gate / Safe Response    |
                    +------------------------------------+
```

### Roles and Capability Scopes
- **`ANALYST`**: In-depth analytical reasoning, premise decomposition, and hypothesis generation. Scope: `REASONING_ONLY`.
- **`RESEARCHER`**: Evidence collection and memory candidate retrieval queries. Scope: `EVIDENCE_RETRIEVAL_ONLY`.
- **`CRITIC`**: Counter-argument generation, assumption auditing, contradiction detection, and risk assessment. Scope: `CONTRADICTION_ANALYSIS_ONLY`.
- **`PLANNER`**: Action planning, step sequencing, dependency graphing, and constraint formulation. Scope: `PLANNING_ONLY`.
- **`SYNTHESIZER`**: Integrating multi-perspective evidence, resolving contradictions, generating synthesis proposals. Scope: `SYNTHESIS_ONLY`.
- **`VERIFIER`**: Fact-checking, consistency verification against ground truth rules. Scope: `VERIFICATION_ONLY`.
- **`OBSERVER`**: Audit trail monitoring and metadata recording. Scope: `OBSERVATION_ONLY`.

### Identity & Contract
Each agent is uniquely identified by `AgentIdentity`:
- `agent_id`: Deterministic role/tenant string (e.g., `agent:analyst:tenant_default`).
- `role`: One of the 7 supported `AgentRole` enum values.
- `tenant_id` & `session_id`: Tenant/session scoping keys.
- `capabilities`: Frozen set of allowed `AgentCapability` permissions.
- `version`: Subsystem version (`"0.1.0"`).
- `status`: Current lifecycle state (`AgentStatus.ACTIVE`, `SUSPENDED`, etc.).

---

## 4. Message Protocol

Agents interact solely via typed, immutable envelopes:

### Envelope Specification
```python
@dataclass(frozen=True)
class AgentMessage:
    message_id: str
    sender_agent_id: str
    receiver_agent_id: str
    tenant_id: str
    session_id: str
    correlation_id: str
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    provenance: Dict[str, Any]
    timestamp: float
    sequence_number: int
    parent_message_id: Optional[str]
    integrity_hash: str
```

### Integrity Hashing & Message Chaining
1. **Payload Hash**: SHA-256 digest of canonical JSON-serialized `(sender, receiver, tenant, session, correlation, type, payload, sequence, parent)`.
2. **Envelope Fingerprint**: Cryptographic validation computed by `FederatedProtocolValidator`.
3. **Chain Verification**: Each message references `parent_message_id` allowing deterministic traversal and audit verification of conversation threads.

---

## 5. Agent Registry

The `AgentRegistry` acts as the single gatekeeper for agent lifecycle and tenant discovery:
- **Tenant Isolation**: Discovery queries (`get_agents_for_tenant`) strictly filter by `tenant_id`. Cross-tenant lookup is prohibited.
- **Hard Ceilings**: Enforces `max_agents` bound ($\le 8$). Attempts to register beyond capacity fail closed with `AgentCapacityExceededError`.
- **Duplicate Prevention**: Re-registering an existing `agent_id` raises `DuplicateAgentError`.
- **Lifecycle Control**: Supports atomic `register_agent`, `unregister_agent`, `update_agent_status`, and contract verification.

---

## 6. Federated Task Decomposition

Incoming unified cognitive tasks are decomposed by `FederatedTaskDecomposer` into a bounded DAG of `AgentTask` instances:
- **Deterministic Assignment**:
  - `EVIDENCE_SEARCH` $\to$ assigned to `RESEARCHER`
  - `DEEP_ANALYSIS` $\to$ assigned to `ANALYST`
  - `CRITICAL_EVALUATION` $\to$ assigned to `CRITIC`
  - `ACTION_PLANNING` $\to$ assigned to `PLANNER`
  - `EVIDENCE_SYNTHESIS` $\to$ assigned to `SYNTHESIZER`
- **Bounds**: Decomposer outputs at most `max_subtasks` (capped at 6) per round to prevent combinatorial explosion.

---

## 7. Execution Lifecycle

The `FederatedCognitionEngine.run_federated_cycle` coordinates an end-to-end multi-agent round:
1. **Pre-Cycle Invariant Verification**: SHA-256 weight hash of `ChakrMicro` is taken before execution.
2. **Context & Decomposition**: Task is evaluated; subtasks are generated for active tenant agents.
3. **Execution Rounds ($\le \text{max\_rounds}$)**:
   - Tasks are dispatched to appropriate agents.
   - Agents run read-only neural candidate reasoning or deterministic symbolic tasks.
   - Outputs are packaged into validated `AgentMessage` instances.
4. **Validation & Protocol Check**: `FederatedProtocolValidator` inspects sender, receiver, tenant boundaries, and cryptographic hash integrity.
5. **Evidence Aggregation**: Claims, assumptions, and supporting evidence are categorized.
6. **Conflict Resolution**: Contradictions between agent claims are identified; counter-evidence is indexed.
7. **Synthesis**: `FederatedSynthesizer` integrates findings into a `FederatedSynthesisCandidate`.
8. **Capability Routing**: If an agent requests capability execution, it is routed through `CapabilityGate`—agents never execute tools directly.
9. **Trace & Experience Recording**: Sanitized trace is created; episodic experience is recorded in continual memory.
10. **Post-Cycle Invariant Verification**: Final SHA-256 weight hash is checked. Any mutation causes an immediate fail-closed `RuntimeError`.

---

## 8. Agent Conflict Resolution

Conflict handling is governed by `FederatedConflictResolver`:
- **Categorization**: Detects whether pairs of agent claims have direct contradictions or divergent conclusions.
- **Resolution States**:
  - `AGREEMENT`: No conflicting assertions.
  - `PARTIAL_AGREEMENT`: Substantial overlap with minor nuances.
  - `CONFLICT`: Mutual contradiction in key claims.
  - `UNRESOLVED`: Conflicting claims lack decisive empirical grounding.
  - `INSUFFICIENT_EVIDENCE`: Claims are speculative or ungrounded.
- **Minority Preservation**: The resolver never performs simplistic majority voting. Conflicting evidence and minority viewpoints are explicitly attached to the `FederatedConflictRecord` and surfaced in the final synthesis.

---

## 9. Evidence Aggregation

`FederatedEvidenceAggregator` parses message payloads into formal epistemic categories:
1. **`Claim`**: An assertion made by an agent (e.g., "Hypothesis A is true").
2. **`Ground Evidence`**: Empirically verified or retrieved data points with provenance.
3. **`Interpretation`**: Logical inference derived from claims or evidence.
4. **`Assumption`**: Unverified presupposition.
5. **`Counter-Evidence`**: Evidence disproving or weakening a claim.

*Weight Formula*: The truth weight of a claim is proportional to verified ground evidence and logical consistency—never to the number of agents asserting it.

---

## 10. Consensus & Synthesis Layer

`FederatedSynthesizer` takes the aggregated evidence, conflict records, and memory candidates to formulate a bounded `FederatedSynthesisCandidate`:
- Determines recommended decision state (`ANSWER`, `ANSWER_WITH_UNCERTAINTY`, `REFUSE`, `REQUEST_CONFIRMATION`).
- Integrates unresolved contradictions into the output summary.
- Attaches the complete evidence chain and confidence score.
- Does **not** authorize execution; output is purely a candidate for the cognitive decision layer.

---

## 11. Fault Isolation

Agents operate in sandboxed roles:
- **Exceptions Caught**: Unhandled exceptions during agent execution are caught by the coordinator.
- **Isolation**: Failing agent is marked `FAILED` in the trace without halting the engine.
- **Retry & Reassignment**: Configurable single retry attempt within round budget.
- **Graceful Degradation**: Remaining operational agents continue the synthesis round; if only a subset survives, the synthesizer operates over available partial evidence.

---

## 12. Capability Boundaries

Strict authority enforcement:
- No agent has access to `os`, `sys`, `socket`, `subprocess`, filesystem writes, or shell execution.
- If a task requires an external action (e.g., tool invocation), the agent submits an intent.
- The intent must pass through `CapabilityGate.verify_capability()` with explicit tenant permissions.
- Unauthorized attempts are rejected and logged in the trace as security violations.

---

## 13. Memory Boundaries

Integration with Step 24 Governed Memory:
- **Tenant Scoping**: Memory queries and writes require matching `tenant_id` and `session_id`.
- **Read-Only Recall**: Agents query semantic and working memory through governed interfaces.
- **Non-Authoritative Experience**: Post-cycle outputs are written to episodic memory as experience records (`record_experience`).
- **Semantic Promotion Protected**: No agent can directly write to or promote items in semantic memory; promotion requires deliberate governance validation.

---

## 14. Critical-Thinking Integration

Multi-agent reasoning integrates Step 23 Critical Thinking protocols:
- `CriticAgent` automatically challenges claims produced by `AnalystAgent` and `ResearcherAgent`.
- Assumptions are cataloged; counter-evidence searches are mandated for high-stakes assertions.
- Contradiction analysis triggers uncertainty elevation rather than premature convergence.

---

## 15. Sanitized Public Trace

The `SafePublicFederatedTrace` records public telemetry without leaking internal states:
- **Included**: `trace_id`, `task_id`, `participating_agents`, `rounds_executed`, `message_count`, `conflict_count`, `decision_state`, `latency_ms`, `final_status`.
- **Strictly Redacted**:
  - Hidden internal chain-of-thought tokens.
  - Raw logits and model internal activations.
  - Private credential or capability payload secrets.
  - Cross-tenant data.

---

## 16. Cryptographic Integrity Foundation

Step 26 introduces tamper-evident auditability without external blockchain or network overhead:
- Canonical JSON serialization of message contents.
- SHA-256 hash generation for every `AgentMessage`.
- Envelope-level message tampering detection (altered payloads trigger `MessageTamperingError`).
- Deterministic trace fingerprint verifying execution logs.

---

## 17. Hardware Adaptation

Execution budgets adapt to hardware profiles while respecting hard safety ceilings:

| Parameter | LOW_RESOURCE | STANDARD | HIGH_RESOURCE | Hard Safety Ceiling |
|---|---|---|---|---|
| `max_agents` | 3 | 5 | 7 | **8** |
| `max_rounds` | 2 | 4 | 6 | **8** |
| `max_messages_per_round` | 12 | 16 | 20 | **24** |
| `max_total_messages` | 24 | 64 | 120 | **128** |
| `max_delegation_depth` | 2 | 3 | 4 | **4** |
| `max_synthesis_attempts` | 1 | 2 | 3 | **3** |

---

## 18. Tenant and Session Isolation

Four-layer isolation guarantees:
1. **Registry Isolation**: Queries cannot enumerate agents from foreign tenants.
2. **Protocol Isolation**: Messages with cross-tenant sender/receiver are rejected at validation.
3. **Engine Isolation**: Task context and decomposition operate strictly within the task's tenant ID.
4. **Memory Isolation**: Continual and working memory lookups are constrained by tenant filters.

---

## 19. Security Model

- **Fail-Closed Weight Audit**: SHA-256 weight hash comparison before and after every federated cycle.
- **Contract Enforcement**: Role capabilities are checked prior to task execution.
- **Input Sanitization**: Messages with invalid sequence numbers, timestamps, or tampered hashes are immediately dropped.
- **Resource Exhaustion Defense**: Strict message count, agent count, and recursion depth ceilings prevent runaway loops or denial-of-service.

---

## 20. Failure Modes & Graceful Degradation

| Failure Mode | Detection Point | System Response |
|---|---|---|
| Agent Crash | Execution Try/Catch | Agent isolated, logged in trace, remaining federation proceeds |
| Message Tampering | Protocol Validator | `MessageTamperingError` raised, message dropped, security incident logged |
| Cross-Tenant Leak | Routing Validator | `TenantRoutingError` raised, message rejected |
| Excessive Delegation | Engine Recursion Check | Delegation halted at `max_delegation_depth` |
| Quorum Loss (0 agents) | Coordinator Init | Graceful fallback to default uncertainty response |
| Neural Weight Mutation | Post-Cycle Audit | Immediate fail-closed `RuntimeError` termination |

---

## 21. Deterministic Execution Guarantees

Under identical inputs, identical agent registration, identical memory state, and identical execution seed:
- Task decomposition generates identical subtasks in identical order.
- Message sequence numbers increment deterministically.
- Conflict detection produces identical conflict records.
- Synthesized output produces identical decision recommendations.

---

## 22. Known Limitations

- **Single-Node Execution**: Current coordinator operates in-process via local CPU queues.
- **Fixed Model Invariant**: Uses only the frozen 3.44M parameter `ChakrMicro` core; reasoning complexity is bounded by small-model context.
- **Non-Preemptive Scheduling**: Agent tasks execute sequentially in order of priority within each round.

---

## 23. Future Distributed Extension Points

Step 26 establishes clean architectural seams for future scaling:
- **Pluggable Message Transport**: `AgentMessage` envelopes are serializable to binary formats (e.g., Protobuf) for IPC or zero-copy shared memory.
- **Asymmetric Cryptography / PKI**: Signature fields can be added to `AgentMessage` for Ed25519 agent keypair verification.
- **Distributed Agent Registry**: Interface allows backing by distributed consensus stores without breaking local contract guarantees.
