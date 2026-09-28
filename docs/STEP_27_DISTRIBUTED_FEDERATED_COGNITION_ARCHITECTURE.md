# CHAKRVIEW — STEP 27 ARCHITECTURE SPECIFICATION
## Distributed Federated Cognition & Secure Agent Transport Foundation

**Authoritative Baseline:** Step 26 Ratified (Commit: `97c06dd`)  
**Frozen Model Invariant:** ChakrMicro v0.1 (3,443,136 parameters, vocab=4096, context=512, BOS=0, EOS=1, PAD=2)  
**Security Invariant:** `NODE != AUTHORITY`, `AGENT != AUTHORITY`, `REMOTE_AGENT != AUTHORITY`, `MESSAGE != AUTHORITY`, `CONSENSUS != AUTHORITY`  
**Execution Environment:** Deterministic In-Process Loopback Transport (No External Network Dependencies)

---

### Status Legend
- **IMPLEMENTED**: Complete, functional, production-ready codebase components.
- **TESTED**: Verified with unit/integration tests and deterministic assertions.
- **DEFERRED**: Explicitly out-of-scope for Step 27, reserved for future stages.

---

## 1. Motivation & Architectural Objectives

ChakrView Step 26 established in-process multi-agent federated cognition. However, agent interactions were constrained to direct function calls and shared in-process references.

Step 27 transforms this foundation into a **network-ready distributed federation architecture** without requiring multi-machine physical deployment or external network dependencies. It introduces:
1. Strongly typed distributed node and role abstractions.
2. Abstract transport interface decoupled from concrete physical sockets.
3. High-integrity, deterministic loopback transport for testing and local verification.
4. Cryptographically signed, canonicalized, tamper-evident message envelopes.
5. Strict bounded replay protection preventing duplicate, delayed, or cross-tenant messages.
6. Deterministic, policy-driven resource-aware task routing.
7. Resilient failure isolation with circuit breakers and exponential backoff retries.
8. Evidence-based distributed synthesis preserving minority and contradictory observations without fabricating majority-rule consensus.
9. Continuous zero-weight-mutation verification enforcing model immutability.

---

## 2. Distributed Architecture Overview

```
                      +---------------------------------------+
                      | DistributedFederatedCognitionEngine   |
                      +-------------------+-------------------+
                                          |
                        [Task Decomposition & Classification]
                                          |
                      +-------------------v-------------------+
                      |       DistributedTaskRouter           |
                      +-------------------+-------------------+
                                          |
         +--------------------------------+-------------------------------+
         |                                                                |
+--------v--------+                                              +--------v--------+
| Local Execution |                                              | Remote Routing  |
|  (Local Node)   |                                              | (Eligible Nodes)|
+--------+--------+                                              +--------+--------+
         |                                                                |
         |                                              [Envelope Creation & Signing]
         |                                              [Replay Registry Registration]
         |                                                                |
         |                                                       +--------v--------+
         |                                                       |    Transport    |
         |                                                       | (Loopback/Local)|
         |                                                       +--------+--------+
         |                                                                |
         |                                              [Transport Response Ingestion]
         |                                              [Integrity & Replay Verification]
         |                                                                |
         +--------------------------------+-------------------------------+
                                          |
                      +-------------------v-------------------+
                      |      FederatedEvidenceAggregator      |
                      |   (Minority & Contradiction Preserved)|
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |      FederatedConsensusEngine         |
                      |     (Synthesize Candidate Decision)   |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |    Unified Cognitive Decision Layer   |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |            CapabilityGate             |
                      |       (Strict External Authority)     |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |   Governed Experience Recording       |
                      |   (Continual Governed Memory)         |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |      Safe Public Distributed Trace    |
                      |   (Redacted Telemetry & Audit Log)    |
                      +---------------------------------------+
```

---

## 3. Node Model & System Invariants

### 3.1 Primitives (`chakrview/cognition/distributed/models.py`)
- **`NodeIdentity`**: Unique node identifier with human-readable name, public key fingerprint, and creation timestamp.
- **`NodeRole`**: Enumeration of technical operational roles:
  - `COORDINATOR`: Ingests, decomposes, and orchestrates distributed cognition.
  - `WORKER`: Executes subtasks and returns structured cognitive evidence.
  - `OBSERVER`: Audits distributed trace and verification without active voting.
  - `ARBITER`: Independent verification and contradiction isolation.
- **`NodeStatus`**: Operational lifecycle states:
  - `REGISTERED`: Registered in registry; awaiting initial health probe.
  - `HEALTHY`: Fully operational and accepting routed tasks.
  - `DEGRADED`: Operational with constrained resource budgets or elevated error rates.
  - `UNAVAILABLE`: Offline or unreachable.
  - `QUARANTINED`: Suspended due to repeated failures or circuit breaker trips.
  - `REVOKED`: Permanently excluded due to integrity or cryptographic violation.
- **`NodeCapabilities`**: Set of supported tasks, agent roles, max concurrent tasks, and memory limits.
- **`NodeResourceProfile`**: Technical hardware metrics (CPU cores, memory MB, device type).
- **`NodeEndpoint`**: Transport scheme, host, port, path, and timeout settings.
- **`NodeHealth`**: Status, active load, error count, latency EWMA, last heartbeat.
- **`NodeRegistration`**: Complete registration record joining identity, role, endpoint, capabilities, profile, and trust state.

### 3.2 Hard Architectural Ceilings
To prevent unbounded resource consumption or denial-of-service in distributed topologies:
- `MAX_FEDERATION_NODES`: $\le 16$
- `MAX_AGENTS_PER_NODE`: $\le 8$
- `MAX_FEDERATION_DEPTH`: $\le 4$
- `MAX_REMOTE_TASKS_PER_CYCLE`: $\le 32$
- `MAX_REPLAY_REGISTRY_ENTRIES`: 10,000

### 3.3 Core Invariant: Node is not Authority
Nodes are physical/logical infrastructure containers hosting agents.
- **`NODE != AGENT`**: A node hosts agents; node identity is separate from agent identity.
- **`NODE != AUTHORITY`**: Nodes cannot authorize actions, capabilities, or state changes.
- **`REMOTE_AGENT != AUTHORITY`**: Remote agent responses are unprivileged input proposals.
- **`CONSENSUS != AUTHORITY`**: Agreement across nodes or agents cannot bypass governance or capability gates.

---

## 4. Transport Abstraction & Loopback Engine

### 4.1 Interface Specification (`chakrview/cognition/distributed/transport.py`)
```python
class Transport(ABC):
    @abstractmethod
    def send(self, envelope: DistributedMessageEnvelope) -> None: ...
    @abstractmethod
    def receive(self, timeout_ms: int = 1000) -> Optional[DistributedMessageEnvelope]: ...
    @abstractmethod
    def request(self, envelope: DistributedMessageEnvelope, timeout_ms: int = 5000) -> TransportResponse: ...
    @abstractmethod
    def broadcast(self, envelope: DistributedMessageEnvelope, target_nodes: Sequence[str]) -> Dict[str, TransportResponse]: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def health_check(self) -> bool: ...
```

### 4.2 In-Process Loopback Transport (`LoopbackTransport`)
- **Status**: IMPLEMENTED & TESTED.
- Purely in-memory, deterministic message transport.
- Dispatches messages across virtual nodes using registered endpoints and handler callbacks.
- Supports fault injection for testing resilience:
  - Simulated latency (deterministic milliseconds)
  - Drop rate simulation ($0.0 \dots 1.0$)
  - Timeout simulation
  - Protocol error simulation
- Thread-safe queues with deterministic timeout handling.

---

## 5. Secure Message Envelope & Integrity Verification

### 5.1 Structure (`DistributedMessageEnvelope`)
Each distributed message contains:
- **`MessageHeader`**:
  - `message_id`: Unique UUIDv4.
  - `protocol_version`: `"chakrview.distributed.v1"`.
  - `message_type`: Task request, response, heartbeat, rejection, or control.
  - `timestamp_ns`: Nanosecond creation timestamp.
  - `logical_clock`: Monotonic sender sequence number.
  - `nonce`: Cryptographically secure unique string.
  - `correlation_id`: Root cognitive cycle identifier.
  - `causation_id`: Immediate parent task or message identifier.
  - `ttl_seconds`: Time-to-live before message expiration.
  - `hop_count`: Current forwarding depth (hard limit: 4).
  - `max_hops`: Maximum permitted hops.
- **`MessageRoute`**:
  - `sender_node`: Source node identifier.
  - `sender_agent`: Source agent identifier.
  - `receiver_node`: Destination node identifier.
  - `receiver_agent`: Destination agent identifier.
  - `tenant_id`: Mandatory tenant boundary.
  - `session_id`: Mandatory session boundary.
- **`payload`**: Deterministically serialized dictionary containing task parameters or cognitive candidate proposals.
- **`MessageIntegrity`**:
  - `sha256_hash`: Hex-encoded SHA-256 fingerprint of the canonical payload representation.
  - `signature`: Hex-encoded cryptographic signature (HMAC-SHA256 for Step 27).
  - `signer_identity`: Identity string of the signing entity.
  - `algorithm`: `"HMAC-SHA256"` (extensible to Ed25519).

### 5.2 Deterministic Canonical Serialization
To guarantee tamper detection:
1. Payloads and headers are sorted by keys recursively.
2. UTF-8 encoded with standard separators `(',', ':')` and no trailing whitespace.
3. Floats are formatted deterministically.
4. Any payload modification breaks `compute_sha256()` validation, causing immediate fail-closed rejection.

---

## 6. Deterministic Replay Protection (`ReplayProtectionTracker`)

- **Status**: IMPLEMENTED & TESTED.
- **Bounded In-Memory Tracking**:
  - Ring-buffer / deque capped at `max_entries` (default: 10,000).
  - Expired entries are evicted when clock moves forward.
- **Rejection Matrix**:
  - `DUPLICATE_MESSAGE_ID`: Message ID already processed.
  - `DUPLICATE_NONCE`: Nonce already used by the same sender.
  - `EXPIRED_TTL`: `current_time > timestamp + ttl_seconds`.
  - `EXCESSIVE_HOPS`: `hop_count > max_hops` or `hop_count > MAX_FEDERATION_DEPTH`.
  - `TENANT_MISMATCH`: Message routing crosses tenant boundaries without authorization.
  - `SESSION_MISMATCH`: Cross-session routing without explicit policy override.
  - `CLOCK_REGRESSION`: Monotonic logical clock regressed for the sender identity.

---

## 7. Cryptographic Identity & Attestation

- **Status**: IMPLEMENTED & TESTED.
- **Interfaces**:
  - `NodeIdentityProvider`: Node identity resolution and key store.
  - `AgentIdentityProvider`: Agent identity resolution.
  - `MessageSigner`: Signs message envelopes.
  - `MessageVerifier`: Verifies signatures against public keys/secrets.
  - `AttestationProvider`: Node integrity attestation (hardware and weight immutability proof).
- **Step 27 Implementation**:
  - `DeterministicHmacMessageSigner` & `DeterministicHmacMessageVerifier`.
  - SHA-256 model weight fingerprint integration into node attestation.
- **Security Rule**: Authentication verifies *who* the sender is; it grants **zero** capability authorization.

---

## 8. Distributed Node Registry (`DistributedNodeRegistry`)

- **Status**: IMPLEMENTED & TESTED.
- **Capabilities**:
  - Register node with validation of hard ceilings ($\le 16$ nodes).
  - Detect and reject duplicate node IDs.
  - Enforce tenant isolation: nodes are bound to allowed tenants. Queries from Tenant A never reveal Tenant B nodes.
  - Node health state machine: `REGISTERED` $\to$ `HEALTHY` $\leftrightarrow$ `DEGRADED` $\to$ `QUARANTINED` / `REVOKED`.
  - Quarantined and revoked nodes are immediately excluded from task routing.

---

## 9. Deterministic Distributed Task Router (`DistributedTaskRouter`)

- **Status**: IMPLEMENTED & TESTED.
- **Selection Criteria**:
  1. Tenant compatibility check.
  2. Health eligibility (`HEALTHY` or `DEGRADED` if policy permits; never `QUARANTINED` or `REVOKED`).
  3. Capability match (node must support subtask's required task type and agent role).
  4. Resource profile adequacy (sufficient CPU and memory budget).
  5. Deterministic sorting:
     - Prefer local node (locality optimization).
     - Lowest active load.
     - Lowest EWMA latency.
     - Lexicographical tie-breaking on `node_id`.
- **Zero Stochasticity**: No random selection; identical registry state produces identical routing.

---

## 10. Resilience, Failure Handling & Circuit Breakers

- **Status**: IMPLEMENTED & TESTED.
- **`TimeoutPolicy`**: Per-hop and per-task bounded timeouts.
- **`RetryPolicy`**:
  - Deterministic exponential backoff: $\text{delay} = \min(\text{initial} \times \text{multiplier}^{\text{attempt}}, \text{max\_backoff})$.
  - Hard retry ceiling: $\le 3$ retries. No infinite retry loops.
- **`CircuitBreaker`**:
  - States: `CLOSED` $\xrightarrow{\ge 3 \text{ failures}}$ `OPEN` $\xrightarrow{\text{reset timeout}}$ `HALF_OPEN` $\xrightarrow{\text{success}}$ `CLOSED`.
  - Fast-fails requests to quarantined or failing nodes without blocking the cognitive cycle.

---

## 11. Resource Adaptation (`DistributedExecutionPolicy`)

Adapts distributed execution budgets based on hardware profile:

| Parameter | LOW_RESOURCE | STANDARD | HIGH_RESOURCE | Hard Ceiling |
|:---|:---:|:---:|:---:|:---:|
| `max_nodes` | 2 | 4 | 8 | 16 |
| `max_remote_agents` | 2 | 4 | 8 | 8 |
| `max_remote_tasks` | 4 | 8 | 16 | 32 |
| `max_hops` | 2 | 3 | 4 | 4 |
| `timeout_ms` | 2,000 | 4,000 | 8,000 | 10,000 |
| `max_retries` | 1 | 2 | 3 | 3 |
| `context_budget` | 256 | 384 | 512 | 512 |

**Core Model Invariant**: Changing hardware profiles adjusts task concurrency and context token windows, but **never** modifies the frozen ChakrMicro v0.1 model weights, vocabulary (4,096), or architecture.

---

## 12. Distributed Evidence Aggregation & Conflict Resolution

- **Status**: IMPLEMENTED & TESTED.
- Integrates Step 26 `FederatedEvidenceAggregator`.
- Ingests responses from local and remote agents.
- **Anti-Majority Principle**:
  - Truth is not determined by headcount.
  - Supporting evidence is corroborated.
  - Contradictory evidence is isolated into `FederatedEvidenceConflict`.
  - Minority evidence is preserved in `FederatedSynthesisCandidate.minority_perspectives`.
  - If evidence is insufficient or conflicts cannot be safely reconciled, returns an uncertainty/insufficient information state (`DecisionState.UNCERTAIN` or `DecisionState.SAFE_STOP`).

---

## 13. Security Invariants & Capability Gate

### 13.1 Rigorous Authority Boundaries
- Remote agents cannot invoke capabilities directly.
- Consensus decisions cannot invoke capabilities directly.
- Every external action must be formed as a `CapabilityRequest` and presented to `CapabilityGate.authorize()`.
- Unauthorized requests fail closed and trigger security audit records.

### 13.2 Weight Immutability Verification
- Before cognitive execution: compute SHA-256 fingerprint of all ChakrMicro v0.1 model parameters.
- After cognitive execution: recompute SHA-256 fingerprint.
- If `pre_hash != post_hash`: raise `WeightMutationError` immediately, invalidate results, and fail closed.

---

## 14. Sanitized Distributed Audit Trace (`SafePublicDistributedTrace`)

Exposes high-level auditability without leaking private neural or cognitive internals:
- **Exposed**: Task ID, federation ID, participating node IDs, participating agent roles, message counts, routing decisions, retries, failures, detected conflicts, synthesis summary, execution latency, final decision state.
- **Redacted & Protected**:
  - Zero private chain-of-thought scratchpads.
  - Zero raw logits or hidden layer activations.
  - Zero cryptographic secret keys or authentication tokens.
  - Zero unredacted raw prompt strings.

---

## 15. Observability & Telemetry

- **Status**: IMPLEMENTED & TESTED.
- **`DistributedObservabilityMetrics`**:
  - Telemetry counters: messages sent, messages received, rejected messages, replay attempts, timeout counts, retry counts, node failures, circuit breaker trips.
  - Latency tracking: sliding window of execution latencies with EWMA calculation.
  - Bounded memory: capped at 1,000 telemetry entries to prevent memory leaks.

---

## 16. Benchmark Performance Analysis

Measured on the local CPU runtime (`scripts/benchmark_distributed_federated_cognition.py`):

| Operation | Measured Latency | Throughput / Overhead |
|:---|:---:|:---:|
| Node Registration | 0.007 ms | ~142,857 ops/sec |
| Envelope Creation | 0.016 ms | ~62,500 ops/sec |
| Envelope Validation | 0.018 ms | ~55,555 ops/sec |
| Loopback Transport Request | 0.021 ms | ~47,619 ops/sec |
| Deterministic Task Routing | 0.007 ms | ~142,857 ops/sec |
| Canonical Serialization | 0.015 ms | ~66,666 ops/sec |
| Evidence Aggregation | 0.009 ms | ~111,111 ops/sec |
| Consensus Synthesis | 0.003 ms | ~333,333 ops/sec |
| Full Distributed Cycle (LOW) | 0.742 ms | ~1,347 cycles/sec |
| Full Distributed Cycle (STANDARD)| 0.816 ms | ~1,225 cycles/sec |
| Full Distributed Cycle (HIGH) | 0.844 ms | ~1,184 cycles/sec |
| Peak Memory Usage | **4.46 MB** | Extremely lightweight |

**Weight Fingerprint Verification**:
- Pre-execution SHA-256: `f8c46cc81cb782d8935986808bd60dcec9d5ac346dc9a23e55e2ab33d6ff8272`
- Post-execution SHA-256: `f8c46cc81cb782d8935986808bd60dcec9d5ac346dc9a23e55e2ab33d6ff8272`
- Mutation Detected: `False` (Verified identical).

---

## 17. Scope & Boundary Matrix

| Capability | Status | Implementation Details |
|:---|:---:|:---|
| Node Model & Identity | **IMPLEMENTED & TESTED** | `chakrview/cognition/distributed/models.py` |
| Transport Abstraction | **IMPLEMENTED & TESTED** | `chakrview/cognition/distributed/transport.py` |
| In-Process Loopback Transport | **IMPLEMENTED & TESTED** | `LoopbackTransport` with fault injection |
| Message Envelope & Integrity | **IMPLEMENTED & TESTED** | SHA-256 canonical hashing + HMAC signing |
| Replay Protection Tracker | **IMPLEMENTED & TESTED** | Bounded tracking, TTL, nonce, hop limit |
| Deterministic Task Router | **IMPLEMENTED & TESTED** | Technical locality and capability-based |
| Resilience & Circuit Breakers | **IMPLEMENTED & TESTED** | Exponential backoff, 3-failure trip |
| Resource Adaptations | **IMPLEMENTED & TESTED** | LOW, STANDARD, HIGH profiles |
| Minority Evidence Preservation | **IMPLEMENTED & TESTED** | Anti-majority evidence aggregation |
| Capability Gate Enforcement | **IMPLEMENTED & TESTED** | Zero remote agent authority |
| Weight Immutability Checks | **IMPLEMENTED & TESTED** | Fail-closed pre/post SHA-256 check |
| Sanitized Distributed Trace | **IMPLEMENTED & TESTED** | Redacted high-level audit telemetry |
| Socket / TCP Physical Transport | **DEFERRED** | Out of scope for Step 27 |
| gRPC / HTTP2 Transport | **DEFERRED** | Out of scope for Step 27 |
| Asymmetric Ed25519 PKI | **DEFERRED** | Extensible interface defined; Step 27 uses HMAC |
| External Distributed Deployment | **DEFERRED** | Reserved for future physical network steps |
