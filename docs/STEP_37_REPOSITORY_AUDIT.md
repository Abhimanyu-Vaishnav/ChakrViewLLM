# CHAKRVIEW STEP 37 — REPOSITORY AUDIT

**Component**: Distributed Resource & Capability Advertisement  
**Date**: September 29, 2026  
**Baseline**: Step 36 Ratified (Commit: `07eaf47` / `722cfae`, 1,045 tests passing)  
**Objective**: Audit existing architecture and define integration plan for Step 37 without modifying the frozen neural core or violating established security boundaries.

---

## 1. Existing Reusable Components

The ChakrView repository already contains extensive foundations that directly support Step 37:

1. **Sovereign Capability Subsystem (`chakrview/capability/`)**:
   - `CapabilityDescriptor` (`contract.py`): Strongly typed manifest specifying `capability_id`, `name`, `version`, `category`, `risk_classification`, and description.
   - `CapabilityCategory` (`contract.py`): Closed enumeration (`SOFTWARE`, `EDGE_DEVICE`, `PHYSICAL_DEVICE`, `SENSOR`, `ACTUATOR`, `UTILITY`).
   - `RiskClassification` (`contract.py`): Risk tiering (`READ_ONLY`, `COMPUTE`, `EXTERNAL_WRITE`, `PHYSICAL_ACTION`, `HIGH_IMPACT`).
   - `CapabilityRegistry` (`registry.py`): Local registry of registered, executable capability instances.
   - `CapabilityGate` (`gate.py`): Sovereign local authorization gate executing capabilities under strict policy constraints.
   - `EnvironmentProfile` (`environment.py`): Hardware constraint specification (`platform`, `architecture`, `memory_limit_mb`, `compute_limit_ms`, `supported_capability_ids`).

2. **Secure Transport Subsystem (`chakrview/cognition/federation/transport/`)**:
   - `FederationMessageEnvelope` (`models.py`): Canonical signed wire envelope with Ed25519 signature, SHA-256 payload digest, monotonic sequence numbers, session IDs, and tenant boundaries.
   - `FederationMessageCodec` (`codec.py`): Deterministic canonical UTF-8 JSON encoder/decoder with recursive prohibited keyword/type scanning (`_assert_no_prohibited_content`).
   - `FederationMessageFramer` (`framing.py`): Length-prefixed binary framing with 1 MB pre-allocation ceiling.
   - `FederationChannel` (`channel.py`): 9-state fail-closed channel lifecycle machine (`DISCONNECTED`, `CONNECTING`, `AUTHENTICATING`, `ESTABLISHED`, `DEGRADED`, `CLOSING`, `CLOSED`, `QUARANTINED`, `REVOKED`).
   - `FederationMessageDispatcher` (`dispatcher.py`): Thread-safe handler dispatch, error containment, bounded timeouts, and local `CapabilityGate` delegation.
   - `FederationMessageType` (`models.py`): Typed message enumeration over federation channels.

3. **Discovery & Membership Subsystem (`chakrview/cognition/federation/discovery/`)**:
   - `FederationNodeEndpoint` (`models.py`): Validated network address, port, protocol, and expected certificate fingerprint.
   - `FederationNodeCandidate` and `FederationNodeMembership` (`models.py`): 8-state membership lifecycle machine (`DISCOVERED`, `PENDING_AUTHENTICATION`, `AUTHENTICATED`, `MEMBER`, `SUSPENDED`, `QUARANTINED`, `REVOKED`, `TERMINATED`).
   - `FederationHeartbeatMonitor` (`membership.py`): Node liveness tracking enforcing `UNREACHABLE != REVOKED`.

4. **Peering & Cryptographic Identity (`chakrview/cognition/peering/`)**:
   - `PeerIdentity` (`models.py`): Immutable peer, zone, and organization identifier.
   - `Ed25519PrivateKeyWrapper`, `Ed25519PublicKeyWrapper` (`crypto.py`): Deterministic cryptographic key wrappers.
   - `CrossZoneFederationEngine` (`engine.py`): Central coordinator linking discovery, membership, transport, and capability gating.
   - `AuditLogger`, `AuditEventType` (`models.py`): Tamper-evident structured audit logging.

5. **Durable Persistence & Recovery (`chakrview/cognition/federation/persistence/` & `recovery.py`)**:
   - `SecurityStateJournal`: Append-only, cryptographically chained write-ahead journal.
   - `SecurityStateStore` (`base.py`, `memory.py`, `sqlite.py`): Atomic snapshot and journal persistence.
   - `FederationRecoveryManager` (`recovery.py`): Cold crash recovery replaying journal mutations.
   - `FederationRuntime` (`runtime.py`): Engine lifecycle coordinator with health tracking and secure rejoin.

---

## 2. Existing Interfaces Step 37 Must Integrate With

Step 37 must seamlessly integrate with the following existing interfaces:

1. **`FederationMessageType` (`chakrview/cognition/federation/transport/models.py`)**:
   - Extend with:
     - `RESOURCE_ADVERTISEMENT`: Broadcast/exchange of resource and capability profiles.
     - `RESOURCE_QUERY`: Query for peers matching capability or resource criteria.
     - `RESOURCE_QUERY_RESPONSE`: Response containing relevant peer resource advertisements.

2. **`FederationMessageDispatcher` (`chakrview/cognition/federation/transport/dispatcher.py`)**:
   - Register handlers for `RESOURCE_ADVERTISEMENT`, `RESOURCE_QUERY`, and `RESOURCE_QUERY_RESPONSE` to route incoming messages through the sovereign local resource registry.

3. **`CrossZoneFederationEngine` (`chakrview/cognition/peering/engine.py`)**:
   - Attach `resource_manager` property managing local discovery, capability publishing, and peer resource caching.

4. **`FederationRuntime` (`chakrview/cognition/federation/runtime.py`)**:
   - Delegate `resource_manager` access for multi-node runtime operations.

5. **`AuditEventType` (`chakrview/cognition/peering/models.py`)**:
   - Add Step 37 audit events:
     - `RESOURCE_PROFILE_DETECTED`: Local hardware resource discovery.
     - `RESOURCE_ADVERTISEMENT_PUBLISHED`: Local advertisement broadcast.
     - `RESOURCE_ADVERTISEMENT_RECEIVED`: Remote advertisement received.
     - `RESOURCE_ADVERTISEMENT_ACCEPTED`: Remote claim validated and registered.
     - `RESOURCE_ADVERTISEMENT_REJECTED`: Invalid, unauthorized, or malformed advertisement rejected.
     - `RESOURCE_ADVERTISEMENT_STALE`: Stale advertisement marked unavailable.
     - `RESOURCE_QUERY_DISPATCHED`: Local query sent to federation.
     - `RESOURCE_QUERY_RECEIVED`: Remote query received.

6. **`JournalEntryType` (`chakrview/cognition/federation/persistence/models.py`)**:
   - Add Step 37 durable journal event types:
     - `CAPABILITY_ADVERTISED`: Durable registration of declared capabilities.
     - `RESOURCE_POLICY_UPDATED`: Durable update to local resource sharing policy.

---

## 3. Potential Architectural Conflicts & Strict Boundaries

To ensure complete architectural coherence, the following potential conflicts have been identified and strictly bounded:

1. **Conflict: Conflating Resource Advertisement with Authorization / Permission**:
   - *Risk*: A remote node advertising "16 CPU cores and 1 GPU" could be mistakenly treated as granting permission to execute jobs on that node, or claiming authority over local jobs.
   - *Resolution*: Enforce strict boundary:
     `RESOURCE_DISCOVERY != RESOURCE_AUTHORIZATION`
     `RESOURCE_ADVERTISEMENT != RESOURCE_PERMISSION`
     `RESOURCE_ADVERTISEMENT != EXECUTION_AUTHORITY`
     `RESOURCE_VISIBILITY != RESOURCE_CONTROL`
     `PEER_RESOURCE_CLAIM != LOCAL_RESOURCE_FACT`
     `LOCAL_RESOURCE_POLICY > REMOTE_RESOURCE_CLAIM`
   - Advertisements are untrusted, informational claims. They confer zero execution authority. Task scheduling and workload execution are strictly out of scope for Step 37.

2. **Conflict: Volatile Utilization vs Durable Security State**:
   - *Risk*: Persisting dynamic resource utilization (e.g., instantaneous CPU load, available RAM at millisecond $t$) into write-ahead journals and disk snapshots would cause journal bloat and stale state recovery.
   - *Resolution*: Strict bifurcation:
     - *Durable*: Node capability declarations, hardware classes, sharing policy limits, static capacity.
     - *Volatile*: Instantaneous utilization, current free memory, transient load.
     - Volatile metrics are NEVER written to the write-ahead journal or disk snapshots. Upon restart, volatile metrics are freshly observed.

3. **Conflict: Heartbeat Overload vs Separation of Concerns**:
   - *Risk*: Coupling resource advertisements directly into periodic heartbeats would bloat heartbeat packets and conflate liveness with resource state.
   - *Resolution*: Heartbeats (`HEARTBEAT` / `HEARTBEAT_ACK`) remain lean liveness pings. Resource advertisements travel over dedicated envelopes (`RESOURCE_ADVERTISEMENT`). However, when a heartbeat monitor detects an `UNREACHABLE` node, the resource registry marks that peer's resource advertisement as `STALE` or `SUSPENDED`.

4. **Conflict: Non-Portable / Heavyweight Hardware Dependencies**:
   - *Risk*: Introducing dependencies like `psutil`, `pynvml`, or Linux-specific `/proc` parsers would break ChakrView on Raspberry Pi, Windows, macOS, or minimal environments.
   - *Resolution*: Use standard library discovery (`os`, `sys`, `platform`, `multiprocessing`) behind modular detector interfaces with clean fallback mocks for accelerators. No external binary dependencies.

5. **Conflict: Privacy & Secret Leakage**:
   - *Risk*: Hardware discovery could leak usernames, home directory paths, MAC addresses, CPU serial numbers, or internal network topology.
   - *Resolution*: Privacy filter sanitizes all hardware profiles. Only coarse, generic descriptors (CPU arch, core count, total RAM in MB, generic accelerator class) are included. The existing `_assert_no_prohibited_content` scanner enforces zero private keys, secrets, model weights, or tensors in payloads.

---

## 4. Existing Security Boundaries

The implementation strictly inherits and upholds all established security boundaries:
- `LOCAL_AUTHORITY > PEER_AUTHORITY`: Remote nodes cannot dictate local policy.
- `REVOKED -> ABSORBING TERMINAL STATE`: Revoked nodes are barred from publishing or querying resources.
- `QUARANTINED -> TRAFFIC HALTED`: Quarantined nodes cannot advertise or query resources.
- Tenant isolation: Cross-tenant resource visibility is blocked by default-deny policy.
- Fail-closed validation: Malformed, unsigned, or mismatched digest envelopes fail closed.
- $\Delta W = 0$: Neural core weights are strictly immutable.

---

## 5. Existing Persistence & Recovery Mechanisms

- `SecurityStateJournal`: Sequential monotonic sequence numbers and SHA-256 chaining.
- `DurableSecuritySnapshot`: Snapshot versions and journal offsets.
- `FederationRecoveryManager`: Cold start applies latest snapshot, replays verified journal entries.
- Step 37 integrates by persisting only stable capability declarations and policy updates, leaving dynamic resource metrics volatile.

---

## 6. Existing Test Infrastructure

- Test framework: `pytest`.
- Total baseline: 1,045 tests passing across 81 test files.
- Test conventions: Hermetic, CPU-first, fast execution (<40s for full suite).
- Step 37 will add `tests/test_federation_resources.py` with comprehensive coverage across all 28 required test vectors.

---

## 7. Existing Benchmark Infrastructure

- Benchmark framework: `tracemalloc` + `time.perf_counter`.
- Metrics: Mean latency, P95 latency, min/max, throughput (ops/sec), peak memory delta.
- Neural verification: Deterministic SHA-256 weight hash check before and after benchmark run.
- Step 37 will add `scripts/benchmark_federation_resources.py` recording to `docs/STEP_37_BENCHMARK_RESULTS.json`.

---

## 8. Files That Must NOT Be Modified

To preserve system integrity, the following modules must remain untouched:
- `chakrview/brain/*`: Frozen neural architecture (ChakrMicro: 3,443,136 params, vocab 4096, context 512, $\Delta W = 0$).
- `chakrview/training/*`: Untouched.
- `chakrview/tokenization/*`: Untouched.
- `chakrview/state/*`: Untouched.

---

## 9. Implementation Plan & Directory Structure

To house the Step 37 functionality cleanly, we will create `chakrview/cognition/federation/resources/`:
1. `models.py`:
   - Hardware descriptors: `CPUResource`, `MemoryResource`, `AcceleratorResource`, `StorageResource`, `PlatformResource`, `NodeResourceProfile`.
   - Capability descriptors: `FederationCapabilityProfile`, `ExecutionType`.
   - Advertisement models: `ResourceAdvertisement`, `AdvertisementFreshness`, `ResourceSharingPolicy`.
2. `discovery.py`:
   - Hardware discovery provider: `LocalResourceDetector` (platform-neutral CPU, RAM, accelerator, OS detection).
3. `registry.py`:
   - `FederationResourceRegistry`: Manages local claims vs peer claims, freshness tracking, stale detection, capability filtering, and tenant isolation.
4. `manager.py`:
   - `FederationResourceManager`: High-level manager coordinating local discovery, advertisement publication, transport dispatch, and query handling.
5. `errors.py`:
   - Typed error hierarchy rooted at `FederationResourceError`.
6. `__init__.py`: Clean public API export.

And integrate into:
- `peering/models.py`: Audit events.
- `transport/models.py`: Message types.
- `persistence/models.py`: Journal entry types.
- `peering/engine.py` & `federation/runtime.py`: Resource manager property attachment.

Audit completed. Ready to proceed to Phase 2 (Architecture/Design) and Phase 3 (Models/Interfaces).
