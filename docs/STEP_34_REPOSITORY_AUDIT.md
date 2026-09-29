# STEP 34 — REPOSITORY AUDIT & ARCHITECTURAL FOUNDATION

**ChakrView Cognitive Engine Architecture**  
**Document Reference**: `docs/STEP_34_REPOSITORY_AUDIT.md`  
**Phase**: Phase 0 Pre-Implementation Audit  
**Classification**: System Persistence, Failure Recovery & Multi-Node Runtime  
**Status**: COMPLETED

---

## 1. Executive Summary & Purpose

This audit establishes the pre-implementation foundation for **Step 34: Multi-Node Federation Runtime, Durable Security State & Failure Recovery**. 
Building on the ratified Step 33 baseline (`7ea9d8c`), this audit reviews existing subsystems across federation, peering, transport, capabilities, audit logging, and state machines to design a crash-recoverable, bounded, deterministic persistence and runtime layer.

The core architectural requirement is:
$$\text{RUNTIME} \neq \text{AUTHORITY}, \quad \text{PERSISTENCE} \neq \text{AUTHORITY}, \quad \text{JOURNAL} \neq \text{AUTHORITY}, \quad \text{RECOVERY} \neq \text{AUTHORITY}$$

Crash recovery must reconstruct exact security state without authority escalation, without dynamic consensus, and with $\Delta W = 0$ over frozen neural weights.

---

## 2. Inventory of Federation & Peering Subsystems

### 2.1 Federation Subsystem (`chakrview/cognition/federation/`)
- `models.py`: Defines immutable `FederationEngineIdentity`, `SecurityStateVersion`, sub-state digests (`ReplayStateDigest`, `TrustStateDigest`, `RevocationStateDigest`, `PeerStateDigest`), master `FederationSecurityStateDigest`, handshake requests/responses, and synchronization messages (`ReplaySyncMessage`, `TrustSyncMessage`, `RevocationSyncMessage`).
- `identity.py`: `FederationEngineIdentityProvider` creates and validates canonical engine identities via deterministic SHA-256 fingerprinting.
- `state.py`: `FederationStateManager` maintains the local monotonic `SecurityStateVersion`, advances logical epochs, computes canonical composite state digests, and evaluates remote versions according to Step 33 Phase 9 rules (Cases A through E).
- `handshake.py`: `FederationHandshakeManager` validates mutual protocol versions and digest matching.
- `replay_sync.py`: `ReplayStateSynchronizer` maintains advisory replay floors and merges message IDs into bounded caches.
- `trust_sync.py`: `TrustStateSynchronizer` enforces ceiling validation and blocks self-escalation claims.
- `revocation_sync.py`: `RevocationStateSynchronizer` deduplicates revocations and cascades local invalidations.
- `coordinator.py`: `DistributedFederationCoordinator` orchestrates engine registration, handshakes, and state exchange.

### 2.2 Peering Subsystem (`chakrview/cognition/peering/`)
- `engine.py`: `CrossZoneFederationEngine` (aliased as `FederationEngine`) coordinates local Ed25519 keypairs, challenge-response authentication, session creation and renewal, peer registry lookups, capability gating, and neural weight invariance checks.
- `session.py`: `SecurePeerSession` manages session lifecycles, session keys (`SessionKeyMetadata`), monotonic sequence verification (`record_and_check_sequence()`), bounded message ID replay caches, and terminal invalidations.
- `crypto.py`: `Ed25519PrivateKeyWrapper`, `Ed25519PublicKeyWrapper`, and `CryptographicPeerIdentity` managing key lifecycle states (`ACTIVE`, `ROTATING`, `REVOKED`, `EXPIRED`) and tracking `retired_keys`.
- `registry.py`: `PeerRegistry` managing peer registrations, capacities ($\le 32$ total, $\le 16$ per zone), and peer revocations.
- `revocation.py`: `RevocationManager` tracking active revocation records.
- `audit.py`: `BoundedAuditLogger` providing thread-safe FIFO ring buffer ($\le 1,000$ entries).

### 2.3 Transport Security Subsystem (`chakrview/cognition/transport/security/`)
- `certificates.py`: Hermetic X.509 PKI builder and `CertificateRevocationRegistry`.
- `binding.py`: `PeerCertificateBinder` binding authenticated TLS certificates to cryptographic peer identities.
- `tls.py`: Context builder for in-memory and TLS connections.

### 2.4 Capability Subsystem (`chakrview/capability/`)
- `gate.py`: `CapabilityGate` mediating all execution requests. Prohibits memory, retrieved data, or remote claims from granting authority.
- `registry.py`: `CapabilityRegistry` tracking registered descriptors and operational statuses.

---

## 3. Existing Persistence Abstractions & Gap Analysis

### 3.1 Existing Repository Persistence Patterns
Across the wider repository, persistence mechanisms exist in specific non-federated modules:
1. `chakrview/memory/storage.py`: `ContinualMemoryStorage` serializes working, episodic, and semantic memory into versioned JSON dictionaries with explicit schema validation (`SCHEMA_VERSION = "24.1"`). Prohibits pickle; uses deterministic encoding.
2. `chakrview/state/snapshot.py`: `CognitiveStateSnapshotManager` takes point-in-time JSON snapshots of cognitive state vectors with SHA-256 integrity digests.
3. `chakrview/training/checkpoint.py`: Checkpoint saving for pre-training weights and optimizer states (Step 05).

### 3.2 Identified Persistence Gaps in Federation
Prior to Step 34, **the federation and peering subsystems are entirely in-memory**:
- If a `CrossZoneFederationEngine` crashes or restarts:
  1. Engine identity is re-created, but ephemeral runtime state is lost.
  2. Registered peers and cryptographic identities disappear.
  3. Active and revoked sessions are wiped; sequence numbers reset to 0.
  4. Revocation records (`RevocationRecord`, `KeyRevocationRecord`, `CertificateRevocationRecord`) are lost, creating a critical vulnerability where previously revoked peers or certificates could be re-admitted after a crash.
  5. Replay caches are cleared, opening a window for replay attacks across engine restarts.
  6. Trust grants and expiration boundaries reset.

Step 34 directly solves these gaps through durable write-ahead journaling and periodic snapshots.

---

## 4. Lifecycle State Machines Across Subsystems

The audit maps the exact state machines that must be preserved, reconstructed, and enforced across failure recovery:

| Subsystem | State Machine Enum | Allowed States | Terminal States | Recovery Invariant |
|:---|:---|:---|:---|:---|
| **Peer Session** | `SessionStatus` | `INITIATED`, `AUTHENTICATING`, `ACTIVE`, `RENEWING`, `EXPIRED`, `TERMINATED`, `REVOKED`, `FAILED` | `EXPIRED`, `TERMINATED`, `REVOKED`, `FAILED` | Terminal states can NEVER transition back to `ACTIVE`. Expired sessions remain fail-closed. |
| **Session Key** | `SessionKeyState` | `CREATED`, `ACTIVE`, `ROTATING`, `EXPIRED`, `REVOKED` | `EXPIRED`, `REVOKED` | Revoked/expired session keys cannot be reused. |
| **Peer Identity Key** | `KeyLifecycleState`| `ACTIVE`, `ROTATING`, `REVOKED`, `EXPIRED` | `REVOKED`, `EXPIRED` | `REVOKED` keys cannot be rotated or restored. Retired keys tracked in `retired_key_fingerprints`. |
| **Peer Discovery** | `DiscoveryStatus` | `DISCOVERED`, `VERIFIED`, `REJECTED`, `REVOKED`, `EXPIRED` | `REJECTED`, `REVOKED`, `EXPIRED` | Revoked peers reject new session establishments. |
| **Trust Grant** | `TrustStatus` | `PENDING`, `ACTIVE`, `EXPIRED`, `REVOKED` | `EXPIRED`, `REVOKED` | Revoked or expired trust grants cannot authorize wire envelopes. |
| **Certificate** | CRL Entry | `ACTIVE`, `REVOKED` | `REVOKED` | Revoked certificates in `CertificateRevocationRegistry` stay permanently revoked. |
| **Federation Coordination**| `HandshakeStatus`| `ACCEPTED`, `REJECTED`, `INCOMPATIBLE_PROTOCOL`, `DIGEST_CONFLICT` | `REJECTED`, `INCOMPATIBLE_PROTOCOL` | Incompatible engines cannot coordinate. |
| **Engine Runtime (Step 34)**| `EngineRuntimeStatus`| `INITIALIZING`, `RUNNING`, `RECOVERING`, `STOPPING`, `STOPPED`, `FAILED` | `STOPPED`, `FAILED` | Bounded runtime lifecycle. |
| **Engine Health (Step 34)** | `EngineHealthStatus` | `UNKNOWN`, `HEALTHY`, `DEGRADED`, `UNREACHABLE`, `RECOVERING`, `QUARANTINED`, `TERMINATED` | `TERMINATED` | `UNREACHABLE != REVOKED`. |

---

## 5. Security-Critical State Required for Recovery

To achieve deterministic crash recovery, the following data must be durably captured:

### 5.1 Must Persist
1. **Engine Identity Metadata**: `engine_id`, `zone_id`, `created_epoch`, `identity_fingerprint`, `protocol_version`.
2. **Security State Version**: Current `version`, `epoch`, `engine_id`.
3. **Peer Registrations**: Peer identities, zone IDs, registered epoch, discovery status.
4. **Peer Cryptographic Identity**: Public key hex, key fingerprint, key lifecycle state, rotation history, retired key fingerprints.
5. **Trust Grants**: Grant ID, trust level, status, granted epoch, expires epoch, permitted scopes.
6. **Active & Revoked Sessions**: Session ID, remote peer ID, status, created epoch, expires epoch, renewal count, sequence floor, highest sequence seen.
7. **Replay Floors**: Highest sequence numbers and message ID hashes per session.
8. **Revocation Records**: Target type (PEER, SESSION, CERTIFICATE, KEY, TRUST), target ID, reason, revoked epoch, revocation hash.
9. **Certificate Bindings & Revocations**: Certificate serial/fingerprint, subject, bound peer IDs, CRL revoked entries.
10. **State Digests**: Composite and facet digests for integrity cross-checking.

### 5.2 Strictly Prohibited from Persistence
- Ed25519 private keys in plaintext or raw seed bytes.
- Ephemeral shared session keys or transport secrets.
- Model weights, biases, or neural core parameters ($\Delta W = 0$).
- Hidden activations, KV caches, or private attention states.
- Scratchpad reasoning traces or continual memory raw text.
- Capability execution secrets, API keys, or caller tokens.

---

## 6. Architecture for Write-Ahead Journal & Recovery

### 6.1 Cryptographically Chained Write-Ahead Journal
Every mutation to security state must be durably appended before the in-memory mutation is finalized:
$$\text{Entry}_N = \left\langle \text{seq}_N, \text{epoch}_N, \text{type}_N, \text{payload}_N, \text{prev\_digest}_{N-1}, \text{digest}_N \right\rangle$$
where:
$$\text{digest}_N = \text{SHA-256}\left(\text{canonical\_json}(\text{payload}_N) \mathbin{\Vert} \text{prev\_digest}_{N-1}\right)$$
For $N = 1$, $\text{prev\_digest}_0 = \text{"0"}^{64}$ (genesis digest).

### 6.2 Snapshot + Journal Compaction
To bound recovery time and prevent unbounded journal growth:
- Snapshots capture the complete normalized state at a known journal sequence index $K$.
- Recovery loads the latest verified Snapshot $K$, checks its SHA-256 integrity hash, and replays journal entries from $K + 1$ to $M$.
- If the journal or snapshot shows cryptographic divergence or tampering, recovery immediately fails closed (`RECOVERY_FAILED_CLOSED`).

---

## 7. Extension Points Identified

1. **`chakrview/cognition/federation/persistence/`**: New package housing store interfaces, memory and SQLite durable implementations, journal engine, and snapshot managers.
2. **`chakrview/cognition/federation/recovery.py`**: Housing `FederationRecoveryManager` implementing the 11 recovery steps and integrity validation.
3. **`chakrview/cognition/federation/runtime.py`**: Housing `FederationRuntime` managing engine registration, lifecycle, health states, and coordination scheduling.
4. **`chakrview/cognition/peering/engine.py`**: Hooks for persistence integration (`CrossZoneFederationEngine` write-ahead journaling on peer registration, key rotation, session updates, and revocations).
5. **`chakrview/cognition/peering/models.py`**: 15 new audit event types for Step 34.

---

## 8. Conclusion & Clearance to Implement

The audit confirms that the Step 33 coordination baseline is modular and cleanly extensible. The persistence, journaling, and recovery layer can be integrated without modifying the neural core, without introducing distributed consensus, and without compromising sovereign local authority boundaries.

Clearance granted to proceed to **Phase 1: Durable Security State**.
