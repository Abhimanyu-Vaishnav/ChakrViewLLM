# CHAKRVIEW STEP 34 — THREAT MODEL

## Multi-Node Federation Runtime, Durable Security State & Failure Recovery

---

### Scope & Architectural Boundaries

Step 34 transitions ChakrView from in-memory distributed coordination into a **bounded, crash-recoverable operational federation runtime**. This document analyzes threats against the persistence subsystem (`chakrview.cognition.federation.persistence`), the write-ahead security journal, snapshot storage, crash recovery (`FederationRecoveryManager`), runtime lifecycle (`FederationRuntime`), engine health tracking, and the 12-step secure rejoin protocol.

The core architectural boundaries enforced throughout this analysis:
* `LOCAL_AUTHORITY > PEER_AUTHORITY`
* `LOCAL_REPLAY_PROTECTION > REMOTE_REPLAY_ADVISORY`
* `LOCAL_REVOCATION > REMOTE_ACTIVE_STATE`
* `RUNTIME != AUTHORITY`
* `PERSISTENCE != AUTHORITY`
* `JOURNAL != AUTHORITY`
* `RECOVERY != AUTHORITY`
* `ENGINE_IDENTITY != AUTHORITY`
* `ENGINE_REJOIN != TRUST_GRANT`
* `ENGINE_REJOIN != CAPABILITY_ESCALATION`
* `NETWORK_FAILURE != AUTOMATIC_REVOCATION`
* `STATE_RECOVERY != MODEL_RECOVERY`
* `SECURITY_STATE != NEURAL_STATE`
* `ΔW = 0`

---

### Threat Analysis Matrix

---

#### 1. Journal Tampering

* **Threat Description**: An adversary with filesystem or storage access modifies payload contents, epoch numbers, or entry types within an existing journal record to forge historical security events (e.g., removing a revocation event or injecting a fictitious peer registration).
* **Attack Boundary**: Durable disk/database store (`SecurityStateStore` / SQLite / filesystem).
* **Detection**: Each journal entry contains a cryptographic SHA-256 digest computed across its canonical JSON payload, sequence number, epoch, and previous entry digest ($E_N.\text{digest} = \text{SHA256}(E_N.\text{canonical} \mathbin{\Vert} E_{N-1}.\text{digest})$). During recovery or load, `SecurityStateJournal.verify_chain` recalculates every digest and detects bitwise modifications.
* **Mitigation**: Strict SHA-256 hash chaining anchored at genesis digest `0000...0000` (or snapshot digest).
* **Fail-Closed Behavior**: `JournalCorruptionError` raised; recovery aborts immediately via `RECOVERY_FAILED_CLOSED`. Engine remains inactive; no corrupted state enters in-memory registry or session pool.
* **Test Coverage**: `tests/test_federation_runtime.py::test_06_journal_corruption_detection`, `test_11_recovery_fails_closed_on_corrupted_journal`.

---

#### 2. Journal Truncation

* **Threat Description**: An attacker deletes trailing journal records from the persistence store to conceal recent administrative actions, such as certificate revocations, session terminations, or peer revocations.
* **Attack Boundary**: Persistence storage medium between snapshot point and current head.
* **Detection**: Journal entries require strictly monotonic, gapless sequence numbers ($S_k = S_{k-1} + 1$). If intermediate records are removed or if sequence numbering regresses, verification fails.
* **Mitigation**: Gapless sequence enforcement and previous-digest chaining prevents selective middle-record truncation. If tail entries are truncated, the recovered state version and composite digest will mismatch any external coordinator handshake or persisted snapshot digest reference.
* **Fail-Closed Behavior**: `JournalSequenceError` or `JournalTruncationError` is raised. Engine fails closed.
* **Test Coverage**: `tests/test_federation_runtime.py::test_07_journal_sequence_regression_detection`, `test_11_recovery_fails_closed_on_corrupted_journal`.

---

#### 3. Journal Replay

* **Threat Description**: An attacker presents or injects historical journal records from an earlier epoch or previous execution run into a freshly initialized engine to reinstate obsolete configurations.
* **Attack Boundary**: Storage initialization and journal replay pipeline.
* **Detection**: Genesis entry verification ensures the journal strictly begins from `JOURNAL_GENESIS_DIGEST` (or snapshot offset). Entries with sequence numbers already satisfied by the snapshot are excluded.
* **Mitigation**: Recovery only replays records where `sequence_number > snapshot.journal_offset`.
* **Fail-Closed Behavior**: Out-of-order, duplicated, or replayed sequences raise `JournalSequenceError` and abort recovery.
* **Test Coverage**: `tests/test_federation_runtime.py::test_08_journal_broken_chain_detection`, `test_09_clean_recovery_from_snapshot_and_journal`.

---

#### 4. Snapshot Corruption

* **Threat Description**: Data corruption, bit-rot, or malicious editing alters the contents of a serialized `DurableSecuritySnapshot` (e.g., modifying peer registrations or resetting replay floors).
* **Attack Boundary**: Snapshot persistence files / SQLite tables.
* **Detection**: Every snapshot contains a canonical SHA-256 `snapshot_digest` calculated over all metadata fields excluding the digest itself (`snapshot.seal()`). When loaded, `snapshot.verify_integrity()` computes and validates the checksum.
* **Mitigation**: Checksum validation before any snapshot field is inspected or applied to in-memory state.
* **Fail-Closed Behavior**: `SnapshotCorruptionError` raised; recovery aborts via `RECOVERY_FAILED_CLOSED`. Corrupted snapshot is not loaded.
* **Test Coverage**: `tests/test_federation_runtime.py::test_01_snapshot_save_and_load`, `test_10_recovery_fails_closed_on_corrupted_snapshot`.

---

#### 5. Snapshot Rollback

* **Threat Description**: An attacker replaces a newer snapshot ($N$) with an older valid snapshot ($N-1$) to resurrect expired trust grants or revert recorded revocations.
* **Attack Boundary**: Snapshot storage retrieval.
* **Detection**: Snapshot versioning ($V_S$), journal offset checks, and cross-peer state version exchange during handshake detect regressive state versions.
* **Mitigation**: The persistence store loads the latest verified snapshot by maximum `snapshot_version` and re-validates the subsequent journal chain. If journal entries exist beyond the older snapshot, they are replayed. Furthermore, local terminal revocation records are permanent and never deleted.
* **Fail-Closed Behavior**: Replaying journal restores all revocations. If state version is older than remote coordinator during rejoin, remote marks engine state as `REMOTE_STATE_STALE` and refuses to downgrade local security state.
* **Test Coverage**: `tests/test_federation_runtime.py::test_03_snapshot_schema_version_validation`, `test_18_rejoin_does_not_grant_authority`.

---

#### 6. Stale Engine Rejoin

* **Threat Description**: An engine partitioned or halted at Epoch 1 rejoins at Epoch 10 and attempts to use its obsolete security state to overwrite active node state or process federated requests.
* **Attack Boundary**: Multi-node coordination handshake (`execute_rejoin`).
* **Detection**: Phase 5 of rejoin exchanges `SecurityStateVersion` (epoch and version number). If local engine has higher version/epoch, remote is flagged stale.
* **Mitigation**: Stale remote engines cannot overwrite local state. Local engine enforces `LOCAL_AUTHORITY > PEER_AUTHORITY` and pushes local terminal revocations to the rejoining node.
* **Fail-Closed Behavior**: Rejoining node receives local revocation and policy boundaries; capabilities are NOT automatically restored.
* **Test Coverage**: `tests/test_federation_runtime.py::test_17_secure_rejoin_success`, `test_19_local_revocation_persists_across_remote_rejoin`.

---

#### 7. Split-Brain Recovery

* **Threat Description**: Two partition halves advance independently, incrementing version numbers with divergent mutation sets, resulting in identical version numbers but conflicting state digests.
* **Attack Boundary**: Inter-engine synchronization post-rejoin.
* **Detection**: Step 33 & 34 composite digest verification: comparing SHA-256 composite digests of local vs remote state.
* **Mitigation**: `STATE_DIGEST_CONFLICT` is triggered when versions match but digests diverge. Coordination fails closed: zero automatic overwrite, zero blind merging.
* **Fail-Closed Behavior**: Divergent state causes `StateDigestConflictError`. Engines refuse state synchronization until administrative reconciliation occurs.
* **Test Coverage**: `tests/test_federation_runtime.py::test_17_secure_rejoin_success`, `tests/test_distributed_federation.py::test_06_conflict_detection_divergent_digests`.

---

#### 8. Duplicate Event Replay

* **Threat Description**: An attacker duplicates a journal record within the store (e.g., duplicate sequence number) to force double-application of state mutations.
* **Attack Boundary**: Journal storage replay.
* **Detection**: Strict sequence monotonicity check ($S_i == S_{i-1} + 1$).
* **Mitigation**: Sequence numbers must be strictly consecutive. Identical or regressive sequence numbers are detected instantly during verification.
* **Fail-Closed Behavior**: `JournalSequenceError` emitted; recovery halts.
* **Test Coverage**: `tests/test_federation_runtime.py::test_07_journal_sequence_regression_detection`.

---

#### 9. Sequence Regression

* **Threat Description**: Journal records arrive with decreasing sequence numbers or non-monotonic ordering.
* **Attack Boundary**: Journal ingestion pipeline.
* **Detection**: `SecurityStateJournal.verify_chain` validates $S_i > S_{i-1}$ and $S_i == S_{i-1} + 1$.
* **Mitigation**: Immediate rejection of out-of-order entries.
* **Fail-Closed Behavior**: `JournalSequenceError` raised; recovery fails closed.
* **Test Coverage**: `tests/test_federation_runtime.py::test_07_journal_sequence_regression_detection`.

---

#### 10. Trust-State Rollback

* **Threat Description**: An engine rebooting from disk attempts to restore trust grants that had previously expired before the restart or crash.
* **Attack Boundary**: Recovery trust restoration.
* **Detection**: `FederationRecoveryManager` checks logical time during recovery: Step 6 explicitly executes `engine.registry.expire_peers(engine.current_epoch)`.
* **Mitigation**: Trust grants have explicit `expires_epoch` ceilings. Any grant where `expires_epoch <= current_epoch` is marked `EXPIRED`.
* **Fail-Closed Behavior**: Expired grants cannot be renewed or used for capability delegation.
* **Test Coverage**: `tests/test_federation_runtime.py::test_20_rejoin_with_expired_trust_blocks_renewal`.

---

#### 11. Revocation Rollback

* **Threat Description**: An attacker restarts an engine using a snapshot created prior to a peer revocation, attempting to restore a revoked peer to active status.
* **Attack Boundary**: Recovery state reconstruction.
* **Detection**: Step 5 of recovery invokes `_enforce_terminal_invariants`: all revocations recorded in the snapshot or replayed journal entries are strictly enforced, and all associated sessions are revoked.
* **Mitigation**: Terminal states (`REVOKED`) are absorbing and cannot transition back to active.
* **Fail-Closed Behavior**: Revoked peers remain revoked. All active sessions associated with a revoked peer are terminated immediately.
* **Test Coverage**: `tests/test_federation_runtime.py::test_09_clean_recovery_from_snapshot_and_journal`, `test_19_local_revocation_persists_across_remote_rejoin`.

---

#### 12. Certificate-State Rollback

* **Threat Description**: An attacker uses crash recovery to un-revoke a compromised peer TLS/Ed25519 certificate.
* **Attack Boundary**: Certificate Revocation Registry restoration.
* **Detection**: Step 1 & 4 of recovery restore `certificate_revocations` from snapshot and journal into `CertificateRevocationRegistry`.
* **Mitigation**: Certificate revocation records are permanent. Re-checking `is_revoked(fingerprint)` returns `True`.
* **Fail-Closed Behavior**: Transport handshake immediately rejects revoked certificate fingerprints.
* **Test Coverage**: `tests/test_federation_runtime.py::test_21_certificate_revocation_remains_revoked_after_recovery`.

---

#### 13. Crash During Mutation

* **Threat Description**: Power outage or system crash occurs mid-mutation, after in-memory state changed but before the write-ahead journal append completed on disk.
* **Attack Boundary**: In-memory state vs durable journal synchronization boundary.
* **Detection**: In-memory state is lost upon process death. On restart, recovery reads only the durable store.
* **Mitigation**: Write-ahead invariant: unpersisted mutations do not survive restart. Durable store is the single source of truth for recovered state.
* **Fail-Closed Behavior**: Uncommitted in-memory mutations vanish completely; no partial or phantom state survives.
* **Test Coverage**: `tests/test_federation_runtime.py::test_12_crash_before_journal_append_does_not_survive`.

---

#### 14. Partial Persistence

* **Threat Description**: Process terminates in the middle of writing a snapshot or journal batch, leaving a half-written file or malformed JSON payload on disk.
* **Attack Boundary**: Storage write serialization.
* **Detection**: JSON deserialization failure, schema validation failure, or SHA-256 checksum mismatch.
* **Mitigation**: In SQLite, database writes execute inside transactions. In memory/file systems, snapshots are sealed and checked against `snapshot_digest`.
* **Fail-Closed Behavior**: `SnapshotCorruptionError` or `DurableSchemaError` raised. Recovery fails closed with `RecoveryFailedClosedError`.
* **Test Coverage**: `tests/test_federation_runtime.py::test_02_sqlite_persistence_roundtrip`, `test_10_recovery_fails_closed_on_corrupted_snapshot`.

---

#### 15. Disk Corruption

* **Threat Description**: Hardware failure or storage corruption flips bits in stored journal records or snapshot files.
* **Attack Boundary**: Physical or virtualized storage layer.
* **Detection**: SHA-256 digest recalculation on every record and snapshot.
* **Mitigation**: Zero tolerance for digest mismatch; no silent error suppression or speculative repair.
* **Fail-Closed Behavior**: `RecoveryFailedClosedError` emitted. Engine refuses to boot in degraded security posture.
* **Test Coverage**: `tests/test_federation_runtime.py::test_06_journal_corruption_detection`, `test_11_recovery_fails_closed_on_corrupted_journal`.

---

#### 16. Unauthorized State Injection

* **Threat Description**: A hostile entity injects private keys, capability secrets, or model weights into persistent state files.
* **Attack Boundary**: State serialization and persistence schema enforcement.
* **Detection**: Schema validation rejects unapproved fields. Strict exclusion of private keys, session secrets, and weight arrays is enforced during snapshot creation and journal appending.
* **Mitigation**: `FederationRecoveryManager` explicitly strips and ignores any forbidden payload keys. Private keys and session secrets are never serialized.
* **Fail-Closed Behavior**: Secrets are never persisted (`test_04_secret_non_leakage_in_persistence`).
* **Test Coverage**: `tests/test_federation_runtime.py::test_04_secret_non_leakage_in_persistence`.

---

#### 17. Persistence Becoming Authority

* **Threat Description**: The persistence storage layer is treated as an authority that can grant trust, elevate permissions, or bypass local capability gates.
* **Attack Boundary**: Persistence-to-Capability gate boundary.
* **Detection**: Architectural enforcement: `PERSISTENCE != AUTHORITY`. State restored from persistence is merely informational state; all capability executions must continue to query `CapabilityGate.authorize()` independently.
* **Mitigation**: Recovery only restores peering topology, replay floors, and revocation ceilings. It never issues new capability tokens or bypasses gate enforcement.
* **Fail-Closed Behavior**: `CapabilityGate` continues to reject unauthorized actions regardless of recovered peering state.
* **Test Coverage**: `tests/test_federation_runtime.py::test_18_rejoin_does_not_grant_authority`.

---

#### 18. Runtime Becoming Authority

* **Threat Description**: `FederationRuntime` or orchestration scheduling logic assumes authority over local policy or capability delegation.
* **Attack Boundary**: Runtime management layer.
* **Detection**: Architectural separation: `RUNTIME != AUTHORITY`. Runtime manages engine registration, lifecycle (start/stop/restart), and health monitoring. It has no authority to grant trust or alter capability scopes.
* **Mitigation**: All security policies reside in local engine and capability gates. Runtime cannot issue trust grants or override local policy.
* **Fail-Closed Behavior**: Capability requests made without gate authorization fail closed with security exceptions.
* **Test Coverage**: `tests/test_federation_runtime.py::test_18_rejoin_does_not_grant_authority`.

---

### Invariant Summary

1. $\Delta W = 0$ strictly preserved under all persistence, crash, journal replay, and runtime lifecycle operations.
2. Rejoin flow never confers capability rights: `ENGINE_REJOIN != TRUST_GRANT`.
3. Network disconnection is distinct from revocation: `UNREACHABLE != REVOKED`.
4. Monotonic replay floors are restored deterministically and cannot be wound back.
5. All revocations are terminal and irreversible across recovery and rejoin.
