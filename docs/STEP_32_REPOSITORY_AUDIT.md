# Step 32: Repository Architectural Audit

## Executive Summary
This architectural audit inspects the ChakrView repository baseline at Step 31 (commit `52bd289` / `9b68d41`) in preparation for **Step 32 — Secure Federation Session & Key Lifecycle Hardening**.

---

## 1. What Already Exists

### A. Peering Subsystem (`chakrview/cognition/peering/`)
- **`session.py` (`SecurePeerSession`)**:
  - Contains basic session attributes: `session_id`, `local_peer_id`, `remote_peer_id`, `local_zone_id`, `remote_zone_id`, `status` (`SessionStatus`), `auth_state`, `created_epoch`, `authenticated_at_epoch`, `expires_at_epoch`.
  - Replay protection cache using `Set[str]` + FIFO `List[str]` bounded by `max_message_history`.
  - Basic methods: `is_active()`, `mark_authenticated()`, `record_and_check_message_id()`, `terminate()`, `revoke()`, `to_dict()`.
- **`crypto.py` (`CryptographicPeerIdentity`, `Ed25519PrivateKeyWrapper`, `Ed25519PublicKeyWrapper`)**:
  - Asymmetric Ed25519 cryptography using PyCA `cryptography`.
  - Zero private key leakage: `__repr__` and `__str__` return redacted strings; `to_dict()` raises `PermissionError`.
  - `KeyLifecycleState`: `ACTIVE`, `ROTATING`, `REVOKED`, `EXPIRED`.
  - `KeyRevocationRecord`: audit record of key revocation.
  - `rotate_key()` on `CryptographicPeerIdentity`: verifies Ed25519 signature from old key authorizing rotation to new key.
- **`authentication.py` (`ChallengeResponseAuthenticator`, `AuthChallenge`, `AuthChallengeResponse`)**:
  - Nonce-based challenge-response authentication protocol with epoch expiration and replay protection.
- **`registry.py` (`PeerRegistry`)**:
  - Thread-safe storage for `PeerRegistration` objects, with `expire_peers()` and `revoke_peer()`.
- **`revocation.py` (`RevocationManager`)**:
  - Central ledger for peer revocation records.
- **`trust.py` & `negotiation.py` (`TrustModel`, `TrustNegotiator`, `TrustGrant`)**:
  - Explicit bounded trust hierarchy (`NONE`, `IDENTIFIED`, `ATTESTED`, `LIMITED_TRUST`, `FEDERATED`, `REVOKED`).
  - Trust grants expire deterministically by epoch (`expires_epoch`).
- **`engine.py` (`CrossZoneFederationEngine`)**:
  - Coordinates peering discovery, attestation, challenge-response auth, manual session creation (`create_secure_session`), and wire envelope mediation (`authorize_and_execute_wire_envelope`).
  - In-memory `self.sessions: Dict[str, SecurePeerSession]`.
  - Neural core weight hash verification before and after operations (`_verify_weight_invariants`).

### B. Transport Security Subsystem (`chakrview/cognition/transport/security/`)
- **`binding.py` (`PeerCertificateBinder`, `PeerCertificateBinding`)**:
  - Binds `peer_id` to TLS certificate SHA-256 fingerprint, expected CN, and expected SAN.
  - Provides `bind_peer()`, `verify_binding()`, and `rotate_binding()`.
- **`certificates.py` & `validation.py`**:
  - Deterministic X.509 parsing, metadata extraction, validation, and in-memory `CertificateRevocationRegistry`.
- **`policy.py` & `tls.py`**:
  - `SecureTransportPolicy` and `TLSContextFactory` creating hardened `ssl.SSLContext` instances.

### C. Capability Subsystem (`chakrview/capability/`)
- **`CapabilityGate`**:
  - Mandated execution boundary. Denies memory/retrieved data provenance authority.
  - Requires explicit registered capability, operational status, and context permissions.

---

## 2. Incomplete Areas & Gaps Identified for Step 32

1. **Session State Machine Gaps**:
   - `SessionStatus` is currently missing `RENEWING` and `FAILED` states.
   - Transitions are not enforced through a formal transition table or `transition_to(target_state, reason)` method. For instance, code could theoretically change `session.status = SessionStatus.ACTIVE` even if the session was previously revoked or terminated.
   - Terminal states (`TERMINATED`, `REVOKED`, `FAILED`, `EXPIRED`) must fail closed and strictly prohibit re-activation.

2. **Session Freshness & Bounded Extension Gaps**:
   - `SecurePeerSession` has `expires_at_epoch`, but lacks an absolute maximum lifetime ceiling (`max_lifetime_epochs` / `max_renewals`). Without a hard ceiling, sessions could theoretically be renewed indefinitely.
   - No explicit renewal handshake/method (`renew_session`) on `CrossZoneFederationEngine` validating fresh cryptographic authentication and active trust grant standing before extending epochs.
   - Renewal boundary checks are missing (e.g. attempting renewal after expiry or revocation).

3. **Session Key Lifecycle Semantics**:
   - While physical transport keys (TLS) and peer identity keys (Ed25519) exist, an explicit `SessionKeyLifecycle` / `SessionKeyMetadata` with formal states (`CREATED`, `ACTIVE`, `ROTATING`, `EXPIRED`, `REVOKED`) is not formally encapsulated.
   - Session keys must be strictly bound to session lifetime, invalidated on termination/revocation, and protected against secret leakage.

4. **Peer Key Rotation Federation Engine Integration**:
   - `CryptographicPeerIdentity.rotate_key` exists in isolation, but `CrossZoneFederationEngine` lacks an atomic `rotate_peer_key()` method that coordinates engine registry updates, session key invalidation, audit logging (`KEY_ROTATION_STARTED`, `KEY_ROTATION_COMPLETED`, `KEY_ROTATION_FAILED`), and ensures trust/capability scope remains immutable.

5. **TLS Certificate Rotation Hardening in Federation Engine**:
   - `PeerCertificateBinder.rotate_binding` exists, but `CrossZoneFederationEngine` lacks a formal `rotate_peer_certificate()` method that validates the replacement certificate, ensures the peer is active and unrevoked, updates bindings, emits audit events, and guarantees zero trust escalation.

6. **Revocation Cascade Deficiencies**:
   - Currently, `engine.revoke_peer()` calls `registry.revoke_peer()`, but does **not** cascade to:
     - Invalidating or revoking active sessions belonging to the peer in `self.sessions`.
     - Revoking the peer's `CryptographicPeerIdentity` (`key_state = KeyLifecycleState.REVOKED`).
     - Revoking or deactivating certificate bindings in `self.certificate_binder`.
     - Invalidating active trust grants and rejecting pending capability executions.
   - This cascade must be deterministic, atomic, and auditable.

7. **Replay & Message Freshness Hardening**:
   - Envelope sequence numbers are not tracked.
   - Replay cache bounding is FIFO-only; need clear verification of behavior under boundary saturation, expired message rejection, and duplicate message detection.

8. **Audit Taxonomy Expansion**:
   - Missing explicit audit event types for session activation, renewal, rotation lifecycle, and revocation cascade.

---

## 3. Reusable Components

- `Ed25519PrivateKeyWrapper` and `Ed25519PublicKeyWrapper` in `crypto.py`.
- `AuthChallenge`, `AuthChallengeResponse`, and `ChallengeResponseAuthenticator` in `authentication.py`.
- `PeerCertificateBinder` in `transport/security/binding.py`.
- `CertificateRevocationRegistry` and `validate_certificate` in `transport/security/`.
- `BoundedAuditLogger` in `peering/audit.py`.
- `CoreIntegrityGuard` in `cognition/diagnostics/integrity.py`.
- `CapabilityGate` in `capability/gate.py`.

---

## 4. Invariants to Harden and Preserve

```text
SESSION != AUTHORITY
SESSION != TRUST
KEY != AUTHORITY
KEY_ROTATION != TRUST_GRANT
CERTIFICATE_ROTATION != AUTHORIZATION
SESSION_RENEWAL != CAPABILITY_ESCALATION

AUTHENTICATION != AUTHORIZATION
TRUST != AUTHORIZATION
TRANSPORT_SECURITY != AUTHORITY

LOCAL_AUTHORITY > PEER_AUTHORITY
CROSS_ZONE_FEDERATION != AUTHORITY_TRANSFER
REMOTE_PEER != LOCAL_CONTROLLER
```

- ChakrMicro neural core remains frozen: 3,443,136 parameters, 4096 vocab, 512 context, $\Delta W = 0$.
- Zero private key leakage in `repr`, `str`, audit records, exceptions, or wire payloads.
- All failures must fail closed.
