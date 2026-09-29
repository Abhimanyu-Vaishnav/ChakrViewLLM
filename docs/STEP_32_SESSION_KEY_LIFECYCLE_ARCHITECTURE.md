# CHAKRVIEW — STEP 32 ARCHITECTURE SPECIFICATION
## Secure Federation Session & Key Lifecycle Hardening

### Status
- **Step**: 32
- **Module**: `chakrview/cognition/peering/`, `chakrview/cognition/transport/`
- **Dependencies**: Step 29 (Peering & Trust), Step 30 (Physical Transport & Ed25519 Identity), Step 31 (Production TLS/mTLS)
- **Neural Invariants**: ChakrMicro v0.1 frozen (`3,443,136` parameters, vocab `4096`, context `512`, `ΔW = 0`)

---

## 1. Axiomatic Separation of Concerns

Step 32 hardens the secure federation session and key lifecycle without conflating distinct architectural layers. In accordance with the foundational axioms of ChakrView:

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

### Definitional Taxonomy:
| Concept | Scope | Authority / Mutability | Lifecycle |
| :--- | :--- | :--- | :--- |
| **IDENTITY** | Long-lived canonical descriptor of a peer (peer ID, zone, org). | Immutable identity anchor. | Registered -> Attested -> Revoked |
| **AUTHENTICATION** | Proof of identity possession via Ed25519 signature or TLS cert. | Confirms claimant identity; confers NO authority. | Per-handshake or challenge-response |
| **TRANSPORT SECURITY** | TLS 1.3 / mTLS channel encryption, framing, and socket integrity. | Encrypts wire bytes; confers NO trust or scope. | Ephemeral socket connection |
| **SESSION** | Bounded logical federation relationship with bounded epoch validity. | Enforces sequence monotonicity and replay rejection. | INITIATED -> AUTHENTICATING -> ACTIVE -> RENEWING -> ACTIVE / TERMINATED |
| **KEY** | Cryptographic keypairs (Ed25519 identity key, session key metadata). | Signs bytes; confers NO capability permissions. | CREATED -> ACTIVE -> ROTATING -> EXPIRED / REVOKED |
| **CERTIFICATE** | X.509 certificate bound to peer identity via fingerprint. | Cryptographically authenticates socket transport. | VALID -> EXPIRING -> EXPIRED / REVOKED |
| **TRUST** | Explicit, bounded, scoped, expiring policy grant issued locally. | Authorizes eligible federation scopes under policy. | PENDING -> ACTIVE -> EXPIRED / REVOKED |
| **AUTHORIZATION** | Per-request gate validation against trust, tenant, and policy. | Evaluated dynamically on every wire message. | Transient per transaction |
| **CAPABILITY** | Discrete executable tool/action mediated by CapabilityGate. | Strictly gated by local permissions; default-deny. | Registered in CapabilityRegistry |
| **AUTHORITY** | Exclusive power to execute, grant capabilities, or alter state. | STRICTLY LOCAL. Never transferred to remote peers. | Invariant, sovereign, immutable |

---

## 2. Session State Machine

A peer session transitions through explicit, deterministic states with closed transitions:

```text
          [ INITIATED ]
                │
                │ transition_to(AUTHENTICATING)
                ▼
        [ AUTHENTICATING ]
                │
                │ mark_authenticated(remote_public_key)
                ▼
          ┌─► [ ACTIVE ] ◄────────┐
          │       │               │
          │       │ renew()       │ renew() succeeds
          │       ▼               │
          │   [ RENEWING ] ───────┘
          │
          │   Terminal / Fail-Closed Transitions
          │   (Cannot transition back to ACTIVE)
          ├─────────────────────────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
     [ EXPIRED ]         [ TERMINATED ]        [ REVOKED ]
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                           [ FAILED ]
```

### Strict Transition Table:
```python
VALID_SESSION_TRANSITIONS = {
    SessionStatus.INITIATED: {
        SessionStatus.AUTHENTICATING,
        SessionStatus.FAILED,
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.AUTHENTICATING: {
        SessionStatus.ACTIVE,
        SessionStatus.FAILED,
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.ACTIVE: {
        SessionStatus.RENEWING,
        SessionStatus.EXPIRED,
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.RENEWING: {
        SessionStatus.ACTIVE,
        SessionStatus.FAILED,
        SessionStatus.EXPIRED,
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.EXPIRED: {
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
    SessionStatus.TERMINATED: set(),
    SessionStatus.REVOKED: set(),
    SessionStatus.FAILED: {
        SessionStatus.TERMINATED,
        SessionStatus.REVOKED,
    },
}
```

### Invariant Rules:
1. Terminal states (`TERMINATED`, `REVOKED`) possess empty outgoing transition sets. Any attempt to reactivate raises `SessionTransitionError`.
2. Expired sessions fail closed: once `current_epoch > expires_at_epoch`, `is_active()` returns `(False, "Session EXPIRED...")` and automatically sets status to `EXPIRED`.
3. An expired session cannot be renewed: `can_renew()` returns `False` and `renew()` raises `SessionTransitionError`.

---

## 3. Session Freshness & Bounded Lifetime Ceilings

To eliminate stale session persistence and indefinite extensions:
1. **Deterministic Logical Epochs**: Lifetimes are evaluated in discrete engine logical epochs rather than erratic wall clocks.
2. **Maximum Session Lifetime Ceiling**: Every session has `max_lifetime_epochs = 200`. The cumulative extension boundary cannot exceed `created_epoch + max_lifetime_epochs`.
3. **Renewal Count Ceiling**: Every session has `max_renewals = 5`. Exceeding this boundary fails closed.
4. **Trust Expiry Boundary**: A session cannot be renewed beyond the peer's active `TrustGrant.expires_epoch`. If the trust grant has expired or reached its ceiling, renewal fails closed with `CrossZoneAuthorizationError`.
5. **No Scope Expansion**: Renewal extends epochs only; capability scopes, trust levels, and tenant bindings remain strictly immutable.

---

## 4. Session Key Lifecycle & Secret Protection

Application-level session keys track an independent state machine:

```text
    CREATED ───► ACTIVE ───► ROTATING ───► ACTIVE
                    │                        │
                    ▼                        ▼
                 EXPIRED                  REVOKED
```

### Key Protections:
- Keys are wrapped in `SessionKeyMetadata` which exposes only metadata (`key_id`, `key_fingerprint`, `created_epoch`, `expires_epoch`, `key_state`).
- Raw private keys and symmetric secrets are NEVER serialized, logged in audit events, or included in `repr()`, `str()`, or `to_dict()`.
- Invalidating a session immediately invalidates the associated key metadata (`invalidate(revoked=...)`).

---

## 5. Peer Ed25519 Identity Key Rotation

Key rotation allows an authenticated peer to transition to a fresh cryptographic keypair while preserving peer identity continuity:

```text
1. CURRENT KEY (K_old)
       │
2. Generate fresh keypair (K_new)
       │
3. Produce cryptographic rotation proof:
   proof = Sign_K_old("ROTATE_KEY:" + FP_old + ":" + FP_new + ":" + epoch)
       │
4. Submit rotation proof to engine.rotate_peer_key(peer_id, K_new, proof)
       │
5. Engine validates proof signature against registered active public key
       │
6. Atomically:
   - Retire K_old -> record in retired_key_fingerprints & retired_keys
   - Activate K_new as primary public key
   - Update active sessions to bind K_new
   - Audit event KEY_ROTATION_COMPLETED
```

### Security Properties:
- **Retired Key Invalidation**: Any wire envelope signed with a retired key is immediately rejected with `SignatureVerificationError`.
- **Identity Continuity**: The peer ID, zone ID, trust grants, and capability profile remain completely unchanged.
- **Zero Escalation**: Key rotation NEVER creates or expands trust grants or capability scopes.
- **Fail-Closed on Revocation**: Key rotation on an already revoked peer identity raises `KeyStateError`.

---

## 6. TLS Certificate Rotation & Identity Binding

Building upon Step 31 mTLS:
1. **Dynamic Re-binding**: When a peer's TLS certificate changes, `engine.rotate_peer_certificate()` validates the replacement certificate against:
   - Certificate format and validity dates (`not_before_epoch`, `not_after_epoch`).
   - Hostname / SAN verification (`beta.example.com`).
   - Local `CertificateRevocationRegistry`.
2. **Peer Identity Binding**: The newly verified certificate fingerprint is atomically bound to the existing peer identity via `PeerCertificateBinder`.
3. **No Automatic Trust**: Certificate renewal validates transport identity only; it confers NO trust renewal.

---

## 7. Revocation Cascade Mechanics

When a peer is revoked (`engine.revoke_peer(peer_id, reason, revoked_by)`), the revocation cascades synchronously and deterministically:

```text
   PEER REVOCATION TRIGGERED (engine.revoke_peer)
                     │
                     ▼
       PeerRegistry Mark REVOKED
                     │
                     ▼
  CryptographicPeerIdentity.revoke() (Key state -> REVOKED)
                     │
                     ▼
       Deactivate Active Sessions
    (sess.revoke() -> SessionKeyMetadata.invalidate(revoked=True))
                     │
                     ▼
       Deactivate TLS Certificate Binding
    (certificate_binder.unbind_peer())
                     │
                     ▼
       Deactivate Active Trust Grants
    (trust_grant.status = REVOKED)
                     │
                     ▼
  Audit Events: REVOCATION_CASCADE_COMPLETED, FEDERATION_REVOKED
```

---

## 8. Replay Protection & Sequence Monotonicity

Every wire interaction is protected against replay and out-of-order execution:
1. **Bounded Message ID Cache**: Each `SecurePeerSession` maintains a bounded set (`_seen_message_ids`) backed by a FIFO queue (`max_message_history = 1000`). Replaying any previously seen `message_id` raises `ReplayAttackError`.
2. **Sequence Monotonicity**: If a payload includes `sequence_number`, the session verifies `seq_num > last_seen_sequence_number`. Duplicate or retrograde sequence numbers are rejected as replay attacks.
3. **Epoch Bounded Freshness**: Messages with `expires_epoch < current_epoch` fail closed.

---

## 9. Neural Core Immutability Guard

Throughout the complete session and key lifecycle:
- Model parameters remain frozen: `3,443,136` parameters.
- Vocabulary size remains `4096`.
- Maximum sequence length remains `512`.
- Parameter hash is computed before and after every cross-zone execution: $\Delta W = 0$.
- Any detected weight variation raises `WeightMutationDetectedError` and fails closed.
