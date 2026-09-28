# Step 30 — Secure Physical Transport & Cryptographic Peer Identity Architecture

## 1. Executive Summary & Motivation

Step 29 established **Cross-Zone Peering & Trust Negotiation**, providing governed discovery, attestation, bounded trust grants, default-deny policies, and `CapabilityGate` mediation for in-process multi-zone federation. However, Step 29 intentionally operated with deterministic local identity claims and deferred physical transport and asymmetric cryptography:
`IDENTITY != CRYPTOGRAPHIC_AUTHENTICATION`.

**Step 30** completes the architectural transition from an in-process simulation to a **secure, transport-independent wire-capable federation layer**. It introduces:
1. **Asymmetric Cryptographic Peer Identity** using standard Ed25519 (RFC 8032) digital signatures.
2. **Bounded Challenge-Response Peer Authentication** with cryptographically secure 256-bit random nonces, session binding, one-time consumption, and strict replay protection.
3. **Secure Peer Session Model** with deterministic epoch validity and bounded message tracking.
4. **Wire Envelope & Canonical Serialization** guaranteeing schema enforcement, payload size ceilings, and detached digital signature verification before trusted processing.
5. **Physical & In-Process Transport Abstractions** including length-prefixed stream framing, thread-safe `LoopbackWireTransport`, physical socket `TCPWireTransport`, and fail-closed adapter boundaries for `HTTP2WireTransport` and `GRPCWireTransport`.
6. **Unyielding Authority & Isolation Invariants**:
   - `TRANSPORT != AUTHORITY`
   - `TRANSPORT != TRUST`
   - `CRYPTOGRAPHIC_IDENTITY != AUTHORITY`
   - `AUTHENTICATION != AUTHORIZATION`
   - `AUTHENTICATION != TRUST`
   - `SIGNATURE_VALIDITY != CAPABILITY_PERMISSION`
   - Mandatory `CapabilityGate` mediation under local authority.
   - Cross-tenant isolation guaranteed even for validly signed sessions.
   - Frozen neural core immutability ($\Delta W = 0$).

---

## 2. Layered Architecture

The Step 30 federation pipeline enforces a strict modular hierarchy where physical and cryptographic layers serve purely as conduits and proofs of possession, never as authorities:

```text
┌─────────────────────────────────────────────────────────────┐
│                       Cognitive Layer                       │
│    (Task Planning, Reasoning, Deliberation, Decision)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Cross-Zone Federation                    │
│    (Discovery, Attestation Claims, Policy, Trust Grants)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                      Secure Peer Session                    │
│       (Session ID, Ephemeral Replay Cache, Epoch TTL)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                     Transport Abstraction                   │
│        (Connect, Listen, Accept, Send, Receive, Health)     │
│   ┌────────────────────┬───────────────┬────────────────┐   │
│   │ Loopback Transport │ TCP Transport │ HTTP/2 Adapter │   │
│   └────────────────────┴───────────────┴────────────────┘   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                  Wire Framing & Serialization               │
│      (4-byte Length Header, Canonical JSON, Payload Limit)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│           Cryptographic Identity & Authentication           │
│     (Ed25519 Keypairs, Challenge-Response, Nonce Verify)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Trust and Authority Boundaries

```text
Incoming Wire Frame
        ↓
[Frame Length & Format Check] (LengthPrefixedFramer)
        ↓
[Canonical Deserialization & Payload Digest Check] (DeterministicWireSerializer)
        ↓
[Registered Peer Identity Check] (PeerRegistry)
        ↓
[Cryptographic Key Validity & Ed25519 Signature Verification] (Ed25519PublicKeyWrapper)
        ↓
[Session State, Freshness & Replay Check] (SecurePeerSession)
        ↓
[Cross-Tenant Boundary Isolation Check] (CrossZoneIsolationGuard)
        ↓
[Trust Grant Scope Check] (TrustModel)
        ↓
[Federation Policy Check] (CrossZoneFederationPolicy)
        ↓
[CapabilityGate Authorization & Execution] (CapabilityGate - Local Authority)
        ↓
[Neural Core Immutability Check (ΔW = 0)] (CoreIntegrityGuard)
        ↓
Outgoing Signed Response WireEnvelope
```

Under this model:
- **Transport reachability confers zero trust**: Being able to connect over TCP or HTTP/2 does not grant any peer rights.
- **Cryptographic authentication proves identity ownership, NOT authority**: A peer verifying private-key possession achieves `CRYPTOGRAPHICALLY_AUTHENTICATED` standing. It cannot invoke capabilities until a separate bounded `TrustGrant` is negotiated.
- **Signatures validate message integrity, NOT capability permission**: A mathematically valid signature on a message requesting capability execution will still be rejected if local policy or `CapabilityGate` denies permission.

---

## 4. Cryptographic Peer Identity Model

### 4.1 Ed25519 Primitive
The cryptographic subsystem uses Ed25519 (RFC 8032) asymmetric keypairs.
- **Public Key**: 32 bytes raw, represented as 64 hex characters.
- **Fingerprint**: SHA-256 hex digest of the raw 32-byte public key.
- **Detached Signature**: 64 bytes raw, represented as 128 hex characters.
- **Deterministic Peer Identifier**: Derived from public key material:
  `peer_id = f"peer_{public_key.fingerprint[:16]}"`

### 4.2 Zero Private Key Leakage
Private keys are wrapped in `Ed25519PrivateKeyWrapper`:
- `__repr__` and `__str__` return redacted placeholders:
  `<Ed25519PrivateKeyWrapper fingerprint=... [PRIVATE KEY REDACTED]>`
- `to_dict()` unconditionally raises `PermissionError("Architectural violation: Private keys must never be serialized or exported to dict.")`
- Private key material is never written to telemetry, audit records, or network envelopes.

### 4.3 Key Lifecycle & Rotation
Keys transition through explicit states:
- `ACTIVE`: Key is valid for signing and verification within logical epoch window.
- `ROTATING`: Key is undergoing scheduled retirement.
- `REVOKED`: Key is compromised or administratively revoked. All subsequent operations fail immediately.
- `EXPIRED`: Key logical epoch validity has elapsed (`current_epoch > expires_epoch`).

Key rotation requires the old key to sign an explicit rotation payload:
`ROTATE_KEY:<old_fingerprint>:<new_fingerprint>:<rotation_epoch>`
ensuring unauthorized parties cannot rotate keys without possessing the active private key.

---

## 5. Bounded Challenge-Response Peer Authentication

Step 30 extends the Step 29 peer lifecycle:

```text
DISCOVERED
   ↓
IDENTITY_PRESENTED
   ↓
CRYPTOGRAPHICALLY_AUTHENTICATED (New in Step 30)
   ↓
ATTESTED
   ↓
TRUST_NEGOTIATED
   ↓
ACTIVE (Federated)
```

### Challenge-Response Sequence
1. **Challenge Issuance**: Challenger generates an `AuthChallenge`:
   - `challenge_id`: Unique identifier (`chal_<hex>`)
   - `session_id`: Bounded session binding
   - `challenger_peer_id`: Identifier of issuing peer
   - `target_peer_id`: Identifier of peer being authenticated
   - `nonce`: Cryptographically secure random 256-bit token (`secrets.token_bytes(32).hex()`)
   - `created_epoch` / `expires_epoch`: Strict logical epoch TTL (default 2 epochs)
2. **Challenge Signing**: Target peer signs the canonical JSON payload:
   `{"protocol_version": "30.0", "challenge_id": ..., "session_id": ..., "nonce": ..., "created_epoch": ..., "expires_epoch": ...}`
   producing `AuthChallengeResponse`.
3. **Challenge Verification**: Challenger verifies:
   - Challenge exists and is in `_active_challenges`.
   - Session ID and Target Peer ID match.
   - Nonce has not been previously consumed (`_consumed_nonces`).
   - Current epoch $\le$ `expires_epoch`.
   - Ed25519 digital signature matches target peer's registered public key.
   - Upon success: challenge is removed, nonce is added to FIFO-bounded consumed history, and peer standing advances to `CRYPTOGRAPHICALLY_AUTHENTICATED`.

---

## 6. Secure Peer Session Model

Established via `SecurePeerSession`:
- `session_id`: Unique identifier (`sess_<hex>`)
- `local_peer_id` & `remote_peer_id`: Mutual peer identifiers
- `local_zone_id` & `remote_zone_id`: Zone isolation identifiers
- `status`: `INITIATED` $\rightarrow$ `AUTHENTICATING` $\rightarrow$ `ACTIVE` $\rightarrow$ `EXPIRED` / `TERMINATED` / `REVOKED`
- `expires_at_epoch`: Deterministic logical epoch expiration
- `remote_public_key`: Associated Ed25519 public key wrapper
- `trust_grant_id`: Pointer to active negotiated `TrustGrant`
- `_seen_message_ids`: Bounded FIFO cache (max 1000 IDs) tracking all message IDs in this session to eliminate replay attacks.

---

## 7. Wire Framing & Protocol Envelope

### 7.1 Binary Framing Layout
For stream-oriented physical transports (TCP), frames use length-prefixed encoding:
```text
┌──────────────────────────────┬──────────────────────────────────────────┐
│   4-Byte Big-Endian Length   │           Canonical JSON Payload         │
│         (Header: >I)         │           (WireEnvelope Bytes)           │
└──────────────────────────────┴──────────────────────────────────────────┘
```
- Hard frame ceiling: 1 MB (`MAX_WIRE_FRAME_BYTES = 1048576`).
- Truncated or oversized headers raise `FrameError` or `OversizedPayloadError`.

### 7.2 WireEnvelope Specification
```json
{
  "protocol_version": "30.0",
  "message_type": "CAPABILITY_REQUEST",
  "message_id": "msg_01a2b3c4",
  "session_id": "sess_f9e8d7c6",
  "sender_peer_id": "peer_alice",
  "receiver_peer_id": "peer_bob",
  "created_epoch": 10,
  "expires_epoch": 20,
  "payload": { ... },
  "payload_digest": "<sha256_hex_of_canonical_payload>",
  "signature": "<ed25519_detached_sig_hex_over_header_and_digest>"
}
```
Validation rules:
1. Strict schema: all required keys must be present.
2. Canonical JSON: sorted keys, compact separators (`','`, `':'`).
3. Payload byte ceiling: $\le 65,536$ bytes default.
4. Digest verification: `payload_digest` must match SHA-256 of `payload`.
5. Signature verification: `signature` must verify against sender's registered public key.

---

## 8. Physical and Logical Transport Adapters

### 8.1 `LoopbackWireTransport`
In-process, thread-safe queue routing between endpoints (`loopback://<zone>`). Provides deterministic sub-millisecond testing on CPU with zero operating-system socket dependencies.

### 8.2 `TCPWireTransport`
Physical socket adapter using standard library `socket`:
- Binds to `127.0.0.1` (IPv4 loopback) on dynamic or configured ports.
- Length-prefixed binary framing via `LengthPrefixedFramer`.
- Explicit socket timeouts on connect, send, and receive.
- Automatic chunked receive buffer reassembly.
- Graceful shutdown with `SHUT_RDWR` and socket closure.

### 8.3 `HTTP2WireTransport` & `GRPCWireTransport` (Adapter Boundaries)
Explicit dependency checks:
- `HTTP2WireTransport`: Checks `h2` or `httpx`.
- `GRPCWireTransport`: Checks `grpc`.
- If missing, `is_available` returns `False`, and connect/listen raise `TransportUnavailableError`. Never fakes transport availability.

---

## 9. CapabilityGate & Tenant Isolation Integration

Even after physical transport delivery, envelope verification, signature validation, and session freshness checks:
1. **Tenant Isolation**:
   `CrossZoneIsolationGuard.validate_tenant_boundary(peer_tenant_id, target_tenant_id)`
   Cross-tenant crossover is rejected with `IsolationViolationError`.
2. **Payload Sanitization**:
   Weights, model activations, private memory, and secrets are stripped via `CrossZoneIsolationGuard.sanitize_payload()`.
3. **Trust & Policy Check**:
   Peer must hold an active `TrustGrant` permitting the requested `FederationScope`.
4. **Authoritative CapabilityGate**:
   If capability invocation is requested, it is dispatched to `CapabilityGate.authorize()` under local authority. Transport code never touches capability functions directly.

---

## 10. Threat Model & Mitigations

| Threat | Attack Vector | Mitigation |
| :--- | :--- | :--- |
| **Impersonation** | Forging sender peer identifier | Ed25519 signature verified against registered public key. |
| **Message Tampering** | Modifying wire payload in flight | Detached signature verifies over SHA-256 payload digest. Tampering breaks signature or digest. |
| **Replay Attacks** | Replaying previously captured valid messages | Session-bound `_seen_message_ids` rejects duplicates; challenge nonces are one-time consumed. |
| **Session Hijacking** | Injecting messages into another session | Envelope sender/receiver IDs are strictly checked against session peer bindings. |
| **Denial of Service** | Oversized wire frames / memory exhaustion | Strict 64 KB payload and 1 MB frame limits enforced at framing layer before parsing. |
| **Cross-Tenant Leakage** | Peer from Tenant A requests Tenant B data | `CrossZoneIsolationGuard` enforces strict tenant boundary matching. |
| **Authority Elevation** | Peer claims capability authorization from signature | `CapabilityGate` remains mandatory; `SIGNATURE_VALIDITY != CAPABILITY_PERMISSION`. |
| **Neural Core Drift** | Peer requests triggering weight mutations | Pre- and post-operation SHA-256 parameter hashing enforces $\Delta W = 0$. |

---

## 11. Known Limitations & Deferred Capabilities

1. **Production PKI / CA Chains**: Step 30 implements direct Ed25519 public key registration and key rotation; hierarchical X.509 certificate authorities remain deferred.
2. **Hardware Security Modules (HSM)**: Cryptographic operations execute via standard PyCA `cryptography`; hardware token PKCS#11 integration remains deferred.
3. **Autonomous Global Mesh**: Discovery remains governed; decentralized DHT / internet peer discovery remains deferred.
4. **Transport Layer Security (TLS)**: Physical TCP transport operates over local IPv4 loopback; production mTLS wrapping with system certificates remains deferred.
