# Step 31 Threat Model: Production Transport Security, TLS/mTLS & Certificate Lifecycle

## 1. Executive Security Philosophy

Step 31 enforces a layered, defense-in-depth security model across physical, transport, cryptographic, session, federation, and local capability execution tiers.
Crucially, the architecture enforces fundamental authority separation invariants:

```
TRANSPORT != AUTHORITY
TRANSPORT != TRUST
CRYPTOGRAPHIC_IDENTITY != AUTHORITY
AUTHENTICATION != AUTHORIZATION
AUTHENTICATION != TRUST
SIGNATURE_VALIDITY != CAPABILITY_PERMISSION
TLS != TRUST
TLS_AUTHENTICATION != FEDERATION_AUTHORIZATION
TLS_IDENTITY != FEDERATION_AUTHORITY
CERTIFICATE_VALIDITY != CAPABILITY_PERMISSION
mTLS != TRUST_GRANT
TRANSPORT_SECURITY != AUTHORITY
```

No single layer is ever sufficient by itself. A valid TLS connection, a trusted certificate, and a verified Ed25519 signature collectively establish an authenticated transport and peer channel, but **confer zero execution authority**. Local authority always strictly supersedes peer authority (`LOCAL_AUTHORITY > PEER_AUTHORITY`), and all execution must be explicitly authorized and mediated by `CapabilityGate`.

---

## 2. Threat Catalog & Mitigation Matrix

| Threat ID | Threat Description | Attack Vector | Mitigating Layer(s) & Mechanisms | Residual Risk & Operational Defense |
|---|---|---|---|---|
| **T-01** | **Man-in-the-Middle (MitM)** | Active network adversary intercepts or tampers with wire packets in transit. | **TLS 1.3 / mTLS** with AEAD cipher suites (ChaCha20-Poly1305, AES-GCM) + **SHA-256 wire digest** + **Ed25519 envelope signature**. | Compromise of root CA key. Mitigated by explicit CA trust roots, certificate pinning, and independent Ed25519 layer. |
| **T-02** | **Certificate Substitution** | Adversary presents a valid TLS certificate issued for an attacker-controlled endpoint. | **Peer Identity Binding (`PeerCertificateBinder`)** + strict CN / SAN matching + deterministic fingerprint verification against registered peer records. | Malicious registration. Mitigated by `CrossZoneFederationPolicy` whitelist and peer registration controls. |
| **T-03** | **Certificate Replay** | Replaying past TLS handshakes or handshake messages. | **TLS 1.3 / TLS 1.2 Handshake Nonces** (server & client randoms) + ephemeral key exchanges (ECDHE). | Replay at application layer mitigated by `SecurePeerSession` message ID FIFO and epoch expiration. |
| **T-04** | **Expired Certificate** | Using expired or stale certificates after credential invalidation. | **Deterministic Validation (`validate_certificate`)** checking `not_valid_before` and `not_valid_after` against current epoch/system clock. Fail closed (`CertificateExpiredError`). | System clock tampering. Mitigated by epoch-based monotonicity checks and NTP synchronization. |
| **T-05** | **Revoked Certificate** | Peer with compromised key attempts connection using an otherwise unexpired certificate. | **CertificateRevocationRegistry** tracking blacklisted SHA-256 fingerprints. Invalidation produces immediate rejection (`CertificateRevokedError`). | Delay in revocation distribution. Mitigated by short validity lifetimes and administrative revocation sync. |
| **T-06** | **Rogue CA** | Attacker convinces node to trust a fraudulent or unauthorized Certificate Authority. | **Explicit Hermetic CA Roots** (`trusted_ca_sources` in `SecureTransportPolicy`). Default-deny on unknown CAs; no automatic trust of operating system keystore without configuration. | Local filesystem write access. Mitigated by read-only filesystem mounts in production environments. |
| **T-07** | **Hostname Confusion** | Attacker issues a certificate for a different domain/IP and routes traffic to legitimate node. | **Mandatory Hostname Verification** in client context (`verify_hostname=True`, `ssl.match_hostname`). SAN IP and DNS element validation. | Wildcard certificates. Bounded by strict policy requiring explicit hostname/IP SAN matching. |
| **T-08** | **TLS Downgrade Attack** | Adversary forces negotiation of obsolete or weak protocol versions (e.g. SSLv3, TLS 1.0, TLS 1.1). | **Hard Protocol Ceilings** in `SecureTransportPolicy` (`minimum_tls_version = TLS_1_3`, TLS 1.2 allowed only if explicitly enabled). Insecure downgrade attempts raise `InsecureDowngradeError`. | None; obsolete protocols are rejected before cipher negotiation. |
| **T-09** | **Plaintext Fallback** | Attacker forces node to fall back to unencrypted socket transport. | **Fail-Closed Mode Enforcement**: `TCPWireTransport` in `TLS` or `MTLS` mode will never fall back to plaintext; connection is terminated with `TLSError`. Plaintext is restricted to `PLAINTEXT_TEST_ONLY`. | Misconfiguration. Mitigated by `SecureTransportPolicy.validate()` requiring explicit test-only acknowledgment. |
| **T-10** | **Peer Identity Substitution** | Valid TLS peer claims identity of a different remote peer in wire envelope (`sender_peer_id`). | **Dual Binding Verification**: TLS cert fingerprint must match registered peer ID binding, and envelope Ed25519 signature must match registered peer public key. | Compromise of both TLS private key and Ed25519 private key. Mitigated by key separation and trust scopes. |
| **T-11** | **Cross-Tenant Access** | Authenticated peer in Tenant A attempts to access resources or invoke capabilities in Tenant B. | **CrossZoneIsolationGuard.validate_tenant_boundary()** enforces strict tenant isolation (`peer_tenant_id == target_tenant_id`). Raises `IsolationViolationError`. | None; cross-tenant crossover is rejected unconditionally prior to capability dispatch. |
| **T-12** | **Stolen Certificate / Key** | Compromise of peer's TLS private key or Ed25519 private key. | **Defense-in-Depth Separation**: TLS key != Ed25519 key != Capability permissions. Compromise of TLS key does not grant Ed25519 signing rights; compromise of Ed25519 key does not bypass CapabilityGate. Immediate revocation in `CertificateRevocationRegistry`. | Window before compromise is detected. Bounded by short session TTLs (e.g. 50 epochs) and CapabilityGate least-privilege scoping. |
| **T-13** | **Replayed Authenticated Session** | Adversary captures valid signed wire envelopes and replays them over network socket. | **SecurePeerSession Replay Cache**: bounded FIFO (`_seen_message_ids`), monotonic epoch expiry (`expires_epoch`), and session status checks. Duplicate message IDs are immediately dropped. | Cache eviction after bounded capacity. Mitigated by strict message epoch windows (`expires_epoch <= current_epoch + TTL`). |
| **T-14** | **Malicious Capability Request** | Authenticated and trusted peer requests dangerous capability (e.g. filesystem write, command execution). | **CapabilityGate Mediation**: capability must be registered, active, permitted by caller context permissions, and allowed by `CrossZoneFederationPolicy` whitelist. | Implementation bugs in individual capabilities. Mitigated by `RiskClassification` constraints and capability sandboxing. |
| **T-15** | **Malformed Network Input** | Malformed wire frames, corrupt headers, or invalid JSON designed to exploit parser vulnerabilities. | **Deterministic Length-Prefixed Binary Framing (`LengthPrefixedFramer`)** + canonical strict JSON serializer (`DeterministicWireSerializer`). Syntax errors fail closed (`TransportProtocolError`). | Parser edge cases. Mitigated by fuzz testing and zero use of unsafe deserializers (no `pickle`). |
| **T-16** | **Oversized Payload / DoS** | Massive payload sent to exhaust node memory. | **Bounded Transport Limits**: `MAX_FRAME_SIZE = 16 MB`, payload size validation in `WireEnvelope.validate()`. Oversized frames are dropped before full socket read. | Socket resource exhaustion. Mitigated by connection limits and socket read timeouts. |
| **T-17** | **Connection Exhaustion** | Flooding node with TCP connections or half-open TLS handshakes. | **Socket Timeouts & Resource Boundaries**: bounded handshake timeouts (`handshake_timeout_seconds`), connection lifecycle cleanup, and strict session limits (`MAX_ACTIVE_PEERS = 8`). | Volumetric DDoS at network edge. Mitigated by upstream firewall / load balancer ingress controls. |

---

## 3. Defense-in-Depth Pipeline

Every inbound network request must traverse each layer sequentially. Failure at ANY layer immediately terminates execution and logs a typed audit event:

```
[ Physical Wire ]
      ↓
[ TLS / mTLS Transport Layer ]
   - Validate TLS protocol version >= minimum (TLS 1.3 preferred)
   - Verify server/client certificate chain against trusted CA
   - Check certificate expiration, revocation, and hostname match
      ↓
[ Length-Prefixed Framing & Wire Serialization ]
   - Validate 8-byte binary frame header & magic bytes
   - Enforce MAX_FRAME_SIZE boundary (16 MB)
   - Parse deterministic canonical JSON without pickle
      ↓
[ WireEnvelope Integrity & Digest ]
   - Verify SHA-256 payload digest matches header
   - Validate message schema, session ID, and epoch boundaries
      ↓
[ Peer Identity Binding ]
   - Extract TLS client certificate SHA-256 fingerprint
   - Verify certificate fingerprint matches bound peer identity
      ↓
[ Cryptographic Peer Authentication ]
   - Verify envelope Ed25519 signature against registered peer public key
   - Verify active SecurePeerSession state & replay cache
      ↓
[ Cross-Zone Isolation Guard ]
   - Enforce strict tenant boundary: peer_tenant_id == target_tenant_id
   - Sanitize payload: strip forbidden keys (model weights, secrets, scratchpads)
      ↓
[ Federation Trust & Policy ]
   - Verify active TrustGrant and permitted FederationScope
   - Verify CrossZoneFederationPolicy allowed_zones and allowed_scopes whitelist
      ↓
[ CapabilityGate Mediation ]
   - Deny memory/hypothesis provenance authority
   - Verify capability registration, active status, and context permissions
      ↓
[ Capability Execution ]
   - Execute sandboxed capability implementation
      ↓
[ Invariant Integrity Check ]
   - Confirm frozen ChakrMicro neural core immutability (ΔW = 0)
```

---

## 4. Private Key Material Safety Guarantees

Step 31 strictly audits and enforces secret zero-leakage across the repository:
1. **Redacted String Representations**:
   `Ed25519PrivateKeyWrapper.__repr__` and `__str__` return `"<Ed25519PrivateKeyWrapper: [REDACTED]>"` with zero key material.
2. **Blocked Serialization**:
   `to_dict()` on private key wrappers raises `PermissionError`.
3. **Audit Log Scrubbing**:
   `BoundedAuditLogger` only accepts structured metadata (peer IDs, fingerprints, epochs, failure rationales). Private keys and raw session keys are prohibited.
4. **Transient File Handling**:
   When loading PEM strings into `ssl.SSLContext`, temporary file descriptors are unlinked immediately after ingestion to prevent disk leakage.
5. **Wire Payloads**:
   `WireEnvelope` schemas explicitly prohibit secret keys, private keys, model weights, and internal activations.
