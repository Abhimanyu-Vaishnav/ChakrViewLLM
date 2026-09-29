# Step 31: Production Transport Security, TLS/mTLS & Certificate Lifecycle Architecture

## 1. Architectural Overview

Step 31 advances the ChakrView federation subsystem from Step 30 physical socket transport to a production-grade, cryptographically hardened transport security tier. It establishes explicit transport encryption (TLS 1.3 / TLS 1.2), mutual certificate authentication (mTLS), deterministic X.509 validation, and formal binding between TLS identity certificates and ChakrView Ed25519 peer identities.

The end-to-end federation execution hierarchy is structured as:

```
+-------------------------------------------------------------------------+
|                              COGNITIVE LAYER                            |
|             (ChakrMicro v0.1: 3,443,136 parameters, frozen, ΔW = 0)     |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                          CROSS-ZONE FEDERATION                          |
|             (Whitelisted peer zones, scoped delegation, TTLs)           |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                           SECURE PEER SESSION                           |
|       (Session ID, monotonic epochs, bounded replay cache FIFO)         |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                       CRYPTOGRAPHIC PEER IDENTITY                       |
|           (Ed25519 asymmetric keypairs, challenge-response auth)       |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                        FEDERATION TRUST & POLICY                        |
|       (Explicit TrustGrant, scope whitelists, default-deny posture)     |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                         SECURE TRANSPORT POLICY                         |
|   (TLS 1.3 minimum preference, cipher suites, hostname checks, CAs)     |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                                TLS / mTLS                               |
|        (Standard TLS 1-way & Mutual TLS 2-way certificate validation)   |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                        TCP / HTTP/2 / gRPC ADAPTER                      |
|            (Socket transport, stream framing, connection management)    |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                               WIRE FRAMING                              |
|           (8-byte binary header, length prefix, 16 MB bounded ceiling)  |
+-------------------------------------------------------------------------+
                                    ↓
+-------------------------------------------------------------------------+
|                       DETERMINISTIC SERIALIZATION                       |
|                (Canonical JSON, SHA-256 payload digest, no pickle)      |
+-------------------------------------------------------------------------+
```

---

## 2. Security Boundaries & Authority Invariants

Step 31 strictly upholds all ratified authority invariants:

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
LOCAL_AUTHORITY > PEER_AUTHORITY
CROSS_ZONE_FEDERATION != AUTHORITY_TRANSFER
```

A valid TLS certificate, successful mTLS handshake, or verified certificate chain:
1. **Never** grants authority over another zone.
2. **Never** confers a federation trust grant (`TrustGrant`).
3. **Never** authorizes a capability execution.
4. **Never** bypasses `CapabilityGate`.
5. **Never** permits cross-tenant crossover.
6. **Never** allows access to neural core weights, activations, scratchpads, or private memory.

---

## 3. TLS Model

The TLS subsystem is abstracted inside `chakrview.cognition.transport.security`:
- **Protocol Minimum**: Strict TLS 1.3 preference. TLS 1.2 is supported only when explicitly enabled (`allow_tls_1_2=True`) for legacy environments, and fails closed otherwise.
- **Prohibited Protocols**: SSLv2, SSLv3, TLS 1.0, and TLS 1.1 are unconditionally disabled.
- **Cipher Suites**: High-grade modern AEAD cipher suites (ChaCha20-Poly1305, AES-256-GCM, AES-128-GCM). Anonymous ciphers and insecure export ciphers are barred.
- **Fail-Closed Verification**: Client sockets require `CERT_REQUIRED` verification against configured CA trust roots. `CERT_NONE` is forbidden except in the explicitly isolated `PLAINTEXT_TEST_ONLY` mode.

---

## 4. Mutual TLS (mTLS) Model

When configured for mutual authentication (`TLSMode.MTLS`):
1. **Server Verification**: The client verifies the server's certificate against its local trusted CA bundle and enforces SAN/hostname validation.
2. **Client Verification**: The server issues a Certificate Request during the TLS handshake, requiring the client to present a valid X.509 client certificate signed by a recognized CA.
3. **Certificate Extraction**: The server and client extract the peer certificate in DER/PEM format post-handshake without exposing private keys.
4. **Subsequent Application Layer**: Establishing mTLS does NOT supersede Step 30 Ed25519 peer authentication; both layers are mandatory in secure federation mode.

---

## 5. Certificate Model

X.509 certificates are encapsulated in the immutable `CertificateMetadata` dataclass:
- `fingerprint`: Colon-delimited uppercase SHA-256 digest (`SHA256:XX:XX:...`).
- `subject`: Dictionary of subject attributes (mapping OIDs to human-readable strings and standard aliases like `CN`, `O`, `OU`).
- `issuer`: Dictionary of issuing CA attributes.
- `serial_number`: Integer certificate serial number.
- `not_before`: UTC timestamp indicating inception of validity.
- `not_after`: UTC timestamp indicating expiration date.
- `key_usage`: Formal set of standard usage permissions (`digital_signature`, `key_encipherment`, etc.).
- `extended_key_usage`: Set of extended key purpose OIDs (`server_auth`, `client_auth`).
- `san_dns`: List of DNS subject alternative names.
- `san_ips`: List of IP subject alternative names.
- `lifecycle_state`: Categorical lifecycle enum.

Private keys are strictly excluded from `CertificateMetadata` and its `to_dict()` representation.

---

## 6. Deterministic Certificate Validation

The `validate_certificate()` and `validate_certificate_or_raise()` helpers provide fail-closed, deterministic checks:
1. **Structural Integrity**: Certificate must parse into valid X.509 data structures without errors.
2. **Temporal Window**: Current epoch/timestamp must fall strictly within `[not_before, not_after]`. Certificates beyond `not_after` produce `CertificateExpiredError`.
3. **Revocation Check**: The certificate fingerprint is checked against `CertificateRevocationRegistry`. Blacklisted fingerprints produce `CertificateRevokedError`.
4. **Hostname / SAN Matching**: When an expected hostname or IP is supplied, the certificate SAN list (DNS names and IP addresses) and Subject Common Name (CN) are matched. Mismatches raise `HostnameMismatchError`.
5. **Fingerprint Pinning**: When `allowed_fingerprints` are provided in policy, untrusted fingerprints produce immediate rejection.
6. **Key Usage**: Mandatory verification that certificates presented for server authentication include `server_auth` and certificates presented for client authentication include `digital_signature` / `client_auth`.

---

## 7. Certificate Lifecycle Management

The `CertificateLifecycleState` enum governs credential status:
- `VALID`: Fully valid and active.
- `EXPIRING`: Within the configurable warning threshold (default: 14 days before `not_after`).
- `EXPIRED`: Timestamp has exceeded `not_after`.
- `REVOKED`: Explicitly revoked in the revocation registry.
- `UNKNOWN`: Uninitialized or unverifiable.
- `INVALID`: Syntactically or structurally corrupted.

Warning events (`AuditEventType.CERTIFICATE_EXPIRED` / `CERTIFICATE_REVOKED`) are emitted to the bounded audit logger to alert operations without interrupting unrelated active sessions.

---

## 8. Certificate Rotation

Certificate rotation is supported cleanly through `PeerCertificateBinder.rotate_binding()`:
- Replaces an active certificate fingerprint binding with a replacement fingerprint for a specified peer ID.
- Increments the rotation sequence number and records the epoch of rotation.
- Inactive or expired bindings can be phased out without disrupting established logical sessions if permitted by policy.
- Zero private key material is involved in rotation events.

---

## 9. Peer Identity to Certificate Binding

The `PeerCertificateBinder` bridges the transport layer and the federation identity layer:
- Maintains an explicit mapping from `peer_id` to `PeerCertificateBinding` records containing:
  - `peer_id`
  - `certificate_fingerprint` (SHA-256)
  - `expected_common_name`
  - `bound_epoch`
  - `rotation_count`
- When an inbound wire envelope is received over a secure TLS/mTLS transport, the engine extracts the peer certificate metadata and calls `verify_binding_or_raise()`.
- If the transport certificate fingerprint does not match the bound peer fingerprint, `PeerBindingMismatchError` is raised, even if the certificate is otherwise valid.

---

## 10. TCP TLS / mTLS Socket Integration

`TCPWireTransport` operates across three explicit modes:
1. `PLAINTEXT_TEST_ONLY`: Raw TCP socket (restricted strictly to testing).
2. `TLS`: Standard TLS with server authentication and encrypted wire transport.
3. `MTLS`: Mutual TLS with mandatory bidirectional certificate validation.

Key operational features:
- Uses `TLSContextFactory` to construct hardened `ssl.SSLContext` objects.
- Wraps accepted incoming sockets (`server_side=True`) and outgoing client sockets (`server_side=False`).
- Extracts peer DER certificates via `sock.getpeercert(binary_form=True)` upon successful handshake.
- Enforces strict socket timeouts on handshakes, frame reads, and writes to prevent socket starvation.
- Seamlessly integrates with Step 30 binary length-prefixed framing and canonical JSON serialization.

---

## 11. HTTP/2 Security Boundary

The `HTTP2WireTransport` adapter provides:
- Clean integration with `SecureTransportPolicy`.
- Dynamic capability reporting (`supports_tls=True`, `supports_mtls=True`).
- Safe absence detection (`is_http2_available()`): returns `TransportUnavailableError` without throwing unhandled exceptions or inventing fake execution when `httpx` / `h2` are absent.

---

## 12. gRPC Security Boundary

The `GRPCWireTransport` adapter provides:
- Clean integration with `SecureTransportPolicy`.
- Dynamic capability reporting (`supports_tls=True`, `supports_mtls=True`).
- Safe absence detection (`is_grpc_available()`): returns `TransportUnavailableError` when `grpcio` is absent.

---

## 13. Transport Security Policy

The `SecureTransportPolicy` dataclass governs transport boundaries:
- `tls_mode`: `PLAINTEXT_TEST_ONLY`, `TLS`, or `MTLS`.
- `minimum_tls_version`: Protocol floor (default: `TLSProtocolVersion.TLS_1_3`).
- `allow_tls_1_2`: Explicit boolean gate for TLS 1.2 compatibility.
- `verify_peer_certificate`: Mandatory verification boolean (default: `True`).
- `verify_hostname`: Hostname/SAN verification boolean (default: `True`).
- `trusted_ca_paths`: List of local filesystem paths to trusted CA PEM bundles.
- `trusted_ca_data`: Optional in-memory PEM string of trusted CAs.
- `allowed_peer_fingerprints`: Set of SHA-256 certificate fingerprints for pinning.
- `certificate_expiry_warning_seconds`: Warning window (default: 14 days).
- `handshake_timeout_seconds`: Connection timeout ceiling (default: 10.0s).

---

## 14. Security Failure Model

All transport security validations **fail closed**:
- Missing certificate -> Rejection (`CertificateValidationError`)
- Expired certificate -> Rejection (`CertificateExpiredError`)
- Revoked certificate -> Rejection (`CertificateRevokedError`)
- Hostname mismatch -> Rejection (`HostnameMismatchError`)
- Untrusted CA -> Rejection (`UntrustedCAError`)
- Missing client certificate during mTLS -> Rejection (`ClientCertificateMissingError`)
- Peer identity to certificate mismatch -> Rejection (`PeerBindingMismatchError`)
- Insecure protocol downgrade attempt -> Rejection (`InsecureDowngradeError`)
- Plaintext fallback attempt in secure mode -> Rejection (`TLSError`)

No silent fallback to insecure transport is ever permitted.

---

## 15. Replay Protection & Session Interaction

Step 31 integrates with the Step 30 `SecurePeerSession` replay defense:
- TLS 1.3 ephemeral ECDHE key exchanges prevent transport-level handshake replays.
- At the wire envelope tier, `WireEnvelope.message_id` is tracked in a bounded FIFO cache (`_seen_message_ids`). Duplicate message IDs within active session windows are discarded immediately.
- `expires_epoch` bounds message validity to prevent delayed replay of captured envelopes.

---

## 16. CapabilityGate Integration

Execution of capabilities requested across zone boundaries follows the exact inviolable chain:
```
Transport Security (TLS / mTLS)
      ↓
Cryptographic Peer Identity (Ed25519)
      ↓
Secure Peer Session (Freshness & Replay Checks)
      ↓
Federation Trust (Active TrustGrant for requested FederationScope)
      ↓
Federation Policy (Allowed zones & scope whitelist)
      ↓
CapabilityGate (Registry lookup, permission check, provenance denial)
      ↓
Sandboxed Capability Execution
```
Transport security code never invokes capability logic directly.

---

## 17. Tenant Isolation

Cross-tenant access remains strictly forbidden:
- `CrossZoneIsolationGuard.validate_tenant_boundary()` enforces that `peer_tenant_id == target_tenant_id`.
- Mismatched tenant IDs immediately raise `IsolationViolationError`.
- Valid TLS certificates, valid mTLS authentication, and valid Ed25519 signatures never grant cross-tenant authority.

---

## 18. Audit & Secret-Safety Verification

- **Bounded Audit Logging**: Safe metadata (peer IDs, SHA-256 fingerprints, TLS mode, protocol version, event types, failure rationales) is logged to `BoundedAuditLogger`.
- **Zero Secret Leakage**:
  - `Ed25519PrivateKeyWrapper` redact repr/str: `"<Ed25519PrivateKeyWrapper: [REDACTED]>"`.
  - `to_dict()` on private keys raises `PermissionError`.
  - TLS private keys are never stored in `CertificateMetadata` or emitted in logs.
  - Ephemeral files created during SSLContext setup are unlinked immediately after ingestion.

---

## 19. Threat Model Summary

Documented comprehensively in `docs/STEP_31_THREAT_MODEL.md`. All 17 identified network threats (MitM, certificate substitution, replay, expiration, revocation, rogue CAs, downgrade, plaintext fallback, cross-tenant crossover, etc.) are formally mapped to deterministic mitigation mechanisms across the architecture.

---

## 20. Known Limitations

1. **In-Memory Revocation Registry**: `CertificateRevocationRegistry` operates within local process memory. Cluster-wide distributed CRL / OCSP stapling is deferred.
2. **Local PKI Model**: Hermetic testing relies on `HermeticPKIBuilder`. Hardware Security Modules (HSM) and PKCS#11 integrations are deferred.
3. **Synchronous TLS Sockets**: Socket TLS wrapping utilizes Python's standard `ssl.SSLContext.wrap_socket()`. Asynchronous event-loop multiplexing (e.g., `asyncio` streams) is deferred.

---

## 21. Deferred Capabilities

The following capabilities are deliberately out of scope for Step 31:
- Hardware Security Module (HSM) / PKCS#11 integration
- Automated public Internet certificate enrollment (ACME / Let's Encrypt)
- OCSP stapling and external CRL distribution point fetching
- Autonomous peer discovery or open Internet peer meshes
- Blockchain or decentralized identifier (DID) frameworks
- Weight sharing, remote gradient synchronization, or distributed model training
- Any modification to the frozen ChakrMicro neural core
