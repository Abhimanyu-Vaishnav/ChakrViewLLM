# ChakrView Step 37: Distributed Resource & Capability Advertisement Architecture

**Status:** Ratified & Production-Hardened  
**Date:** September 2026  
**Module:** `chakrview.cognition.federation.resources`  
**Integration:** `chakrview.cognition.federation.transport`, `chakrview.cognition.peering.engine`, `chakrview.capability.gate`

---

## 1. Executive Architectural Summary

Step 37 implements **Production Distributed Resource & Capability Advertisement** for ChakrView. It allows nodes in a secure federation to voluntarily discover, inspect, package, cryptographically sign, and advertise their available hardware resources (CPU, RAM, accelerators, platform) and cognitive capabilities without surrendering local sovereignty.

### Core Non-Negotiable Principles
```
LOCAL_RESOURCE_POLICY > REMOTE_RESOURCE_CLAIM
RESOURCE_ADVERTISEMENT != RESOURCE_PERMISSION
RESOURCE_ADVERTISEMENT != EXECUTION_AUTHORITY
PEER_RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
UNREACHABLE != REVOKED
REJOIN != TRUST_GRANT
VOLATILE_STATE != DURABLE_AUTHORITY
ZERO NEURAL MUTATION: ΔW = 0
```

1. **Non-Authoritative Informational Sharing:** Remote claims are informational only. An advertisement informs peers of what a node *can* do, never granting authority to *command* the node or schedule workloads without explicit sovereign approval.
2. **Strict Cryptographic Integrity:** Every advertisement is uniquely identified, binds a monotonic sequence version, is anchored by a canonical SHA-256 payload digest, and is signed with an Ed25519 digital signature.
3. **Fail-Closed Privacy & Masking:** By default, hardware observations are coarse-grained (`coarse_privacy_enabled=True`). Raw microarchitectural details, physical socket layouts, disk mount topologies, and sensitive environmental variables are strictly stripped.
4. **Transport Decoupling:** Step 37 uses the Step 36 production message transport (`FederationMessageEnvelope`, `FederationChannel`, `FederationMessageDispatcher`) for message delivery (`RESOURCE_ADVERTISEMENT`, `RESOURCE_QUERY`), but keeps transport mechanisms cleanly separate from resource evaluation logic.

---

## 2. Component Hierarchy & Subsystem Interactions

```
+-----------------------------------------------------------------------------------+
|                            CrossZoneFederationEngine                              |
|  +---------------------------+             +-----------------------------------+  |
|  |     CapabilityGate        |             |    FederationMessageDispatcher    |  |
|  +-------------+-------------+             +-----------------+-----------------+  |
+----------------|---------------------------------------------|--------------------+
                 |                                             |
                 v                                             v
+-----------------------------------------------------------------------------------+
|                        FederationResourceManager                                  |
|  - Coordinates local detection, refreshes, publishing, and inbound handling       |
|                                                                                   |
|  +-----------------------------+        +--------------------------------------+  |
|  |    LocalResourceDetector    |        |      FederationResourceRegistry      |  |
|  | - Hardware inspection       |        | - Local profile & policy isolation   |  |
|  | - Capability extraction     |        | - Peer claims cache & floors         |  |
|  | - Privacy sanitization      |        | - Freshness state machine            |  |
|  +-----------------------------+        | - Capability claim filtering         |  |
|                                         +--------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### Component Roles:
1. **`LocalResourceDetector`**:
   - Inspects operating system, CPU architecture, logical core counts, total/available memory, and optional hardware accelerators (CUDA/ROCm/DirectML/Metal/NPU).
   - Extracts registered capabilities from the local sovereign `CapabilityRegistry`.
   - Sanitizes raw telemetry through coarse quantization.
2. **`ResourceAdvertisement`**:
   - Signed container encapsulating node ID, zone, tenant, version, epoch, timestamp, TTL, `NodeResourceProfile`, and `AdvertisedCapability` list.
   - Computes canonical SHA-256 payload digest and Ed25519 digital signature.
3. **`FederationResourceRegistry`**:
   - Maintains strict separation between `_local_profile` (authoritative local reality) and `_peer_advertisements` (untrusted remote claims).
   - Enforces replay protection via monotonic `_peer_version_floors`.
   - Executes the 5-state freshness lifecycle (`FRESH`, `AGING`, `STALE`, `EXPIRED`, `UNAVAILABLE`).
   - Maintains terminal absorbing revocation barrier (`_barred_nodes`).
4. **`FederationResourceManager`**:
   - Top-level manager attached to `CrossZoneFederationEngine.resource_manager`.
   - Handles automatic periodic volatile metric refresh.
   - Registers Step 37 message handlers (`RESOURCE_ADVERTISEMENT`, `RESOURCE_QUERY`) with `FederationMessageDispatcher`.
   - Handles transport disconnect events (`on_peer_disconnected`) to mark peer claims `UNAVAILABLE`.

---

## 3. Data Structures & Wire Representations

### A. Hardware Profile (`NodeResourceProfile`)
```json
{
  "profile_id": "prof_8600e7c019239c17",
  "cpu": {
    "architecture": "amd64",
    "logical_cores": 20,
    "physical_cores": null,
    "instruction_capabilities": [],
    "total_capacity_mhz": null,
    "available_cores": 20.0,
    "utilization_percent": null
  },
  "memory": {
    "total_memory_mb": 31744,
    "available_memory_mb": 14336,
    "reservable_memory_mb": null,
    "utilization_percent": null
  },
  "platform": {
    "os_family": "Windows",
    "os_release": "11",
    "python_version": "3.14.7",
    "chakrview_version": "37.0",
    "node_epoch": 1
  },
  "accelerator": null,
  "storage": null,
  "timestamp": 1790698707.51,
  "is_coarse": true
}
```

### B. Capability Profile (`AdvertisedCapability`)
```json
{
  "capability_id": "core.inference",
  "name": "ChakrMicro Neural Core",
  "version": "1.0.0",
  "execution_type": "INFERENCE",
  "supported_inputs": ["application/x-token-ids"],
  "supported_outputs": ["application/x-logits"],
  "concurrency_limit": 2,
  "requires_accelerator": false,
  "is_exposed": true,
  "description": "ChakrMicro 3.44M parameter autoregressive inference."
}
```

### C. Signed Advertisement Wire Envelope (`ResourceAdvertisement`)
```json
{
  "advertisement_id": "adv_fa7abeb13d1ddb5a",
  "node_id": "peer_beta",
  "engine_id": "eng_peer_beta",
  "zone_id": "zone-beta",
  "tenant_id": "default",
  "version": 1,
  "epoch": 1,
  "timestamp": 1790698707.52,
  "ttl_seconds": 60.0,
  "federation_scope": "ALLOW_CAPABILITY_METADATA",
  "resource_profile": { ... },
  "capabilities": [ ... ],
  "payload_digest": "f4a2a4b8b16248c6461ede4158b5dfac6563d61c9a6d1c1acb5fdbec16d4ecbb",
  "signature": "f3eab8f0c5cccdab587b332005d9b2fadf7d708c9b5bd0150f8adfa2cc35a42b..."
}
```

---

## 4. Cryptographic Integrity & Signing Pipeline

```
  +------------------------------------------------------------------------+
  |                   Advertisement Payload Construction                   |
  +-----------------------------------+-+----------------------------------+
                                      |
                                      v
  +------------------------------------------------------------------------+
  |              Prohibited Secret & Keyword Scanner                       |
  | (Rejects private_key, model_weights, tensors, non-primitive classes)   |
  +-----------------------------------+-+----------------------------------+
                                      |
                                      v
  +------------------------------------------------------------------------+
  |             Canonical UTF-8 JSON Sorting & Formatting                  |
  |              json.dumps(data, sort_keys=True, separators=(',', ':'))   |
  +-----------------------------------+-+----------------------------------+
                                      |
                                      v
  +------------------------------------------------------------------------+
  |                     SHA-256 Payload Digest                             |
  |             payload_digest = sha256(canonical_utf8).hexdigest()        |
  +-----------------------------------+-+----------------------------------+
                                      |
                                      v
  +------------------------------------------------------------------------+
  |                   Deterministic Signing Bytes Format                   |
  |    CHAKR_RES_ADV:{adv_id}:{node_id}:{version}:{epoch}:{payload_digest}  |
  +-----------------------------------+-+----------------------------------+
                                      |
                                      v
  +------------------------------------------------------------------------+
  |                     Ed25519 Digital Signature                          |
  |                  signature = private_key.sign(signing_bytes)           |
  +------------------------------------------------------------------------+
```

---

## 5. Advertisement Freshness State Machine

Every peer claim adheres to an unambiguous 5-state lifecycle:

```
               [Receive Advertisement]
                         |
                         v
                    +----------+
                    |  FRESH   |  (0 <= elapsed < 0.5 * TTL)
                    +----+-----+
                         |  (elapsed >= 0.5 * TTL)
                         v
                    +----+-----+
                    |  AGING   |  (0.5 * TTL <= elapsed < TTL)
                    +----+-----+
                         |  (elapsed >= TTL)
                         v
                    +----+-----+
                    |  STALE   |  (TTL <= elapsed < 2.0 * TTL)
                    +----+-----+
                         |  (elapsed >= 2.0 * TTL)
                         v
                    +----+-----+
                    | EXPIRED  |  (Scheduled for pruning)
                    +----------+

           +------------------------------+
           | On Node Disconnect / Drop    | ---> [ UNAVAILABLE ]
           +------------------------------+
           | On Revocation Event          | ---> [ REVOKED / BARRED ]
           +------------------------------+
```

### State Definitions:
- **`FRESH`**: Active, verified, usable for capability matching.
- **`AGING`**: Approaching expiry; candidate for background refresh query.
- **`STALE`**: Beyond TTL; excluded from default capability matching (`allow_stale=False`).
- **`EXPIRED`**: Beyond double TTL; eligible for automated pruning.
- **`UNAVAILABLE`**: Peer disconnected or partitioned; claim retained for version tracking but hidden from queries.

---

## 6. Ingestion & Validation Pipeline

When an inbound advertisement arrives via `FederationMessageDispatcher`:

```
INBOUND MESSAGE
  |
  +--> 1. Validate Channel State (Reject QUARANTINED, REVOKED, CLOSED)
  |
  +--> 2. Validate Peer Identity & Membership (Reject local node spoofing)
  |
  +--> 3. Revocation Check (Check against _barred_nodes; Fail-closed)
  |
  +--> 4. Quarantine Check (Reject quarantined peer claims)
  |
  +--> 5. Multi-Tenant Check (Verify tenant_id match or wildcard)
  |
  +--> 6. Payload Digest Verification (Recompute SHA-256 and assert match)
  |
  +--> 7. Ed25519 Signature Verification (Validate cryptographic signature)
  |
  +--> 8. Monotonic Sequence Floor Check (Assert version > _peer_version_floors[node_id])
  |
  +--> 9. Freshness Check (Assert not already STALE upon receipt)
  |
  +--> 10. Atomically Update State & Audit Log
```

---

## 7. Performance Characteristics & Benchmark Summary

Results from empirical benchmark run on Windows x64 Python 3.14.7 environment:

| Operation | Mean Latency | 95th Percentile | Throughput |
|---|---|---|---|
| **Local Resource Discovery** | 0.2113 ms | 0.2517 ms | 4,733.59 ops/sec |
| **CPU Inspection** | 0.0069 ms | 0.0073 ms | 145,692.92 ops/sec |
| **Memory Inspection** | 0.1259 ms | 0.1405 ms | 7,945.97 ops/sec |
| **Accelerator Discovery** | 0.0051 ms | 0.0053 ms | 196,126.48 ops/sec |
| **Capability Declaration** | 0.0172 ms | 0.0175 ms | 58,228.40 ops/sec |
| **Advertisement Creation & Signing** | 1.5634 ms | 1.8263 ms | 639.63 ops/sec |
| **Advertisement Serialization** | 0.0731 ms | 0.0772 ms | 13,687.61 ops/sec |
| **Advertisement Deserialization** | 0.0220 ms | 0.0225 ms | 45,415.84 ops/sec |
| **Payload Digest Computation** | 0.3981 ms | 0.4116 ms | 2,511.66 ops/sec |
| **Ed25519 Signature Generation** | 0.4320 ms | 0.5231 ms | 2,314.85 ops/sec |
| **Ed25519 Signature Verification** | 0.4810 ms | 0.6498 ms | 2,078.93 ops/sec |
| **Advertisement Ingestion Throughput** | 0.8858 ms | 1.2198 ms | 1,128.87 ops/sec |
| **Replay Rejection Latency** | 0.8760 ms | 1.1123 ms | 1,141.54 ops/sec |
| **Tampered Rejection Latency** | 0.3998 ms | 0.5178 ms | 2,501.05 ops/sec |
| **Capability Filtering (100 Peers)** | 0.2428 ms | 0.3245 ms | 4,118.87 queries/sec |
| **Transport Query Roundtrip** | 1.4320 ms | 1.7519 ms | 698.31 ops/sec |
| **Transport Adv Roundtrip** | 0.5668 ms | 0.6424 ms | 1,764.35 ops/sec |

- **Peak Memory Consumption:** 3.42 MB delta.
- **Neural Invariant:** $\Delta W = 0$ strictly preserved.
  - Parameters: 3,443,136
  - Vocabulary: 4096
  - Max Context: 512
  - SHA-256 Weight Hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`

---

## 8. Step 38 Boundary & Future Evolution

Step 37 establishes the informational layer for resource discovery. It intentionally does **NOT** implement:
- Dynamic workload orchestration or remote task scheduling.
- Cognitive task delegation protocols.
- Automated peer-to-peer load balancing.
- Decentralized model routing.

These capabilities belong exclusively to subsequent steps (such as Step 38: Sovereign Workload Scheduling & Cognitive Delegation). Step 37 stops strictly at secure, authenticated, sovereign resource & capability advertisement.
