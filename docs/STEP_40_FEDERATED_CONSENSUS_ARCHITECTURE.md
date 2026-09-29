# ChakrView Step 40: Federated Consensus, Byzantine Fault Tolerance & Multi-Node State Agreement

**Status:** RATIFIED & LOCKED  
**Step:** 40  
**Version:** 1.0.0  
**Axiom:** `CONSENSUS != AUTHORITY` | `LOCAL_POLICY > CONSENSUS_DECISION` | `EQUIVOCATION == BYZANTINE_FAULT`  
**Neural Core Delta:** $\Delta W = 0$ (Frozen Weights & Architecture Verified)

---

## 1. Executive Architectural Summary

Prior to Step 40, ChakrView established secure message transport (Step 36), distributed resource advertisement (Step 37), task orchestration (Step 38), and checkpointed work migration (Step 39). While individual worker nodes could fail and recover, multi-node agreement on cluster topology, global node revocations, and authoritative task finalization relied on pairwise exchanges without distributed consensus.

Step 40 introduces **production-grade federated consensus, Byzantine fault tolerance (BFT), and replicated state machine agreement**:
1. **Three-Phase BFT State Replication:** `PROPOSE` $\to$ `PREVOTE` $\to$ `PRECOMMIT` $\to$ `COMMIT` protocol.
2. **Cryptographic Quorum Certificates (QC):** Every state transition requires verifiable multi-signatures from a supermajority ($\ge 2f + 1$ out of $3f + 1$ in BFT mode, or $\ge \lfloor N/2 \rfloor + 1$ in CFT mode).
3. **Byzantine Fault Detection:** Immediate detection and quarantine of proposer equivocation (double-proposing different payloads for the same slot) and validator double-voting.
4. **Deterministic Replicated State Machine (RSM):** Append-only log with tamper-evident cryptographic parent hash chaining.
5. **Sovereign Local Gate Axiom (`CONSENSUS != AUTHORITY`):** A quorum agreement across remote peers can **never** override local sovereign safety policies or force unauthorized local capability invocation (`LOCAL_POLICY > CONSENSUS_DECISION`).

---

## 2. Non-Negotiable Invariants

```
CONSENSUS != AUTHORITY
LOCAL_POLICY > CONSENSUS_DECISION
ADVERTISEMENT != PERMISSION
UNREACHABLE != REVOKED
WORKER_FAILURE != TASK_FAILURE
DUPLICATE_EXECUTION != DUPLICATE_COMMIT
STALE_PROPOSAL != VALID_TRANSITION
EQUIVOCATION == BYZANTINE_FAULT
QUORUM_CERTIFICATE MUST BE AUTHENTICATED
SUPERMAJORITY REQUIRED (2f + 1 of 3f + 1)
TENANT_ISOLATION IS MANDATORY
ZERO SECRET EXPOSURE
ΔW = 0
```

---

## 3. Consensus Subsystem Architecture

### 3.1 Consensus Models & Transitions (`models.py`)
- **`ConsensusProposal`:** Identifies `(epoch, round, height)`, proposer node ID, transition type, payload, parent hash, tenant ID, and canonical SHA-256 digest `proposal_digest`.
- **`ConsensusVote`:** Identifies voter node ID, vote type (`PREVOTE`, `PRECOMMIT`, `VIEW_CHANGE`), voted proposal digest, and signature.
- **`QuorumCertificate`:** Cryptographic container binding proposal digest, vote type, voter IDs, and signatures proving supermajority quorum.
- **`EquivocationEvidence`:** Immutable proof of Byzantine double-proposal containing both conflicting proposal digests signed by the same offender.
- **`ConsensusTransitionType`:**
  - `MEMBERSHIP_TOPOLOGY_UPDATE`: cluster validator set changes.
  - `PEER_REVOCATION_AGREEMENT`: cluster-wide quarantine/revocation of a compromised node.
  - `TASK_COMMIT_FINALIZATION`: authoritative cluster agreement on finalized task outputs.
  - `CHECKPOINT_AUTHORIZATION`: cluster agreement on authoritative checkpoint references.
  - `EPOCH_ADVANCEMENT`: monotonic advancement of the cluster epoch.

### 3.2 Consensus Validator & Byzantine Detection (`validator.py`)
- **Syntax, Size & Payload Bounds:** Enforces maximum payload ceiling (1 MB) and recursive prohibited keyword/regex scanning to prevent private key or model weight leakage.
- **Monotonic Slot Progression:** Rejects stale epochs (`epoch < current_epoch`) and stale heights (`height <= current_height`).
- **Equivocation Detection:** Tracks seen proposals per slot `(epoch, round, height, proposer_id)`. If two distinct proposal digests are observed from the same proposer, records `EquivocationEvidence` and raises `EquivocationError`.
- **Double Voting Detection:** Tracks seen votes per slot `(epoch, round, height, voter_id, vote_type)`. If conflicting votes are received from the same validator, drops the second vote and raises `DoubleVotingError`.
- **Quorum Certificate Verification:** Checks unique voter count against the computed quorum threshold $Q = \lfloor \frac{2N}{3} \rfloor + 1$ (or $\lfloor \frac{N}{2} \rfloor + 1$) and verifies all voters are authorized.

### 3.3 Deterministic Replicated State Machine (`state_machine.py`)
- **Cryptographic Parent Hash Chain:** Every committed entry binds `parent_hash == latest_state_root`. New state root computed as:
  $$\text{state\_root}_N = \text{SHA-256}(\text{state\_root}_{N-1} : \text{height} : \text{proposal\_digest} : \text{qc\_digest})$$
- **Tamper Evidence:** Any log divergence or parent hash tampering raises `StateDivergenceError`.
- **Sovereign Local Defense (`CONSENSUS != AUTHORITY`):**
  - Before applying any transition locally, the state machine evaluates the transition payload against the local `CapabilityGate`.
  - If local policy forbids the capability, local actuation is refused with `SovereignPolicyViolationError`.
  - Remote quorums cannot unilaterally force self-revocation or self-quarantine on the local node.

### 3.4 Consensus Engine (`engine.py`)
- **Deterministic Round-Robin Leader Schedule:**
  $$\text{leader} = \text{validators}[(\text{height} + \text{round}) \pmod N]$$
- **Three-Phase Protocol:**
  1. Designated leader creates and broadcasts `ConsensusProposal`.
  2. Validators validate proposal and broadcast `PREVOTE`. When $\ge Q$ prevotes are gathered, a PREVOTE QC is assembled.
  3. Validators broadcast `PRECOMMIT` upon observing the PREVOTE QC. When $\ge Q$ precommits are gathered, a PRECOMMIT QC is assembled.
  4. Node executes `commit_block`, applying the transition to the Replicated State Machine and resetting round state.
- **View Change & Liveness:**
  - If a leader fails or times out, validators broadcast `VIEW_CHANGE` votes.
  - Advancing to the next round triggers the next deterministic leader in the round-robin schedule.

---

## 4. WAL & Audit Trail Integration

Consensus lifecycle events are recorded in WAL (`JournalEntryType`) and Security Audit logs (`AuditEventType`):
- `CONSENSUS_PROPOSAL_BROADCAST` / `CONSENSUS_PROPOSAL_CREATED`
- `CONSENSUS_VOTE_RECORDED` / `CONSENSUS_VOTE_CAST` / `CONSENSUS_VOTE_RECEIVED`
- `CONSENSUS_QUORUM_REACHED` / `CONSENSUS_QUORUM_CERTIFIED`
- `CONSENSUS_BLOCK_COMMITTED` / `CONSENSUS_COMMIT_EXECUTED`
- `CONSENSUS_VIEW_CHANGED` / `CONSENSUS_VIEW_TIMEOUT`
- `CONSENSUS_EQUIVOCATION_DETECTED`
- `CONSENSUS_SOVEREIGN_OVERRIDE`

---

## 5. Neural Core Immutability ($\Delta W = 0$)

Neural core parameters and architecture remain strictly frozen:
- **Total Parameters:** $3,443,136$
- **Vocabulary Size:** $4,096$
- **Context Length:** $512$
- **Model SHA-256 Hash:** `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Verification:** Verified unchanged before and after multi-node consensus operations.
