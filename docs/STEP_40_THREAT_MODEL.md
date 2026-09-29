# ChakrView Step 40: Formal Threat Model
## Federated Consensus, Byzantine Fault Tolerance & Multi-Node State Agreement

**Document Status:** ACTIVE & RATIFIED  
**Step:** 40  
**Version:** 1.0.0  
**Scope:** Distributed Consensus Protocol, Quorum Certificates, Byzantine Detection, Replicated State Machine, Sovereign Local Defense.

---

## 1. Security Context & Threat Landscape

In a distributed multi-node federated deployment, nodes cannot assume that all participating peers are benign, well-coordinated, or failure-free. Nodes may:
1. Crash, lose power, or drop network connectivity (Crash Faults).
2. Transmit delayed, reordered, or duplicated messages (Network Faults).
3. Exhibit arbitrary or malicious behavior: proposing contradictory state transitions to different peers, casting conflicting votes, forging signatures, or withholding proposals (Byzantine Faults).
4. Attempt to exploit consensus to override local sovereign safety policies or access cross-tenant state.

This threat model outlines the explicit threat vectors (TV-01 through TV-15) and their fail-closed mitigations in ChakrView Step 40.

---

## 2. Threat Vectors & Fail-Closed Mitigations

### TV-01: Proposer Equivocation (Double Proposing)
- **Threat:** A compromised or Byzantine proposer sends two different proposals for the same height/round to different subsets of the federation to cause network partition or divergent state.
- **Mitigation:**
  - Nodes maintain an immutable record of observed proposals for `(epoch, round, height)`.
  - If a second, distinct proposal is observed from the same proposer for the same slot, it is flagged as an `EQUIVOCATION_EVIDENCE` Byzantine violation.
  - The node immediately drops the conflicting proposal and triggers proposer slashing/quarantine.
- **Fail-Closed State:** The conflicting proposal is dropped; node only votes for the first authenticated proposal.

### TV-02: Double Voting / Contradictory Prevotes
- **Threat:** A Byzantine validator signs two contradictory votes (`PREVOTE` or `PRECOMMIT`) for different proposal digests in the same round.
- **Mitigation:**
  - Nodes maintain a per-round voter record. If a node submits more than one vote for different block hashes in the same round, the conflicting vote is rejected and logged as `BYZANTINE_DOUBLE_VOTE`.
  - Quorum verification verifies unique voter node IDs; duplicates from the same node ID are never counted toward quorum.
- **Fail-Closed State:** Only the first authenticated vote per validator is accepted.

### TV-03: Stale Epoch / View Replay Attack
- **Threat:** An attacker captures messages from an earlier epoch/round and replays them to force rollback of active consensus state.
- **Mitigation:**
  - All consensus envelopes require strictly monotonic `(epoch, round, height)` tuples.
  - Any message where `epoch < current_epoch` or `(epoch == current_epoch and round < current_round)` is discarded with `StaleConsensusMessageError`.
- **Fail-Closed State:** Stale messages dropped immediately.

### TV-04: Sybil / Unauthorized Consensus Node Participation
- **Threat:** An unknown or unvetted node attempts to inject proposals or votes into the consensus round.
- **Mitigation:**
  - Consensus membership is strictly restricted to authenticated members of the `FederationMembershipRegistry` with active `TrustGrant` records.
  - Senders not in the authorized validator set are rejected with `UnauthorizedValidatorError`.
- **Fail-Closed State:** Non-member messages ignored.

### TV-05: Quorum Forgery / Fabricated Quorum Certificate
- **Threat:** An attacker constructs an invalid Quorum Certificate (QC) claiming that a supermajority agreed to an unauthorized state transition.
- **Mitigation:**
  - Every QC must carry cryptographic evidence containing verified votes from $\ge 2f + 1$ distinct authorized validators.
  - Each constituent vote must verify against the voter's Ed25519 public key and the proposal digest.
- **Fail-Closed State:** Invalid QC rejected with `InvalidQuorumCertificateError`; transition is not applied.

### TV-06: Split-Brain Network Partition
- **Threat:** A network split divides the cluster into two equal or unequal partitions, causing both sides to attempt progress independently.
- **Mitigation:**
  - Strict supermajority quorum requirement: $Q = \lfloor \frac{2N}{3} \rfloor + 1$ (or majority $Q = \lfloor \frac{N}{2} \rfloor + 1$ for crash fault tolerance configurations).
  - In any partition with fewer than $Q$ reachable nodes, consensus progress halts safely.
- **Fail-Closed State:** Sub-quorum partition enters read-only paused state; no state transition can be committed without $Q$.

### TV-07: Byzantine Censorship / Proposal Withholding
- **Threat:** A designated leader/proposer goes offline or intentionally refuses to propose a block, stalling the federation.
- **Mitigation:**
  - Deterministic round-timer state machine.
  - If a valid proposal is not received within `proposal_timeout_sec`, validators broadcast a `VIEW_CHANGE` vote to advance to `round + 1` with a deterministic round-robin leader schedule.
- **Fail-Closed State:** Cluster transitions to next round; Byzantine leader is bypassed.

### TV-08: Liveness Starvation via Perpetual View Changes
- **Threat:** Adversarial nodes intentionally trigger premature view change timeouts to prevent commit finalization.
- **Mitigation:**
  - View changes require a quorum certificate of view-change votes before advancing the round.
  - Exponential backoff on round timeouts (`timeout * 1.5^round`) to guarantee network synchrony windows.
- **Fail-Closed State:** Round duration scales dynamically until consensus completes.

### TV-09: Cross-Tenant Consensus Contamination
- **Threat:** Proposals carrying tenant A state or tasks are evaluated or committed by a cluster not authorized for tenant A.
- **Mitigation:**
  - All consensus proposals carry explicit `tenant_id`.
  - Nodes enforce tenant isolation: validators verify that `proposal.tenant_id` matches their allowed tenant scope or cluster-wide scope.
- **Fail-Closed State:** Tenant mismatch triggers `TenantConsensusIsolationError`.

### TV-10: State Rollback / Log Divergence
- **Threat:** A node attempts to rewrite previously committed log slots.
- **Mitigation:**
  - The Replicated State Machine (RSM) log is strictly append-only and indexed by monotonic `height`.
  - Every proposal binds `parent_hash`, establishing a tamper-evident cryptographic hash chain.
- **Fail-Closed State:** Any attempt to overwrite or diverge from committed log slots triggers `StateDivergenceError`.

### TV-11: Consensus Override of Local Sovereignty (`CONSENSUS != AUTHORITY`)
- **Threat:** A supermajority of peers agrees on a malicious proposal that commands a node to perform an unauthorized local action, execute arbitrary code, or access local secrets.
- **Mitigation:**
  - Non-negotiable architectural invariant: `LOCAL_POLICY > CONSENSUS_DECISION`.
  - Before applying any committed transition locally, the node submits the payload to its local `CapabilityGate`. If local policy forbids the action, local execution is refused.
- **Fail-Closed State:** Quorum commitment is recorded in the audit log, but local execution fails closed.

### TV-12: Poisoned State Machine Transition Payloads
- **Threat:** Proposer submits syntactically valid JSON containing malicious commands, negative resources, or invalid enum values.
- **Mitigation:**
  - Strongly typed state machine transitions (`ConsensusTransitionType`).
  - Strict payload schema validation before voting. Payloads failing schema checks are voted `NIL` / rejected.
- **Fail-Closed State:** Invalid payload rejected in PREVOTE stage.

### TV-13: Resource Exhaustion / Denial of Service via Proposal Flooding
- **Threat:** Malicious node floods the network with oversized or rapid-fire proposals.
- **Mitigation:**
  - Max proposal payload size hard ceiling (1 MB).
  - Rate limiting on proposal broadcast per node.
  - Strict height indexing (nodes reject proposals for `height > current_height + 1`).
- **Fail-Closed State:** Flooding messages dropped; offending peer quarantined.

### TV-14: Secret Leakage through Replicated State Payloads
- **Threat:** Confidential session keys, passwords, or model weights are placed inside replicated consensus proposals.
- **Mitigation:**
  - Prohibited keyword and regex scanner recursively inspects all proposal payloads.
  - Private keys, session secrets, token vectors, and tensor weights are strictly banned.
- **Fail-Closed State:** Scanner raises `ProhibitedPayloadError`; proposal is rejected.

### TV-15: Neural Core Weight Tampering through Consensus State ($\Delta W = 0$)
- **Threat:** A consensus transition attempts to inject new neural weights, modify model architecture, or retrain the core brain.
- **Mitigation:**
  - Frozen neural core invariant: no consensus state transition handler has access to model parameters.
  - Pre- and post-consensus programmatic check of parameter count ($3,443,136$) and SHA-256 weight hash (`c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`).
- **Fail-Closed State:** Immediate fatal abort if weight hash differs ($\Delta W = 0$).

---

## 3. Threat Matrix & Test Mapping

| Threat Vector | Attack Type | Primary Defense | Verification Test |
|---|---|---|---|
| TV-01 | Proposer Equivocation | Single proposal per slot & equivocation detection | `test_proposer_equivocation_detection` |
| TV-02 | Double Voting | Round voter registry & unique voter checks | `test_double_voting_rejection` |
| TV-03 | Stale Epoch Replay | Monotonic height/epoch validation | `test_stale_epoch_message_rejection` |
| TV-04 | Unauthorized Validator | Membership registry & trust grant verification | `test_unauthorized_validator_rejection` |
| TV-05 | Quorum Forgery | Multi-signature QC verification with $\ge 2f + 1$ | `test_quorum_certificate_verification` |
| TV-06 | Split-Brain Partition | Supermajority requirement ($Q = \lfloor 2N/3 \rfloor + 1$) | `test_split_brain_partition_safety` |
| TV-07 | Leader Censorship | Deterministic round timeout & view change | `test_view_change_timeout_recovery` |
| TV-08 | View Change Liveness | Exponential backoff on rounds | `test_view_change_quorum_progression` |
| TV-09 | Cross-Tenant Contamination | Tenant scope check on proposals | `test_cross_tenant_consensus_isolation` |
| TV-10 | State Log Divergence | Cryptographic parent-hash chain | `test_replicated_state_machine_hash_chain` |
| TV-11 | Sovereignty Override | `LOCAL_POLICY > CONSENSUS` via `CapabilityGate` | `test_local_sovereign_policy_override` |
| TV-12 | Poisoned Payloads | Strongly typed schema validation | `test_poisoned_payload_rejection` |
| TV-13 | Proposal Flooding | 1 MB ceiling & height bounded ingress | `test_oversized_proposal_rejection` |
| TV-14 | Secret Leakage | Recursive prohibited payload filter | `test_zero_secret_leakage_in_consensus` |
| TV-15 | Neural Weight Mutation | Frozen core parameters and weight SHA-256 | `test_neural_weight_immutability_delta_w_zero` |
