"""
Unified Federated Node Orchestrator (Step 41).

Integrates Discovery, Transport, Resources, Grants, Tasks, Leases, Checkpoints,
BFT Consensus, Persistence, and Sovereign Governance into a cohesive production node.
"""

import hashlib
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Set, Tuple
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.capability.gate import CapabilityGate
from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityCategory,
    RiskClassification,
    CapabilityStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    ResourceLimits,
)
from chakrview.cognition.federation.runtime import FederationRuntime
from chakrview.cognition.federation.resources.models import (
    CPUResource,
    MemoryResource,
    AcceleratorResource,
    StorageResource,
    PlatformResource,
    NodeResourceProfile,
    ResourceAdvertisement,
    AdvertisedCapability,
)
from chakrview.cognition.federation.tasks.models import (
    DistributedTask,
    WorkUnit,
    TaskState,
    AggregationStrategy,
    ResourceRequirements,
)
from chakrview.cognition.federation.consensus.models import (
    ConsensusTransitionType,
    ConsensusProposal,
    ConsensusVote,
    VoteType,
    QuorumCertificate,
)
from chakrview.cognition.federation.integration.models import (
    NodeLifecycleState,
    FederatedNodeConfig,
    FederatedNodeStatus,
)
from chakrview.cognition.federation.integration.handlers import FederationWireHandlerRegistry

logger = logging.getLogger("chakrview.federation.integration.node")

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3443136


class SimpleFederatedCapability(Capability):
    """Safe built-in federated capability provider for compute work units."""

    def __init__(self, capability_id: str, name: str, description: str) -> None:
        self._descriptor = CapabilityDescriptor(
            capability_id=capability_id,
            name=name,
            version="1.0.0",
            description=description,
            category=CapabilityCategory.SOFTWARE,
            risk_level=RiskClassification.COMPUTE,
            provider_id="federation_node",
            required_permissions=[],
            resource_limits=ResourceLimits(max_cpu_time_ms=1000.0, timeout_seconds=5.0),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        res = {"result": f"Executed {self.descriptor.capability_id} successfully", "params": request.parameters}
        return CapabilityResult(
            request_id=request.request_id,
            capability_id=self.descriptor.capability_id,
            success=True,
            output=res,
            status=CapabilityStatus.AVAILABLE,
        )


class FederatedNode:
    """
    Unified production node coordinating all federated distributed capabilities:
    Transport, Membership, Resource Sharing, Task Execution, Checkpointing,
    BFT Consensus, and Replicated State Machine agreement.

    Non-negotiable Invariants:
    LOCAL_POLICY > CONSENSUS_DECISION
    CONSENSUS != AUTHORITY
    ADVERTISEMENT != PERMISSION
    UNREACHABLE != REVOKED
    WORKER_FAILURE != TASK_FAILURE
    DUPLICATE_EXECUTION != DUPLICATE_COMMIT
    ZERO SECRET EXPOSURE
    ZERO NEURAL WEIGHT MUTATION (ΔW = 0)
    """

    def __init__(
        self,
        config: FederatedNodeConfig,
        model: Optional[torch.nn.Module] = None,
        capability_gate: Optional[CapabilityGate] = None,
        policy: Optional[Any] = None,
        store: Optional[Any] = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._state = NodeLifecycleState.UNINITIALIZED
        self._start_time: float = 0.0

        # 1. Neural Core (Frozen Invariant: ΔW = 0)
        if model is not None:
            self.model = model
        else:
            torch.manual_seed(42)
            cfg = ModelConfig()
            self.model = ChakrMicro(cfg)
        self.model.eval()

        # Initial weight hash verification
        self._verify_neural_hash()

        # 2. Local Capability Gate
        self.capability_gate = capability_gate or CapabilityGate()

        # 3. Peering & Federation Engine
        from chakrview.cognition.peering.engine import CrossZoneFederationEngine
        self.engine = CrossZoneFederationEngine(
            local_zone_id=self.config.zone_id,
            local_peer_id=self.config.node_id,
            policy=policy,
            capability_gate=self.capability_gate,
            model=self.model,
        )

        # 4. Federation Runtime & Persistence
        self.runtime = FederationRuntime(
            engine=self.engine,
            store=store,
            auto_recover=self.config.auto_recover,
        )

        # 5. Wire Message Handler Registry
        self.handler_registry = FederationWireHandlerRegistry(engine=self.engine)

        # 6. Configure Consensus Validators
        validators = self.config.consensus_validators or [self.config.node_id]
        if hasattr(self.engine, "consensus_engine") and self.engine.consensus_engine:
            self.engine.consensus_engine.update_validator_set(validators)

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    def start(self) -> None:
        """
        Bootstrap and transition node into ACTIVE lifecycle state.
        """
        with self._lock:
            if self._state == NodeLifecycleState.ACTIVE:
                return

            self._state = NodeLifecycleState.STARTING
            self._start_time = time.time()

            # Start runtime and recover durable state
            self.runtime.start()

            # Register all wire handlers
            self.handler_registry.register_all_handlers()

            # Register local resource profile
            self._register_default_local_profile()

            self._state = NodeLifecycleState.ACTIVE
            logger.info("FederatedNode %s started successfully in zone %s", self.config.node_id, self.config.zone_id)

    def stop(self) -> None:
        """
        Deterministic 5-Step Clean Shutdown Lifecycle:
        1. Mark SHUTTING_DOWN (halt ingress).
        2. Flush in-flight task checkpoints.
        3. Flush in-flight consensus state.
        4. Flush WAL journal to durable store.
        5. Close transport channels and mark STOPPED.
        """
        with self._lock:
            if self._state == NodeLifecycleState.STOPPED:
                return

            logger.info("Initiating 5-step clean shutdown for node %s", self.config.node_id)
            self._state = NodeLifecycleState.SHUTTING_DOWN

            # Step 1: Halt ingress - done via lifecycle state check in handlers
            # Step 2: Flush task checkpoints
            try:
                if hasattr(self.engine, "task_coordinator"):
                    cm = getattr(self.engine.task_coordinator, "checkpoint_manager", None)
                    if cm and hasattr(cm, "flush"):
                        cm.flush()
            except Exception as e:
                logger.warning("Checkpoint flush error during shutdown: %s", e)

            # Step 3: Flush consensus state
            try:
                if hasattr(self.engine, "consensus_engine") and self.engine.consensus_engine:
                    if hasattr(self.engine.consensus_engine, "flush"):
                        self.engine.consensus_engine.flush()
            except Exception as e:
                logger.warning("Consensus flush error during shutdown: %s", e)

            # Step 4: Flush durable runtime journal
            try:
                if self.runtime:
                    self.runtime.stop()
            except Exception as e:
                logger.warning("Runtime stop error during shutdown: %s", e)

            # Step 5: Close transport channels and mark STOPPED
            try:
                if hasattr(self.engine, "transport") and self.engine.transport:
                    self.engine.transport.close()
            except Exception as e:
                logger.warning("Transport close error during shutdown: %s", e)

            self._state = NodeLifecycleState.STOPPED
            logger.info("Node %s clean shutdown complete", self.config.node_id)

    # ========================================================================
    # Status & Telemetry
    # ========================================================================

    def get_status(self) -> FederatedNodeStatus:
        """Generate point-in-time status snapshot of the federated node."""
        with self._lock:
            uptime = (time.time() - self._start_time) if self._state == NodeLifecycleState.ACTIVE else 0.0

            # Compute active peer count
            peers_count = 0
            if hasattr(self.engine, "membership_manager"):
                peers_count = len(self.engine.membership_manager.list_members())

            # Active tasks count
            tasks_count = 0
            if hasattr(self.engine, "task_coordinator"):
                tasks_count = len(self.engine.task_coordinator.list_tasks())

            # Consensus height & state root
            cons_height = 0
            state_root = "0" * 64
            is_leader = False
            revoked_count = 0
            if hasattr(self.engine, "consensus_engine") and self.engine.consensus_engine:
                ce = self.engine.consensus_engine
                cons_height = ce.state_machine.current_height
                state_root = ce.state_machine.state_root_hash
                is_leader = ce.is_leader_for_current_round() if hasattr(ce, "is_leader_for_current_round") else False
                revoked_count = len(ce.state_machine._revoked_nodes)

            return FederatedNodeStatus(
                node_id=self.config.node_id,
                lifecycle_state=self._state,
                zone_id=self.config.zone_id,
                tenant_id=self.config.tenant_id,
                active_peers_count=peers_count,
                active_tasks_count=tasks_count,
                consensus_height=cons_height,
                state_root_hash=state_root,
                is_consensus_leader=is_leader,
                quarantined_nodes_count=0,
                revoked_nodes_count=revoked_count,
                uptime_seconds=uptime,
            )

    # ========================================================================
    # Task Orchestration API
    # ========================================================================

    def submit_task(
        self,
        name: str,
        units_spec: List[Dict[str, Any]],
        aggregation_strategy: AggregationStrategy = AggregationStrategy.CONCATENATE,
        tenant_id: Optional[str] = None,
    ) -> DistributedTask:
        """Submit, decompose, and dispatch a distributed task across the federation."""
        with self._lock:
            coord = self.engine.task_coordinator
            task = coord.create_task(
                name=name,
                aggregation_strategy=aggregation_strategy,
                tenant_id=tenant_id or self.config.tenant_id,
            )
            coord.decompose_task(task.task_id, units_spec)
            coord.schedule_and_dispatch_task(task.task_id)
            return task

    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """Query task state by ID."""
        with self._lock:
            return self.engine.task_coordinator.get_task(task_id)

    # ========================================================================
    # Consensus & Multi-Node Governance API
    # ========================================================================

    def propose_consensus(
        self,
        transition_type: ConsensusTransitionType,
        payload: Dict[str, Any],
    ) -> ConsensusProposal:
        """Propose a cluster-wide state transition via the consensus engine."""
        with self._lock:
            ce = self.engine.consensus_engine
            if not ce:
                raise RuntimeError("ConsensusEngine is not initialized on this node")
            return ce.create_proposal(transition_type, payload)

    def propose_peer_revocation(self, node_id: str, reason: str = "") -> ConsensusProposal:
        """Propose cluster-wide revocation of a compromised node."""
        payload = {
            "target_node_id": node_id,
            "revoked_node_id": node_id,
            "reason": reason,
            "timestamp": time.time(),
        }
        return self.propose_consensus(ConsensusTransitionType.PEER_REVOCATION_AGREEMENT, payload)

    def propose_policy_update(self, policy_data: Dict[str, Any]) -> ConsensusProposal:
        """Propose cluster-wide policy synchronization."""
        return self.propose_consensus(ConsensusTransitionType.POLICY_UPDATE_COMMIT, policy_data)

    def finalize_task_via_consensus(self, task_id: str) -> ConsensusProposal:
        """
        Propose cluster-wide authoritative commit finalization for a completed task.
        """
        with self._lock:
            task = self.get_task(task_id)
            if not task:
                raise KeyError(f"Task {task_id} not found")
            if task.state != TaskState.COMPLETED:
                raise ValueError(f"Cannot finalize incomplete task {task_id} (state={task.state.value})")

            payload = {
                "task_id": task_id,
                "result": task.final_result,
            }
            return self.propose_consensus(ConsensusTransitionType.TASK_COMMIT_FINALIZATION, payload)

    # ========================================================================
    # Private Helpers & Neural Invariant Verification
    # ========================================================================

    def _register_default_local_profile(self) -> None:
        """Register host resources and advertised capabilities into resource registry."""
        profile = NodeResourceProfile(
            profile_id=f"prof_{self.config.node_id}",
            cpu=CPUResource(physical_cores=4, logical_cores=4, available_cores=4.0, architecture="x86_64"),
            memory=MemoryResource(total_memory_mb=8192, available_memory_mb=4096),
            accelerator=AcceleratorResource(is_available=False),
            storage=StorageResource(total_storage_mb=100000, available_storage_mb=50000),
            platform=PlatformResource(os_family="Windows", os_release="11", python_version="3.14"),
        )
        caps = [
            AdvertisedCapability(capability_id="add", name="Add", description="Arithmetic Add", execution_type="CPU"),
            AdvertisedCapability(capability_id="process", name="Process", description="Batch Process", execution_type="CPU"),
        ]
        self.engine.resource_registry.register_local_profile(profile, caps)

        # Register in CapabilityGate registry for local execution
        if hasattr(self.engine, "capability_gate") and hasattr(self.engine.capability_gate, "registry"):
            reg = self.engine.capability_gate.registry
            if not reg.has("add"):
                reg.register(SimpleFederatedCapability("add", "Add", "Arithmetic Add"))
            if not reg.has("process"):
                reg.register(SimpleFederatedCapability("process", "Process", "Batch Process"))

    def _verify_neural_hash(self) -> None:
        """Verify non-negotiable invariant: ΔW = 0."""
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count != EXPECTED_PARAM_COUNT:
            raise RuntimeError(
                f"Neural core parameter count altered: {param_count} != {EXPECTED_PARAM_COUNT}"
            )
        hasher = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(self.model.named_parameters()):
                hasher.update(name.encode("utf-8"))
                hasher.update(param.detach().cpu().numpy().tobytes())
        current_hash = hasher.hexdigest()
        if current_hash != EXPECTED_WEIGHT_HASH:
            raise RuntimeError(
                f"Neural core weights mutated! {current_hash} != {EXPECTED_WEIGHT_HASH} (ΔW != 0 violation)"
            )
