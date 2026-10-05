"""
ChakrView Step 115-120: Multi-Agent and Federated Cognition Package.
"""

from chakrview.cognition.multi_agent.contracts import (
    WorkerRole,
    ReviewerVerdict,
    WorkerContract,
    WorkerPackage,
)
from chakrview.cognition.multi_agent.result import (
    WorkerExecutionStatus,
    WorkerResult,
    ReviewerCritique,
    SynthesisResult,
)
from chakrview.cognition.multi_agent.worker import (
    BaseCognitiveWorker,
    ProjectAnalystWorker,
    PlannerWorker,
    ImplementerWorker,
    TestEngineerWorker,
    ReviewerWorker,
)
from chakrview.cognition.multi_agent.coordinator import MultiAgentCoordinator
from chakrview.cognition.multi_agent.protocol import MultiAgentProtocol
from chakrview.cognition.multi_agent.transport import (
    PROTOCOL_VERSION_V1,
    RequestEnvelope,
    ResponseEnvelope,
    EnvelopeType,
)
from chakrview.cognition.multi_agent.federation import (
    FederatedWorkerMetadata,
    FederatedScheduler,
)
from chakrview.cognition.multi_agent.resource_federation import (
    ResourceCapacityLevel,
    WorkerResourceProfile,
    TaskResourceRequirements,
    ResourceAwareWorkerSelector,
)
from chakrview.cognition.multi_agent.transport_channel import (
    BaseTransportChannel,
    SubprocessTransportChannel,
    TransportMessageCorrelation,
)
from chakrview.cognition.multi_agent.distributed_scheduler import (
    DistributedDAGScheduler,
)
from chakrview.cognition.multi_agent.fault_tolerance import (
    FailureRecoveryAudit,
    FederationFaultToleranceManager,
)
from chakrview.cognition.multi_agent.curriculum_integration import (
    run_extended_curriculum_wave,
)

__all__ = [
    "WorkerRole",
    "ReviewerVerdict",
    "WorkerContract",
    "WorkerPackage",
    "WorkerExecutionStatus",
    "WorkerResult",
    "ReviewerCritique",
    "SynthesisResult",
    "BaseCognitiveWorker",
    "ProjectAnalystWorker",
    "PlannerWorker",
    "ImplementerWorker",
    "TestEngineerWorker",
    "ReviewerWorker",
    "MultiAgentCoordinator",
    "MultiAgentProtocol",
    "PROTOCOL_VERSION_V1",
    "RequestEnvelope",
    "ResponseEnvelope",
    "EnvelopeType",
    "FederatedWorkerMetadata",
    "FederatedScheduler",
    "ResourceCapacityLevel",
    "WorkerResourceProfile",
    "TaskResourceRequirements",
    "ResourceAwareWorkerSelector",
    "BaseTransportChannel",
    "SubprocessTransportChannel",
    "TransportMessageCorrelation",
    "DistributedDAGScheduler",
    "FailureRecoveryAudit",
    "FederationFaultToleranceManager",
    "run_extended_curriculum_wave",
]
