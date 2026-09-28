"""
Adaptive Execution Strategy Selection for Step 28 Cognitive Orchestration.

Maps WorkloadClass and system resource state to an overarching OrchestrationStrategy.
"""

from enum import Enum
from typing import Dict, Any, Optional

from chakrview.cognition.orchestration.models import WorkloadClass
from chakrview.cognition.adaptation.profiles import ResourceProfile


class OrchestrationStrategy(str, Enum):
    """Execution strategy determining coordination topology and rigor."""
    LOCAL_MINIMAL = "LOCAL_MINIMAL"                  # 1 local agent, zero distributed overhead
    LOCAL_FEDERATED = "LOCAL_FEDERATED"              # Multi-agent federation executing locally in-process
    DISTRIBUTED_ROUTED = "DISTRIBUTED_ROUTED"        # Multi-node distributed task routing via transport
    VERIFICATION_PIPELINE = "VERIFICATION_PIPELINE"  # Rigid verification-first pipeline
    FAIL_SAFE_DEGRADED = "FAIL_SAFE_DEGRADED"        # Degraded safety-first minimal execution


class AdaptiveStrategySelector:
    """
    Selects the optimal OrchestrationStrategy for a given workload and environment.
    """

    @classmethod
    def select_strategy(
        cls,
        workload_class: WorkloadClass,
        resource_profile: ResourceProfile,
        available_nodes_count: int = 1,
        has_node_failures: bool = False,
    ) -> OrchestrationStrategy:
        """
        Derive the appropriate OrchestrationStrategy deterministically.
        """
        # 1. Resource constrained or failure-degraded
        if workload_class == WorkloadClass.RESOURCE_CONSTRAINED or has_node_failures:
            return OrchestrationStrategy.FAIL_SAFE_DEGRADED

        # 2. Simple tasks: always local minimal
        if workload_class == WorkloadClass.SIMPLE:
            return OrchestrationStrategy.LOCAL_MINIMAL

        # 3. Explicit verification required or conflicted tasks
        if workload_class in (WorkloadClass.VERIFICATION_REQUIRED, WorkloadClass.CONFLICTED):
            return OrchestrationStrategy.VERIFICATION_PIPELINE

        # 4. Complex or Standard tasks with distributed nodes available
        if available_nodes_count > 1 and resource_profile in (ResourceProfile.STANDARD, ResourceProfile.HIGH_RESOURCE):
            return OrchestrationStrategy.DISTRIBUTED_ROUTED

        # 5. Default to local federated coordination
        return OrchestrationStrategy.LOCAL_FEDERATED
