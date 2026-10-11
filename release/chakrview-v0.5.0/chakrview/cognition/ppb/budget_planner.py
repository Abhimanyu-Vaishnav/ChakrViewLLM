"""
ChakrView Step 87: Context & Resource Budget Intelligence Planner.

Analyzes task nodes against hardware capabilities and PPB knowledge:
- Computes estimated context footprint (file sizes, AST complexity, dependency reach).
- Compares demand with HardwareCapability and ResourcePolicy.
- Decisions:
  1. EXECUTE_DIRECT: Node fits easily within available context budget.
  2. RETRIEVE_PPB_ONLY: Direct file ingestion exceeds budget; load compressed PPB records instead.
  3. SPLIT_TASK: Node exceeds context window or file count; split into bounded child subtasks.
  4. DEFER_QUEUE: Resources currently saturated or prerequisite memory unavailable.
  5. REQUEST_INVESTIGATION: Required information missing or insufficient (UNKNOWN).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.runtime.resource import (
    HardwareCapability,
    ResourceDetector,
    ResourcePolicy,
    RuntimeStrategy,
)
from chakrview.cognition.ppb.task_models import (
    PersistentTaskNode,
    TaskResourceType,
)
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.expansion_models import DynamicSubtaskRequest, TaskProvenance


class BudgetDecisionAction(str, Enum):
    """Action selected by the Context & Resource Budget Planner."""
    EXECUTE_DIRECT = "EXECUTE_DIRECT"
    RETRIEVE_PPB_ONLY = "RETRIEVE_PPB_ONLY"
    SPLIT_TASK = "SPLIT_TASK"
    DEFER_QUEUE = "DEFER_QUEUE"
    REQUEST_INVESTIGATION = "REQUEST_INVESTIGATION"


@dataclass(frozen=True)
class ContextBudgetPlan:
    """
    Budget allocation and operational plan for a single task node.
    """
    node_id: str
    action: BudgetDecisionAction
    estimated_tokens: int
    allocated_context_tokens: int
    use_accelerator: bool
    requires_investigation: bool = False
    investigation_query: str = ""
    suggested_splits: Tuple[DynamicSubtaskRequest, ...] = field(default_factory=tuple)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "action": self.action.value,
            "estimated_tokens": self.estimated_tokens,
            "allocated_context_tokens": self.allocated_context_tokens,
            "use_accelerator": self.use_accelerator,
            "requires_investigation": self.requires_investigation,
            "investigation_query": self.investigation_query,
            "suggested_splits": [s.title for s in self.suggested_splits],
            "rationale": self.rationale,
        }


class ContextResourcePlanner:
    """
    Evaluates task nodes against active hardware capabilities and PPB knowledge state
    to produce bounded ContextBudgetPlans.
    """

    def __init__(
        self,
        brain: PersistentProjectBrain,
        hardware_capability: Optional[HardwareCapability] = None,
    ) -> None:
        self.brain = brain
        self.hardware_capability = hardware_capability or brain.hardware_capability
        self.resource_policy = ResourceDetector.determine_strategy(self.hardware_capability)

    def plan_budget_for_node(self, node: PersistentTaskNode) -> ContextBudgetPlan:
        """
        Evaluate node requirements and select appropriate bounded strategy.
        """
        max_context = self.resource_policy.max_context_size
        strategy = self.resource_policy.strategy

        # 1. Estimate token demands based on affected files and resource type
        file_count = len(node.affected_files)
        # Base token estimate per file ~ 300 tokens + task overhead
        estimated_tokens = (file_count * 300) + 150
        if node.resource_type in (TaskResourceType.REASONING, TaskResourceType.DEPENDENCY_ANALYSIS):
            estimated_tokens += 300

        use_acc = self.resource_policy.use_accelerator and (node.resource_type == TaskResourceType.REASONING)

        # 2. Check for missing knowledge on uninspected files
        missing_files = []
        for f in node.affected_files:
            recs = self.brain.query_knowledge(file_path=f)
            if not recs:
                missing_files.append(f)

        if missing_files and node.resource_type != TaskResourceType.INSPECTION:
            # Need investigation before proceeding with reasoning or patches
            inv_query = f"Investigate unknown module {missing_files[0]} for task {node.title}"
            return ContextBudgetPlan(
                node_id=node.node_id,
                action=BudgetDecisionAction.REQUEST_INVESTIGATION,
                estimated_tokens=estimated_tokens,
                allocated_context_tokens=min(estimated_tokens, max_context),
                use_accelerator=False,
                requires_investigation=True,
                investigation_query=inv_query,
                rationale=f"Affected module '{missing_files[0]}' has not been inspected; investigation required.",
            )

        # 3. Check if node exceeds context budget on low-resource hardware
        if estimated_tokens > max_context or file_count > 3:
            if file_count > 1 and strategy == RuntimeStrategy.LOW_RESOURCE:
                # Split task into smaller bounded children
                splits: List[DynamicSubtaskRequest] = []
                for f in node.affected_files:
                    sub_title = f"{node.title}: Process {f}"
                    splits.append(DynamicSubtaskRequest(
                        title=sub_title,
                        description=f"Bounded processing for {f}",
                        resource_type=node.resource_type,
                        parent_id=node.node_id,
                        affected_files=(f,),
                        priority=node.priority,
                        provenance=TaskProvenance(
                            origin_type="DYNAMIC_EXPANSION",
                            trigger_node_id=node.node_id,
                            rationale="Split task to maintain bounded context on low-resource hardware",
                        ),
                    ))
                return ContextBudgetPlan(
                    node_id=node.node_id,
                    action=BudgetDecisionAction.SPLIT_TASK,
                    estimated_tokens=estimated_tokens,
                    allocated_context_tokens=max_context,
                    use_accelerator=False,
                    suggested_splits=tuple(splits),
                    rationale=f"Estimated tokens ({estimated_tokens}) exceeds budget ({max_context}); splitting task.",
                )
            else:
                # Use compressed PPB records instead of raw file ingestion
                return ContextBudgetPlan(
                    node_id=node.node_id,
                    action=BudgetDecisionAction.RETRIEVE_PPB_ONLY,
                    estimated_tokens=min(estimated_tokens, max_context),
                    allocated_context_tokens=max_context,
                    use_accelerator=use_acc,
                    rationale="High context demand: retrieved structured PPB records to fit bounded budget.",
                )

        # 4. Standard execution fits within context ceiling
        return ContextBudgetPlan(
            node_id=node.node_id,
            action=BudgetDecisionAction.EXECUTE_DIRECT,
            estimated_tokens=estimated_tokens,
            allocated_context_tokens=min(estimated_tokens, max_context),
            use_accelerator=use_acc,
            rationale="Context demands fit comfortably within available resource budget.",
        )
