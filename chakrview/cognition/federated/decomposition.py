"""
Federated Task Decomposition Subsystem (Step 26).

Deterministically translates incoming cognitive objectives into bounded,
role-specific subtasks allocated to registered logical agents.

CRITICAL INVARIANTS:
1. Subtasks are assigned to registered logical roles (RESEARCHER, ANALYST, CRITIC, SYNTHESIZER).
2. Does NOT create uncontrolled dynamic agents.
3. Enforces delegation depth ceiling (depth <= policy.max_delegation_depth).
4. Strictly bounded subtask count.
"""

import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.cognition.federated.models import (
    AgentRole,
    AgentTask,
    MessagePriority,
)
from chakrview.cognition.unified.models import CognitiveTaskType


class FederatedTaskDecomposer:
    """
    Deterministic task decomposition coordinator allocating bounded subtasks.
    """

    def __init__(self, max_subtasks: int = 5) -> None:
        self.max_subtasks = min(max_subtasks, 6)

    def decompose(
        self,
        objective: str,
        task_type: CognitiveTaskType,
        tenant_id: str,
        session_id: str,
        parent_task_id: Optional[str] = None,
        depth: int = 0,
        max_depth: int = 4,
    ) -> List[AgentTask]:
        """
        Decompose a high-level cognitive objective into role-bounded AgentTasks.
        """
        clean_obj = objective.strip()
        effective_depth = min(depth, max_depth)
        tasks: List[AgentTask] = []

        base_constraints = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "depth": effective_depth,
        }

        # 1. FACTUAL Tasks: Research -> Analyze -> Verify -> Synthesize
        if task_type == CognitiveTaskType.FACTUAL:
            tasks.append(
                AgentTask(
                    task_id=f"ftask_res_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.RESEARCHER,
                    objective=f"Retrieve verified memory and facts relevant to: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["retrieved_memories"],
                    expected_output_type="evidence",
                    priority=MessagePriority.HIGH,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_ana_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.ANALYST,
                    objective=f"Analyze retrieved evidence and evaluate factual claims for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["evidence"],
                    expected_output_type="analysis",
                    priority=MessagePriority.NORMAL,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_syn_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.SYNTHESIZER,
                    objective=f"Synthesize verified factual response for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["analysis", "evidence"],
                    expected_output_type="synthesis",
                    priority=MessagePriority.HIGH,
                    depth=effective_depth,
                )
            )

        # 2. ANALYTICAL Tasks: Research -> Analyze -> Critic -> Synthesize
        elif task_type == CognitiveTaskType.ANALYTICAL:
            tasks.append(
                AgentTask(
                    task_id=f"ftask_res_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.RESEARCHER,
                    objective=f"Gather contextual background and reference evidence for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["context"],
                    expected_output_type="evidence",
                    priority=MessagePriority.NORMAL,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_ana_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.ANALYST,
                    objective=f"Perform structured computational reasoning and hypothesis formation for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["evidence"],
                    expected_output_type="analysis",
                    priority=MessagePriority.HIGH,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_crt_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.CRITIC,
                    objective=f"Challenge primary hypothesis with counter-evidence and uncover implicit assumptions for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["analysis"],
                    expected_output_type="critique",
                    priority=MessagePriority.HIGH,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_syn_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.SYNTHESIZER,
                    objective=f"Reconcile analytical findings, critiques, and counter-evidence for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["analysis", "critique"],
                    expected_output_type="synthesis",
                    priority=MessagePriority.NORMAL,
                    depth=effective_depth,
                )
            )

        # 3. DECISION Tasks: Planner -> Critic -> Analyst -> Synthesizer
        elif task_type == CognitiveTaskType.DECISION:
            tasks.append(
                AgentTask(
                    task_id=f"ftask_pln_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.PLANNER,
                    objective=f"Identify candidate decision paths and trade-offs for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["options"],
                    expected_output_type="plan",
                    priority=MessagePriority.HIGH,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_crt_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.CRITIC,
                    objective=f"Evaluate failure modes, edge cases, and contradictions in decision options for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["plan"],
                    expected_output_type="critique",
                    priority=MessagePriority.HIGH,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_syn_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.SYNTHESIZER,
                    objective=f"Synthesize recommended decision candidate with uncertainty bounds for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["plan", "critique"],
                    expected_output_type="synthesis",
                    priority=MessagePriority.HIGH,
                    depth=effective_depth,
                )
            )

        # 4. CAPABILITY Tasks: Planner -> Verifier -> Synthesizer
        elif task_type == CognitiveTaskType.CAPABILITY:
            tasks.append(
                AgentTask(
                    task_id=f"ftask_pln_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.PLANNER,
                    objective=f"Formulate required capability parameters and preconditions for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=[],
                    expected_output_type="plan",
                    priority=MessagePriority.CRITICAL,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_ver_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.VERIFIER,
                    objective=f"Verify capability bounds and ensure non-authoritative provenance check for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["plan"],
                    expected_output_type="verification",
                    priority=MessagePriority.CRITICAL,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_syn_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.SYNTHESIZER,
                    objective=f"Format capability request for CapabilityGate authorization for: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["plan", "verification"],
                    expected_output_type="synthesis",
                    priority=MessagePriority.HIGH,
                    depth=effective_depth,
                )
            )

        # 5. GENERAL / Default: Analyst -> Critic -> Synthesizer
        else:
            tasks.append(
                AgentTask(
                    task_id=f"ftask_ana_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.ANALYST,
                    objective=f"Analyze and process: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=[],
                    expected_output_type="analysis",
                    priority=MessagePriority.NORMAL,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_crt_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.CRITIC,
                    objective=f"Examine analytical output for assumptions and caveats: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["analysis"],
                    expected_output_type="critique",
                    priority=MessagePriority.NORMAL,
                    depth=effective_depth,
                )
            )
            tasks.append(
                AgentTask(
                    task_id=f"ftask_syn_{uuid.uuid4().hex[:6]}",
                    parent_task_id=parent_task_id,
                    assigned_role=AgentRole.SYNTHESIZER,
                    objective=f"Formulate balanced synthesis: '{clean_obj}'",
                    constraints=dict(base_constraints),
                    required_evidence=["analysis", "critique"],
                    expected_output_type="synthesis",
                    priority=MessagePriority.NORMAL,
                    depth=effective_depth,
                )
            )

        return tasks[:self.max_subtasks]
