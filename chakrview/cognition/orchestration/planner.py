"""
Adaptive Task Planner for Step 28 Cognitive Orchestration.

Translates cognitive task objectives and their WorkloadClass into bounded,
role-driven TaskPlans adhering to the Principle of Minimum Sufficient Bounded Cognition.

CRITICAL INVARIANTS:
1. STRICT BOUNDS: No unbounded recursive decomposition, no self-spawning agents.
2. MINIMUM SUFFICIENT COGNITION:
   Simple tasks allocate minimal agent sets; complex or conflicted tasks allocate
   necessary specialized roles without exceeding hard ceilings.
"""

from typing import Dict, List, Optional, Any
import uuid

from chakrview.cognition.orchestration.models import (
    TaskPlan,
    WorkloadClass,
    MAX_ORCHESTRATION_AGENTS,
    MAX_ORCHESTRATION_NODES,
    MAX_DELIBERATION_ROUNDS,
)
from chakrview.cognition.federated.models import AgentRole


class AdaptiveTaskPlanner:
    """
    Creates bounded, role-driven TaskPlans based on workload classification.
    """

    def plan_task(
        self,
        task_id: str,
        objective: str,
        workload_class: WorkloadClass,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        """
        Generate a deterministic, bounded TaskPlan tailored to the WorkloadClass.
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        context = context or {}

        if workload_class == WorkloadClass.SIMPLE:
            # Minimal single-agent set; no complex coordination overhead
            required_roles = [AgentRole.ANALYST]
            optional_roles = []
            role_dependencies = {}
            max_rounds = 1
            max_agent_count = 1
            max_node_count = 1
            execution_budget_ms = 2000
            verification_required = False
            termination_conditions = ["single_round_complete", "evidence_sufficient"]

        elif workload_class == WorkloadClass.STANDARD:
            # Balanced multi-agent federation
            required_roles = [AgentRole.ANALYST, AgentRole.RESEARCHER, AgentRole.SYNTHESIZER]
            optional_roles = [AgentRole.CRITIC]
            role_dependencies = {
                AgentRole.SYNTHESIZER.value: [AgentRole.ANALYST.value, AgentRole.RESEARCHER.value]
            }
            max_rounds = 2
            max_agent_count = 3
            max_node_count = 2
            execution_budget_ms = 5000
            verification_required = False
            termination_conditions = ["synthesis_consensus_reached", "max_rounds_reached"]

        elif workload_class == WorkloadClass.COMPLEX:
            # Expanded specialized roles: planning, research, analysis, critique, synthesis
            required_roles = [
                AgentRole.PLANNER,
                AgentRole.RESEARCHER,
                AgentRole.ANALYST,
                AgentRole.CRITIC,
                AgentRole.SYNTHESIZER,
            ]
            optional_roles = [AgentRole.VERIFIER]
            role_dependencies = {
                AgentRole.RESEARCHER.value: [AgentRole.PLANNER.value],
                AgentRole.ANALYST.value: [AgentRole.PLANNER.value],
                AgentRole.CRITIC.value: [AgentRole.ANALYST.value, AgentRole.RESEARCHER.value],
                AgentRole.SYNTHESIZER.value: [AgentRole.CRITIC.value],
            }
            max_rounds = 3
            max_agent_count = min(5, MAX_ORCHESTRATION_AGENTS)
            max_node_count = min(3, MAX_ORCHESTRATION_NODES)
            execution_budget_ms = 8000
            verification_required = False
            termination_conditions = ["critique_passed", "synthesis_complete", "max_rounds_reached"]

        elif workload_class == WorkloadClass.AMBIGUOUS:
            # Evidence gathering, analytical breakdown, and critique to resolve uncertainty
            required_roles = [
                AgentRole.RESEARCHER,
                AgentRole.ANALYST,
                AgentRole.CRITIC,
                AgentRole.SYNTHESIZER,
            ]
            optional_roles = [AgentRole.VERIFIER]
            role_dependencies = {
                AgentRole.CRITIC.value: [AgentRole.ANALYST.value, AgentRole.RESEARCHER.value],
                AgentRole.SYNTHESIZER.value: [AgentRole.CRITIC.value],
            }
            max_rounds = 2
            max_agent_count = 4
            max_node_count = 2
            execution_budget_ms = 6000
            verification_required = True
            termination_conditions = [
                "ambiguity_resolved",
                "uncertainty_explicitly_acknowledged",
                "max_rounds_reached",
            ]

        elif workload_class == WorkloadClass.CONFLICTED:
            # Disagreement isolation: Researcher, Analyst, Critic, Verifier, Synthesizer
            required_roles = [
                AgentRole.ANALYST,
                AgentRole.CRITIC,
                AgentRole.VERIFIER,
                AgentRole.SYNTHESIZER,
            ]
            optional_roles = [AgentRole.RESEARCHER]
            role_dependencies = {
                AgentRole.CRITIC.value: [AgentRole.ANALYST.value],
                AgentRole.VERIFIER.value: [AgentRole.CRITIC.value],
                AgentRole.SYNTHESIZER.value: [AgentRole.VERIFIER.value],
            }
            max_rounds = 3
            max_agent_count = 4
            max_node_count = 2
            execution_budget_ms = 7000
            verification_required = True
            termination_conditions = [
                "conflict_evaluated",
                "minority_evidence_preserved",
                "max_rounds_reached",
            ]

        elif workload_class == WorkloadClass.VERIFICATION_REQUIRED:
            # Mandatory independent verifier stage; no premature synthesis
            required_roles = [AgentRole.ANALYST, AgentRole.VERIFIER, AgentRole.SYNTHESIZER]
            optional_roles = [AgentRole.CRITIC]
            role_dependencies = {
                AgentRole.VERIFIER.value: [AgentRole.ANALYST.value],
                AgentRole.SYNTHESIZER.value: [AgentRole.VERIFIER.value],
            }
            max_rounds = 2
            max_agent_count = 3
            max_node_count = 2
            execution_budget_ms = 6000
            verification_required = True
            termination_conditions = ["verification_concluded", "max_rounds_reached"]

        elif workload_class == WorkloadClass.RESOURCE_CONSTRAINED:
            # Stripped-down minimal federation; preserves safety checks if mandated
            must_verify = context.get("safety_critical", False)
            if must_verify:
                required_roles = [AgentRole.ANALYST, AgentRole.VERIFIER]
                max_agent_count = 2
                verification_required = True
            else:
                required_roles = [AgentRole.ANALYST]
                max_agent_count = 1
                verification_required = False

            optional_roles = []
            role_dependencies = {}
            max_rounds = 1
            max_node_count = 1
            execution_budget_ms = 2500
            termination_conditions = ["minimal_round_complete"]

        else:
            raise ValueError(f"Unknown workload_class: {workload_class}")

        return TaskPlan(
            plan_id=plan_id,
            task_id=task_id,
            workload_class=workload_class,
            primary_objective=objective,
            required_roles=required_roles,
            optional_roles=optional_roles,
            role_dependencies=role_dependencies,
            execution_budget_ms=execution_budget_ms,
            max_deliberation_rounds=max_rounds,
            max_agent_count=max_agent_count,
            max_node_count=max_node_count,
            verification_required=verification_required,
            termination_conditions=termination_conditions,
        )
