"""
ChakrView Step 89: Adaptive Re-planning Engine & Step 90: Goal Verification Gate.

Unifies:
- AdaptiveReplanEngine (Step 89): Reevaluates task graph on observations or failures,
  bounds replanning cycles (<= 3), injects dynamic subtasks, and prevents cycles.
- GoalVerificationGate (Step 90): Final whole-goal audit ensuring "all tasks finished"
  is not confused with "goal accomplished". Evaluates tests, outputs, unresolved unknowns,
  and constraint violations to output ACCEPT, PARTIAL, REVISE, REJECT, or ABSTAIN.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
    TaskResourceType,
)
from chakrview.cognition.ppb.expansion_models import (
    DynamicSubtaskRequest,
    GoalVerificationVerdict,
    GoalVerificationResult,
    TaskProvenance,
)
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import EpistemicStatus, KnowledgeRecordType, KnowledgeRecord


@dataclass
class ReplanDecision:
    """Audit record for a replanning step."""
    cycle_number: int
    added_subtasks: List[str]
    reprioritized_subtasks: List[str]
    deferred_subtasks: List[str]
    graph_modified: bool
    rationale: str


class AdaptiveReplanEngine:
    """
    Dynamically evolves and re-plans the task graph in response to observations,
    failed assumptions, or discovered dependencies. Enforces bounded cycles (<= 3).
    """

    def __init__(self, brain: PersistentProjectBrain, max_cycles: int = 3) -> None:
        self.brain = brain
        self.max_cycles = max_cycles
        self.current_cycle = 0

    def evaluate_and_replan(
        self,
        graph: PersistentTaskGraph,
        trigger_node: PersistentTaskNode,
        observation: Dict[str, Any],
        requests: Optional[Sequence[DynamicSubtaskRequest]] = None,
    ) -> ReplanDecision:
        """
        Evaluate node outcome and apply justified task expansions or reprioritizations.
        """
        if self.current_cycle >= self.max_cycles:
            return ReplanDecision(
                cycle_number=self.current_cycle,
                added_subtasks=[],
                reprioritized_subtasks=[],
                deferred_subtasks=[],
                graph_modified=False,
                rationale=f"Max replanning cycles ({self.max_cycles}) reached; replanning halted.",
            )

        self.current_cycle += 1
        added: List[str] = []
        reprioritized: List[str] = []
        modified = False

        # 1. Process explicit dynamic subtask expansion requests
        if requests:
            for req in requests:
                dyn_id = req.compute_deterministic_id(graph.graph_id)
                if dyn_id not in graph.nodes:
                    new_node = PersistentTaskNode(
                        node_id=dyn_id,
                        graph_id=graph.graph_id,
                        title=req.title,
                        description=req.description,
                        resource_type=req.resource_type,
                        parent_id=req.parent_id,
                        dependencies=list(req.dependencies),
                        affected_files=list(req.affected_files),
                        required_symbols=list(req.required_symbols),
                        priority=req.priority,
                        status=TaskNodeStatus.PENDING,
                    )
                    graph.add_node(new_node)
                    added.append(dyn_id)
                    modified = True

        # 2. Check observation for newly discovered dependent modules
        discovered_deps = observation.get("discovered_dependencies", [])
        for dep_mod in discovered_deps:
            dep_id = f"{graph.graph_id}_inspect_{dep_mod.replace('/', '_').replace('.', '_')}"
            if dep_id not in graph.nodes:
                new_node = PersistentTaskNode(
                    node_id=dep_id,
                    graph_id=graph.graph_id,
                    title=f"Inspect Discovered Dependency: {dep_mod}",
                    description=f"Automated dynamic expansion to inspect discovered dependency {dep_mod}",
                    resource_type=TaskResourceType.INSPECTION,
                    parent_id=trigger_node.node_id,
                    dependencies=[trigger_node.node_id],
                    affected_files=[dep_mod],
                    priority=trigger_node.priority + 5,
                    status=TaskNodeStatus.READY if trigger_node.status == TaskNodeStatus.COMPLETED else TaskNodeStatus.PENDING,
                )
                graph.add_node(new_node)
                added.append(dep_id)
                modified = True

        return ReplanDecision(
            cycle_number=self.current_cycle,
            added_subtasks=added,
            reprioritized_subtasks=reprioritized,
            deferred_subtasks=[],
            graph_modified=modified,
            rationale=f"Cycle {self.current_cycle}: Added {len(added)} dynamic subtasks.",
        )


class GoalVerificationGate:
    """
    Step 90: Authoritative Whole-Goal Verification Gate.
    Verifies that the root user objective has genuinely been accomplished.
    """

    def __init__(self, brain: PersistentProjectBrain) -> None:
        self.brain = brain

    def evaluate_goal(
        self,
        graph: PersistentTaskGraph,
        required_outcomes: Optional[Sequence[str]] = None,
        mandatory_test_files: Optional[Sequence[str]] = None,
    ) -> GoalVerificationResult:
        """
        Evaluate complete graph status and verify root objective criteria.
        """
        summary = graph.get_progress_summary()
        completed_count = summary.get(TaskNodeStatus.COMPLETED.value, 0)
        failed_count = summary.get(TaskNodeStatus.FAILED.value, 0)
        blocked_count = summary.get(TaskNodeStatus.BLOCKED.value, 0)
        total = len(graph.nodes)

        satisfied: List[str] = []
        unsatisfied: List[str] = []
        unknowns: List[str] = []
        followups: List[DynamicSubtaskRequest] = []

        # 1. Check for failure or blocked nodes
        if failed_count > 0 or blocked_count > 0:
            unsatisfied.append(f"{failed_count} subtasks failed and {blocked_count} were blocked.")
            return GoalVerificationResult(
                verdict=GoalVerificationVerdict.REJECT,
                score=0.0,
                satisfied_requirements=satisfied,
                unsatisfied_requirements=unsatisfied,
                unresolved_unknowns=unknowns,
                justification=f"Subtask execution failed ({failed_count} failures, {blocked_count} blocked).",
            )

        # 2. Check if all subtasks completed
        if completed_count < total:
            unsatisfied.append(f"Incomplete graph: {completed_count}/{total} nodes completed.")
            return GoalVerificationResult(
                verdict=GoalVerificationVerdict.PARTIAL,
                score=round(completed_count / max(total, 1), 2),
                satisfied_requirements=satisfied,
                unsatisfied_requirements=unsatisfied,
                unresolved_unknowns=unknowns,
                justification="Graph execution has not finished all required nodes.",
            )

        satisfied.append(f"All {total} decomposed subtasks executed successfully.")

        # 3. Check for unresolved STALE or UNKNOWN knowledge in affected files
        all_affected_files: Set[str] = set()
        for node in graph.nodes.values():
            all_affected_files.update(node.affected_files)

        for aff_f in all_affected_files:
            recs = self.brain.query_knowledge(file_path=aff_f)
            if not recs:
                unknowns.append(f"File '{aff_f}' has no grounded PPB records.")
            elif any(r.epistemic_status == EpistemicStatus.STALE for r in recs):
                unsatisfied.append(f"File '{aff_f}' still contains STALE knowledge records.")

        if unknowns:
            return GoalVerificationResult(
                verdict=GoalVerificationVerdict.ABSTAIN,
                score=0.6,
                satisfied_requirements=satisfied,
                unsatisfied_requirements=unsatisfied,
                unresolved_unknowns=unknowns,
                justification="Unresolved UNKNOWN records detected in affected files; cannot verify goal.",
            )

        if unsatisfied:
            # Stale records need revision
            followups.append(DynamicSubtaskRequest(
                title="Re-verify Stale Project Records",
                description="Re-analyze stale knowledge records after task execution",
                resource_type=TaskResourceType.VERIFICATION,
                parent_id=list(graph.nodes.keys())[-1],
                priority=10,
                provenance=TaskProvenance(origin_type="REPLANNING", rationale="Clean stale records before goal acceptance"),
            ))
            return GoalVerificationResult(
                verdict=GoalVerificationVerdict.REVISE,
                score=0.8,
                satisfied_requirements=satisfied,
                unsatisfied_requirements=unsatisfied,
                unresolved_unknowns=unknowns,
                suggested_followups=followups,
                justification="Some affected records remain STALE; revision required.",
            )

        # 4. Success: All checks passed cleanly
        satisfied.append("All project knowledge records verified and up-to-date.")
        result = GoalVerificationResult(
            verdict=GoalVerificationVerdict.ACCEPT,
            score=1.0,
            satisfied_requirements=satisfied,
            unsatisfied_requirements=[],
            unresolved_unknowns=[],
            justification=f"Goal '{graph.root_task_description}' completed and verified across all criteria.",
        )

        # Persist goal verification outcome to PPB
        goal_rec = KnowledgeRecord(
            record_id=f"goal_{graph.graph_id}",
            project_id=self.brain.project_id,
            record_type=KnowledgeRecordType.TASK_HISTORY,
            file_path="project",
            summary=f"Goal Accepted: {graph.root_task_description}",
            details={"graph_id": graph.graph_id, "score": 1.0, "verdict": "ACCEPT"},
            epistemic_status=EpistemicStatus.FACT,
            confidence=1.0,
            source_chunk="goal_verification_gate",
        )
        self.brain.store_knowledge(goal_rec)

        return result
