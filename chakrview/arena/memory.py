"""
ChakrView Project Arena: Memory & RIL Experience Bridge.

Converts project execution histories and closed-loop repair trajectories
into structured experience records for ChakrView's EpisodicMemoryStore and RIL.

Axiom:
    MODEL WEIGHTS ≠ MEMORY ≠ RAG ≠ TOOLS ≠ RIL ≠ SELF-MODIFICATION
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional

from chakrview.arena.models import ExecutionHistory, ProjectSpecification, FailureCategory


class ArenaMemoryBridge:
    """
    Bridge connecting Project Arena execution histories to Episodic Memory.
    """
    @staticmethod
    def create_experience_record(
        specification: ProjectSpecification,
        history: ExecutionHistory,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Package a project execution trajectory into a structured experience dictionary.
        """
        # Determine initial and final pass rates
        first_iter = history.iterations[0] if history.iterations else None
        last_iter = history.iterations[-1] if history.iterations else None

        initial_pass_rate = first_iter.test_result.pass_rate if first_iter else 0.0
        final_pass_rate = history.final_pass_rate
        improvement = final_pass_rate - initial_pass_rate

        # Extract repair patterns (errors that were resolved)
        repair_patterns = []
        for i, it in enumerate(history.iterations[:-1]):
            next_it = history.iterations[i + 1]
            if it.failure_category != FailureCategory.SUCCESS and (next_it.test_result.passed > it.test_result.passed):
                repair_patterns.append({
                    "from_iteration": it.iteration,
                    "to_iteration": next_it.iteration,
                    "error_type": it.failure_category.value,
                    "diagnosis": it.diagnosis,
                    "files_patched": it.files_modified,
                    "pass_increase": next_it.test_result.passed - it.test_result.passed,
                })

        return {
            "experience_id": f"exp_arena_{specification.project_id}_{int(time.time())}",
            "project_id": specification.project_id,
            "project_name": specification.project_name,
            "description": specification.description,
            "language": specification.language,
            "converged": history.converged,
            "total_iterations": history.total_iterations,
            "initial_pass_rate": round(initial_pass_rate, 4),
            "final_pass_rate": round(final_pass_rate, 4),
            "improvement_delta": round(improvement, 4),
            "initial_failure_category": first_iter.failure_category.value if first_iter else None,
            "final_failure_category": last_iter.failure_category.value if last_iter else None,
            "repair_patterns": repair_patterns,
            "execution_history": history.to_dict(),
            "metadata": metadata or {},
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
