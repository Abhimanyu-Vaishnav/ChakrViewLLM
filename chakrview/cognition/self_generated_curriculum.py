"""
Self-Generated Curriculum Engine.
Wave 532: Analyzes observed runtime failures, identifies capability gaps,
and synthesizes isolated practice tasks to address recurring weaknesses.

Safety Guardrail:
- Practice tasks are isolated strictly to candidate training.
- The system CANNOT generate its own test tasks and declare itself improved.
- Evaluation must occur on an independently authored held-out evaluation set.
"""

from typing import List, Dict, Any
from collections import defaultdict
import hashlib

class SelfGeneratedCurriculumEngine:
    def __init__(self):
        self.failure_history: List[Dict[str, Any]] = []

    def record_failure(self, task: Dict[str, Any], error_type: str) -> None:
        self.failure_history.append({
            "task_id": task.get("id", "unk"),
            "task_family": task.get("task_family", "generic"),
            "hops": task.get("hops", 1),
            "error_type": error_type
        })

    def analyze_gaps(self) -> List[Dict[str, Any]]:
        """Group failures and identify recurring weaknesses."""
        gap_counts = defaultdict(lambda: {"count": 0, "hops": set(), "errors": defaultdict(int)})
        for f in self.failure_history:
            fam = f["task_family"]
            gap_counts[fam]["count"] += 1
            gap_counts[fam]["hops"].add(f["hops"])
            gap_counts[fam]["errors"][f["error_type"]] += 1

        gaps = []
        for fam, data in gap_counts.items():
            if data["count"] >= 2: # Recurring gap
                dominant_error = max(data["errors"].items(), key=lambda x: x[1])[0]
                gaps.append({
                    "task_family": fam,
                    "failure_count": data["count"],
                    "hops_involved": list(data["hops"]),
                    "dominant_error": dominant_error,
                    "priority": "HIGH" if data["count"] >= 3 else "MEDIUM"
                })
        return gaps

    def generate_practice_curriculum(self, gap: Dict[str, Any], num_tasks: int = 5) -> List[Dict[str, Any]]:
        """
        Synthesizes practice tasks targeting the identified gap.
        These are marked role='practice' and cannot be used as evaluation tasks.
        """
        family = gap["task_family"]
        hops = max(gap["hops_involved"]) if gap["hops_involved"] else 3
        practice_tasks = []

        for i in range(num_tasks):
            t_id = f"practice_{family}_{i}_{hashlib.sha256(str(i).encode()).hexdigest()[:6]}"
            entities = [f"Ent_{chr(65+j)}_{i}" for j in range(hops + 1)]
            premises = [(entities[j], "connects_to", entities[j+1]) for j in range(hops)]
            query = (entities[0], f"recurrent_hop_{hops}", "?")
            target = entities[-1]

            practice_tasks.append({
                "id": t_id,
                "role": "practice", # Strictly practice
                "task_family": family,
                "hops": hops,
                "premise": premises,
                "query": query,
                "target": target
            })
        return practice_tasks
