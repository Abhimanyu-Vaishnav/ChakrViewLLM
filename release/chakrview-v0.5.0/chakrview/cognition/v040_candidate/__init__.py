"""
ChakrView v0.4.0 Candidate: Integrated Cognitive Architecture & Engine.

Encompasses Waves 581-590:
- Wave 581: Working Memory 2.0 (Bounded, task-scoped, structured)
- Wave 582: Long-Context Chunk Processor (Low-RAM, chunk continuity, synthesis)
- Wave 583: Hierarchical Adaptive Planner (Goals -> Subgoals -> Actions, replanning)
- Wave 584: Multi-Step Goal Executor (Ground truth verification, recovery)
- Wave 585: Memory <-> Reasoning Integrator (Evidence > Stale Memory precedence)
- Wave 586: Experience <-> Reasoning Integrator (Strategy selection without weight mutation)
- Wave 587: Language <-> Cognitive State Bridge (Intent representation, semantic grounding)
- Wave 588: Adaptive Strategy Solver (Resource & complexity aware strategy selector)
- Wave 589: Unified Cognitive Capability Benchmark
- Wave 590: v0.4.0 Release Candidate Audit Gate
"""

from .working_memory import WorkingMemory2, ActiveTaskState, TaskConstraint
from .chunk_processor import LongContextChunkProcessor, ChunkSynthesisResult
from .hierarchical_planner import HierarchicalPlanner, SubGoal, HierarchicalPlan
from .multi_step_executor import MultiStepGoalExecutor, ExecutionOutcome, ActionStatus
from .memory_reasoning_bridge import MemoryReasoningBridge, MemoryBeliefRecord
from .experience_reasoning_bridge import ExperienceReasoningBridge, StrategyCandidate
from .language_cognitive_bridge import LanguageCognitiveBridge, SemanticCognitiveFrame
from .adaptive_solver import AdaptiveProblemSolver, ProblemSolvingStrategy
from .v040_master_gate import V040CandidateMasterGate

__all__ = [
    "WorkingMemory2",
    "ActiveTaskState",
    "TaskConstraint",
    "LongContextChunkProcessor",
    "ChunkSynthesisResult",
    "HierarchicalPlanner",
    "SubGoal",
    "HierarchicalPlan",
    "MultiStepGoalExecutor",
    "ExecutionOutcome",
    "ActionStatus",
    "MemoryReasoningBridge",
    "MemoryBeliefRecord",
    "ExperienceReasoningBridge",
    "StrategyCandidate",
    "LanguageCognitiveBridge",
    "SemanticCognitiveFrame",
    "AdaptiveProblemSolver",
    "ProblemSolvingStrategy",
    "V040CandidateMasterGate",
]
