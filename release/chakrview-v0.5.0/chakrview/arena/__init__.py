"""
ChakrKshetra (formerly Project Arena): Subsystem Package Initialization.

ChakrKshetra is ChakrView's controlled execution, experimentation,
observation, and evaluation environment. It provides isolated disposable workspaces,
sandboxed test execution, path traversal protection, closed-loop repair controller,
episodic memory bridge, AST evaluation, and metrics collection.

Architectural Invariant:
    ChakrKshetra is an external environment, NOT the neural brain.
"""

from chakrview.arena.models import (
    PathTraversalError,
    WorkspaceQuotaExceededError,
    FailureCategory,
    FileRole,
    SourceFile,
    ProjectSpecification,
    ProjectManifest,
    TestResult,
    PatchDiff,
    IterationRecord,
    ExecutionHistory,
    ArenaExecutionResult,
    EvaluationMetrics,
)
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.arena.executor import SandboxedExecutor
from chakrview.arena.evaluator import ArenaEvaluator
from chakrview.arena.dataset import CodingCorpusManager
from chakrview.arena.loop import ArenaClosedLoopController
from chakrview.arena.memory import ArenaMemoryBridge

# ChakrKshetra Conceptual Aliases
ChakrKshetraWorkspace = IsolatedWorkspace
ChakrKshetraExecutor = SandboxedExecutor
ChakrKshetraEvaluator = ArenaEvaluator
ChakrKshetraController = ArenaClosedLoopController
ChakrKshetraMemoryBridge = ArenaMemoryBridge

__all__ = [
    "PathTraversalError",
    "WorkspaceQuotaExceededError",
    "FailureCategory",
    "FileRole",
    "SourceFile",
    "ProjectSpecification",
    "ProjectManifest",
    "TestResult",
    "PatchDiff",
    "IterationRecord",
    "ExecutionHistory",
    "ArenaExecutionResult",
    "EvaluationMetrics",
    "IsolatedWorkspace",
    "SandboxedExecutor",
    "ArenaEvaluator",
    "CodingCorpusManager",
    "ArenaClosedLoopController",
    "ArenaMemoryBridge",
    "ChakrKshetraWorkspace",
    "ChakrKshetraExecutor",
    "ChakrKshetraEvaluator",
    "ChakrKshetraController",
    "ChakrKshetraMemoryBridge",
]
