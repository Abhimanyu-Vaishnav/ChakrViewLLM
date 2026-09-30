"""
ChakrView Project Arena: Subsystem Package Initialization.

Provides isolated project workspaces, sandboxed test execution,
path traversal protection, closed-loop repair controller, memory bridge,
AST evaluation, and metrics collection.
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
]
