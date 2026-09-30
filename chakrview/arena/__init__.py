"""
ChakrView Project Arena: Subsystem Package Initialization.

Provides isolated project workspaces, sandboxed test execution,
project-level dataset splitting, AST evaluation, and metrics collection.
"""

from chakrview.arena.models import (
    FailureCategory,
    FileRole,
    SourceFile,
    ProjectSpecification,
    ProjectManifest,
    TestResult,
    ArenaExecutionResult,
    EvaluationMetrics,
)
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.arena.executor import SandboxedExecutor
from chakrview.arena.evaluator import ArenaEvaluator
from chakrview.arena.dataset import CodingCorpusManager

__all__ = [
    "FailureCategory",
    "FileRole",
    "SourceFile",
    "ProjectSpecification",
    "ProjectManifest",
    "TestResult",
    "ArenaExecutionResult",
    "EvaluationMetrics",
    "IsolatedWorkspace",
    "SandboxedExecutor",
    "ArenaEvaluator",
    "CodingCorpusManager",
]
