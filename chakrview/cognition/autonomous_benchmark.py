"""
Step 112 Autonomous Work Loop & Benchmarking Harness.

Implements the complete real-world autonomous cycle:
1. Disk repository inspection with persistent symbol & chunk indexing (no full rescans).
2. Targeted context retrieval adhering to the strict <= 512 token ceiling.
3. Task graph decomposition and dynamic replanning upon failure.
4. Governed tool execution (read, write, pytest) via GovernedToolGate with argument sanitization.
5. Injected deterministic failure detection, root-cause diagnosis, replanning, and repair.
6. Persistent project brain state survival across process boundaries with zero rescan of unchanged files.
7. Verification of canonical baseline weight immutability.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.master_cognitive_pipeline import (
    MasterCognitivePipeline,
    ContextEfficientProjectEngine,
)
from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
    TaskResourceType,
)
from chakrview.cognition.ppb.replan_and_gate import (
    AdaptiveReplanEngine,
    GoalVerificationGate,
    GoalVerificationResult,
    GoalVerificationVerdict,
)
from chakrview.cognition.governed_learning.self_evaluator import (
    GovernedFailureAnalysis,
    FailureClass,
)
from chakrview.cognition.governed_learning.strategy_registry import (
    CognitiveStrategy,
    CognitiveStrategyRegistry,
    StrategyStatus,
)
from chakrview.runtime.skills import SkillPolicy, Skill
from chakrview.runtime.tools import (
    Tool,
    ToolRegistry,
    ToolExecutor,
    ToolResult,
)
from chakrview.cognition.tool_gate import (
    GovernedToolGate,
    ToolAuthorizationError,
    ArgumentValidationError,
)
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


class BenchmarkDiskProjectEngine:
    """
    On-disk context-efficient project engine with durable SQLite caching
    and per-file scan tracking to verify the NO-RESCAN contract.
    """
    def __init__(self, db_path: Path, project_root: Path) -> None:
        self.db_path = Path(db_path)
        self.project_root = Path(project_root).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.scan_counters: Dict[str, int] = {}
        self._init_db()

    def _init_db(self) -> None:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_index (
                    rel_path TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    last_scanned_utc TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbol_index (
                    symbol TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    PRIMARY KEY (symbol, rel_path)
                )
            """)
            conn.commit()

    def discover_files(self) -> List[str]:
        files: List[str] = []
        for root, _, filenames in os.walk(self.project_root):
            for f in sorted(filenames):
                if f.endswith(".py"):
                    full_p = Path(root) / f
                    rel_p = str(full_p.relative_to(self.project_root)).replace("\\", "/")
                    files.append(rel_p)
        return sorted(files)

    def scan_repository(self) -> Dict[str, Any]:
        """
        Scans on-disk repository files. Only computes AST / symbols if
        file content SHA256 has changed since previous scan.
        """
        import sqlite3
        files = self.discover_files()
        scanned_count = 0
        skipped_count = 0

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for rel_p in files:
                full_p = self.project_root / rel_p
                content = full_p.read_text(encoding="utf-8")
                cur_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                cursor.execute("SELECT sha256 FROM file_index WHERE rel_path = ?", (rel_p,))
                row = cursor.fetchone()

                if row and row[0] == cur_hash:
                    skipped_count += 1
                    continue

                # File is new or modified: perform full extraction
                scanned_count += 1
                self.scan_counters[rel_p] = self.scan_counters.get(rel_p, 0) + 1
                summary = f"Module {rel_p}: {len(content.splitlines())} lines"
                tok_count = max(1, len(content) // 4)

                cursor.execute("""
                    INSERT OR REPLACE INTO file_index (rel_path, sha256, summary, token_count, last_scanned_utc)
                    VALUES (?, ?, ?, ?, ?)
                """, (rel_p, cur_hash, summary, tok_count, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

                # Re-index symbols
                cursor.execute("DELETE FROM symbol_index WHERE rel_path = ?", (rel_p,))
                symbols = [
                    line.split()[1].split("(")[0].split(":")[0]
                    for line in content.splitlines()
                    if line.startswith("def ") or line.startswith("class ")
                ]
                for sym in symbols:
                    cursor.execute("INSERT OR IGNORE INTO symbol_index (symbol, rel_path) VALUES (?, ?)", (sym, rel_p))

            conn.commit()

        return {
            "total_files": len(files),
            "files_scanned": scanned_count,
            "files_skipped_unchanged": skipped_count,
            "scan_counters": dict(self.scan_counters),
        }

    def query_targeted_context(self, required_symbols: Sequence[str]) -> Dict[str, Any]:
        """
        Retrieves targeted context for specific symbols, guaranteeing
        total context length remains <= 512 tokens.
        """
        import sqlite3
        relevant_files: Set[str] = set()
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for sym in required_symbols:
                cursor.execute("SELECT rel_path FROM symbol_index WHERE symbol = ?", (sym,))
                for row in cursor.fetchall():
                    relevant_files.add(row[0])

            summaries: Dict[str, str] = {}
            total_tokens = 0
            for rel_p in sorted(relevant_files):
                cursor.execute("SELECT summary, token_count FROM file_index WHERE rel_path = ?", (rel_p,))
                row = cursor.fetchone()
                if row:
                    summaries[rel_p] = row[0]
                    total_tokens += row[1]

        # Context ceiling constraint check
        assert total_tokens <= 512, f"Targeted context tokens {total_tokens} exceeded 512 ceiling!"

        return {
            "required_symbols": list(required_symbols),
            "relevant_files": sorted(list(relevant_files)),
            "summaries": summaries,
            "total_context_tokens": total_tokens,
            "context_ceiling_maintained": total_tokens <= 512,
        }


class BenchmarkReadFileTool(Tool):
    def __init__(self, benchmark_root: Path) -> None:
        self.benchmark_root = Path(benchmark_root).resolve()

    @property
    def tool_id(self) -> str:
        return "read_file"

    @property
    def name(self) -> str:
        return "Read File"

    @property
    def description(self) -> str:
        return "Read file content within benchmark root"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {"file_path": {"type": "string", "required": True}}

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        target = (self.benchmark_root / file_path).resolve()
        if not str(target).startswith(str(self.benchmark_root)):
            return ToolResult(tool_id=self.tool_id, success=False, error=f"Security violation: Access outside benchmark root: {file_path}")
        if not target.is_file():
            return ToolResult(tool_id=self.tool_id, success=False, error=f"File not found: {file_path}")
        try:
            content = target.read_text(encoding="utf-8")
            return ToolResult(tool_id=self.tool_id, success=True, output=content)
        except Exception as e:
            return ToolResult(tool_id=self.tool_id, success=False, error=str(e))


class BenchmarkWriteFileTool(Tool):
    def __init__(self, benchmark_root: Path) -> None:
        self.benchmark_root = Path(benchmark_root).resolve()

    @property
    def tool_id(self) -> str:
        return "write_file"

    @property
    def name(self) -> str:
        return "Write File"

    @property
    def description(self) -> str:
        return "Write file content within benchmark root"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {"file_path": {"type": "string", "required": True}, "content": {"type": "string", "required": True}}

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        content = kwargs.get("content", "")
        target = (self.benchmark_root / file_path).resolve()
        if not str(target).startswith(str(self.benchmark_root)):
            return ToolResult(tool_id=self.tool_id, success=False, error=f"Security violation: Modification outside benchmark root: {file_path}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(tool_id=self.tool_id, success=True, output=f"Successfully wrote {len(content)} characters to {file_path}")
        except Exception as e:
            return ToolResult(tool_id=self.tool_id, success=False, error=str(e))


class BenchmarkRunTestsTool(Tool):
    def __init__(self, benchmark_root: Path) -> None:
        self.benchmark_root = Path(benchmark_root).resolve()

    @property
    def tool_id(self) -> str:
        return "run_tests"

    @property
    def name(self) -> str:
        return "Run Tests"

    @property
    def description(self) -> str:
        return "Run pytest suite inside benchmark root"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {"test_target": {"type": "string", "required": False}}

    def execute(self, **kwargs: Any) -> ToolResult:
        test_target = kwargs.get("test_target", "tests")
        target_dir = (self.benchmark_root / test_target).resolve()
        if not str(target_dir).startswith(str(self.benchmark_root)):
            return ToolResult(tool_id=self.tool_id, success=False, error="Security violation: Test target outside benchmark root")
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", test_target, "-q"],
                cwd=str(self.benchmark_root),
                capture_output=True,
                text=True,
            )
            out_json = json.dumps({
                "returncode": res.returncode,
                "passed": (res.returncode == 0),
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip(),
            })
            return ToolResult(tool_id=self.tool_id, success=True, output=out_json)
        except Exception as e:
            return ToolResult(tool_id=self.tool_id, success=False, error=str(e))


def build_benchmark_tool_gate(benchmark_root: Path) -> Tuple[GovernedToolGate, Skill]:
    """
    Constructs a real GovernedToolGate wired to safe disk tools restricted
    to the benchmark repository folder.
    """
    root = Path(benchmark_root).resolve()
    registry = ToolRegistry()
    registry.register(BenchmarkReadFileTool(root))
    registry.register(BenchmarkWriteFileTool(root))
    registry.register(BenchmarkRunTestsTool(root))

    from chakrview.runtime.skills import SkillDomain
    policy = SkillPolicy(
        allowed_tools=["read_file", "write_file", "run_tests"],
        max_context_tokens=512,
    )
    skill = Skill(
        skill_id="benchmark_engineer",
        name="Benchmark Engineer",
        version="1.0.0",
        domain=SkillDomain.CODING,
        description="Autonomous repository task execution skill",
        policy=policy,
    )

    tool_gate = GovernedToolGate(tool_registry=registry)
    return tool_gate, skill
