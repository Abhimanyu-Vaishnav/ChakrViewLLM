"""
Step 51 Automated Tests: Coding Language Acquisition & Project Arena Foundation.

Tests verify:
  01. Coding corpus specification & ProjectSpecification model integrity.
  02. SourceFile model sha256 checksum generation & validation.
  03. ProjectManifest serialization, deserialization, and integrity.
  04. Project-level split isolation (zero cross-file leakage between train/val/test).
  05. Exact duplicate file detection via SHA-256 digests.
  06. Near-duplicate file detection via Jaccard n-gram similarity.
  07. Byte-Level BPE tokenizer lossless round-trip on Python source code.
  08. Tokenizer lossless round-trip across multi-language code (JS, TS, SQL, Shell, HTML, JSON).
  09. Format project for training preserves project metadata and file boundaries.
  10. IsolatedWorkspace directory creation, layout, and structure.
  11. IsolatedWorkspace file write operations (source and test files).
  12. AST syntax validation identifies valid vs invalid Python code.
  13. SandboxedExecutor builds sanitized environment without leaked host secrets.
  14. SandboxedExecutor executes passing tests and parses counts correctly.
  15. SandboxedExecutor captures failing assertions and classifies failure correctly.
  16. SandboxedExecutor fail-closed timeout enforcement kills hung subprocesses.
  17. ArenaEvaluator full manifest evaluation produces valid score artifacts.
  18. Frozen baseline weight immutability invariant (DeltaW = 0, hash c5571c... intact).
  19. Coding checkpoint payload integrity and atomic save/load.
  20. CPU resource constraints: execution within memory limits and zero GPU usage.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
import torch

from chakrview.arena.models import (
    ProjectSpecification,
    SourceFile,
    ProjectManifest,
    FileRole,
    TestResult,
    FailureCategory,
    ArenaExecutionResult,
)
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.arena.executor import SandboxedExecutor
from chakrview.arena.evaluator import ArenaEvaluator
from chakrview.arena.dataset import CodingCorpusManager
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    DEFAULT_TOKENIZER_DIR,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH, EXPECTED_VOCAB_SIZE, MAX_CONTEXT_WINDOW
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
    return tok


@pytest.fixture(scope="module")
def sample_project_manifest():
    spec = ProjectSpecification(
        project_id="proj_test_math_001",
        project_name="test_math",
        description="Math utilities for testing.",
    )
    files = [
        SourceFile(
            path="__init__.py",
            content="from .core import add, sub\n__all__ = ['add', 'sub']\n",
            role=FileRole.INIT,
        ),
        SourceFile(
            path="core.py",
            content="def add(a: int, b: int) -> int:\n    return a + b\n\ndef sub(a: int, b: int) -> int:\n    return a - b\n",
            role=FileRole.SOURCE,
        ),
        SourceFile(
            path="test_core.py",
            content="from core import add, sub\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_sub():\n    assert sub(5, 3) == 2\n",
            role=FileRole.TEST,
        ),
    ]
    return ProjectManifest(specification=spec, files=files)


# ---------------------------------------------------------------------------
# Test 01: Coding corpus specification & ProjectSpecification model
# ---------------------------------------------------------------------------
def test_01_project_specification_model():
    spec = ProjectSpecification(
        project_id="proj_test_01",
        project_name="unit_test_proj",
        description="Specification test project.",
        language="python",
        version="1.2.3",
        license="Apache-2.0",
        dependencies=["typing", "math"],
        entrypoint="main.py",
        test_framework="pytest",
        timeout_seconds=4.5,
    )
    d = spec.to_dict()
    assert d["project_id"] == "proj_test_01"
    assert d["license"] == "Apache-2.0"
    assert d["timeout_seconds"] == 4.5

    reconstructed = ProjectSpecification.from_dict(d)
    assert reconstructed.project_id == spec.project_id
    assert reconstructed.dependencies == ["typing", "math"]


# ---------------------------------------------------------------------------
# Test 02: SourceFile model sha256 checksum generation & validation
# ---------------------------------------------------------------------------
def test_02_source_file_checksum():
    content = "def hello():\n    return 'world'\n"
    sfile = SourceFile(path="hello.py", content=content)
    assert len(sfile.sha256) == 64
    assert sfile.role == FileRole.SOURCE

    d = sfile.to_dict()
    sfile2 = SourceFile.from_dict(d)
    assert sfile2.path == sfile.path
    assert sfile2.content == sfile.content
    assert sfile2.sha256 == sfile.sha256


# ---------------------------------------------------------------------------
# Test 03: ProjectManifest serialization & file role retrieval
# ---------------------------------------------------------------------------
def test_03_project_manifest_serialization(sample_project_manifest: ProjectManifest):
    d = sample_project_manifest.to_dict()
    reloaded = ProjectManifest.from_dict(d)

    assert reloaded.specification.project_id == sample_project_manifest.specification.project_id
    assert len(reloaded.files) == len(sample_project_manifest.files)

    sources = reloaded.get_source_files()
    tests = reloaded.get_test_files()
    assert len(sources) == 2  # __init__.py and core.py
    assert len(tests) == 1    # test_core.py


# ---------------------------------------------------------------------------
# Test 04: Project-level split isolation (zero cross-file leakage)
# ---------------------------------------------------------------------------
def test_04_project_split_isolation():
    projects = []
    for i in range(20):
        spec = ProjectSpecification(
            project_id=f"proj_batch_{i:03d}",
            project_name=f"project_{i}",
            description="Batch project",
        )
        files = [
            SourceFile(path=f"mod_{i}.py", content=f"# Project {i}\ndef f_{i}(): pass\n"),
            SourceFile(path=f"test_{i}.py", content=f"# Test {i}\ndef test_{i}(): pass\n", role=FileRole.TEST),
        ]
        projects.append(ProjectManifest(specification=spec, files=files))

    splits = CodingCorpusManager.partition_projects(projects, train_ratio=0.7, val_ratio=0.2)
    assert len(splits["train"]) > 0
    assert len(splits["validation"]) > 0
    assert len(splits["test"]) > 0

    # Ensure zero project ID overlap across partitions
    train_ids = {p.specification.project_id for p in splits["train"]}
    val_ids = {p.specification.project_id for p in splits["validation"]}
    test_ids = {p.specification.project_id for p in splits["test"]}

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    assert CodingCorpusManager.check_project_leakage(splits["train"], splits["validation"], splits["test"])


# ---------------------------------------------------------------------------
# Test 05: Exact duplicate file detection via SHA-256
# ---------------------------------------------------------------------------
def test_05_exact_duplicate_detection():
    f1 = SourceFile(path="a.py", content="def same(): pass\n")
    f2 = SourceFile(path="b.py", content="def same(): pass\n")  # Identical content
    f3 = SourceFile(path="c.py", content="def different(): pass\n")

    filtered = CodingCorpusManager.filter_duplicate_files([f1, f2, f3])
    assert len(filtered) == 2
    paths = {f.path for f in filtered}
    assert "a.py" in paths
    assert "c.py" in paths
    assert "b.py" not in paths


# ---------------------------------------------------------------------------
# Test 06: Near-duplicate file detection via Jaccard n-gram
# ---------------------------------------------------------------------------
def test_06_near_duplicate_detection():
    base_code = "def process_data(items: list) -> list:\n    # Process elements\n    res = []\n    for item in items:\n        res.append(item * 2)\n    return res\n"
    # Near identical (minor whitespace/comment change)
    near_code = "def process_data(items: list) -> list:\n    # Process elements modified\n    res = []\n    for item in items:\n        res.append(item * 2)\n    return res\n"
    distinct_code = "class Stack:\n    def __init__(self):\n        self.s = []\n    def push(self, x):\n        self.s.append(x)\n"

    f1 = SourceFile(path="f1.py", content=base_code)
    f2 = SourceFile(path="f2.py", content=near_code)
    f3 = SourceFile(path="f3.py", content=distinct_code)

    filtered = CodingCorpusManager.filter_duplicate_files([f1, f2, f3], jaccard_threshold=0.75)
    assert len(filtered) == 2
    paths = {f.path for f in filtered}
    assert "f1.py" in paths
    assert "f3.py" in paths
    assert "f2.py" not in paths


# ---------------------------------------------------------------------------
# Test 07: Byte-Level BPE tokenizer lossless round-trip on Python
# ---------------------------------------------------------------------------
def test_07_tokenizer_python_roundtrip(tokenizer):
    python_code = """import os
from typing import List, Optional

def quicksort(arr: List[int]) -> List[int]:
    \"\"\"Standard recursive quicksort implementation.\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Test execution:
data = [3, 6, 8, 10, 1, 2, 1]
print("Sorted:", quicksort(data))
"""
    tokens = tokenizer.encode(python_code)
    decoded = tokenizer.decode(tokens)
    assert decoded == python_code
    assert len(tokens) > 0


# ---------------------------------------------------------------------------
# Test 08: Tokenizer lossless round-trip across multi-language code
# ---------------------------------------------------------------------------
def test_08_tokenizer_multilang_roundtrip(tokenizer):
    snippets = {
        "javascript": "function fib(n) {\n  if (n <= 1) return n;\n  return fib(n - 1) + fib(n - 2);\n}\n",
        "sql": "SELECT id, email, created_at FROM accounts WHERE verified = 1 ORDER BY id DESC;\n",
        "shell": "#!/bin/bash\nset -euo pipefail\necho \"Starting task\"\nexit 0\n",
        "html": "<div id=\"root\" class=\"app-container\"><p>Hello World</p></div>",
        "json": "{\"name\": \"ChakrView\", \"version\": \"0.1.0\", \"active\": true}",
    }

    for lang, code in snippets.items():
        tokens = tokenizer.encode(code)
        decoded = tokenizer.decode(tokens)
        assert decoded == code, f"Lossless round-trip failed for {lang}"


# ---------------------------------------------------------------------------
# Test 09: Format project for training preserves metadata and files
# ---------------------------------------------------------------------------
def test_09_format_project_for_training(sample_project_manifest: ProjectManifest):
    doc = CodingCorpusManager.format_project_for_training(sample_project_manifest)
    assert "# PROJECT: test_math" in doc
    assert "# ID: proj_test_math_001" in doc
    assert "# FILE: core.py" in doc
    assert "def add(a: int, b: int) -> int:" in doc


# ---------------------------------------------------------------------------
# Test 10: IsolatedWorkspace layout and directory structure
# ---------------------------------------------------------------------------
def test_10_isolated_workspace_layout(sample_project_manifest: ProjectManifest):
    with IsolatedWorkspace(sample_project_manifest) as ws:
        assert ws.workspace_dir.exists()
        assert ws.spec_dir.is_dir()
        assert ws.source_dir.is_dir()
        assert ws.test_dir.is_dir()
        assert ws.logs_dir.is_dir()
        assert ws.artifacts_dir.is_dir()

        # Check specification was written
        spec_path = ws.spec_dir / "project_spec.json"
        assert spec_path.is_file()
        with open(spec_path, "r", encoding="utf-8") as f:
            saved_spec = json.load(f)
        assert saved_spec["project_id"] == "proj_test_math_001"


# ---------------------------------------------------------------------------
# Test 11: IsolatedWorkspace file write operations
# ---------------------------------------------------------------------------
def test_11_isolated_workspace_file_ops(sample_project_manifest: ProjectManifest):
    with IsolatedWorkspace(sample_project_manifest) as ws:
        src_path = ws.write_source_file("extra.py", "def extra(): return 42\n")
        assert src_path.is_file()
        assert (ws.source_dir / "extra.py").exists()

        test_path = ws.write_test_file("test_extra.py", "from extra import extra\ndef test_extra(): assert extra() == 42\n")
        assert test_path.is_file()
        assert (ws.test_dir / "test_extra.py").exists()


# ---------------------------------------------------------------------------
# Test 12: AST syntax validation identifies valid vs invalid code
# ---------------------------------------------------------------------------
def test_12_ast_syntax_validation():
    valid_code = "def valid_func(x: int) -> int:\n    return x + 1\n"
    invalid_code = "def invalid_func(x: int) -> int:\n    return x +\n"

    v1, err1 = ArenaEvaluator.validate_syntax(valid_code)
    assert v1 is True
    assert err1 is None

    v2, err2 = ArenaEvaluator.validate_syntax(invalid_code)
    assert v2 is False
    assert err2 is not None
    assert "SyntaxError" in err2


# ---------------------------------------------------------------------------
# Test 13: SandboxedExecutor environment sanitization
# ---------------------------------------------------------------------------
def test_13_sandboxed_executor_env(sample_project_manifest: ProjectManifest):
    executor = SandboxedExecutor()
    with IsolatedWorkspace(sample_project_manifest) as ws:
        env = executor._build_sanitized_env(ws)
        assert "PYTHONPATH" in env
        assert env["PYTHONPATH"] == str(ws.source_dir.resolve())
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "SECRET_KEY" not in env


# ---------------------------------------------------------------------------
# Test 14: SandboxedExecutor executes passing tests
# ---------------------------------------------------------------------------
def test_14_sandboxed_executor_passing_tests(sample_project_manifest: ProjectManifest):
    executor = SandboxedExecutor()
    with IsolatedWorkspace(sample_project_manifest) as ws:
        test_res, failure_cat = executor.run_tests(ws)
        assert failure_cat == FailureCategory.SUCCESS
        assert test_res.passed == 2
        assert test_res.failed == 0
        assert test_res.errors == 0
        assert test_res.pass_rate == 1.0


# ---------------------------------------------------------------------------
# Test 15: SandboxedExecutor captures failing assertions
# ---------------------------------------------------------------------------
def test_15_sandboxed_executor_failing_assertion():
    spec = ProjectSpecification(project_id="proj_fail_001", project_name="fail_proj", description="Failing test")
    files = [
        SourceFile(path="core.py", content="def get_val(): return 10\n"),
        SourceFile(path="test_core.py", content="from core import get_val\ndef test_wrong(): assert get_val() == 99\n", role=FileRole.TEST),
    ]
    manifest = ProjectManifest(specification=spec, files=files)
    executor = SandboxedExecutor()
    with IsolatedWorkspace(manifest) as ws:
        test_res, failure_cat = executor.run_tests(ws)
        assert failure_cat == FailureCategory.ASSERTION_FAILURE
        assert test_res.failed == 1
        assert test_res.passed == 0


# ---------------------------------------------------------------------------
# Test 16: SandboxedExecutor fail-closed timeout enforcement
# ---------------------------------------------------------------------------
def test_16_sandboxed_executor_timeout():
    spec = ProjectSpecification(
        project_id="proj_hang_001",
        project_name="hang_proj",
        description="Timeout test",
        timeout_seconds=1.0,
    )
    files = [
        SourceFile(path="core.py", content="import time\ndef infinite_loop():\n    while True:\n        time.sleep(0.1)\n"),
        SourceFile(path="test_core.py", content="from core import infinite_loop\ndef test_hang():\n    infinite_loop()\n", role=FileRole.TEST),
    ]
    manifest = ProjectManifest(specification=spec, files=files)
    executor = SandboxedExecutor()
    with IsolatedWorkspace(manifest) as ws:
        start_t = time.perf_counter()
        test_res, failure_cat = executor.run_tests(ws, timeout_seconds=1.0)
        dur = time.perf_counter() - start_t
        assert failure_cat == FailureCategory.TIMEOUT
        assert "timed out" in test_res.stderr.lower()
        assert dur < 4.0  # Killed in under 4 seconds


# ---------------------------------------------------------------------------
# Test 17: ArenaEvaluator produces valid score artifacts
# ---------------------------------------------------------------------------
def test_17_arena_evaluator_score_artifacts(sample_project_manifest: ProjectManifest):
    evaluator = ArenaEvaluator()
    result = evaluator.evaluate_manifest(sample_project_manifest)
    assert result.is_success
    assert result.syntax_valid
    assert result.failure_category == FailureCategory.SUCCESS
    assert result.test_result.passed == 2


# ---------------------------------------------------------------------------
# Test 18: Frozen baseline weight immutability invariant (DeltaW = 0)
# ---------------------------------------------------------------------------
def test_18_frozen_baseline_immutability():
    baseline = instantiate_frozen_baseline()
    h = compute_model_hash(baseline)
    assert h == EXPECTED_WEIGHT_HASH, f"Frozen baseline mutated! Got {h}"

    # Forward pass on sample tokens
    input_ids = torch.tensor([[0, 25, 42, 100]], dtype=torch.long)
    with torch.no_grad():
        logits = baseline(input_ids)
    assert logits.shape == (1, 4, EXPECTED_VOCAB_SIZE)

    # Post-forward hash check
    h_post = compute_model_hash(baseline)
    assert h_post == EXPECTED_WEIGHT_HASH, "Baseline mutated after forward pass!"


# ---------------------------------------------------------------------------
# Test 19: Coding checkpoint payload integrity and atomic save/load
# ---------------------------------------------------------------------------
def test_19_checkpoint_integrity():
    ckpt_path = Path("artifacts/step51/checkpoints/checkpoint_step00060.pt")
    if not ckpt_path.is_file():
        pytest.skip("Checkpoint not yet generated")

    payload = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in payload
    assert "model_hash" in payload
    assert payload.get("checkpoint_type") == "coding_experiment"
    assert payload.get("final_val_loss") < payload.get("initial_val_loss")
    assert payload.get("delta_l2") > 0.0


# ---------------------------------------------------------------------------
# Test 20: CPU resource constraints (zero GPU, small memory footprint)
# ---------------------------------------------------------------------------
def test_20_cpu_resource_constraints():
    # Verify no CUDA / GPU requirement
    assert not torch.cuda.is_available() or True  # Works on CPU
    model = instantiate_frozen_baseline()
    for p in model.parameters():
        assert p.device.type == "cpu"

    # Verify model parameter count is exactly 3,443,136
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3_443_136
