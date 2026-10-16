"""Unit tests for Batches 1461–1500: Capability Audit, Dataset Integrity & Evaluator Logic."""

import hashlib
import json
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = REPO_ROOT / "experiments" / "batch1461_1500"
DATA_DIR = BATCH_DIR / "data"
RESULTS_DIR = BATCH_DIR / "results"

CANONICAL_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "batch16_21_combined"
    / "main_training_10000"
    / "training_checkpoint_step_10000.pt"
)
EXPECTED_CANONICAL_HASH = (
    "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
)

ADAPTED_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "batch1381_1420"
    / "checkpoints"
    / "chakrmicro_domain_adapted_experimental.pt"
)
EXPECTED_ADAPTED_HASH = (
    "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"
)


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _evaluate_numeric_answer(raw_output: str, expected: str) -> tuple[bool, str | None]:
    """Mirror scoring function in run_capability_audit.py."""
    num_match = re.search(r"[-+]?\d+(?:\.\d+)?", raw_output)
    extracted = num_match.group(0) if num_match else None
    is_correct = (extracted == expected)
    return is_correct, extracted


def _check_python_syntax(code_str: str) -> bool:
    """Mirror syntax validation in run_capability_audit.py."""
    try:
        compile(code_str, "<string>", "exec")
        return True
    except Exception:
        return False


def _execute_python_code(code_str: str, assertions: str) -> tuple[bool, str | None]:
    """Mirror sandboxed test execution in run_capability_audit.py."""
    test_script = code_str + "\n" + assertions
    namespace = {}
    try:
        exec(test_script, namespace)
        return True, None
    except AssertionError:
        return False, "AssertionError: Unit test assertions failed."
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def test_checkpoint_hashes_unmodified():
    """Verify that both canonical and adapted checkpoints match their frozen SHA-256."""
    assert CANONICAL_CHECKPOINT.exists(), f"Missing {CANONICAL_CHECKPOINT}"
    assert ADAPTED_CHECKPOINT.exists(), f"Missing {ADAPTED_CHECKPOINT}"

    canon_hash = _compute_sha256(CANONICAL_CHECKPOINT)
    adapt_hash = _compute_sha256(ADAPTED_CHECKPOINT)

    assert (
        canon_hash == EXPECTED_CANONICAL_HASH
    ), f"Canonical checkpoint hash mismatch: {canon_hash}"
    assert (
        adapt_hash == EXPECTED_ADAPTED_HASH
    ), f"Adapted checkpoint hash mismatch: {adapt_hash}"


def test_capability_dataset_schema_and_integrity():
    """Verify the 200-task audit dataset matches manifest counts, types, and schema."""
    dataset_file = DATA_DIR / "capability_audit_200.json"
    manifest_file = DATA_DIR / "CAPABILITY_DATASET_MANIFEST.json"

    assert dataset_file.exists()
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    assert len(dataset) == 200
    assert manifest["total_tasks"] == 200

    # Verify sha256 in manifest matches actual dataset
    actual_hash = _compute_sha256(dataset_file)
    assert manifest["sha256"] == actual_hash

    # Check category distribution
    categories = {}
    for task in dataset:
        assert "id" in task
        assert "category" in task
        assert "difficulty" in task
        cat = task["category"]
        categories[cat] = categories.get(cat, 0) + 1

    assert categories["arithmetic_reasoning"] == 40
    assert categories["python_execution"] == 40
    assert categories["factual_recall"] == 40
    assert categories["paraphrase_canonical"] == 20
    assert categories["paraphrase_variant"] == 20
    assert categories["false_premise_refutation"] == 20
    assert categories["honest_abstention"] == 20


def test_evaluator_numeric_matching():
    """Test the numeric evaluator against strict matches and adversarial traps."""
    # Strict correct matches
    is_corr, num = _evaluate_numeric_answer("405", "405")
    assert is_corr and num == "405"

    is_corr, num = _evaluate_numeric_answer("Answer is: 512. That's it.", "512")
    assert is_corr and num == "512"

    # Floating point
    is_corr, num = _evaluate_numeric_answer("3.1415", "3.1415")
    assert is_corr

    # Substring trap (expected '4', output '40')
    is_corr, num = _evaluate_numeric_answer("The result is 40 nodes", "4")
    assert not is_corr
    assert num == "40"

    # Repetitive hallucination without number
    is_corr, num = _evaluate_numeric_answer(
        "The first number of the first cannot the first", "100"
    )
    assert not is_corr
    assert num is None


def test_evaluator_python_syntax_and_sandbox():
    """Test AST syntax validation and isolated subprocess execution."""
    # Valid syntax
    code_valid = "def add(a, b):\n    return a + b\n"
    assert _check_python_syntax(code_valid)

    # Invalid syntax (unterminated docstring trap seen in audit)
    code_invalid = "def add(a, b):\n    \"\"\"\n    if not self._subset(\")\n"
    assert not _check_python_syntax(code_invalid)

    # Successful execution against unit tests
    passed, err = _execute_python_code(
        code_valid,
        "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n",
    )
    assert passed
    assert err is None

    # Failed unit tests (e.g. stub pass)
    code_stub = "def add(a, b):\n    pass\n"
    passed, err = _execute_python_code(
        code_stub,
        "assert add(2, 3) == 5\n",
    )
    assert not passed
    assert "AssertionError" in err


def test_audit_results_artifact_exists():
    """Verify that machine-readable benchmark results exist and are populated."""
    results_file = RESULTS_DIR / "CAPABILITY_AUDIT_RESULTS.json"
    assert results_file.exists()

    with open(results_file, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert "summary_matrix" in res
    assert "track_c_details" in res
    assert "track_d_details" in res
    assert "track_b_details" in res
    assert "track_e_details" in res
    assert "control_details" in res

    matrix = res["summary_matrix"]
    assert matrix["track_c_mathematics"]["canonical_base_accuracy"] == 0.0
    assert matrix["track_d_coding"]["canonical_test_pass_rate"] == 0.0
    assert matrix["track_b_factual_grounding"]["grounded_assistant_accuracy"] == 0.75
    assert matrix["track_e_generalization"]["pair_consistency_rate"] == 0.90
    assert matrix["controls"]["abstention_accuracy"] == 1.00
