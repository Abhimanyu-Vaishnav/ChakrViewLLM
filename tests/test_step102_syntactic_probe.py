"""
Tests for Step 102 Phase 2: Syntactic Probe Benchmark.

Verifies:
1. Benchmark fixture loads cleanly (40 probes, valid schemas).
2. All probe prompts and targets encode deterministically with v4096 tokenizer.
3. Baseline evaluation on 40 probes matches established measurement (5% top-1, 5% top-5).
4. Evaluator executes with zero parameter mutations (ΔW = 0).
"""

from pathlib import Path
import json
import pytest

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.syntactic_probe import SyntacticProbeEvaluator

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
PROBES_FILE = ROOT_DIR / "tests" / "fixtures" / "step102_syntactic_probes.json"


@pytest.fixture
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


def test_syntactic_probe_fixture_integrity(tokenizer):
    """Verify that the 40 syntactic probes fixture exists and satisfies all schema rules."""
    assert PROBES_FILE.is_file()
    with open(PROBES_FILE, "r", encoding="utf-8") as f:
        probes = json.load(f)

    assert len(probes) == 40
    categories = {"python_syntax", "delimiters", "control_flow", "natural_language"}
    found_cats = set()

    for p in probes:
        assert "probe_id" in p
        assert "category" in p
        assert "prompt" in p
        assert "target" in p
        assert p["category"] in categories
        found_cats.add(p["category"])

        p_tokens = tokenizer.encode(p["prompt"], add_bos=True, add_eos=False)
        t_tokens = tokenizer.encode(p["target"], add_bos=False, add_eos=False)
        assert len(p_tokens) > 0
        assert len(t_tokens) > 0

    assert found_cats == categories


def test_syntactic_probe_baseline_execution(tokenizer):
    """Verify that running SyntacticProbeEvaluator on the canonical baseline reproduces exact baseline metrics."""
    model = instantiate_frozen_baseline()
    evaluator = SyntacticProbeEvaluator(tokenizer=tokenizer, probes_path=PROBES_FILE)
    metrics = evaluator.evaluate(model)

    assert metrics.total_probes == 40
    assert metrics.top1_correct_count == 2
    assert metrics.top5_correct_count == 2
    assert metrics.top1_accuracy == 5.0
    assert metrics.top5_accuracy == 5.0

    # Ensure zero weight mutation
    assert compute_model_hash(model) == EXPECTED_WEIGHT_HASH
