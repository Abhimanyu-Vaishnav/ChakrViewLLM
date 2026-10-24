import pytest
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent

from experiments.research_v09_conversation_sft.exp02_context_grounding.evaluate_context_grounding import (
    score_example,
    is_repetitive_ngram
)

def test_canonical_artifact_immutability():
    expected = {
        "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
        "data/experiments/vocab_4096/vocab.json": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
        "data/experiments/vocab_4096/merges.json": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
        "chakrview/runtime/session.py": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    }
    for rel_path, exp_hash in expected.items():
        p = REPO_ROOT / rel_path
        assert p.exists(), f"Missing file: {p}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == exp_hash, f"Hash mismatch on {rel_path}: {h} vs {exp_hash}"

def test_metric_repetition_rejection():
    # Repetitive loops must fail regardless of content
    text = "The fire of the fire of the fire of the fire of the fire."
    assert is_repetitive_ngram(text) is True
    success, reason = score_example("dynamic_entity_binding", text, {"target_entity": "fire"})
    assert success is False
    assert reason == "REPETITIVE_ATTRACTOR_COLLAPSE"

def test_metric_empty_output_rejection():
    success, reason = score_example("dynamic_entity_binding", "   ", {"target_entity": "sensor_1"})
    assert success is False
    assert reason == "EMPTY_OUTPUT"

def test_metric_entity_binding_distinguishes_training_leakage():
    meta = {"target_entity": "sensor_cluster_test_0"}
    
    # 1. Memorized training artifact must fail
    leak_output = "The sensor_cluster_train_0 is online."
    s_leak, r_leak = score_example("dynamic_entity_binding", leak_output, meta)
    assert s_leak is False
    assert r_leak == "ENTITY_NOT_FOUND"
    
    # 2. Genuine target binding succeeds
    valid_output = "The sensor_cluster_test_0 recorded 120."
    s_val, r_val = score_example("dynamic_entity_binding", valid_output, meta)
    assert s_val is True
    assert r_val == "EXACT_ENTITY_BOUND"

def test_metric_user_correction_drops_superseded_target():
    meta = {"new_target": "Unit_Beta", "cancelled_target": "Unit_Alpha"}
    
    # Keeps old target -> FAIL
    s1, r1 = score_example("user_correction_handling", "Assigned to Unit_Alpha and Unit_Beta", meta)
    assert s1 is False
    assert r1 == "FAILED_TO_UPDATE_TARGET"
    
    # Drops old target, uses new target -> PASS
    s2, r2 = score_example("user_correction_handling", "Cancelled Unit_Alpha. Assigned to Unit_Beta.", meta)
    assert s2 is True
    assert r2 == "CORRECTED_TARGET_BOUND"

def test_metric_distractor_resistance():
    meta = {"target_entity": "Subsystem_Core", "target_state": "operational", "distractors": ["degraded", "offline"]}
    
    # Intruding distractor state -> FAIL
    s1, r1 = score_example("distractor_resistance", "Subsystem_Core is operational but also offline", meta)
    assert s1 is False
    assert r1 == "DISTRACTOR_INTRUSION_OR_MISSING"
    
    # Clean target state -> PASS
    s2, r2 = score_example("distractor_resistance", "Subsystem_Core is operational.", meta)
    assert s2 is True
    assert r2 == "TARGET_BOUND_DISTRACTORS_REJECTED"

def test_metric_code_synthesis_ast_parser():
    meta = {"function_name": "is_valid_test_0"}
    
    # Invalid syntax -> FAIL
    s1, r1 = score_example("constrained_code_synthesis", "def is_valid_test_0(val): return val >", meta)
    assert s1 is False
    assert "AST_SYNTAX_ERROR" in r1
    
    # Wrong function name -> FAIL
    s2, r2 = score_example("constrained_code_synthesis", "def is_valid_train_0(val):\n    return val >= 10", meta)
    assert s2 is False
    assert r2 == "FUNCTION_NAME_MISMATCH"
    
    # Valid AST and name -> PASS
    s3, r3 = score_example("constrained_code_synthesis", "def is_valid_test_0(val):\n    return val >= 10", meta)
    assert s3 is True
    assert r3 == "VALID_AST_AND_NAME"

def test_benchmark_manifest_integrity():
    m_path = EXP_DIR / "BENCHMARK_MANIFEST.json"
    assert m_path.exists()
    m = json.loads(m_path.read_text(encoding="utf-8"))
    assert m["splits"]["train"]["count"] == 100
    assert m["splits"]["val"]["count"] == 30
    assert m["splits"]["test"]["count"] == 60
