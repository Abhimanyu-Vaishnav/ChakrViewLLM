"""
Unit and Integration Tests for Batches 701-740:
Neural Subtask Decomposition, Learnability, Validation and Generalization.
Location: tests/test_batch701_740_neural_decomposition.py
"""
from pathlib import Path
import tempfile
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.subtask_decomposition_model import (
    ACTION_TO_ID,
    GOAL_TO_ID,
    ID_TO_ACTION,
    ID_TO_GOAL,
    ID_TO_TOOL,
    TOOL_TO_ID,
    ChakrSubtaskDecompositionModel,
)
from experiments.batch701_740.run_neural_planning_evaluation import (
    TaskGraphValidator,
    load_neural_planner,
    predict_neural_plan,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "experiments" / "batch701_740" / "checkpoints" / "candidate_step150_neural_planner.pt"


def test_neural_decomposition_checkpoint_presence_and_weights():
    """Verifies that the trained neural decomposition checkpoint exists and has valid weights."""
    assert CKPT_PATH.exists(), f"Candidate checkpoint not found at {CKPT_PATH}"
    device = torch.device("cpu")
    weights = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    assert "goal_head.2.weight" in weights
    assert "st1_action_head.weight" in weights
    assert "dependency_head.2.weight" in weights
    assert "clarification_head.weight" in weights


def test_task_graph_validator_acyclicity_and_tool_rules():
    """Verifies deterministic task graph validation without gold plan substitution."""
    # Valid sequential plan
    valid_plan = {
        "subtasks": [
            {"action": "tool_call", "tool": "calculator", "deps": []},
            {"action": "tool_call", "tool": "calculator", "deps": [0]},
        ],
        "clarification_needed": False,
    }
    is_valid, msg = TaskGraphValidator.validate_plan(valid_plan)
    assert is_valid is True
    assert msg == "Valid plan structure"

    # Invalid cyclic plan (subtask 0 depends on subtask 1)
    cyclic_plan = {
        "subtasks": [
            {"action": "tool_call", "tool": "calculator", "deps": [1]},
            {"action": "tool_call", "tool": "calculator", "deps": []},
        ],
        "clarification_needed": False,
    }
    is_valid, msg = TaskGraphValidator.validate_plan(cyclic_plan)
    assert is_valid is False
    assert "Invalid dependency" in msg

    # Unauthorized tool usage
    illegal_tool_plan = {
        "subtasks": [
            {"action": "tool_call", "tool": "root_shell_exec", "deps": []},
        ],
        "clarification_needed": False,
    }
    is_valid, msg = TaskGraphValidator.validate_plan(illegal_tool_plan)
    assert is_valid is False
    assert "Unauthorized or unknown tool" in msg


def test_neural_subtask_prediction_compound_math():
    """Tests raw neural prediction on a compound arithmetic query."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_neural_planner(device)
    query = "Calculate 15 * 12 and then add 20 to that result"
    pred = predict_neural_plan(model, tok, query, device)

    assert pred["goal_type"] == "compound_arithmetic"
    assert len(pred["subtasks"]) >= 2
    assert pred["subtasks"][0]["tool"] == "calculator"
    assert pred["subtasks"][1]["tool"] == "calculator"
    assert pred["subtasks"][1]["deps"] == [0]
    assert pred["clarification_needed"] is False


def test_neural_subtask_prediction_text_chain():
    """Tests raw neural prediction on a text drafting and transformation query."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_neural_planner(device)
    query = "Draft phrase 'Neural Architecture' and convert it to uppercase"
    pred = predict_neural_plan(model, tok, query, device)

    assert pred["goal_type"] == "text_transform_chain"
    assert len(pred["subtasks"]) >= 2
    assert pred["subtasks"][0]["tool"] == "text_utility"
    assert pred["subtasks"][1]["tool"] == "text_utility"
    assert pred["subtasks"][1]["deps"] == [0]


def test_neural_subtask_prediction_hindi_multilingual():
    """Tests generalization to Hindi/Hinglish multi-step phrasing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_neural_planner(device)
    query = "15 को 4 से गुणा करें और फिर परिणाम में 10 जोड़ें"
    pred = predict_neural_plan(model, tok, query, device)

    assert pred["goal_type"] == "compound_arithmetic"
    assert pred["subtasks"][0]["tool"] == "calculator"
    assert pred["subtasks"][1]["deps"] == [0]


def test_causal_clarification_responsiveness():
    """Tests that missing inputs trigger the neural clarification head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_neural_planner(device)
    query = "Compute total discounted cost for item without rate"
    pred = predict_neural_plan(model, tok, query, device)

    assert pred["goal_type"] == "clarification_needed"
    assert pred["clarification_needed"] is True
