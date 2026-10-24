"""
test_span_aligner.py
ChakrView Exp04 — Unit Tests for Fail-Closed Span Aligner.
"""

from pathlib import Path
import pytest
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp04_pointer_robustness.span_aligner import (
    align_character_span_to_tokens,
    AlignmentError
)

CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)
    return tok

def test_trailing_space_entity_alignment(tokenizer):
    prompt = "We connected both Pump_Alpha and Pump_Beta to the main coolant line."
    target = "Pump_Alpha"
    c_start = prompt.find(target)
    c_end = c_start + len(target)
    
    t_start, t_end, tokens = align_character_span_to_tokens(
        prompt, c_start, c_end, target, tokenizer
    )
    assert t_start > 0 and t_end >= t_start
    decoded = tokenizer.decode(tokens[t_start:t_end+1]).strip()
    assert decoded == target

def test_entity_with_punctuation(tokenizer):
    prompt = "Operator note: target status is [CRITICAL_FAULT]!"
    target = "CRITICAL_FAULT"
    c_start = prompt.find(target)
    c_end = c_start + len(target)
    
    t_start, t_end, tokens = align_character_span_to_tokens(
        prompt, c_start, c_end, target, tokenizer
    )
    assert t_start > 0
    decoded = tokenizer.decode(tokens[t_start:t_end+1]).strip()
    assert decoded == target

def test_entity_at_beginning_of_prompt(tokenizer):
    prompt = "Sensor_Gamma reported a temperature anomaly."
    target = "Sensor_Gamma"
    c_start = 0
    c_end = len(target)
    
    t_start, t_end, tokens = align_character_span_to_tokens(
        prompt, c_start, c_end, target, tokenizer
    )
    assert t_start == 1  # 0 is BOS
    decoded = tokenizer.decode(tokens[t_start:t_end+1]).strip()
    assert decoded == target

def test_entity_at_end_of_prompt(tokenizer):
    prompt = "Active pod name: pod_omega_9"
    target = "pod_omega_9"
    c_start = prompt.find(target)
    c_end = c_start + len(target)
    
    t_start, t_end, tokens = align_character_span_to_tokens(
        prompt, c_start, c_end, target, tokenizer
    )
    assert t_end == len(tokens) - 1
    decoded = tokenizer.decode(tokens[t_start:t_end+1]).strip()
    assert decoded == target

def test_abstention_returns_position_zero(tokenizer):
    prompt = "Is there an update on the deployment schedule?"
    t_start, t_end, tokens = align_character_span_to_tokens(
        prompt, 0, 0, "", tokenizer, is_abstention=True
    )
    assert t_start == 0 and t_end == 0

def test_unalignable_span_raises_explicit_error(tokenizer):
    prompt = "The system is running on server_node_1."
    # Intentionally misaligned character span
    with pytest.raises(AlignmentError):
        align_character_span_to_tokens(
            prompt, 5, 20, "nonexistent_target", tokenizer, is_abstention=False
        )
