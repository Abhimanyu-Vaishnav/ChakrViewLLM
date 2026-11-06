import os
import sys
import json
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

def test_tokenizer_loading():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    assert tok is not None
    assert tok.vocab_size == 4096

def test_utf8_partial_multibyte_decoding():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    # Encode a non-ascii Hindi word
    text = "नमस्ते दिल्ली"
    enc = tok.encode(text)
    # Check that decoding full ids reconstructs text
    decoded = tok.decode(enc)
    assert decoded == text
    
    # Check partial truncation behavior: does it crash or handle cleanly?
    # Truncate halfway through
    half_enc = enc[:len(enc)//2]
    try:
        partial_dec = tok.decode(half_enc)
        # Should be a string
        assert isinstance(partial_dec, str)
    except UnicodeDecodeError:
        # If raw decode raises UnicodeDecodeError, our benchmark evaluator must catch/replace
        pass

def test_normalization_rules():
    # Verify exact answer checking logic
    def check_hit(pred, expected, cat):
        norm_pred = pred.strip().lower()
        norm_exp = expected.strip().lower()
        if cat in ["single_hop_and_counterfactual", "multi_entity_distractor", "multiturn_state_tracking"]:
            return norm_exp in norm_pred
        elif cat == "missing_info_abstention":
            return any(phrase in norm_pred for phrase in ["not state", "does not mention", "unknown", "cannot be determined", "not mentioned", "not specify"])
        elif cat == "conditional_logical_reasoning":
            first_word = norm_pred.split()[0] if norm_pred.split() else ""
            return norm_exp == first_word or norm_exp in norm_pred[:len(norm_exp)+2]
        else:
            return norm_exp in norm_pred

    # Test cases
    assert check_hit("Delhi.", "Delhi", "single_hop_and_counterfactual")
    assert check_hit("The context does not state where Ravi lives.", "The context does not state", "missing_info_abstention")
    assert check_hit("Yes, she visits.", "Yes", "conditional_logical_reasoning")
    assert not check_hit("No, she does not.", "Yes", "conditional_logical_reasoning")
    assert check_hit("Rohan is in Jaipur.", "Jaipur", "multiturn_state_tracking")

if __name__ == "__main__":
    test_tokenizer_loading()
    test_utf8_partial_multibyte_decoding()
    test_normalization_rules()
    print("ALL EVALUATOR UNIT TESTS PASSED.")

