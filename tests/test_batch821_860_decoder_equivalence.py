import pytest
from chakrview.cognition.grammar_constrained_decoder import GrammarConstrainedDecoder
from chakrview.cognition.optimized_grammar_decoder import OptimizedGrammarDecoder
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    return tok


def test_decoder_equivalence_on_sample_prefixes(tokenizer):
    base_decoder = GrammarConstrainedDecoder(tokenizer)
    opt_decoder = OptimizedGrammarDecoder(tokenizer)

    test_prefixes = [
        "",
        "G",
        "GOAL: ",
        "GOAL: compound_arithmetic\n",
        "GOAL: compound_arithmetic\nSTEP ",
        "GOAL: compound_arithmetic\nSTEP 0: tool_call | ",
        "GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={",
        "GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={expr=12*4} | deps=[]\n",
        "GOAL: clarification_needed\nCLARIFICATION: ",
    ]

    for p in test_prefixes:
        base_allowed = set(base_decoder.get_allowed_tokens(p))
        opt_allowed = set(opt_decoder.get_allowed_tokens(p))
        # Optimized decoder candidates should be a valid subset containing the essential grammar transitions
        assert len(opt_allowed) > 0, f"Optimized decoder produced 0 candidates for prefix: {repr(p)}"
        # All tokens in opt_allowed must be valid according to base grammar checker
        assert opt_allowed.issubset(base_allowed), f"Optimized produced non-grammar tokens for prefix: {repr(p)}"


def test_empty_set_on_invalid_prefix(tokenizer):
    opt_decoder = OptimizedGrammarDecoder(tokenizer)
    invalid_prefix = "GOAL: invalid_goal_name\n"
    allowed = opt_decoder.get_allowed_tokens(invalid_prefix)
    assert len(allowed) == 0
