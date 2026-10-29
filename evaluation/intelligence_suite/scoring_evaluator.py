"""
ChakrView Evaluation: Robust Scoring Logic
Provides robust evaluation functions for exact match, normalized token match,
digit boundaries, and isolated functional coding tests.
"""
import re
import ast
from typing import Tuple, Dict, Any, Optional

try:
    from evaluation.coding_suite.isolated_executor import run_process_isolated_test
    ISOLATED_EXECUTOR_AVAILABLE = True
except ImportError:
    ISOLATED_EXECUTOR_AVAILABLE = False


def evaluate_response(
    generated_text: str,
    expected_concept: str,
    scoring_rule: str,
    test_code: str = "",
    full_prompt: str = ""
) -> Tuple[bool, str]:
    """
    Evaluates generated output against expected concept using robust rule matching.
    
    Returns:
        (passed, reason_message)
    """
    clean_gen = generated_text.strip()
    if not clean_gen:
        return False, "EMPTY_RESPONSE"

    clean_exp = expected_concept.strip()

    if scoring_rule == "exact_strip":
        # Strict exact match ignoring leading/trailing whitespace
        passed = (clean_gen.lower() == clean_exp.lower())
        return passed, "EXACT_MATCH" if passed else f"MISMATCH: '{clean_gen}' != '{clean_exp}'"

    elif scoring_rule == "word_boundary" or scoring_rule == "number_match":
        # Substring checks must NOT accept accidental matches (e.g. 7 inside 17, or 'yes' in 'yesterday')
        pattern = r"(?<![a-zA-Z0-9_])" + re.escape(clean_exp.lower()) + r"(?![a-zA-Z0-9_])"
        match = re.search(pattern, clean_gen.lower())
        passed = bool(match)
        return passed, "BOUNDARY_MATCH" if passed else f"NO_WORD_BOUNDARY_MATCH: '{clean_exp}' not isolated in text"

    elif scoring_rule == "code_functional":
        # First check syntax
        code_to_eval = (full_prompt + clean_gen) if full_prompt else clean_gen
        # If test_code exists, run in isolated test runner
        if test_code and ISOLATED_EXECUTOR_AVAILABLE:
            res = run_process_isolated_test(code_to_eval, test_code, timeout_seconds=2.0)
            if res.get("rejected_by_policy"):
                return False, f"POLICY_REJECTED: {res.get('error')}"
            if not res.get("syntax_valid"):
                return False, f"SYNTAX_INVALID: {res.get('error')}"
            passed = bool(res.get("passed"))
            return passed, "FUNCTIONAL_TEST_PASSED" if passed else f"ASSERTION_FAILED: {res.get('error')}"
        else:
            # Fall back to syntax validity check
            try:
                ast.parse(code_to_eval)
                return True, "SYNTAX_VALID"
            except SyntaxError as e:
                return False, f"SYNTAX_ERROR: {e.msg}"

    elif scoring_rule == "three_word_sentence":
        # Strictly validates a complete three-word sentence ending with sentence punctuation (. ! ?)
        first_line = clean_gen.splitlines()[0].strip()
        words = [w for w in re.findall(r"[a-zA-Z0-9'-]+", first_line) if w]
        has_terminal_punct = bool(re.search(r"[.!?]$", first_line))
        passed = (len(words) == 3 and has_terminal_punct)
        if passed:
            return True, f"THREE_WORD_SENTENCE_VALID: '{first_line}'"
        return False, f"INVALID_THREE_WORD_SENTENCE: word_count={len(words)} (expected 3), terminal_punct={has_terminal_punct}"

    elif scoring_rule == "substring":
        # Case-insensitive substring with minimum check
        passed = clean_exp.lower() in clean_gen.lower()
        return passed, "SUBSTRING_MATCH" if passed else f"SUBSTRING_NOT_FOUND: '{clean_exp}'"

    elif scoring_rule == "heuristic_summary":
        # Summarization is treated as a limited heuristic check
        passed = clean_exp.lower() in clean_gen.lower()
        return passed, "HEURISTIC_KEYWORD_MATCH" if passed else f"KEYWORD_MISSING: '{clean_exp}'"

    return False, f"UNKNOWN_SCORING_RULE: {scoring_rule}"