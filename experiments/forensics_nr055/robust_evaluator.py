import re
import unicodedata

def normalize_text(text: str) -> str:
    """Normalize text by trimming, lowercasing, and normalizing unicode."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    return text

def extract_numbers(text: str):
    """Extract all distinct integer and decimal numbers from text."""
    # Matches integers, decimals, negative numbers
    pattern = r"[-+]?\d*\.?\d+"
    matches = re.findall(pattern, text)
    nums = []
    for m in matches:
        try:
            if "." in m:
                nums.append(float(m))
            else:
                nums.append(int(m))
        except ValueError:
            pass
    return nums

def score_item(response: str, benchmark_item: dict) -> dict:
    """
    Score a generated response against a benchmark item using a strict 0-3 rubric.
    
    Rubric:
      0: Irrelevant, hallucinated, malformed, empty, or wrong.
      1: Partially relevant or contains partial topical overlap but materially incorrect.
      2: Mostly correct or contains correct factual answer with minor extraneous tokens.
      3: Completely correct, strictly satisfying the prompt and constraints.
      
    Never awards score 2 or 3 on substring matching alone without boundary validation.
    """
    clean_resp = normalize_text(response)
    expected = normalize_text(benchmark_item["expected_answer"])
    alts = [normalize_text(a) for a in benchmark_item.get("acceptable_alternatives", [])]
    all_targets = [expected] + alts
    
    r_type = benchmark_item.get("rubric_type", "exact_or_substring")
    constraints = benchmark_item.get("constraints", [])
    
    # Check for empty or whitespace-only response
    if not clean_resp or len(clean_resp) == 0:
        return {"score": 0, "verdict": "empty_response", "reason": "No text generated"}
        
    # Check for prompt echo / repetitive garble
    tokens = clean_resp.split()
    if len(tokens) > 4:
        # Check if identical 3-gram repeats 3+ times
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
        if len(trigrams) > 3 and max(trigrams.count(t) for t in trigrams) >= 3:
            return {"score": 0, "verdict": "repetition_loop", "reason": "Severe repetition detected"}

    # 1. Numerical Exact Matching
    if r_type == "numerical_exact":
        resp_nums = extract_numbers(clean_resp)
        target_nums = []
        for t in all_targets:
            target_nums.extend(extract_numbers(t))
            
        if not target_nums:
            # Fallback for word numbers like 'seven'
            for t in all_targets:
                pattern = rf"\b{re.escape(t)}\b"
                if re.search(pattern, clean_resp):
                    if len(tokens) <= 3:
                        return {"score": 3, "verdict": "exact_match", "reason": "Target number word matched exactly"}
                    else:
                        return {"score": 2, "verdict": "mostly_correct", "reason": "Target number word matched in prose"}
            return {"score": 0, "verdict": "no_number_found", "reason": "Target number not present"}
            
        expected_val = target_nums[0]
        if expected_val in resp_nums:
            # Check if there are contradicting numbers
            if len(resp_nums) == 1:
                if len(tokens) <= 3:
                    return {"score": 3, "verdict": "exact_numerical", "reason": "Exact single number"}
                else:
                    return {"score": 2, "verdict": "mostly_correct_prose", "reason": "Correct number embedded in sentence"}
            elif len(resp_nums) <= 3:
                # E.g. "5 minus 3 = 2"
                # If target is the last number or explicitly preceded by answer indicator
                if resp_nums[-1] == expected_val or f"= {expected_val}" in clean_resp or f"is {expected_val}" in clean_resp:
                    return {"score": 2, "verdict": "equation_containment", "reason": "Valid equation context"}
                return {"score": 1, "verdict": "ambiguous_numbers", "reason": "Multiple conflicting numbers present"}
            else:
                return {"score": 0, "verdict": "numerical_noise", "reason": "Excessive random numbers"}
        else:
            return {"score": 0, "verdict": "incorrect_number", "reason": f"Expected {expected_val}, got {resp_nums}"}

    # 2. Strict Exact Matching (e.g. single word constraint, spelling, boolean)
    elif r_type == "strict_exact":
        # Check single word / single char constraints
        if "single_word" in constraints or "single_char" in constraints:
            # Strip punctuation
            clean_token = re.sub(r"[^\w\s]", "", clean_resp).strip()
            if clean_token in all_targets:
                return {"score": 3, "verdict": "strict_single_token_match", "reason": "Clean exact single token"}
            # Check if first word is target
            words = clean_token.split()
            if len(words) > 0 and words[0] in all_targets:
                return {"score": 2, "verdict": "target_prefixed", "reason": "Target followed by extra tokens"}
            return {"score": 0, "verdict": "constraint_violation", "reason": f"Expected single token '{expected}', got '{clean_resp}'"}
            
        # Standard strict exact
        for t in all_targets:
            if clean_resp == t or clean_resp.strip(".!?,") == t:
                return {"score": 3, "verdict": "exact_string_match", "reason": "Exact response string"}
            pattern = rf"^{re.escape(t)}[.!?]?$"
            if re.match(pattern, clean_resp):
                return {"score": 3, "verdict": "exact_string_match", "reason": "Exact response string"}
                
        # Check if target is isolated by word boundary
        for t in all_targets:
            pattern = rf"\b{re.escape(t)}\b"
            if re.search(pattern, clean_resp):
                if len(tokens) <= len(t.split()) + 3:
                    return {"score": 2, "verdict": "target_in_short_phrase", "reason": "Target present in brief answer"}
                return {"score": 1, "verdict": "target_diluted", "reason": "Target word found in long uncontrolled prose"}
                
        return {"score": 0, "verdict": "unmatched_strict", "reason": "Target string not matched"}

    # 3. Token Set Match (e.g., list of colors)
    elif r_type == "token_set_match":
        # Check if all required items in expected are present
        target_elements = [normalize_text(x) for x in re.split(r"[,/ ]+", expected) if x]
        present = sum(1 for el in target_elements if re.search(rf"\b{re.escape(el)}\b", clean_resp))
        if present == len(target_elements):
            return {"score": 3, "verdict": "complete_set_match", "reason": "All set elements present"}
        elif present >= len(target_elements) / 2:
            return {"score": 1, "verdict": "partial_set_match", "reason": f"{present}/{len(target_elements)} elements present"}
        return {"score": 0, "verdict": "missing_set_elements", "reason": "Elements missing"}

    # 4. Standard Factual / Substring with Word Boundaries
    else:
        for t in all_targets:
            # Must match as whole word / whole phrase (never inside another word like 'ice' in 'notice')
            pattern = rf"\b{re.escape(t)}\b"
            if re.search(pattern, clean_resp):
                # Check length to award 3 vs 2
                target_len = len(t.split())
                if len(tokens) <= target_len + 4:
                    return {"score": 3, "verdict": "clean_factual_hit", "reason": f"Factual target '{t}' directly stated"}
                elif len(tokens) <= 25:
                    return {"score": 2, "verdict": "factual_hit_in_sentence", "reason": f"Factual target '{t}' embedded in sentence"}
                else:
                    return {"score": 1, "verdict": "factual_hit_in_verbose_text", "reason": f"Factual target '{t}' found in verbose paragraph"}
                    
        return {"score": 0, "verdict": "factual_miss", "reason": "No target factual entities matched with word boundaries"}


# Adversarial Unit Tests
def run_adversarial_evaluator_tests():
    test_cases = [
        # Adversarial 1: 'ice' inside 'notice' or 'service'
        {
            "response": "Please notice that customer service is open.",
            "item": {"expected_answer": "ice", "rubric_type": "exact_or_substring", "constraints": ["factual"]},
            "expected_score": 0
        },
        # Adversarial 2: '2' in date '2024' or long noise
        {
            "response": "The article was written in 2024 by 5 authors.",
            "item": {"expected_answer": "2", "rubric_type": "numerical_exact", "constraints": ["exact_number"]},
            "expected_score": 0
        },
        # Adversarial 3: Single digit 2 in legitimate equation
        {
            "response": "5 minus 3 is 2.",
            "item": {"expected_answer": "2", "rubric_type": "numerical_exact", "constraints": ["exact_number"]},
            "expected_score": 2
        },
        # Adversarial 4: Clean single number 2
        {
            "response": "2",
            "item": {"expected_answer": "2", "rubric_type": "numerical_exact", "constraints": ["exact_number"]},
            "expected_score": 3
        },
        # Adversarial 5: Repetitive babble containing factual entity
        {
            "response": "Tokyo is Tokyo is Tokyo is Tokyo is Tokyo is Tokyo is Tokyo",
            "item": {"expected_answer": "Tokyo", "rubric_type": "exact_or_substring", "constraints": ["factual"]},
            "expected_score": 0
        },
        # Adversarial 6: Clean factual entity
        {
            "response": "The capital is Tokyo.",
            "item": {"expected_answer": "Tokyo", "rubric_type": "exact_or_substring", "constraints": ["factual"]},
            "expected_score": 3
        },
        # Adversarial 7: Strict instruction violation
        {
            "response": "YES, absolutely I agree with that.",
            "item": {"expected_answer": "YES", "rubric_type": "strict_exact", "constraints": ["single_word"]},
            "expected_score": 2
        },
        # Adversarial 8: Strict instruction perfect
        {
            "response": "YES",
            "item": {"expected_answer": "YES", "rubric_type": "strict_exact", "constraints": ["single_word"]},
            "expected_score": 3
        }
    ]
    
    passed = 0
    for i, tc in enumerate(test_cases):
        res = score_item(tc["response"], tc["item"])
        score = res["score"]
        if score == tc["expected_score"]:
            passed += 1
            print(f"Test {i+1} PASSED: {res['verdict']} (Score {score})")
        else:
            print(f"Test {i+1} FAILED! Expected {tc['expected_score']}, got {score}. Reason: {res['reason']}")
            
    print(f"\nEvaluator Adversarial Suite: {passed}/{len(test_cases)} tests passed.")
    assert passed == len(test_cases), "Not all adversarial tests passed!"

if __name__ == "__main__":
    run_adversarial_evaluator_tests()
