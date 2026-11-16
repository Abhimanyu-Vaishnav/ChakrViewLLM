import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import json
import re

# Comprehensive rubric evaluator for baseline_benchmark.json
def evaluate_response(probe: dict, generated_text: str) -> dict:
    gen_clean = generated_text.strip()
    category = probe["category"]
    expected_kw = probe.get("expected_keywords", [])
    exact_match_flag = probe.get("exact_match", False)
    
    # 1. Non-empty check
    if not gen_clean:
        return {"passed": False, "score": 0.0, "reason": "Empty output"}
        
    # 2. Severe repetition / degenerate loop check
    words = gen_clean.split()
    if len(words) >= 6:
        # Check adjacent word repetitions (e.g. 'is a is a is a', 'an an an an')
        adjacent_repeats = sum(1 for i in range(len(words)-1) if words[i].lower() == words[i+1].lower())
        if adjacent_repeats >= 4:
            return {"passed": False, "score": 0.0, "reason": "Severe adjacent repetition"}
        # Check 3-gram repetition
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) > 3 and len(set(trigrams)) <= len(trigrams) // 2:
            return {"passed": False, "score": 0.0, "reason": "Severe n-gram repetition loop"}

    # 3. Category-specific evaluation
    if exact_match_flag:
        # Normalize punctuation and casing for fair evaluation
        def norm(s):
            return re.sub(r'[^\w\s]', '', s).strip().lower()
        target = expected_kw[0]
        if norm(gen_clean) == norm(target) or norm(target) in norm(gen_clean):
            return {"passed": True, "score": 1.0, "reason": "Exact match satisfied"}
        else:
            return {"passed": False, "score": 0.0, "reason": f"Expected exact match for {repr(target)}"}
            
    if category == "Elementary Arithmetic":
        # Look for the required number in the generated response
        # Numbers can be digits or word representations
        has_num = any(kw.lower() in gen_clean.lower() for kw in expected_kw)
        if has_num:
            return {"passed": True, "score": 1.0, "reason": f"Correct arithmetic number present ({expected_kw})"}
        else:
            return {"passed": False, "score": 0.0, "reason": f"Missing expected arithmetic result ({expected_kw})"}
            
    if category == "Basic Instructions":
        # Check single-word or constraint fulfillment
        inp_lower = probe["input"].lower()
        if "one word" in inp_lower or "planet name" in inp_lower or "capital city name" in inp_lower:
            # Should have the target keyword and not be excessively rambling
            has_kw = any(kw.lower() in gen_clean.lower() for kw in expected_kw)
            is_concise = len(words) <= 10
            if has_kw and is_concise:
                return {"passed": True, "score": 1.0, "reason": "Instruction constraint and keyword satisfied"}
            else:
                return {"passed": False, "score": 0.0, "reason": "Failed conciseness or missing target keyword"}
        elif "yes or no" in inp_lower:
            has_kw = any(kw.lower() in [w.lower().strip(".,!?") for w in words[:3]] for kw in expected_kw)
            return {"passed": has_kw, "score": 1.0 if has_kw else 0.0, "reason": "Yes/No match"}
        else:
            has_kw = sum(1 for kw in expected_kw if kw.lower() in gen_clean.lower())
            passed = has_kw >= 1 and len(words) >= 3
            return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Keyword overlap: {has_kw}"}

    if category == "Devanagari Hindi":
        # Must contain Devanagari Unicode characters (U+0900 to U+097F)
        devanagari_chars = [c for c in gen_clean if '\u0900' <= c <= '\u097f']
        if len(devanagari_chars) < 5:
            return {"passed": False, "score": 0.0, "reason": "Insufficient Devanagari characters"}
        # Check keyword presence or semantic relevance
        matches = sum(1 for kw in expected_kw if kw in gen_clean)
        # Pass if at least 1 relevant keyword matches and no collapse
        passed = matches >= 1 and len(devanagari_chars) >= 10
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Devanagari keywords matched: {matches}"}

    # For general English and Hinglish categories:
    matches = sum(1 for kw in expected_kw if kw.lower() in gen_clean.lower())
    if matches >= 1 and len(words) >= 4:
        return {"passed": True, "score": 1.0, "reason": f"Semantic keywords matched: {matches}"}
    else:
        return {"passed": False, "score": 0.0, "reason": f"Low relevance (matches={matches}, words={len(words)})"}
