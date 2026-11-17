import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(r"D:\Project\ChakrView")
BENCHMARK_PATH = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "sealed_language_intelligence_500.json"

def is_repetitive(text: str) -> bool:
    """Detect repetitive n-gram loops or stuttering."""
    words = [w.strip() for w in text.split() if w.strip()]
    if len(words) < 4:
        return False
    # Check for consecutive identical words (>= 3 times)
    for i in range(len(words) - 2):
        if words[i].lower() == words[i+1].lower() == words[i+2].lower():
            return True
    # Check for 2-word phrase looping >= 3 times
    phrases = [" ".join(words[i:i+2]).lower() for i in range(len(words) - 1)]
    for i in range(len(phrases) - 4):
        if phrases[i] == phrases[i+2] == phrases[i+4]:
            return True
    return False

def evaluate_probe(probe: Dict[str, Any], generated: str) -> Dict[str, Any]:
    cleaned = generated.strip()
    norm_out = cleaned.lower()
    cat = probe["category"]

    if not cleaned:
        return {"passed": False, "status": "EMPTY_OUTPUT", "reason": "Empty string"}

    if is_repetitive(cleaned):
        return {"passed": False, "status": "REPETITIVE_DEGENERATION", "reason": "Repetitive phrase or token looping"}

    # 1. Exact Copying
    if cat == "Exact copying" or probe.get("exact_match", False):
        target = probe.get("target", "").strip()
        # Clean both target and output of boundary punctuation
        t_clean = re.sub(r"^[^\w]+|[^\w]+$", "", target.lower())
        g_clean = re.sub(r"^[^\w]+|[^\w]+$", "", norm_out)
        passed = (t_clean == g_clean) or (target.lower() in norm_out and len(cleaned.split()) <= len(target.split()) + 3)
        return {
            "passed": passed,
            "status": "CORRECT" if passed else "INCORRECT",
            "reason": f"Target: '{target}' vs Generated: '{cleaned}'"
        }

    # 2. Elementary Arithmetic
    if cat == "Elementary arithmetic":
        target_num = probe.get("target_number")
        found_nums = [int(m) for m in re.findall(r"\b\d+\b", cleaned)]
        num_words = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-two": 22, "twenty-five": 25,
            "thirty": 30, "thirty-two": 32, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70
        }
        for w, val in num_words.items():
            if re.search(rf"\b{w}\b", norm_out):
                found_nums.append(val)

        if target_num in found_nums:
            return {"passed": True, "status": "CORRECT", "reason": f"Found target {target_num} in {found_nums}"}
        else:
            return {"passed": False, "status": "INCORRECT", "reason": f"Expected {target_num}, found {found_nums}"}

    # 3. Devanagari Hindi
    if cat == "Devanagari Hindi":
        dev_chars = len(re.findall(r"[\u0900-\u097F]", cleaned))
        if dev_chars < 6:
            return {"passed": False, "status": "INCORRECT_SCRIPT", "reason": f"Insufficient Devanagari script: {dev_chars} chars"}
        exp_kw = probe.get("expected_keywords", [])
        kw_hits = [k for k in exp_kw if k in cleaned]
        if len(kw_hits) >= 1 and len(cleaned.split()) >= 3:
            return {"passed": True, "status": "CORRECT", "reason": f"Devanagari verified, kw_hits: {kw_hits}"}
        return {"passed": False, "status": "IRRELEVANT", "reason": f"Missing expected Hindi keywords: {exp_kw}"}

    # 4. Semantic Consistency (Negative Constraints)
    if cat == "Semantic consistency":
        forb = probe.get("forbidden", [])
        for f in forb:
            if re.search(rf"\b{f.lower()}\b", norm_out):
                return {"passed": False, "status": "CONSTRAINT_VIOLATED", "reason": f"Used forbidden word: '{f}'"}
        exp_kw = probe.get("expected_keywords", [])
        kw_hits = [k for k in exp_kw if re.search(rf"\b{k.lower()}\b", norm_out)]
        if len(kw_hits) >= 1 and len(cleaned.split()) >= 3:
            return {"passed": True, "status": "CORRECT", "reason": f"Consistent, kw_hits: {kw_hits}"}
        return {"passed": False, "status": "IRRELEVANT", "reason": f"Missing expected positive keywords: {exp_kw}"}

    # 5. English Explanations (Strict Multi-Concept Co-occurrence)
    if cat == "English explanations":
        exp_kw = probe.get("expected_keywords", [])
        kw_hits = [k for k in exp_kw if k.lower() in norm_out]
        # Require co-occurrence of at least 2 distinct concepts
        if len(kw_hits) >= 2 and len(cleaned.split()) >= 4:
            return {"passed": True, "status": "CORRECT", "reason": f"Multi-concept hit: {kw_hits}"}
        elif len(kw_hits) == 1:
            return {"passed": False, "status": "PARTIAL_CONCEPT", "reason": f"Only 1 keyword hit: {kw_hits} (needed >= 2)"}
        return {"passed": False, "status": "IRRELEVANT", "reason": f"Missing conceptual keywords: {exp_kw}"}

    # 6. General Conversational / Hinglish / Roman Hindi / Multi-Turn / Instruction Following
    exp_kw = probe.get("expected_keywords", [])
    kw_hits = [k for k in exp_kw if k.lower() in norm_out]
    if len(kw_hits) >= 1 and len(cleaned.split()) >= 1:
        return {"passed": True, "status": "CORRECT", "reason": f"Keyword verified: {kw_hits}"}

    return {"passed": False, "status": "IRRELEVANT", "reason": f"Missing keywords: {exp_kw}"}
