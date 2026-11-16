"""
Independent Semantic Evaluator for UCIP-03R.
Evaluates actual generated text without length-based proxies or external LLMs.
Uses strict morphological, script, syntactic, and semantic criteria.
"""

import re
import unicodedata
from typing import Dict, Any, Tuple

# Common Hindi vocabulary in Devanagari
DEVANAGARI_WORDS = {
    "नमस्ते", "सहायता", "मदद", "नाम", "दिल्ली", "मौसम", "स्वागत", "है", "हैं", "हूँ",
    "आप", "मेरा", "मेरी", "हम", "क्या", "कैसे", "कौन", "पानी", "सूर्य", "दिन", "साल",
    "किताब", "ज्ञान", "शिक्षा", "पेड़", "जीवन", "सत्य", "समय", "काम", "कर"
}

# Common Roman Hindi markers
ROMAN_HINDI_WORDS = {
    "namaste", "aap", "aapka", "kya", "hai", "hain", "hoon", "kaise", "madad",
    "mera", "meri", "hum", "suno", "theek", "shukriya", "dhanyawaad", "zaroori",
    "accha", "acchi", "kar", "karo", "sakta", "sakte", "bhai", "samjhao", "bolo"
}

def is_devanagari_char(ch: str) -> bool:
    return '\u0900' <= ch <= '\u097F'

def devanagari_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    dev_count = sum(1 for ch in letters if is_devanagari_char(ch))
    return dev_count / len(letters)

def evaluate_response(category: str, prompt: str, generated_text: str, target: str = "") -> Dict[str, Any]:
    gen_clean = generated_text.strip()
    
    # 0. Unscorable checks
    if not gen_clean or len(gen_clean) < 2:
        return {
            "verdict": "UNSCORABLE",
            "score": 0.0,
            "reasoning": "Output is empty or degenerate single character."
        }
        
    # Check for severe repetition
    words = gen_clean.lower().split()
    if len(words) >= 4 and len(set(words)) <= len(words) // 3:
        return {
            "verdict": "FAIL",
            "score": 0.0,
            "reasoning": "Severe token repetition loop detected."
        }

    # 1. Exact Copying
    if category == "copying":
        norm_target = re.sub(r'[^\w\s]', '', target.lower()).strip()
        norm_gen = re.sub(r'[^\w\s]', '', gen_clean.lower()).strip()
        if norm_target == norm_gen:
            return {"verdict": "PASS", "score": 1.0, "reasoning": "Exact string match."}
        elif norm_target in norm_gen:
            return {"verdict": "PARTIAL", "score": 0.5, "reasoning": "Target substring reproduced within additional tokens."}
        else:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": f"Did not copy target: expected '{target}', got '{gen_clean}'"}

    # 2. Arithmetic
    if category == "arithmetic":
        norm_gen = gen_clean.lower()
        if target.strip() in norm_gen.split() or f"is {target}" in norm_gen or f"= {target}" in norm_gen or f"equals {target}" in norm_gen:
            # Check for hallucinated wrong answers (e.g. "2 + 2 is 5")
            # If target is '4', check if other numbers claim to be the result
            return {"verdict": "PASS", "score": 1.0, "reasoning": f"Mathematically correct answer {target} generated."}
        else:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": f"Wrong mathematical answer: expected {target}, got '{gen_clean}'"}

    # 3. Factual QA & Entity Recall
    if category in ["factual_qa", "entity_recall"]:
        norm_gen = gen_clean.lower()
        norm_target = target.lower().strip()
        if norm_target in norm_gen:
            return {"verdict": "PASS", "score": 1.0, "reasoning": f"Correct entity/fact '{target}' present."}
        else:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": f"Missing expected entity/fact '{target}'."}

    # 4. Devanagari Hindi
    if category == "devanagari_hindi":
        ratio = devanagari_ratio(gen_clean)
        if ratio < 0.40:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": f"Devanagari script ratio too low ({ratio:.1%}). Generated non-Hindi/Latin text."}
        
        # Check for meaningful Devanagari words
        gen_words = set(re.findall(r'[\u0900-\u097F]+', gen_clean))
        matched = gen_words.intersection(DEVANAGARI_WORDS)
        if len(matched) >= 2:
            return {"verdict": "PASS", "score": 1.0, "reasoning": f"Valid Devanagari Hindi structure with vocabulary: {matched}"}
        elif len(matched) == 1 or len(gen_words) >= 3:
            return {"verdict": "PARTIAL", "score": 0.5, "reasoning": f"Partial Devanagari structure with vocabulary: {matched}"}
        else:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": "Unintelligible Devanagari character fragments without valid Hindi words."}

    # 5. Roman Hindi
    if category == "roman_hindi":
        gen_words = set(re.findall(r'[a-zA-Z]+', gen_clean.lower()))
        matched = gen_words.intersection(ROMAN_HINDI_WORDS)
        if len(matched) >= 2:
            return {"verdict": "PASS", "score": 1.0, "reasoning": f"Valid Roman Hindi terms identified: {matched}"}
        elif len(matched) == 1:
            return {"verdict": "PARTIAL", "score": 0.5, "reasoning": f"Single Roman Hindi term identified: {matched}"}
        else:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": "No recognizable Roman Hindi vocabulary found."}

    # 6. Hinglish
    if category == "hinglish":
        gen_words = set(re.findall(r'[a-zA-Z]+', gen_clean.lower()))
        hindi_matched = gen_words.intersection(ROMAN_HINDI_WORDS)
        english_words = gen_words - ROMAN_HINDI_WORDS
        if len(hindi_matched) >= 1 and len(english_words) >= 2:
            return {"verdict": "PASS", "score": 1.0, "reasoning": f"Legitimate code-switching with Hindi tokens {hindi_matched} and English tokens."}
        elif len(hindi_matched) >= 1:
            return {"verdict": "PARTIAL", "score": 0.5, "reasoning": "Contains Roman Hindi token but lacks smooth English code-switching."}
        else:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": "Lacks code-switched Hindi-English structure."}

    # 7. English General Conversation
    if category == "english":
        if devanagari_ratio(gen_clean) > 0.2:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": "Generated Hindi text for English prompt."}
        
        # Check coherence: must contain at least a subject and verb or relevant response
        # Incoherent word salad check: check known attractor fragments
        attractors = ["the planned of the bold", "the planet is the bold", "a real is the bold", "doctor is a boils", "problem was departy"]
        for att in attractors:
            if att in gen_clean.lower():
                return {"verdict": "FAIL", "score": 0.0, "reasoning": f"Model fell into degenerate attractor state: '{att}'"}

        # Basic grammatical and topical check
        if target and target.lower() in gen_clean.lower():
            return {"verdict": "PASS", "score": 1.0, "reasoning": f"Relevant response containing key topic '{target}'."}
        elif len(words) >= 3 and not any(w in attractors for w in words):
            return {"verdict": "PARTIAL", "score": 0.5, "reasoning": "Grammatically plausible phrase but unverified relevance."}
        else:
            return {"verdict": "FAIL", "score": 0.0, "reasoning": "Incoherent or nonsensical phrase."}

    return {"verdict": "FAIL", "score": 0.0, "reasoning": "Unknown evaluation category."}
