"""
Deterministic Workload Feature Extraction for Adaptive Orchestration (Step 28).

Provides rule-based, deterministic syntactic and structural feature extraction
to characterize cognitive tasks without stochasticity or hidden model calls.
"""

from typing import Dict, Any, List, Optional, Set
import re


# Deterministic lexical patterns for feature extraction
MATH_LOGIC_TERMS = {
    "calculate", "compute", "sum", "average", "multiply", "divide", "subtract",
    "equation", "formula", "proof", "deduce", "infer", "logic", "algorithm",
    "matrix", "vector", "derivative", "integral", "theorem", "lemma"
}

MULTI_PART_TERMS = {
    "furthermore", "moreover", "in addition", "subsequently", "firstly",
    "secondly", "thirdly", "step 1", "step 2", "part a", "part b", ";",
    "on the other hand", "alternatively", "compare and contrast"
}

CONTRADICTION_TERMS = {
    "contradict", "contradiction", "conflict", "opposing", "dispute",
    "inconsistent", "versus", "vs", "counter-claim", "disagree",
    "clash", "dilemma", "paradox"
}

AMBIGUITY_TERMS = {
    "maybe", "perhaps", "possibly", "unclear", "vague", "ambiguous",
    "might", "could be", "undetermined", "approximate", "guess",
    "hypothetically", "suppose"
}

VERIFICATION_TERMS = {
    "verify", "validate", "check safety", "audit", "confirm", "certify",
    "authenticate", "ensure", "compliance", "security check", "integrity check"
}

ANALYTICAL_TERMS = {
    "analyze", "evaluate", "investigate", "assess", "examine",
    "compare", "correlate", "synthesize", "review", "implications"
}

CAPABILITY_TERMS = {
    "execute", "tool", "capability", "device", "sensor", "actuator",
    "system call", "hardware", "invoke", "api"
}


def extract_workload_features(
    objective: str,
    context: Optional[Dict[str, Any]] = None,
    contradiction_count: int = 0,
    has_unverified_claims: bool = False,
    is_resource_constrained: bool = False,
) -> Dict[str, Any]:
    """
    Extract deterministic structural, semantic, and environmental features
    for workload classification.
    """
    cleaned = (objective or "").strip().lower()
    words = set(re.findall(r"\b[a-z0-9_\-\.]+\b", cleaned))
    word_count = len(cleaned.split())

    has_math = bool(words.intersection(MATH_LOGIC_TERMS) or any(op in cleaned for op in ["+", "*", "=", "/", "^"]))
    has_analytical = bool(words.intersection(ANALYTICAL_TERMS))
    has_multi_part = any(term in cleaned for term in MULTI_PART_TERMS) or (word_count > 40 and "," in cleaned)
    has_contradiction = bool(words.intersection(CONTRADICTION_TERMS)) or (contradiction_count > 0)
    has_ambiguity = bool(words.intersection(AMBIGUITY_TERMS))
    has_verification = bool(words.intersection(VERIFICATION_TERMS)) or has_unverified_claims
    has_capability = bool(words.intersection(CAPABILITY_TERMS))

    return {
        "word_count": word_count,
        "has_math_or_logic": has_math,
        "has_analytical": has_analytical,
        "has_multi_part": has_multi_part,
        "has_contradiction": has_contradiction,
        "contradiction_count": contradiction_count,
        "has_ambiguity": has_ambiguity,
        "has_verification": has_verification,
        "has_capability": has_capability,
        "is_resource_constrained": is_resource_constrained,
        "context_keys": list((context or {}).keys()),
    }
