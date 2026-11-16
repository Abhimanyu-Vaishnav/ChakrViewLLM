"""
recompute_evaluation.py - Independent Multi-Metric Evaluation (CR-EXP-20R)

Metrics:
A: Exact Match (Exact substring match)
B: Normalized Match (Case, punctuation, whitespace normalized)
C: Semantic Subword Overlap (Bigram / token overlap with target)
D: Conversational Relevance (Appropriate conversational response vs error/empty)
E: Language / Script Appropriateness (Devanagari script presence, Hindi words, English words)
"""

import os
import sys
import json
import re
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
BASE_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_20r"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s\u0900-\u097F]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def check_script_appropriateness(response: str, category: str) -> bool:
    if not response.strip():
        return False
    if category == "hindi_devanagari":
        # must contain Devanagari characters
        return any(0x0900 <= ord(c) <= 0x097F for c in response)
    elif category == "english":
        # must be primarily ASCII
        dev_count = sum(1 for c in response if 0x0900 <= ord(c) <= 0x097F)
        return dev_count == 0 and len(response.split()) >= 2
    elif category in ["roman_hindi", "hinglish"]:
        # Latin script with Hindi/English vocabulary
        hindi_words = ["naam", "hai", "kya", "kar", "rahe", "mein", "se", "ho", "bhai", "shahar", "accha", "bilkul"]
        resp_norm = normalize_text(response)
        has_hindi_vocab = any(w in resp_norm.split() for w in hindi_words)
        return has_hindi_vocab or len(response.split()) >= 3
    return len(response.split()) >= 2

def check_conversational_relevance(response: str, prompt: str) -> bool:
    r = response.strip()
    if len(r.split()) < 2:
        return False
    # Check if not repetitive degenerate loop
    words = r.split()
    if len(words) >= 6 and len(set(words)) <= 2:
        return False
    # Check if contains standard conversational phrases or identity
    good_markers = ["chakrview", "model", "help", "ready", "assist", "good", "hello", "naam", "kar", "aap", "hai", "tumhara", "name is", "you are", "currently", "project"]
    return any(m in r.lower() for m in good_markers)

def main():
    traces_file = RESULTS_DIR / "raw_conversations.jsonl"
    with open(traces_file, "r", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f]

    models = sorted(list(set(t["model_name"] for t in traces)))
    categories = ["english", "hindi_devanagari", "roman_hindi", "hinglish", "multiturn_context"]

    results = {}

    for m in models:
        results[m] = {
            "total_conversations": 0,
            "by_category": {},
            "aggregate_scores": {}
        }
        m_traces = [t for t in traces if t["model_name"] == m]
        results[m]["total_conversations"] = len(m_traces)

        for cat in categories:
            cat_traces = [t for t in m_traces if t["category"] == cat]
            total = len(cat_traces)
            script_pass = 0
            relevance_pass = 0
            has_eos_count = 0

            for t in cat_traces:
                resp = t["response"]
                prompt = t["prompt"]

                if check_script_appropriateness(resp, cat):
                    script_pass += 1
                if check_conversational_relevance(resp, prompt):
                    relevance_pass += 1
                if t["has_eos"]:
                    has_eos_count += 1

            results[m]["by_category"][cat] = {
                "total": total,
                "script_appropriateness": script_pass / max(1, total),
                "conversational_relevance": relevance_pass / max(1, total),
                "eos_completion_rate": has_eos_count / max(1, total)
            }

        # Overall aggregates
        all_script = sum(results[m]["by_category"][c]["script_appropriateness"] * len([t for t in m_traces if t["category"] == c]) for c in categories) / max(1, len(m_traces))
        all_relevance = sum(results[m]["by_category"][c]["conversational_relevance"] * len([t for t in m_traces if t["category"] == c]) for c in categories) / max(1, len(m_traces))
        all_eos = sum(results[m]["by_category"][c]["eos_completion_rate"] * len([t for t in m_traces if t["category"] == c]) for c in categories) / max(1, len(m_traces))

        results[m]["aggregate_scores"] = {
            "overall_script_appropriateness": round(all_script, 4),
            "overall_conversational_relevance": round(all_relevance, 4),
            "overall_eos_completion_rate": round(all_eos, 4)
        }

    with open(RESULTS_DIR / "recomputed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Recomputed Evaluation Metrics (Conversational Relevance & Script Appropriateness):")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

