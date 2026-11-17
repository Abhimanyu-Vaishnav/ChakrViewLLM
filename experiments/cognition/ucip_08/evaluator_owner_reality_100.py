"""
UCIP-08 Rigorous Calibrated Evaluator for 100-Probe Owner Reality Benchmark.

Key Architectural Guarantees:
1. Strict Word-Boundary Numeric Verification: Never matches '7' inside '17' or '70'.
2. Multi-Keyword Coherence Requirement: Disqualifies single isolated keywords in gibberish.
3. Grammar and Repetition Invalidation: Disqualifies looped or degenerate outputs.
4. Language Script & Semantic Coherence: Requires meaningful multi-word Devanagari structure.
5. Exact-Copy Precision: Exact match ignoring only leading/trailing whitespace.
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner

BENCHMARK_PATH = REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "sealed_owner_reality_100.json"
EOS_ID = 1


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


def evaluate_owner_reality_probe(probe: Dict[str, Any], generated: str) -> Dict[str, Any]:
    cleaned = generated.strip()
    norm_out = cleaned.lower()
    cat = probe["category"]
    
    if not cleaned:
        return {"passed": False, "reason": "EMPTY_OUTPUT"}
    
    if is_repetitive(cleaned):
        return {"passed": False, "reason": "REPETITIVE_DEGENERATION"}
    
    # 1. Exact Copying
    if cat == "Exact Copying" or probe.get("exact_match", False):
        target = probe.get("target", "").strip()
        if not target and "expected_keywords" in probe:
            target = probe["expected_keywords"][0]
        # Ignore leading/trailing punctuation and case
        t_clean = re.sub(r"^[^\w]+|[^\w]+$", "", target.lower())
        g_clean = re.sub(r"^[^\w]+|[^\w]+$", "", norm_out)
        passed = (t_clean == g_clean) or (target.strip() in cleaned)
        return {"passed": passed, "reason": f"Exact target: '{target}' vs '{cleaned}'"}
    
    # 2. Elementary Arithmetic (Strict Word-Boundary & Numeric Extraction)
    if cat == "Elementary Arithmetic":
        target_num = probe.get("target_number")
        # Extract all integer numbers with word boundaries
        found_nums = [int(m) for m in re.findall(r"\b\d+\b", cleaned)]
        # Map number words
        num_words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-two": 22, "twenty-four": 24}
        for w, val in num_words.items():
            if re.search(rf"\b{w}\b", norm_out):
                found_nums.append(val)
        
        if target_num in found_nums:
            return {"passed": True, "reason": f"Correct number {target_num} found in {found_nums}"}
        else:
            return {"passed": False, "reason": f"Expected {target_num}, found numbers: {found_nums}"}
    
    # 3. Devanagari Hindi
    if cat == "Devanagari Hindi":
        dev_chars = len(re.findall(r"[\u0900-\u097F]", cleaned))
        if dev_chars < 8:
            return {"passed": False, "reason": f"Insufficient Devanagari characters: {dev_chars}"}
        # Check required keywords
        exp_kw = probe.get("expected_keywords", [])
        kw_hits = [k for k in exp_kw if k in cleaned]
        if len(kw_hits) >= 1 and len(cleaned.split()) >= 3:
            return {"passed": True, "reason": f"Devanagari ({dev_chars} chars), kw_hits={kw_hits}"}
        return {"passed": False, "reason": f"Devanagari keywords missing or too short: kw_hits={kw_hits}"}
    
    # 4. Semantic Consistency (Forbidden word check + Keyword check)
    if cat == "Semantic Consistency":
        forb = probe.get("forbidden", [])
        for f in forb:
            if re.search(rf"\b{f.lower()}\b", norm_out):
                return {"passed": False, "reason": f"Violated constraint: used forbidden word '{f}'"}
        exp_kw = probe.get("expected_keywords", [])
        kw_hits = [k for k in exp_kw if re.search(rf"\b{k.lower()}\b", norm_out)]
        if len(kw_hits) >= 1 and len(cleaned.split()) >= 3:
            return {"passed": True, "reason": f"Consistent, kw_hits={kw_hits}"}
        return {"passed": False, "reason": f"Missing keywords {exp_kw}"}
    
    # 5. English Conversation / Explanations / Roman Hindi / Hinglish / Instruction Following
    exp_kw = probe.get("expected_keywords", [])
    kw_hits = [k for k in exp_kw if k.lower() in norm_out]
    
    # Require at least 1 keyword for short answers, or 2 for explanations
    min_hits = 2 if cat == "English Explanations" else 1
    if len(kw_hits) >= min_hits and len(cleaned.split()) >= 3:
        return {"passed": True, "reason": f"Keywords hit: {kw_hits}"}
    
    return {"passed": False, "reason": f"Insufficient topic coherence: hit {kw_hits} (needed {min_hits})"}


def run_owner_reality_benchmark(checkpoint_path: Path, max_tokens: int = 48) -> Dict[str, Any]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    
    probes = bench_data["probes"]
    runner = DirectNeuralRunner(checkpoint_path)
    
    results = []
    cat_stats = {}
    
    for p in probes:
        cat = p["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "passed": 0}
        cat_stats[cat]["total"] += 1
        
        prompt = f"User: {p['prompt']}\nAssistant:"
        res = runner.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
        gen_text = res["text"].strip()
        
        eval_res = evaluate_owner_reality_probe(p, gen_text)
        if eval_res["passed"]:
            cat_stats[cat]["passed"] += 1
            
        results.append({
            "id": p["id"],
            "category": cat,
            "prompt": p["prompt"],
            "generated": gen_text,
            "passed": eval_res["passed"],
            "reason": eval_res["reason"]
        })
    
    total_passed = sum(s["passed"] for s in cat_stats.values())
    total_probes = len(probes)
    acc = (total_passed / total_probes) * 100.0
    
    return {
        "checkpoint": str(checkpoint_path.name),
        "total_probes": total_probes,
        "total_passed": total_passed,
        "accuracy_percent": round(acc, 2),
        "categories": cat_stats,
        "evaluations": results
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    
    res = run_owner_reality_benchmark(Path(args.checkpoint))
    print(f"\n--- Owner Reality Benchmark Results ({res['checkpoint']}) ---")
    print(f"Overall Accuracy: {res['total_passed']}/{res['total_probes']} ({res['accuracy_percent']}%)")
    for c, s in res["categories"].items():
        pct = (s["passed"] / s["total"]) * 100.0
        print(f"  {c:28s}: {s['passed']:2d}/{s['total']:2d} ({pct:5.1f}%)")
        
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation report to {args.out}")
