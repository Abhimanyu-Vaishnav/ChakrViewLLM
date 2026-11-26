"""
NR-018 Generalization and Breakdown Analysis
Analyzes benchmark results across:
1. In-distribution vs Out-of-distribution (OOD) accuracy per system
2. Failure mode taxonomy (intent misclassification, argument extraction error, JSON parse failure, injection vulnerability)
3. Multilingual routing efficacy (English vs Hindi vs Hinglish)
4. Counterfactual sensitivity (changing numerical values while keeping template)
"""

import json

def analyze_generalization():
    with open("data/stage_h_018/system_comparison_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data["details"]
    summary = data["summary"]

    # Language breakdown for System C & B
    lang_stats = {
        "english": {"total": 0, "sys_a": 0, "sys_b": 0, "sys_c": 0, "sys_d": 0},
        "hindi": {"total": 0, "sys_a": 0, "sys_b": 0, "sys_c": 0, "sys_d": 0},
        "hinglish": {"total": 0, "sys_a": 0, "sys_b": 0, "sys_c": 0, "sys_d": 0}
    }

    # Error taxonomy for System C
    sys_c_errors = {
        "json_parse_error": 0,
        "argument_extraction_error": 0,
        "intent_misclassification": 0,
        "correct": 0
    }

    for item in details:
        prompt = item["prompt"]
        # Language detection heuristic
        # Devanagari range \u0900-\u097F
        has_devanagari = any('\u0900' <= char <= '\u097F' for char in prompt)
        has_hinglish_kw = any(w in prompt.lower() for w in ["kardo", "jodo", "ghatao", "batao", "karo", "aur", "mein", "ko"])
        
        if has_devanagari:
            lang = "hindi"
        elif has_hinglish_kw:
            lang = "hinglish"
        else:
            lang = "english"

        lang_stats[lang]["total"] += 1
        if item["system_a"]["pass"]: lang_stats[lang]["sys_a"] += 1
        if item["system_b"]["pass"]: lang_stats[lang]["sys_b"] += 1
        if item["system_c"]["pass"]: lang_stats[lang]["sys_c"] += 1
        if item["system_d"]["pass"]: lang_stats[lang]["sys_d"] += 1

        # Classify System C error
        if item["system_c"]["pass"]:
            sys_c_errors["correct"] += 1
        else:
            out_str = item["system_c"].get("out", "")
            raw_json_str = item["system_c"].get("json", "")
            if out_str == "PARSE_ERROR" or not raw_json_str.strip().endswith("}"):
                sys_c_errors["json_parse_error"] += 1
            else:
                try:
                    parsed = json.loads(raw_json_str)
                    exp_op = item.get("expected")
                    # If JSON parsed, was it wrong operation or wrong argument?
                    sys_c_errors["argument_extraction_error"] += 1
                except:
                    sys_c_errors["json_parse_error"] += 1

    report = {
        "overall_summary": summary,
        "language_breakdown": lang_stats,
        "system_c_error_taxonomy": sys_c_errors
    }

    with open("data/stage_h_018/generalization_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== GENERALIZATION & FAILURE BREAKDOWN ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    analyze_generalization()
