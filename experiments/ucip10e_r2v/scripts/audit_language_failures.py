import json
from collections import Counter

models = {
    "ucip09": "experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json",
    "ucip09r": "experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json",
    "ucip09s": "experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json"
}

lang_audit = {}

for name, path in models.items():
    data = json.load(open(path, encoding="utf-8"))
    lang_probes = [e for e in data["evaluations"] if e["domain"] == "language"]
    
    subdomain_counts = Counter(p["subdomain"] for p in lang_probes)
    reason_categories = Counter()
    
    # Classify failure modes
    detailed_failures = []
    for p in lang_probes:
        r = p["reason"]
        out = p["raw_output"]
        
        mode = "other"
        if "DEGENERATE_REPETITION" in r or "EMPTY_OR_DEGENERATE" in r:
            mode = "degenerative_loop"
        elif "TOO_SHORT" in r:
            mode = "too_short"
        elif "NO_HINGLISH_GRAMMAR_MARKERS" in r:
            mode = "hinglish_syntax_failure"
        elif "NO_HINDI_CONVERSATIONAL_MARKERS" in r:
            mode = "hindi_conversational_marker_missing"
        elif "Insufficient keywords" in r or "Missing expected" in r:
            mode = "missing_domain_keywords"
        
        reason_categories[mode] += 1
        detailed_failures.append({
            "id": p["id"],
            "subdomain": p.get("subdomain"),
            "prompt": p["prompt"],
            "raw_output": out,
            "reason": r,
            "failure_mode": mode
        })
    
    lang_audit[name] = {
        "total_language_probes": len(lang_probes),
        "subdomain_distribution": dict(subdomain_counts),
        "failure_modes": dict(reason_categories),
        "detailed_failures": detailed_failures
    }

with open("experiments/ucip10e_r2v/evaluations/language_failures_forensics.json", "w", encoding="utf-8") as f:
    json.dump(lang_audit, f, indent=2)

print("Language failures breakdown:")
for name, res in lang_audit.items():
    print(f"\nModel: {name}")
    print(f"Subdomains: {res['subdomain_distribution']}")
    print(f"Modes: {res['failure_modes']}")
