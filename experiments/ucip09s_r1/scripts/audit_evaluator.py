import json
import re
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

manifest = json.load(open(REPO_ROOT / "experiments" / "ucip09s" / "manifests" / "owner_reproduction_suite.json", encoding="utf-8"))
baseline_eval = json.load(open(REPO_ROOT / "experiments" / "ucip09s" / "evaluations" / "baseline_reproduction_results.json", encoding="utf-8"))
pilot_eval = json.load(open(REPO_ROOT / "experiments" / "ucip09s" / "evaluations" / "pilot_reproduction_results.json", encoding="utf-8"))

models = {
    "UCIP-09": baseline_eval["UCIP-09"]["evaluations"],
    "UCIP-09R": baseline_eval["UCIP-09R"]["evaluations"],
    "UCIP-09S_Pilot": pilot_eval["UCIP-09S_Pilot"]["evaluations"]
}

# Strict evaluator functions
def strict_evaluate(probe, generated):
    cat = probe["category"]
    text = generated.strip()
    norm = text.lower()
    
    if not text:
        return False, "EMPTY_OUTPUT"
    
    # Check simple repetitive loops
    words = text.split()
    if len(words) >= 4:
        for i in range(len(words)-2):
            if words[i].lower() == words[i+1].lower() == words[i+2].lower():
                return False, "REPETITIVE_DEGENERATION"

    if cat == "English conversation":
        pid = probe["id"]
        if pid == "REPRO_ENG_01": # Greeting & identity
            # Must mention ChakrView or assistant and not be broken
            valid = ("chakrview" in norm or "assistant" in norm) and not ("contast" in norm or "tax" in norm or "dower" in norm)
            return valid, "Identity/greeting check"
        elif pid == "REPRO_ENG_02": # Creator
            # Expected creator recall: Abhimanyu / creator / designer
            valid = ("abhimanyu" in norm or "creator" in norm or "created" in norm or "designed" in norm) and not ("cpu cpu" in norm)
            return valid, "Creator check"
        elif pid == "REPRO_ENG_03": # CPU
            # Must mention instructions, execute, processor, program, compute coherently
            valid = any(k in norm for k in ["instruction", "execute", "process", "program", "cycle", "fetch"]) and len(words) >= 5 and not text.endswith("-")
            return valid, "CPU semantic check"
        elif pid == "REPRO_ENG_04": # RAM vs storage
            # Must contrast volatile/fast memory vs non-volatile/storage/disk
            has_ram = "ram" in norm or "memory" in norm or "volatile" in norm
            has_storage = "storage" in norm or "permanent" in norm or "disk" in norm
            valid = (has_ram and has_storage) and not ("curves of rest" in norm or "holatier holy" in norm)
            return valid, "RAM vs storage semantic contrast"

    elif cat == "Arithmetic":
        tgt = probe["target_number"]
        # Strict arithmetic: target number MUST be the asserted answer, not just any operand mentioned
        # Look for equation format or word problem answer format
        # Check if the final or result number is target
        # E.g. "What is 2 + 2?" -> "2 + 1 = 3" had [2, 1, 3] which did not match 4
        # "What is 5 - 2?" -> "5 - 3 = 4" had [5, 3, 4] where 3 was an operand, NOT the answer!
        # "5 apples - 2" -> "You will have 3 apple remaining (37..." had 3 as the answer!
        # "10 + 5" -> "10 + 5 = 3" had 3!
        # So we check if the sentence asserts answer == tgt
        # Direct regex for '= <tgt>' or 'have <tgt>' or '<tgt>'
        # Disallow when target is merely an operand in a bogus equation like '5 - 3 = 4' for '5 - 2'
        found_nums = [int(m) for m in re.findall(r"\b\d+\b", text)]
        
        # Check if target is explicitly present
        if tgt not in found_nums:
            return False, f"Target {tgt} not in found numbers {found_nums}"
        
        # Verify it's not a hallucinated wrong equation where tgt is an operand
        # e.g. "5 - 3 = 4" is false for "What is 5 - 2?"
        if "=" in text:
            lhs, rhs = text.split("=", 1)
            rhs_nums = [int(m) for m in re.findall(r"\b\d+\b", rhs)]
            if rhs_nums:
                return (rhs_nums[0] == tgt), f"Equation RHS is {rhs_nums[0]}, expected {tgt}"
        
        # If it says "have X remaining"
        m = re.search(r"have\s+(\d+)", norm)
        if m:
            val = int(m.group(1))
            return (val == tgt), f"Answer states {val}, expected {tgt}"
            
        # If text is solely a number
        if text.isdigit():
            return (int(text) == tgt), f"Single number {text} vs {tgt}"

        # Otherwise require that tgt is present and no conflicting equation exists
        return (tgt in found_nums), f"Found {tgt} in {found_nums}"

    elif cat == "Exact copying":
        tgt = probe["target_text"].strip().lower()
        t_clean = re.sub(r"^[^\w]+|[^\w]+$", "", tgt)
        g_clean = re.sub(r"^[^\w]+|[^\w]+$", "", norm)
        # Strict: must match target text exactly (modulo punctuation)
        passed = (t_clean == g_clean) or (norm == tgt)
        return passed, f"Exact copy target '{tgt}' vs '{text}'"

    elif cat == "Hindi":
        # Must be valid Devanagari script and contain relevant keywords
        has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in text)
        if not has_devanagari:
            return False, "No Devanagari characters"
        exp_kws = probe.get("expected_keywords", [])
        hits = [k for k in exp_kws if k in text]
        passed = (len(hits) >= 1) and (len(words) >= 4)
        return passed, f"Devanagari keywords: {hits}"

    elif cat == "Hinglish":
        exp_kws = probe.get("expected_keywords", [])
        hits = [k for k in exp_kws if k.lower() in norm]
        # Must also have Hindi/Hinglish marker words
        hing_markers = ["hai", "mein", "ko", "hota", "karte", "kare", "karna", "ka", "ki", "ke"]
        has_hing = any(re.search(rf"\b{m}\b", norm) for m in hing_markers)
        passed = (len(hits) >= 1) and has_hing and (len(words) >= 4)
        return passed, f"Hinglish markers: {has_hing}, Hits: {hits}"

    elif cat == "Instruction following":
        if "forbidden" in probe:
            for f in probe["forbidden"]:
                if re.search(rf"\b{f.lower()}\b", norm):
                    return False, f"Forbidden word used: {f}"
        pid = probe["id"]
        if pid == "REPRO_INST_01": # only word confirmed
            clean_w = [re.sub(r"[^\w]", "", w) for w in norm.split()]
            return (clean_w == ["confirmed"]), f"Only 'confirmed': found {clean_w}"
        elif pid == "REPRO_INST_02": # yes or no
            clean_w = [re.sub(r"[^\w]", "", w) for w in norm.split()]
            return ("yes" in clean_w and "no" not in clean_w), f"Binary Yes: {clean_w}"
        elif pid == "REPRO_INST_03": # winter without snow
            exp_kws = probe.get("expected_keywords", [])
            hits = [k for k in exp_kws if re.search(rf"\b{k.lower()}\b", norm)]
            return (len(hits) >= 1 and len(words) >= 3), f"Winter keywords: {hits}"

    return False, "Unknown category"

audit_results = {}
for m_name, evals in models.items():
    m_res = []
    cat_orig = {}
    cat_strict = {}
    for p, e in zip(manifest, evals):
        cat = p["category"]
        cat_orig.setdefault(cat, {"total": 0, "passed": 0})
        cat_strict.setdefault(cat, {"total": 0, "passed": 0})
        cat_orig[cat]["total"] += 1
        cat_strict[cat]["total"] += 1
        
        orig_pass = e["passed"]
        if orig_pass:
            cat_orig[cat]["passed"] += 1
            
        strict_pass, strict_reason = strict_evaluate(p, e["generated"])
        if strict_pass:
            cat_strict[cat]["passed"] += 1
            
        m_res.append({
            "id": p["id"],
            "category": cat,
            "input": p["input"],
            "generated": e["generated"],
            "original_passed": orig_pass,
            "original_reason": e.get("reason"),
            "strict_passed": strict_pass,
            "strict_reason": strict_reason,
            "discrepancy": (orig_pass != strict_pass)
        })
        
    audit_results[m_name] = {
        "summary_original": {
            "total_passed": sum(c["passed"] for c in cat_orig.values()),
            "total_probes": 20,
            "accuracy": sum(c["passed"] for c in cat_orig.values()) / 20 * 100,
            "breakdown": cat_orig
        },
        "summary_strict": {
            "total_passed": sum(c["passed"] for c in cat_strict.values()),
            "total_probes": 20,
            "accuracy": sum(c["passed"] for c in cat_strict.values()) / 20 * 100,
            "breakdown": cat_strict
        },
        "probes": m_res
    }

with open(REPO_ROOT / "experiments" / "ucip09s_r1" / "evaluations" / "evaluator_audit_detailed.json", "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)

print("=== EVALUATOR AUDIT COMPARISON ===")
for m_name, res in audit_results.items():
    o = res["summary_original"]
    s = res["summary_strict"]
    print(f"\nModel: {m_name}")
    print(f"  Original Evaluator: {o['total_passed']}/20 ({o['accuracy']:.1f}%)")
    print(f"  Strict Evaluator:   {s['total_passed']}/20 ({s['accuracy']:.1f}%)")
    print("  Category Comparison (Original vs Strict):")
    for cat in o["breakdown"]:
        op = o["breakdown"][cat]["passed"]
        sp = s["breakdown"][cat]["passed"]
        tot = o["breakdown"][cat]["total"]
        print(f"    {cat:22s}: Orig {op}/{tot} -> Strict {sp}/{tot}")
