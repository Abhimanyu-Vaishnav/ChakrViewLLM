"""
UCIP-11 Benchmark Evaluation Harness.
Evaluates all 5 conditions on:
1. UCIP-11 Development Suite (100 probes)
2. UCIP-11 Sealed Suite (100 probes)
Uses ModelAdapter with deterministic greedy argmax generation.
Saves probe-level evaluations and summary metrics.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ucip10e.runners.model_adapter import ModelAdapter

# Checkpoints to evaluate
checkpoints_to_eval = {
    "Condition_A_Baseline": REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt",
    "Condition_B_CPT": REPO_ROOT / "experiments" / "ucip11" / "checkpoints" / "condition_b_cpt" / "condition_b_cpt_best.pt",
    "Condition_C_SFT": REPO_ROOT / "experiments" / "ucip11" / "checkpoints" / "condition_c_sft" / "condition_c_sft_best.pt",
    "Condition_D_CPT_SFT": REPO_ROOT / "experiments" / "ucip11" / "checkpoints" / "condition_d_cpt_sft" / "condition_d_cpt_sft_best.pt",
    "Condition_E_Scratch": REPO_ROOT / "experiments" / "ucip11" / "checkpoints" / "condition_e_scratch" / "condition_e_scratch_best.pt"
}

def evaluate_probe(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    eval_type = probe.get("eval_type", "semantic_rubric")
    out_lower = raw_output.lower().strip()
    
    # 1. Exact Math
    if eval_type == "exact_math":
        expected = str(probe.get("expected_answer", "")).lower()
        # Find integers
        import re
        nums = re.findall(r"\b\d+\b", raw_output)
        passed = (expected in nums)
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Expected {expected}, found {nums}"}
        
    # 2. Binary Logic
    elif eval_type == "binary_logic":
        expected = str(probe.get("expected_answer", "")).lower()
        import re
        words = re.findall(r"\b\w+\b", out_lower)
        has_yes = "yes" in words
        has_no = "no" in words
        
        if expected == "yes":
            passed = (has_yes and not has_no)
        else:
            passed = (has_no and not has_yes)
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Expected {expected}, words: {words[:6]}"}
        
    # 3. Exact Target Match / Verbatim Copy
    elif eval_type in ["exact_target_match", "verbatim_copy"]:
        expected = str(probe.get("expected_answer", "")).lower()
        passed = (expected in out_lower)
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Target '{expected}' in output: {passed}"}
        
    # 4. Semantic Rubric / Concept Match
    else:
        concepts = probe.get("expected_concepts", [])
        if not concepts:
            passed = len(out_lower) > 10
            return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": "Length check"}
            
        matched = [c for c in concepts if c.lower() in out_lower]
        # Require at least 2 distinct key concepts and non-degenerate output
        # Check for degenerate repetition
        words = out_lower.split()
        has_loop = any(len(words) >= 4 and words[i] == words[i+1] == words[i+2] for i in range(len(words)-2))
        
        passed = (len(matched) >= 2 and not has_loop and len(words) >= 5)
        score = len(matched) / len(concepts) if concepts else 0.0
        return {
            "passed": passed,
            "score": score,
            "matched_concepts": matched,
            "reason": f"Matched {len(matched)}/{len(concepts)} concepts. Loop: {has_loop}"
        }

def run_suite_eval(adapter: ModelAdapter, probes: List[Dict[str, Any]], suite_name: str) -> Dict[str, Any]:
    records = []
    total_passed = 0
    domain_stats = {}
    
    t0 = time.time()
    for idx, p in enumerate(probes):
        dom = p.get("domain", "general")
        domain_stats.setdefault(dom, {"total": 0, "passed": 0})
        domain_stats[dom]["total"] += 1
        
        raw = adapter.generate(p["prompt"], max_new_tokens=48)
        res = evaluate_probe(p, raw)
        
        passed = res.get("passed", False)
        if passed:
            total_passed += 1
            domain_stats[dom]["passed"] += 1
            
        records.append({
            "id": p["id"],
            "domain": dom,
            "subdomain": p.get("subdomain"),
            "prompt": p["prompt"],
            "raw_output": raw,
            "passed": passed,
            "score": res.get("score", 0.0),
            "reason": res.get("reason")
        })
        
    elapsed = time.time() - t0
    acc = (total_passed / len(probes) * 100.0) if probes else 0.0
    
    return {
        "suite": suite_name,
        "total_probes": len(probes),
        "total_passed": total_passed,
        "accuracy_pct": acc,
        "elapsed_seconds": round(elapsed, 2),
        "domain_stats": domain_stats,
        "records": records
    }

if __name__ == "__main__":
    dev_probes = json.load(open(REPO_ROOT / "experiments" / "ucip11" / "benchmarks" / "development" / "ucip11_dev.json", encoding="utf-8"))
    sealed_probes = json.load(open(REPO_ROOT / "experiments" / "ucip11" / "benchmarks" / "sealed" / "ucip11_sealed.json", encoding="utf-8"))
    
    full_benchmark_results = {}
    
    for cond_name, ckpt_p in checkpoints_to_eval.items():
        print(f"\n=======================================================")
        print(f"Evaluating {cond_name}: {ckpt_p.name}")
        print(f"=======================================================")
        
        adapter = ModelAdapter(ckpt_p)
        
        dev_res = run_suite_eval(adapter, dev_probes, "ucip11_dev")
        sealed_res = run_suite_eval(adapter, sealed_probes, "ucip11_sealed")
        
        print(f"[{cond_name}] Dev Score: {dev_res['total_passed']}/{dev_res['total_probes']} ({dev_res['accuracy_pct']:.1f}%)")
        print(f"[{cond_name}] Sealed Score: {sealed_res['total_passed']}/{sealed_res['total_probes']} ({sealed_res['accuracy_pct']:.1f}%)")
        
        full_benchmark_results[cond_name] = {
            "checkpoint": str(ckpt_p.relative_to(REPO_ROOT)),
            "sha256": adapter.sha256,
            "dev_results": {
                "total": dev_res["total_probes"],
                "passed": dev_res["total_passed"],
                "accuracy_pct": dev_res["accuracy_pct"],
                "domain_stats": dev_res["domain_stats"]
            },
            "sealed_results": {
                "total": sealed_res["total_probes"],
                "passed": sealed_res["total_passed"],
                "accuracy_pct": sealed_res["accuracy_pct"],
                "domain_stats": sealed_res["domain_stats"]
            },
            "dev_evaluations": dev_res["records"],
            "sealed_evaluations": sealed_res["records"]
        }
        
    out_p = REPO_ROOT / "experiments" / "ucip11" / "results" / "full_benchmark_evaluation_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(full_benchmark_results, f, indent=2)
    print(f"\nComplete benchmark evaluation saved to {out_p}")
