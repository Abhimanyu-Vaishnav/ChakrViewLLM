import sys
import json
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.chat import ConversationalChatCLI

def evaluate_benchmarks():
    bench_file = REPO_ROOT / "experiments/cognition/ucip_02/benchmark/core_acceptance_benchmark_300.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        probes = json.load(f)
        
    v011_path = REPO_ROOT / "releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt"
    
    # 1. Neural Only Mode (--memory off)
    cli_neural = ConversationalChatCLI(v011_path, memory_enabled=False)
    
    # 2. Hybrid Mode (--memory on)
    cli_hybrid = ConversationalChatCLI(v011_path, memory_enabled=True)
    
    results = {}
    
    for p in probes:
        cat = p["category"]
        if cat not in results:
            results[cat] = {
                "total": 0,
                "neural_correct": 0,
                "hybrid_correct": 0
            }
            
        results[cat]["total"] += 1
        exp = p["expected"].lower()
        
        # Test Neural Mode
        cli_neural.reset()
        if "setup" in p:
            cli_neural.interact_step(p["setup"])
        resp_neural = cli_neural.interact_step(p["prompt"]).lower()
        if exp in resp_neural:
            results[cat]["neural_correct"] += 1
            
        # Test Hybrid Mode
        cli_hybrid.reset()
        if "setup" in p:
            cli_hybrid.interact_step(p["setup"])
        resp_hybrid = cli_hybrid.interact_step(p["prompt"]).lower()
        if exp in resp_hybrid:
            results[cat]["hybrid_correct"] += 1
            
    print(f"{'Category':30s} | {'Total':5s} | {'Neural Only':12s} | {'Hybrid Mode':12s}")
    print("-" * 70)
    for cat, data in results.items():
        tot = data["total"]
        n_c = data["neural_correct"]
        h_c = data["hybrid_correct"]
        print(f"{cat:30s} | {tot:5d} | {n_c:4d} ({n_c/tot*100:5.1f}%) | {h_c:4d} ({h_c/tot*100:5.1f}%)")
        
    out_file = REPO_ROOT / "experiments/cognition/ucip_02/results/benchmark_300_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {out_file}")

if __name__ == "__main__":
    evaluate_benchmarks()
