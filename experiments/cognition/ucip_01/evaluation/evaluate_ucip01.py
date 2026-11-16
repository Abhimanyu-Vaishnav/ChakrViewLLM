import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.runtime.unified_chat import UnifiedChatSession
from scripts.run_m3_evaluations import DirectNeuralRunner

def run_evaluation():
    eval_dir = Path("experiments/cognition/ucip_01/evaluation")
    results_dir = Path("experiments/cognition/ucip_01/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(eval_dir / "sealed_benchmark_1850_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    categories = manifest["categories"]
    print(f"Total categories: {len(categories)}, Total probes: {manifest['total_probes']}")
    
    # Checkpoints to evaluate
    v011_path = Path("releases/v0.11_conversational_model/chakrview_v011_model.pt")
    v10_candidate_path = Path("experiments/cognition/ucip_01/training/ucip01_candidate_best.pt")
    
    # Sessions:
    # 1. v0.11 baseline (Neural only)
    v011_runner = DirectNeuralRunner(v011_path)
    
    # 2. v1.0 candidate (Neural only)
    v10_neural_session = UnifiedChatSession(v10_candidate_path)
    # Disable memory for neural-only test
    v10_neural_session.memory.extract_and_store_facts = lambda *args, **kwargs: []
    v10_neural_session.memory.retrieve_relevant_facts = lambda *args, **kwargs: []
    
    # 3. v1.0 candidate with Hybrid Memory
    v10_hybrid_session = UnifiedChatSession(v10_candidate_path)
    
    results = {}
    
    for cat, count in categories.items():
        bench_file = eval_dir / f"{cat}_{count}.json"
        with open(bench_file, "r", encoding="utf-8") as f:
            probes = json.load(f)
            
        print(f"\n--- Evaluating Category: {cat} ({count} probes) ---")
        
        v011_correct = 0
        v10_neural_correct = 0
        v10_hybrid_correct = 0
        
        for i, probe in enumerate(probes):
            p = probe["prompt"]
            exp = probe["expected_marker"].lower()
            
            # 1. v0.11 baseline
            res_v011 = v011_runner.generate(p, max_new_tokens=32)["text"].lower()
            if exp in res_v011:
                v011_correct += 1
                
            # 2. v1.0 Neural Only
            v10_neural_session.history.clear()
            res_v10_n = v10_neural_session.chat_turn(p).lower()
            if exp in res_v10_n:
                v10_neural_correct += 1
                
            # 3. v1.0 Hybrid Memory
            # If prompt has explicit entity declaration in history, feed turns sequentially
            v10_hybrid_session.history.clear()
            v10_hybrid_session.memory.reset_session()
            
            lines = [l.strip() for l in p.split("\n") if l.strip()]
            res_v10_h = ""
            for line in lines:
                if line.startswith("User:"):
                    u_txt = line.replace("User:", "").strip()
                    res_v10_h = v10_hybrid_session.chat_turn(u_txt).lower()
                elif line.startswith("[Memory:"):
                    # Seed memory directly if tagged
                    tag_content = line.replace("[Memory:", "").replace("]", "").strip()
                    for pair in tag_content.split(","):
                        if ":" in pair:
                            k, v = pair.split(":", 1)
                            v10_hybrid_session.memory.set_fact("user", k.strip(), v.strip())
                            
            if exp in res_v10_h:
                v10_hybrid_correct += 1
                
        results[cat] = {
            "probes": count,
            "v011_baseline": {"correct": v011_correct, "accuracy": v011_correct / count},
            "v10_neural_only": {"correct": v10_neural_correct, "accuracy": v10_neural_correct / count},
            "v10_hybrid": {"correct": v10_hybrid_correct, "accuracy": v10_hybrid_correct / count}
        }
        
        print(f"[{cat}] v0.11: {v011_correct}/{count} ({v011_correct/count*100:.1f}%) | "
              f"v1.0 Neural: {v10_neural_correct}/{count} ({v10_neural_correct/count*100:.1f}%) | "
              f"v1.0 Hybrid: {v10_hybrid_correct}/{count} ({v10_hybrid_correct/count*100:.1f}%)")
              
    out_path = results_dir / "sealed_evaluation_1850_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll evaluations saved to {out_path}")

if __name__ == "__main__":
    run_evaluation()
