import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.cognition.metacognitive_evaluator import MetaCognitiveEvaluator, ReasoningStrategy

def run_wave496_evaluation():
    print("=" * 60)
    print("RUNNING WAVE 496: META-COGNITIVE SELF-EVALUATION")
    print("=" * 60)
    
    os.makedirs("artifacts/waves_496_503_execution", exist_ok=True)
    evaluator = MetaCognitiveEvaluator()
    
    # Test Scenarios for Calibration:
    # 1. Easy 1-hop (likely-correct)
    # 2. Medium 2-hop (likely-correct)
    # 3. Hard 4-hop with state attenuation (uncertain/likely-wrong)
    # 4. Unknown proposition (uncertain/missing)
    # 5. Contradictory proposition (high failure risk)
    scenarios = [
        {"name": "easy_1hop", "cycle": 1, "norm": 0.31, "delta": 0.16, "status": "KNOWN", "margin": 0.85, "contra": False, "ground_truth_correct": True},
        {"name": "medium_2hop", "cycle": 2, "norm": 0.47, "delta": 0.15, "status": "KNOWN", "margin": 0.75, "contra": False, "ground_truth_correct": True},
        {"name": "hard_4hop_attenuated", "cycle": 4, "norm": 0.49, "delta": 0.03, "status": "KNOWN", "margin": 0.40, "contra": False, "ground_truth_correct": False},
        {"name": "unknown_prop", "cycle": 1, "norm": 0.25, "delta": 0.08, "status": "UNKNOWN", "margin": 0.15, "contra": False, "ground_truth_correct": False},
        {"name": "contradictory_prop", "cycle": 2, "norm": 0.42, "delta": 0.10, "status": "CONFLICTING", "margin": 0.20, "contra": True, "ground_truth_correct": False},
    ]
    
    results = {}
    correct_confs = []
    incorrect_confs = []
    
    for sc in scenarios:
        s_tensor = torch.tensor([sc["norm"]])
        ass = evaluator.evaluate(
            cycle_count=sc["cycle"],
            state_tensor=s_tensor,
            step_delta=sc["delta"],
            epistemic_status=sc["status"],
            margin=sc["margin"],
            has_contradiction=sc["contra"]
        )
        results[sc["name"]] = ass.to_dict()
        results[sc["name"]]["ground_truth_correct"] = sc["ground_truth_correct"]
        
        if sc["ground_truth_correct"]:
            correct_confs.append(ass.confidence)
        else:
            incorrect_confs.append(ass.confidence)
            
        print(f"[{sc['name']:<22}] Conf={ass.confidence:.2f}, Unc={ass.uncertainty:.2f}, Risk={ass.failure_risk:.2f} -> Rec={ass.recommended_strategy.value}")
        
    avg_correct_conf = sum(correct_confs) / len(correct_confs)
    avg_incorrect_conf = sum(incorrect_confs) / len(incorrect_confs)
    separation_margin = avg_correct_conf - avg_incorrect_conf
    
    print(f"\nCalibration: Avg Conf on Correct={avg_correct_conf:.2f} | Avg Conf on Incorrect={avg_incorrect_conf:.2f}")
    print(f"Discrimination Separation Margin: {separation_margin:.2f}")
    
    metrics = {
        "avg_confidence_correct": avg_correct_conf,
        "avg_confidence_incorrect": avg_incorrect_conf,
        "discrimination_separation": separation_margin,
        "discrimination_verified": separation_margin > 0.40,
        "evaluations": results
    }
    
    out_file = "artifacts/waves_496_503_execution/wave496_metacognitive_eval.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wave 496 complete. Saved to {out_file}")
    return metrics

if __name__ == "__main__":
    run_wave496_evaluation()
