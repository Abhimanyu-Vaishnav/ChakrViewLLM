import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_wave491_learning_signals():
    print("=" * 60)
    print("RUNNING WAVE 491: EXPERIENCE -> LEARNING SIGNAL")
    print("=" * 60)
    
    in_file = "artifacts/waves_488_495_execution/wave489_captured_experiences.json"
    with open(in_file, "r") as f:
        experiences = json.load(f)
        
    # Potential learning signal candidates to evaluate:
    # A. Supervised correction signal (uniform weighting)
    # B. Trajectory-level loss weighting (weight = 1.0 + 0.2 * depth)
    # C. Failure-category weighted learning (boost state degradation examples by 2.5x)
    # D. Experience replay (standard unweighted buffer)
    # E. Curriculum prioritization (prioritize failed longer-horizon tasks)
    # F. Confidence-weighted learning (weight = 1.0 - confidence)
    
    training_exps = [e for e in experiences if e["evaluation_split"] == "TRAINING_EXPERIENCE"]
    print(f"Total training experiences available (Strictly non-heldout): {len(training_exps)}")
    
    signal_weights = {}
    for exp in training_exps:
        exp_id = exp["experience_id"]
        depth = exp["reasoning_depth"]
        conf = exp["confidence"]
        is_fail = not exp["success"]
        
        w_supervised = 1.0
        w_trajectory = 1.0 + 0.2 * depth
        w_failure_boost = 2.5 if is_fail else 1.0
        w_confidence_inv = 1.0 - conf + 0.1
        
        # Governed Combined Signal (RIL Signal): Prioritize failure modes and longer horizons
        w_ril_combined = w_trajectory * w_failure_boost * w_confidence_inv
        
        signal_weights[exp_id] = {
            "task_id": exp["task_id"],
            "depth": depth,
            "success": exp["success"],
            "supervised_weight": round(w_supervised, 3),
            "trajectory_weight": round(w_trajectory, 3),
            "failure_weighted": round(w_failure_boost, 3),
            "confidence_weighted": round(w_confidence_inv, 3),
            "ril_combined_weight": round(w_ril_combined, 3)
        }
        print(f"Task {exp['task_id']:<10} (Depth {depth}, Succ={exp['success']}): Supervised={w_supervised:.2f} | RIL Signal Weight={w_ril_combined:.2f}")
        
    metrics = {
        "candidate_signals_evaluated": ["supervised", "trajectory", "failure_boosted", "confidence_inverse", "ril_combined"],
        "selected_signal": "ril_combined",
        "signal_derivation": "w_trajectory * w_failure_boost * (1 - confidence + 0.1)",
        "non_heldout_isolated": True,
        "sample_weights": signal_weights
    }
    
    out_file = "artifacts/waves_488_495_execution/wave491_learning_signals.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wave 491 complete. Saved to {out_file}")
    return metrics

if __name__ == "__main__":
    run_wave491_learning_signals()
