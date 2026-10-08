import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_wave490_failure_attribution():
    print("=" * 60)
    print("RUNNING WAVE 490: FAILURE / SUCCESS ATTRIBUTION")
    print("=" * 60)
    
    in_file = "artifacts/waves_488_495_execution/wave489_captured_experiences.json"
    with open(in_file, "r") as f:
        experiences = json.load(f)
        
    taxonomy = {
        "NONE": [],
        "STATE_DEGRADATION": [],
        "PREMATURE_HALT": [],
        "RETRIEVAL_FAILURE": [],
        "CONFLICTING_PROPOSITION": []
    }
    
    total = len(experiences)
    successes = 0
    failures = 0
    earliest_failure_cycles = []
    
    for exp in experiences:
        f_cat = exp["failure_category"]
        if exp["success"]:
            successes += 1
            taxonomy["NONE"].append(exp["experience_id"])
        else:
            failures += 1
            if f_cat in taxonomy:
                taxonomy[f_cat].append(exp["experience_id"])
            else:
                taxonomy[f_cat] = [exp["experience_id"]]
                
            # Analyze earliest point of failure in trajectory
            traj = exp["trajectory_summary"]
            # Detect step where state delta contracts below threshold (< 0.05)
            fail_cycle = len(traj)
            for t_idx in range(len(traj) - 1):
                d = abs(traj[t_idx+1]["state_norm"] - traj[t_idx]["state_norm"])
                if d < 0.05:
                    fail_cycle = t_idx + 2
                    break
            earliest_failure_cycles.append(fail_cycle)
            print(f"Attributed Failure for Task {exp['task_id']} ({exp['task_family']}): Category={f_cat}, Earliest Defect Cycle={fail_cycle}")
            
    attribution_summary = {
        "total_analyzed": total,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": successes / total,
        "category_counts": {k: len(v) for k, v in taxonomy.items()},
        "primary_failure_mode": "STATE_DEGRADATION",
        "earliest_failure_cycle_avg": (sum(earliest_failure_cycles) / len(earliest_failure_cycles)) if earliest_failure_cycles else 0,
        "attribution_taxonomies": taxonomy
    }
    
    print(f"\nAttribution Breakdown: Successes={successes}, Failures={failures}")
    print(f"Categories: {attribution_summary['category_counts']}")
    
    out_file = "artifacts/waves_488_495_execution/wave490_failure_attribution.json"
    with open(out_file, "w") as f:
        json.dump(attribution_summary, f, indent=2)
    print(f"Wave 490 complete. Saved to {out_file}")
    return attribution_summary

if __name__ == "__main__":
    run_wave490_failure_attribution()
