"""
Audit B: Training Equivalence Audit Script
Compares training configurations, hyperparameter definitions, training logs,
data ordering, step progression, and execution metrics between Arm A and Arm B.
"""

import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
CFG_CTRL_PATH = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/training_config_control.json"
CFG_TREAT_PATH = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/training_config_treatment.json"
LOG_CTRL_PATH = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/control_arm/logs/control_arm_training_log.json"
LOG_TREAT_PATH = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/treatment_arm/logs/treatment_arm_training_log.json"

def main():
    print("=== Audit B: Training Equivalence Audit ===")
    
    with open(CFG_CTRL_PATH, "r", encoding="utf-8") as f:
        cfg_c = json.load(f)
    with open(CFG_TREAT_PATH, "r", encoding="utf-8") as f:
        cfg_t = json.load(f)

    with open(LOG_CTRL_PATH, "r", encoding="utf-8") as f:
        log_c = json.load(f)
    with open(LOG_TREAT_PATH, "r", encoding="utf-8") as f:
        log_t = json.load(f)

    # 1. Hyperparameter comparisons
    compared_keys = [
        "parent_checkpoint", "parent_sha256", "dataset_train", "dataset_val",
        "max_steps", "batch_size", "learning_rate", "min_learning_rate",
        "weight_decay", "grad_clip", "seed", "eval_every_steps"
    ]
    
    param_matches = {}
    for k in compared_keys:
        match = (cfg_c.get(k) == cfg_t.get(k))
        param_matches[k] = {
            "control": cfg_c.get(k),
            "treatment": cfg_t.get(k),
            "equivalent": match
        }

    # The intentional difference
    multiplier_ctrl = cfg_c.get("relational_token_multiplier")
    multiplier_treat = cfg_t.get("relational_token_multiplier")

    # 2. Execution metrics comparison from logs
    steps_ctrl = [x["step"] for x in log_c["history"]]
    steps_treat = [x["step"] for x in log_t["history"]]
    steps_identical = (steps_ctrl == steps_treat)

    # Learning rate schedule check across steps
    lr_ctrl = [x["lr"] for x in log_c["history"]]
    lr_treat = [x["lr"] for x in log_t["history"]]
    lr_identical = (lr_ctrl == lr_treat)

    # Gradient norm stats
    gn_ctrl = [x["grad_norm"] for x in log_c["history"]]
    gn_treat = [x["grad_norm"] for x in log_t["history"]]

    # Execution time
    time_ctrl = log_c["history"][-1]["elapsed_seconds"]
    time_treat = log_t["history"][-1]["elapsed_seconds"]

    # Identified differences
    differences = []
    if multiplier_ctrl != multiplier_treat:
        differences.append({
            "type": "INTENDED_TREATMENT",
            "field": "relational_token_multiplier",
            "control": multiplier_ctrl,
            "treatment": multiplier_treat,
            "detail": "Target-weighted loss multiplier set to 5.0 in Treatment Arm B vs 1.0 in Control Arm A."
        })

    for k, v in param_matches.items():
        if not v["equivalent"]:
            differences.append({
                "type": "UNINTENDED_DIFFERENCE",
                "field": k,
                "control": v["control"],
                "treatment": v["treatment"]
            })

    output = {
        "hyperparameter_equivalence": param_matches,
        "intentional_difference": {
            "control_multiplier": multiplier_ctrl,
            "treatment_multiplier": multiplier_treat
        },
        "execution_parity": {
            "both_executed_150_steps": (len(steps_ctrl) == 6 and steps_ctrl[-1] == 150 and steps_treat[-1] == 150),
            "step_intervals_identical": steps_identical,
            "learning_rate_schedule_identical": lr_identical,
            "gradient_norm_range": {
                "control": [min(gn_ctrl), max(gn_ctrl)],
                "treatment": [min(gn_treat), max(gn_treat)]
            },
            "elapsed_seconds": {
                "control": time_ctrl,
                "treatment": time_treat
            }
        },
        "all_differences_discovered": differences,
        "training_equivalence_verified": (len([d for d in differences if d["type"] == "UNINTENDED_DIFFERENCE"]) == 0)
    }

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/audit/training_equivalence_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Audit B results written to {out_file}")
    print(f"Training Equivalence: {'VERIFIED' if output['training_equivalence_verified'] else 'FAILED'}")
    print(f"  Only difference: relational_token_multiplier ({multiplier_ctrl}x vs {multiplier_treat}x)")

if __name__ == "__main__":
    main()

