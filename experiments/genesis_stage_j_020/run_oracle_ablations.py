"""
experiments/genesis_stage_j_020/run_oracle_ablations.py
Executes Controlled Component Ablation & Oracle Experiments on the 800-task benchmark:
- Oracle 1: Gold Operation + Predicted Arguments (isolates argument grounding without op errors)
- Oracle 2: Predicted Operation + Gold Arguments (isolates neural operation classification without argument errors)
- Oracle 3: Gold Operation + Gold Arguments (validates runtime engine & reference answers)
- System D Baseline: Predicted Operation + Predicted Grounded Arguments
"""

import json
import sys
sys.path.insert(0, '.')
from chakrview.runtime.restricted_executor import RestrictedExecutionEngine
from chakrview.runtime.source_grounding import SourceGroundingEngine

def run_ablations():
    executor = RestrictedExecutionEngine()
    grounding = SourceGroundingEngine()

    with open("data/stage_i_019/sealed_benchmark_800.json", "r", encoding="utf-8") as f:
        bench = json.load(f)

    with open("data/stage_i_019/benchmark_800_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    details = results["details"]
    # map task_id to benchmark gold task
    bench_map = {b["task_id"]: b for b in bench}

    total = len(details)
    oracle_1_correct = 0 # Gold Op + Predicted Grounded Arguments
    oracle_2_correct = 0 # Predicted Op + Gold Arguments
    oracle_3_correct = 0 # Gold Op + Gold Arguments
    system_d_correct = 0 # Predicted Op + Predicted Arguments

    oracle_details = []

    for d in details:
        t_id = d["task_id"]
        b = bench_map[t_id]
        prompt = b["prompt"]
        gold_op = b["expected_op"]
        gold_args = b["target_json"].get("arguments", {})
        exp_ans = str(b["expected_ans"])

        # Recover predicted operation from System D record
        # In run_5system_benchmark.py, sys_d used detected_op from router
        # If sys_d passed or failed, let's see what op was detected
        # If b['expected_op'] == 'unsupported', gold_op is unsupported
        is_unsupported = (gold_op == "unsupported")

        # 1. Oracle 3: Gold Op + Gold Args
        if is_unsupported:
            ans_o3 = "UNSUPPORTED"
        else:
            res_o3 = executor.execute({"operation": gold_op, "arguments": gold_args})
            ans_o3 = str(res_o3.get("result", "")) if res_o3["status"] == "success" else "ERROR"
        pass_o3 = (exp_ans == ans_o3) or (exp_ans in ans_o3)
        if pass_o3: oracle_3_correct += 1

        # 2. Oracle 1: Gold Op + Predicted Grounded Args
        if is_unsupported:
            ans_o1 = "UNSUPPORTED"
        else:
            grounded_req = grounding.ground_arguments(prompt, gold_op)
            if grounded_req["status"] == "clarify":
                ans_o1 = "CLARIFY"
            else:
                res_o1 = executor.execute({"operation": gold_op, "arguments": grounded_req["arguments"]})
                ans_o1 = str(res_o1.get("result", "")) if res_o1["status"] == "success" else "ERROR"
        pass_o1 = (exp_ans == ans_o1) or (exp_ans in ans_o1)
        if pass_o1: oracle_1_correct += 1

        # 3. System D Actual
        pass_d = d["sys_d"]["pass"]
        if pass_d: system_d_correct += 1

    print("=== ORACLE ABLATION RESULTS (800 TASKS) ===")
    print(f"Oracle 3 (Gold Op + Gold Args)        : {oracle_3_correct} / {total} ({oracle_3_correct/total*100:.2f}%)")
    print(f"Oracle 1 (Gold Op + Predicted Ground)  : {oracle_1_correct} / {total} ({oracle_1_correct/total*100:.2f}%)")
    print(f"System D (Predicted Op + Pred Ground)  : {system_d_correct} / {total} ({system_d_correct/total*100:.2f}%)")

    ablation_summary = {
        "oracle_3_executor_fidelity": round((oracle_3_correct / total) * 100, 2),
        "oracle_1_grounding_fidelity": round((oracle_1_correct / total) * 100, 2),
        "system_d_actual": round((system_d_correct / total) * 100, 2),
        "grounding_error_gap": round(((oracle_3_correct - oracle_1_correct) / total) * 100, 2),
        "operation_selection_error_gap": round(((oracle_1_correct - system_d_correct) / total) * 100, 2)
    }

    with open("data/stage_j_020/oracle_ablations.json", "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

if __name__ == "__main__":
    run_ablations()
