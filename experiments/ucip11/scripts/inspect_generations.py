import json

p = 'experiments/ucip11/results/full_benchmark_evaluation_results.json'
d = json.load(open(p))

samples_to_show = ["SEA_ENG_001", "SEA_ENG_006", "SEA_HIN_001", "SEA_HNG_001", "SEA_MTH_001", "SEA_LOG_001"]

for pid in samples_to_show:
    print(f"\n==================== PROBE: {pid} ====================")
    for cond in ["Condition_A_Baseline", "Condition_B_CPT", "Condition_C_SFT", "Condition_D_CPT_SFT", "Condition_E_Scratch"]:
        evals = d[cond]["sealed_evaluations"]
        match = [e for e in evals if e["id"] == pid]
        if match:
            e = match[0]
            if cond == "Condition_A_Baseline":
                print(f"Prompt: {e['prompt']}")
            print(f"  [{cond}] -> {e['raw_output']}")
