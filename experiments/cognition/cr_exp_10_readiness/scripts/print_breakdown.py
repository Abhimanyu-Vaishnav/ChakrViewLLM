import json

with open("experiments/cognition/cr_exp_10_readiness/final_multi_suite_evaluation.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("| Category (800 OOD Suite) | Candidate 3 Baseline | CR-10 Arm A | CR-10 Arm B (Champion) | Delta (Arm B vs Base) |")
print("|---|---|---|---|---|")

cats = [k for k in d["candidate_3_baseline"]["benchmark_800"].keys() if k != "overall"]
for c in cats:
    b = d["candidate_3_baseline"]["benchmark_800"][c]
    a = d["cr10_arm_a"]["benchmark_800"][c]
    b_arm = d["cr10_arm_b"]["benchmark_800"][c]
    delta = b_arm["accuracy_pct"] - b["accuracy_pct"]
    print(f"| {c} | {b['accuracy_pct']:5.2f}% ({b['hits']}/100) | {a['accuracy_pct']:5.2f}% ({a['hits']}/100) | **{b_arm['accuracy_pct']:5.2f}% ({b_arm['hits']}/100)** | **{delta:+6.2f}%** |")

tot_b = d["candidate_3_baseline"]["benchmark_800"]["overall"]
tot_a = d["cr10_arm_a"]["benchmark_800"]["overall"]
tot_barm = d["cr10_arm_b"]["benchmark_800"]["overall"]
delta_tot = tot_barm["overall_accuracy_pct"] - tot_b["overall_accuracy_pct"]
print(f"| **OVERALL 800-PROBE SCORE** | **{tot_b['overall_accuracy_pct']:5.2f}% ({tot_b['grand_total_hits']}/800)** | **{tot_a['overall_accuracy_pct']:5.2f}% ({tot_a['grand_total_hits']}/800)** | **{tot_barm['overall_accuracy_pct']:5.2f}% ({tot_barm['grand_total_hits']}/800)** | **{delta_tot:+6.2f}%** |")
