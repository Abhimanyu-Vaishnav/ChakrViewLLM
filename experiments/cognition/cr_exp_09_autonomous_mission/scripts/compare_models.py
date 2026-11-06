import json

with open('experiments/cognition/cr_exp_09_autonomous_mission/baseline_benchmark_arm_b.json') as f:
    b0 = json.load(f)['summary']
with open('experiments/cognition/cr_exp_09_autonomous_mission/benchmark_candidate_1.json') as f:
    c1 = json.load(f)['summary']
with open('experiments/cognition/cr_exp_09_autonomous_mission/candidate_2_benchmark.json') as f:
    c2 = json.load(f)['summary']
with open('experiments/cognition/cr_exp_09_autonomous_mission/candidate_3_benchmark.json') as f:
    c3 = json.load(f)['summary']

print("| Capability Dimension | Baseline Arm B | Candidate 1 | Candidate 2 | Candidate 3 | Net Progression (C3 vs Base) |")
print("|---|---|---|---|---|---|")
cats = [k for k in b0.keys() if k != "overall"]
for k in cats:
    p0 = b0[k]['accuracy_pct']
    p1 = c1[k]['accuracy_pct']
    p2 = c2[k]['accuracy_pct']
    p3 = c3[k]['accuracy_pct']
    h0 = b0[k]['hits']
    h1 = c1[k]['hits']
    h2 = c2[k]['hits']
    h3 = c3[k]['hits']
    diff = p3 - p0
    print(f"| {k} | {p0:5.2f}% ({h0}/75) | {p1:5.2f}% ({h1}/75) | {p2:5.2f}% ({h2}/75) | **{p3:5.2f}% ({h3}/75)** | **{diff:+6.2f}%** |")

tot0 = b0['overall']['overall_accuracy_pct']
tot1 = c1['overall']['overall_accuracy_pct']
tot2 = c2['overall']['overall_accuracy_pct']
tot3 = c3['overall']['overall_accuracy_pct']
th0 = b0['overall']['grand_total_hits']
th1 = c1['overall']['grand_total_hits']
th2 = c2['overall']['grand_total_hits']
th3 = c3['overall']['grand_total_hits']
diff_tot = tot3 - tot0
print(f"| **OVERALL ACCURACY** | **{tot0:5.2f}% ({th0}/600)** | **{tot1:5.2f}% ({th1}/600)** | **{tot2:5.2f}% ({th2}/600)** | **{tot3:5.2f}% ({th3}/600)** | **{diff_tot:+6.2f}%** |")
