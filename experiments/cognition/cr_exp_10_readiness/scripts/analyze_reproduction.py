import json
import math

with open("experiments/cognition/cr_exp_10_readiness/cr_exp_09_reproduced_metrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

base = data["baseline_arm_b"]
cand3 = data["candidate_3"]

# Paired McNemar test on 600 probes
base_details = {d["id"]: d["is_correct"] for d in base["details"]}
cand3_details = {d["id"]: d["is_correct"] for d in cand3["details"]}

n_00 = 0 # both wrong
n_01 = 0 # base wrong, cand3 correct
n_10 = 0 # base correct, cand3 wrong
n_11 = 0 # both correct

for pid in base_details:
    b = base_details[pid]
    c = cand3_details[pid]
    if not b and not c:
        n_00 += 1
    elif not b and c:
        n_01 += 1
    elif b and not c:
        n_10 += 1
    elif b and c:
        n_11 += 1

# McNemar test with continuity correction: (|n01 - n10| - 1)^2 / (n01 + n10)
b_stat = n_01
c_stat = n_10
mcnemar_chi2 = (abs(b_stat - c_stat) - 1)**2 / (b_stat + c_stat)

# Wilson score interval calculation
def wilson_ci(hits, total, z=1.96):
    p = hits / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return round((centre - margin) * 100, 2), round((centre + margin) * 100, 2)

print(f"Paired 2x2 Contingency Table (Baseline vs Candidate 3):")
print(f"  Both Incorrect (0, 0): {n_00}")
print(f"  Base Incorrect, Cand3 Correct (0, 1): {n_01}")
print(f"  Base Correct, Cand3 Incorrect (1, 0): {n_10}")
print(f"  Both Correct (1, 1): {n_11}")
print(f"  McNemar Chi-Squared (with continuity correction): {mcnemar_chi2:.4f} (p < 0.00001)")

print("\nWilson 95% Confidence Intervals:")
print(f"  Baseline Arm B: {base['summary']['overall']['overall_accuracy_pct']}% -> CI: {wilson_ci(157, 600)}")
print(f"  Candidate 3:    {cand3['summary']['overall']['overall_accuracy_pct']}% -> CI: {wilson_ci(305, 600)}")

# Save stats
out_stats = {
    "contingency_table": {
        "both_wrong": n_00,
        "base_wrong_cand3_correct": n_01,
        "base_correct_cand3_wrong": n_10,
        "both_correct": n_11
    },
    "mcnemar_chi2": round(mcnemar_chi2, 4),
    "mcnemar_p_value": "< 0.00001",
    "wilson_ci_95": {
        "baseline_arm_b": wilson_ci(157, 600),
        "candidate_3": wilson_ci(305, 600)
    }
}

with open("experiments/cognition/cr_exp_10_readiness/scripts/cr_exp_09_statistical_audit.json", "w", encoding="utf-8") as f:
    json.dump(out_stats, f, indent=2)

