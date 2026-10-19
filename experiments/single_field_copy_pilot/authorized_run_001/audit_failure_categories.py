import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

results_path = Path("experiments/single_field_copy_pilot/authorized_run_001/PILOT_RESULTS.json")
with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

step200 = data["step_200_metrics"]
details = step200["details"]

categories = Counter()
output_counts = Counter()
expected_counts = Counter()

print(f"Total Validation Records: {len(details)}")

for d in details:
    gen = d["generated"].strip()
    exp = d["expected"].strip()
    output_counts[gen] += 1
    expected_counts[exp] += 1

print("\n--- Unique Generated Outputs (Total: 40) ---")
for out, count in output_counts.most_common():
    print(f"  '{out}': {count} occurrences ({count/40*100:.1f}%)")

print("\n--- Breakdown by Task Category ---")
# Let's inspect by family
families = {"val_srv": [], "val_met": [], "val_inv": [], "val_atc": []}
for d in details:
    prefix = d["id"][:7]
    families[prefix].append(d)

for fam, recs in families.items():
    print(f"\nFamily {fam} (Count: {len(recs)}):")
    fam_gens = Counter(r["generated"].strip() for r in recs)
    for g, c in fam_gens.items():
        print(f"  Generated: '{g}' (count: {c}) | Expected examples: {[r['expected'] for r in recs if r['generated'].strip() == g][:3]}")
