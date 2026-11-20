import json
from pathlib import Path

ROOT = Path(".")

# 1. Inspect all files matching **/*sealed*.json
files = sorted(ROOT.glob("experiments/**/*sealed*.json"))
print(f"Total files matching **/*sealed*.json: {len(files)}")
file_probe_counts = {}
all_probes_raw = []
all_probes_clean = set()

for f in files:
    try:
        data = json.load(open(f, encoding="utf-8"))
        if isinstance(data, list):
            cnt = 0
            for item in data:
                if isinstance(item, dict) and "prompt" in item:
                    p = item["prompt"]
                    all_probes_raw.append(p)
                    all_probes_clean.add(p.strip())
                    cnt += 1
            file_probe_counts[str(f.relative_to(ROOT))] = cnt
        elif isinstance(data, dict) and "probes" in data:
            cnt = 0
            for item in data["probes"]:
                if isinstance(item, dict) and "prompt" in item:
                    p = item["prompt"]
                    all_probes_raw.append(p)
                    all_probes_clean.add(p.strip())
                    cnt += 1
            file_probe_counts[str(f.relative_to(ROOT))] = cnt
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\nFile breakdown:")
for fname, cnt in file_probe_counts.items():
    print(f"  {cnt:4d} probes: {fname}")

print(f"\nTotal raw probe entries across all files: {len(all_probes_raw)}")
print(f"Total unique stripped probe strings: {len(all_probes_clean)}")
