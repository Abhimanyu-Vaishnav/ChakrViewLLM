import json
import collections
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
sealed_path = REPO_ROOT / "experiments" / "ucip10e" / "datasets" / "sealed" / "benchmark_sealed.json"
probes = json.load(open(sealed_path, encoding="utf-8"))

cats = collections.Counter()
for p in probes:
    cats[p["domain"]] += 1

print("Total probes in sealed suite:", len(probes))
selected = []
for dom in cats:
    dom_probes = [p for p in probes if p["domain"] == dom]
    # sample 10 from each domain
    selected.extend(dom_probes[:10])

print(f"Curated balanced suite: {len(selected)} probes across {len(cats)} domains ({len(selected)//len(cats)} per domain)")
out_path = REPO_ROOT / "experiments" / "ucip10e" / "datasets" / "sealed" / "benchmark_sealed_core100.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

# Register in manifest registry
reg_path = REPO_ROOT / "experiments" / "ucip10e" / "manifests" / "manifest_registry.json"
reg = json.load(open(reg_path, encoding="utf-8"))
import hashlib
reg["benchmark_suites"]["sealed_core100"] = {
    "path": str(out_path.relative_to(REPO_ROOT)),
    "total_probes": len(selected),
    "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest()
}
with open(reg_path, "w", encoding="utf-8") as f:
    json.dump(reg, f, indent=2)
print("Registered sealed_core100 suite successfully!")
