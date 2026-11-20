import json
from pathlib import Path

ROOT = Path(".")

# Compare build_ucip14r vs build_ucip17
def index_14r():
    prompts = set()
    for p in ROOT.glob("experiments/**/*sealed*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "prompt" in item:
                            prompts.add(item["prompt"].strip())
                elif isinstance(data, dict):
                    for k in ["probes", "benchmark_cases", "test_cases"]:
                        if k in data and isinstance(data[k], list):
                            for item in data[k]:
                                if isinstance(item, dict) and "prompt" in item:
                                    prompts.add(item["prompt"].strip())
        except Exception:
            pass
    return prompts

def index_17():
    prompts = set()
    for p in ROOT.glob("experiments/**/*sealed*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "prompt" in item:
                            prompts.add(item["prompt"].strip())
        except Exception:
            pass
    return prompts

p14 = index_14r()
p17 = index_17()
print(f"p14 len: {len(p14)}, p17 len: {len(p17)}")

# Check contamination against all 8,215 unique probes
all_sealed = set()
for f in ROOT.glob("experiments/**/*sealed*.json"):
    try:
        data = json.load(open(f, encoding="utf-8"))
        items = data if isinstance(data, list) else (data.get("probes", []) if isinstance(data, dict) else [])
        for item in items:
            if isinstance(item, dict) and "prompt" in item:
                all_sealed.add(item["prompt"].strip())
    except:
        pass

print(f"Total exhaustive sealed prompts: {len(all_sealed)}")

# Check ucip17 master
ucip17_prompts = [json.loads(l)["prompt"].strip() for l in open("data/sft/ucip17/sft_master_15000.jsonl", encoding="utf-8")]
clean_17 = [p.replace("User: ", "").replace("\nAssistant: ", "").strip() for p in ucip17_prompts]

overlap_raw = set(ucip17_prompts).intersection(all_sealed)
overlap_clean = set(clean_17).intersection(all_sealed)
print(f"Overlap with raw: {len(overlap_raw)}")
print(f"Overlap with clean: {len(overlap_clean)}")
