"""
UCIP-12R Phase H: Benchmark Contamination Verification Tool.
Audits sealed benchmarks against Stage C pretraining data and historical SFT sets.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R2_SEALED = ROOT / "experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json"
STAGE_C_MANIFEST = ROOT / "data/manifests/stage_c_manifest.json"

with open(R2_SEALED, "r", encoding="utf-8") as f:
    r2_probes = json.load(f)

with open(STAGE_C_MANIFEST, "r", encoding="utf-8") as f:
    stage_c_manifest = json.load(f)

manifest_text = json.dumps(stage_c_manifest).lower()

print(f"Scanning {len(r2_probes)} R2 sealed benchmark probes against Stage C manifest...")
leaks = []
for p in r2_probes:
    prompt = p.get("prompt", "").strip().lower()
    if len(prompt) > 20 and prompt in manifest_text:
        leaks.append((p.get("id"), prompt))

print(f"Leaks found between R2 sealed benchmark and Stage C: {len(leaks)}")
assert len(leaks) == 0, f"Leaks found: {leaks}"

# Verify presence of memory leak probe in legacy SFT
legacy_sft = ROOT / "data/sft/exp34_candidate_v3/splits/exp34_train_split_900.jsonl"
with open(legacy_sft, "r", encoding="utf-8") as f:
    sft_content = f.read().lower()

assert "what is a memory leak and how does it degrade system performance?" in sft_content
print("Confirmed legacy SFT contains probe SEA_LANG_ENG_117 (Quarantine confirmed).")
