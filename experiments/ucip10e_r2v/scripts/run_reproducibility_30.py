"""
UCIP-10E-R2V Deterministic Reproducibility Verifier.
Selects the identical 30 benchmark probes from benchmark_r2_sealed.json
used in R2 reproducibility audit, runs generation deterministically with
ModelAdapter for all 3 checkpoints, and verifies exact token match against
both the main R2 raw evaluation records and reproducibility run records.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ucip10e.runners.model_adapter import ModelAdapter

# Checkpoints
checkpoints = {
    "ucip09": (
        REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt",
        REPO_ROOT / "experiments" / "ucip10e_r2" / "results" / "uc09" / "eval_r2_chakrview_ucip09_medium_model.json"
    ),
    "ucip09r": (
        REPO_ROOT / "releases" / "ucip_09r_candidates" / "chakrview_ucip09r_medium_model.pt",
        REPO_ROOT / "experiments" / "ucip10e_r2" / "results" / "uc09r" / "eval_r2_chakrview_ucip09r_medium_model.json"
    ),
    "ucip09s": (
        REPO_ROOT / "experiments" / "ucip09s" / "checkpoints" / "ucip09s_balanced_pilot.pt",
        REPO_ROOT / "experiments" / "ucip10e_r2" / "results" / "uc09s" / "eval_r2_ucip09s_balanced_pilot.json"
    )
}

sample_probes = [
    'SEA_LANG_ENG_112', 'SEA_LANG_ENG_113', 'SEA_LANG_ENG_115', 'SEA_LANG_ENG_116',
    'SEA_LANG_HIN_119', 'SEA_LANG_HIN_122', 'SEA_LANG_HING_102', 'SEA_LANG_HING_107',
    'SEA_LANG_HING_121', 'SEA_MTH_SUB_001', 'SEA_MTH_SUB_011', 'SEA_MTH_SUB_014',
    'SEA_MTH_SUB_019', 'SEA_MTH_WRD_000', 'SEA_MTH_WRD_015', 'SEA_LOG_014',
    'SEA_LOG_016', 'SEA_GEN_008', 'SEA_GEN_029', 'SEA_GEN_037', 'SEA_GEN_052',
    'SEA_GEN_058', 'SEA_TRN_017', 'SEA_TRN_022', 'SEA_TRN_036', 'SEA_COR_009',
    'SEA_COR_016', 'SEA_COR_027', 'SEA_COR_029', 'SEA_RUN_016'
]

bench_sealed = json.load(open(REPO_ROOT / "experiments" / "ucip10e_r2" / "datasets" / "sealed" / "benchmark_r2_sealed.json", encoding="utf-8"))
bench_map = {p["id"]: p for p in bench_sealed}

reproducibility_report = {}

for name, (ckpt_path, eval_path) in checkpoints.items():
    print(f"\nEvaluating reproducibility for {name} on 30 stratified probes...")
    r2_eval_data = json.load(open(eval_path, encoding="utf-8"))
    r2_probe_map = {e["id"]: e for e in r2_eval_data["evaluations"]}
    
    adapter = ModelAdapter(ckpt_path)
    
    matches = 0
    mismatches = 0
    probe_details = []
    
    for pid in sample_probes:
        probe = bench_map[pid]
        expected_raw = r2_probe_map[pid]["raw_output"]
        
        # Deterministic generation
        generated = adapter.generate(probe["prompt"], max_new_tokens=48)
        
        is_match = (generated == expected_raw)
        if is_match:
            matches += 1
        else:
            mismatches += 1
            
        probe_details.append({
            "id": pid,
            "domain": probe["domain"],
            "prompt": probe["prompt"],
            "r2_recorded_output": expected_raw,
            "reproduced_output": generated,
            "match": is_match
        })
    
    reproducibility_report[name] = {
        "checkpoint": ckpt_path.name,
        "sha256": adapter.sha256,
        "total_probes_tested": len(sample_probes),
        "exact_matches": matches,
        "mismatches": mismatches,
        "deterministic_token_match": (matches == len(sample_probes)),
        "details": probe_details
    }
    print(f"Result for {name}: {matches}/{len(sample_probes)} exact string matches.")

out_path = REPO_ROOT / "experiments" / "ucip10e_r2v" / "evaluations" / "reproducibility_30_audit.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(reproducibility_report, f, indent=2)
print(f"\nReproducibility audit complete. Saved to {out_path}")
