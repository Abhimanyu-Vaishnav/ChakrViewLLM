"""
Verify hashes and generate CR001_REPRODUCIBILITY_MANIFEST.json.
"""

import sys
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs" / "cognition"
MANIFEST_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_001"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

PROTECTED_FILES = {
    "chakrview/brain/model.py": "chakrview/brain/model.py",
    "chakrview/runtime/session.py": "chakrview/runtime/session.py",
    "chakrview/brain/config.py": "chakrview/brain/config.py",
    "data/experiments/vocab_4096/vocab.json": "data/experiments/vocab_4096/vocab.json",
    "artifacts/step102_stage_b_curriculum/checkpoints/checkpoint_0000500.pt": "artifacts/step102_stage_b_curriculum/checkpoints/checkpoint_0000500.pt"
}

DOCS_CREATED = [
    "CHAKRVIEW_COGNITIVE_ARCHITECTURE.md",
    "COGNITIVE_CAPABILITY_TAXONOMY.md",
    "NEURAL_REASONING_DESIGN.md",
    "REASONING_ACQUISITION_CURRICULUM.md",
    "COGNITIVE_EVALUATION_FRAMEWORK.md",
    "EXTERNAL_VS_INTERNAL_COGNITION.md",
    "COGNITIVE_SCALING_ROADMAP.md",
    "CR001_ARCHITECTURE_DECISIONS.md"
]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    protected_hashes = {}
    for k, rel in PROTECTED_FILES.items():
        p = REPO_ROOT / rel
        if p.exists():
            protected_hashes[k] = {
                "exists": True,
                "sha256": sha256_file(p)
            }
        else:
            protected_hashes[k] = {"exists": False}
            
    doc_hashes = {}
    for doc in DOCS_CREATED:
        p = DOCS_DIR / doc
        if p.exists():
            doc_hashes[doc] = {
                "exists": True,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p)
            }
        else:
            doc_hashes[doc] = {"exists": False}
            
    manifest = {
        "mission_id": "CR-001",
        "timestamp_utc": "2026-11-05T05:30:00Z",
        "repository_head": "8e866b495c9693566b39c6a715701e220a028227",
        "branch": "exp/cbr-01-conversational-breakthrough",
        "status": "SPECIFICATION_COMPLETE_STOP_GATE_HONORED",
        "protected_baseline": {
            "model": "ChakrMicro v0.1",
            "parameters": 3443136,
            "layers": 6,
            "heads": 6,
            "d_model": 192,
            "vocab_size": 4096,
            "context_length": 512,
            "weight_tying": True,
            "protected_file_hashes": protected_hashes
        },
        "documentation_deliverables": doc_hashes,
        "classification_of_proposals": {
            "supported_by_existing_evidence": [
                "Curriculum data engineering on variable binding (CBR-SFT-08 demonstrated removing 'context' sink)",
                "CPU throughput feasibility (>1,200 tok/s on 3.44M)",
                "Strict external vs internal evaluation separation"
            ],
            "plausible_but_unverified": [
                "Induction circuit formation in 6 layers without external pretraining",
                "Auxiliary consistency head for contradiction detection without negative transfer",
                "ChakrSmall (9.15M) scaling providing dedicated induction layers"
            ],
            "requires_experiment": [
                "CR-EXP-01: Synthetic key-value binding curriculum on 3.44M baseline",
                "CR-EXP-02: Decomposed scratchpad arithmetic training on random operands",
                "CR-EXP-03: Two-pass self-correction fine-tuning"
            ],
            "not_currently_justified": [
                "Premature scaling to 43M parameters before expanding training tokens beyond 7M",
                "Adding complex recurrent or external memory networks before measuring standard transformer curriculum limits"
            ]
        },
        "smallest_sensible_next_experiment": {
            "experiment_id": "CR-EXP-01",
            "name": "Synthetic Variable Binding and Induction Circuit Formation",
            "target": "Train 1,000 synthetic key-value binding examples on ChakrMicro v0.1 to evaluate whether induction heads form in layers 3-5",
            "compute_cost": "< 15 minutes on CPU",
            "ram": "< 100 MB"
        }
    }
    
    out_file = MANIFEST_DIR / "CR001_REPRODUCIBILITY_MANIFEST.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"CR001_REPRODUCIBILITY_MANIFEST.json generated at {out_file}")

if __name__ == "__main__":
    main()
