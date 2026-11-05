import os
import sys
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

baseline = {
    "git": {
        "branch": "exp/cbr-01-conversational-breakthrough",
        "head_commit": "8e866b495c9693566b39c6a715701e220a028227",
        "head_commit_msg": "research(chakrview): diagnose and improve conversational model overnight"
    },
    "investigated_checkpoint": {
        "path": "experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt",
        "expected_sha256": "0cb5df5fa840d20cea737da47758d96cfe884782c9801f28e51660bb236de86b",
        "actual_sha256": sha256_file(REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt"),
        "match": None
    },
    "canonical_invariants": {
        "model_py": {
            "path": "chakrview/brain/model.py",
            "expected_sha256": "1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361",
            "actual_sha256": sha256_file(REPO_ROOT / "chakrview/brain/model.py")
        },
        "session_py": {
            "path": "chakrview/runtime/session.py",
            "expected_sha256": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67",
            "actual_sha256": sha256_file(REPO_ROOT / "chakrview/runtime/session.py")
        },
        "chat_py": {
            "path": "chakrview/chat.py",
            "actual_sha256": sha256_file(REPO_ROOT / "chakrview/chat.py")
        }
    },
    "tokenizer_artifacts": {}
}

baseline["investigated_checkpoint"]["match"] = (
    baseline["investigated_checkpoint"]["expected_sha256"] == baseline["investigated_checkpoint"]["actual_sha256"]
)

tok_dir = REPO_ROOT / "data/experiments/vocab_4096"
for fname in ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_meta.json"]:
    fpath = tok_dir / fname
    if fpath.exists():
        baseline["tokenizer_artifacts"][fname] = {
            "path": str(fpath.relative_to(REPO_ROOT)),
            "sha256": sha256_file(fpath),
            "size_bytes": fpath.stat().st_size
        }

out_p = REPO_ROOT / "experiments/cbr01_sft/conversational_forensics_01/baseline_manifest.json"
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2)

print("Baseline verification complete:")
print("  Checkpoint SHA-256 match:", baseline["investigated_checkpoint"]["match"])
print("  model.py match:", baseline["canonical_invariants"]["model_py"]["expected_sha256"] == baseline["canonical_invariants"]["model_py"]["actual_sha256"])
print("  session.py match:", baseline["canonical_invariants"]["session_py"]["expected_sha256"] == baseline["canonical_invariants"]["session_py"]["actual_sha256"])

