"""
Stage J: Final Protected Artifact Re-verification and Manifest Generation.
Recomputes SHA-256 for all 8 protected artifacts, verifies matches,
and outputs:
  - FINAL_PROTECTED_HASHES.json
  - INTEGRITY_MANIFEST.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("d:/Project/ChakrView").resolve()
MISSION_DIR = REPO_ROOT / "experiments" / "overnight_runs" / "mission_v08_overnight_20261023_232924"

PROTECTED_FILES = {
    "canonical_checkpoint": (
        REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
        "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
    ),
    "tokenizer_vocab": (
        REPO_ROOT / "data" / "experiments" / "vocab_4096" / "vocab.json",
        "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
    ),
    "tokenizer_merges": (
        REPO_ROOT / "data" / "experiments" / "vocab_4096" / "merges.json",
        "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
    ),
    "runtime_session": (
        REPO_ROOT / "chakrview" / "runtime" / "session.py",
        "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67",
    ),
    "model_source": (
        REPO_ROOT / "chakrview" / "brain" / "model.py",
        "1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361",
    ),
    "exp17_train": (
        REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp17_instruction_sft_v1" / "data" / "train_exp17_v1.jsonl",
        "074fafe5c1dbcbd7467430cb6f5eccd2c11f2f8679ff27e13c0d38cc90bac181",
    ),
    "exp17_val": (
        REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp17_instruction_sft_v1" / "data" / "val_exp17_v1.jsonl",
        "15d9c942c01fb6afaf6280b7515d9e621ff8df7740be8010daa2be98841f4564",
    ),
    "exp17_test": (
        REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp17_instruction_sft_v1" / "data" / "test_exp17_v1.jsonl",
        "be10afaa3957713cead5399c2b01e5ad2f4fb61d6355835432c7d6e0fdcc6430",
    ),
}

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_final():
    results = {}
    all_match = True
    for key, (path, expected_hash) in PROTECTED_FILES.items():
        if not path.is_file():
            results[key] = {
                "path": str(path),
                "expected": expected_hash,
                "actual": "FILE_NOT_FOUND",
                "match": False,
            }
            all_match = False
            continue
        actual_hash = compute_sha256(path)
        match = (actual_hash == expected_hash)
        if not match:
            all_match = False
        results[key] = {
            "path": str(path),
            "expected": expected_hash,
            "actual": actual_hash,
            "match": match,
        }

    with open(MISSION_DIR / "FINAL_PROTECTED_HASHES.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Git status
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=str(REPO_ROOT), text=True)
    with open(MISSION_DIR / "FINAL_GIT_STATUS.txt", "w", encoding="utf-8") as f:
        f.write(git_status)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "mission_id": "mission_v08_overnight_20261023_232924",
        "all_protected_hashes_match": all_match,
        "protected_assets_count": len(results),
        "git_status_lines": len(git_status.strip().splitlines()),
        "results": results,
    }
    with open(MISSION_DIR / "INTEGRITY_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Final Protected Hashes Verified. All match: {all_match}")

if __name__ == "__main__":
    verify_final()
