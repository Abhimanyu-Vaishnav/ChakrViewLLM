"""
Stage G: Isolate Canonical Checkpoint Copy and Prepare In-Memory Splits for SFT.
Creates:
  - Isolated checkpoint copy: experiments/overnight_runs/mission_v08_overnight_20261023_232924/checkpoints/canonical_copy_step_10000.pt
  - Temporary isolated training and validation JSONL files in the mission directory.
  - Verifies hashes before and after.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path("d:/Project/ChakrView").resolve()
MISSION_DIR = REPO_ROOT / "experiments" / "overnight_runs" / "mission_v08_overnight_20261023_232924"
CHECKPOINTS_DIR = MISSION_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def prepare_stage_g():
    print("Preparing Stage G isolated environment...")
    
    # 1. Copy canonical checkpoint
    src_ckpt = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
    src_hash = compute_sha256(src_ckpt)
    expected_hash = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert src_hash == expected_hash, f"Source canonical checkpoint hash mismatch: {src_hash}"
    
    dst_ckpt = CHECKPOINTS_DIR / "canonical_copy_step_10000.pt"
    shutil.copy2(src_ckpt, dst_ckpt)
    dst_hash = compute_sha256(dst_ckpt)
    assert dst_hash == expected_hash, f"Copied checkpoint hash mismatch: {dst_hash}"
    print(f"Isolated canonical checkpoint copy created: {dst_ckpt} (SHA256: {dst_hash})")
    
    # 2. Generate in-memory splits and write ONLY to mission directory
    data_dir = MISSION_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    gen = Exp18DatasetGenerator()
    splits = gen.generate_all_splits()
    
    train_file = data_dir / "exp18_train_isolated.jsonl"
    val_file = data_dir / "exp18_val_isolated.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for r in splits["train"]:
            f.write(json.dumps(r) + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for r in splits["val"]:
            f.write(json.dumps(r) + "\n")
            
    print(f"Isolated train split written: {train_file} ({len(splits['train'])} records)")
    print(f"Isolated val split written: {val_file} ({len(splits['val'])} records)")
    print(f"Test split strictly held out (NOT written to training inputs).")

if __name__ == "__main__":
    prepare_stage_g()
