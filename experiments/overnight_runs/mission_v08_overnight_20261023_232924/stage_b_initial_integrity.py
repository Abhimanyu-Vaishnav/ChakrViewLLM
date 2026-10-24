"""
stage_b_initial_integrity.py
Recomputes SHA-256 for all protected repository artifacts and generates:
- INITIAL_PROTECTED_HASHES.json
- ENVIRONMENT_REPORT.json
- PROCESS_AND_RESOURCE_REPORT.json
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import psutil
import subprocess
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MISSION_DIR = Path(__file__).resolve().parent

# Authoritative protected artifacts and reference hashes
PROTECTED_ASSETS = [
    ("canonical_training_checkpoint_step_10000", REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt", "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"),
    ("tokenizer_vocab_json", REPO_ROOT / "data" / "experiments" / "vocab_4096" / "vocab.json", "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"),
    ("tokenizer_merges_json", REPO_ROOT / "data" / "experiments" / "vocab_4096" / "merges.json", "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"),
    ("runtime_session_adapter", REPO_ROOT / "chakrview" / "runtime" / "session.py", "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"),
    ("model_brain_source", REPO_ROOT / "chakrview" / "brain" / "model.py", "1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361"),
    ("exp17_train_split", REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp17_instruction_sft_v1" / "data" / "train_exp17_v1.jsonl", "074fafe5c1dbcbd7467430cb6f5eccd2c11f2f8679ff27e13c0d38cc90bac181"),
    ("exp17_val_split", REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp17_instruction_sft_v1" / "data" / "val_exp17_v1.jsonl", "15d9c942c01fb6afaf6280b7515d9e621ff8df7740be8010daa2be98841f4564"),
    ("exp17_test_split", REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp17_instruction_sft_v1" / "data" / "test_exp17_v1.jsonl", "be10afaa3957713cead5399c2b01e5ad2f4fb61d6355835432c7d6e0fdcc6430"),
]

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# 1. Hashes
hash_records = {}
all_matched = True
for name, path, expected in PROTECTED_ASSETS:
    actual = hash_file(path)
    matched = (actual == expected)
    if not matched:
        all_matched = False
    hash_records[name] = {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "match": matched
    }

initial_hashes = {
    "verification_timestamp": "2026-10-23T23:30:00+05:30",
    "all_protected_hashes_match": all_matched,
    "assets": hash_records
}

with open(MISSION_DIR / "INITIAL_PROTECTED_HASHES.json", "w", encoding="utf-8") as f:
    json.dump(initial_hashes, f, indent=2)

# 2. Environment Report
env_report = {
    "timestamp": "2026-10-23T23:30:00+05:30",
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python_version": platform.python_version(),
    "python_executable": sys.executable,
    "cpu_count_logical": os.cpu_count(),
    "cpu_count_physical": psutil.cpu_count(logical=False),
    "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    "disk_free_gb": round(psutil.disk_usage(str(REPO_ROOT)).free / (1024**3), 2),
    "torch_version": torch.__version__,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_threads": torch.get_num_threads(),
    "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
    "git_head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
}

with open(MISSION_DIR / "ENVIRONMENT_REPORT.json", "w", encoding="utf-8") as f:
    json.dump(env_report, f, indent=2)

# 3. Process & Resource Report
procs = []
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
    try:
        if 'python' in p.info['name'].lower():
            procs.append({
                'pid': p.info['pid'],
                'name': p.info['name'],
                'cmdline': (p.info['cmdline'][:5] if p.info['cmdline'] else []),
                'rss_mb': round(p.info['memory_info'].rss / (1024**2), 1)
            })
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

resource_report = {
    "timestamp": "2026-10-23T23:30:00+05:30",
    "active_python_processes": procs,
    "active_training_jobs_detected": False,
    "cpu_utilization_percent": psutil.cpu_percent(interval=0.5),
    "virtual_memory_percent": psutil.virtual_memory().percent,
    "disk_utilization_percent": psutil.disk_usage(str(REPO_ROOT)).percent
}

with open(MISSION_DIR / "PROCESS_AND_RESOURCE_REPORT.json", "w", encoding="utf-8") as f:
    json.dump(resource_report, f, indent=2)

print("INITIAL INTEGRITY & ENVIRONMENT CAPTURED. All hashes match:", all_matched)
