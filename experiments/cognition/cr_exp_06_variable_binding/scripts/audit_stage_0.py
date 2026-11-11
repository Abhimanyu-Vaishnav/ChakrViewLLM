"""
Stage 0: Safety, Lineage & Repository Audit Script for CR-EXP-06
Verifies parent, candidate, canonical baseline, tokenizer, and runtime hashes.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.release_manifest_verifier import EXPECTED_TOKENIZER_HASHES

EXPECTED = {
    "candidate_path": "experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt",
    "candidate_sha256": "ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd",
    "parent_path": "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
    "parent_sha256": "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153",
    "vocab_path": "data/experiments/vocab_4096/vocab.json",
    "vocab_sha256": EXPECTED_TOKENIZER_HASHES["vocab.json"],
    "merges_path": "data/experiments/vocab_4096/merges.json",
    "merges_sha256": EXPECTED_TOKENIZER_HASHES["merges.json"],
    "config_path": "data/experiments/vocab_4096/config.json",
    "config_sha256": EXPECTED_TOKENIZER_HASHES["config.json"],
    "runtime_session_path": "chakrview/runtime/session.py",
    "runtime_session_sha256": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67",
}

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== CR-EXP-06 Stage 0: Safety & Lineage Audit ===")
    results = {}
    
    cand_p = REPO_ROOT / EXPECTED["candidate_path"]
    cand_hash = sha256_file(cand_p)
    results["candidate_checkpoint"] = {
        "path": str(cand_p),
        "sha256": cand_hash,
        "expected_sha256": EXPECTED["candidate_sha256"],
        "match": cand_hash == EXPECTED["candidate_sha256"]
    }

    parent_p = REPO_ROOT / EXPECTED["parent_path"]
    parent_hash = sha256_file(parent_p)
    results["parent_checkpoint"] = {
        "path": str(parent_p),
        "sha256": parent_hash,
        "expected_sha256": EXPECTED["parent_sha256"],
        "match": parent_hash == EXPECTED["parent_sha256"]
    }

    v_p = REPO_ROOT / EXPECTED["vocab_path"]
    results["tokenizer_vocab"] = {
        "path": str(v_p),
        "sha256": sha256_file(v_p),
        "expected_sha256": EXPECTED["vocab_sha256"],
        "match": sha256_file(v_p) == EXPECTED["vocab_sha256"]
    }

    m_p = REPO_ROOT / EXPECTED["merges_path"]
    results["tokenizer_merges"] = {
        "path": str(m_p),
        "sha256": sha256_file(m_p),
        "expected_sha256": EXPECTED["merges_sha256"],
        "match": sha256_file(m_p) == EXPECTED["merges_sha256"]
    }

    c_p = REPO_ROOT / EXPECTED["config_path"]
    results["tokenizer_config"] = {
        "path": str(c_p),
        "sha256": sha256_file(c_p),
        "expected_sha256": EXPECTED["config_sha256"],
        "match": sha256_file(c_p) == EXPECTED["config_sha256"]
    }

    torch.manual_seed(42)
    canonical_model = ChakrMicro(ModelConfig())
    canonical_model.eval()
    canonical_model_hash = compute_model_hash(canonical_model)
    results["canonical_baseline"] = {
        "sha256": canonical_model_hash,
        "expected_sha256": EXPECTED_WEIGHT_HASH,
        "match": canonical_model_hash == EXPECTED_WEIGHT_HASH
    }

    sess_p = REPO_ROOT / EXPECTED["runtime_session_path"]
    sess_hash = sha256_file(sess_p)
    results["runtime_session"] = {
        "path": str(sess_p),
        "sha256": sess_hash,
        "expected_sha256": EXPECTED["runtime_session_sha256"],
        "match": sess_hash == EXPECTED["runtime_session_sha256"]
    }

    all_ok = all(v["match"] for v in results.values())
    print(f"All identity and integrity checks passed: {all_ok}")
    for k, v in results.items():
        print(f"  {k:22s}: {'PASS' if v['match'] else 'FAIL'} ({v['sha256'][:16]}...)")

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/checkpoint_lineage_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Audit written to {out_file}")

    return all_ok

if __name__ == "__main__":
    if not main():
        sys.exit(1)

