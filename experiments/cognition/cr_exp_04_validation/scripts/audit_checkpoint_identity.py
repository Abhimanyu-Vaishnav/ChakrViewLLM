"""
Phase 1: Safety & Checkpoint Identity Audit Script
Verifies SHA-256 hashes of checkpoints, tokenizer artifacts, canonical baseline, and runtime files.
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
}

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Phase 1: Checkpoint & Artifact Identity Audit ===")
    results = {}
    
    # 1. Candidate Checkpoint
    cand_p = REPO_ROOT / EXPECTED["candidate_path"]
    cand_hash = sha256_file(cand_p) if cand_p.exists() else None
    cand_size = cand_p.stat().st_size if cand_p.exists() else 0
    results["candidate_checkpoint"] = {
        "path": str(cand_p),
        "exists": cand_p.exists(),
        "size_bytes": cand_size,
        "sha256": cand_hash,
        "expected_sha256": EXPECTED["candidate_sha256"],
        "match": cand_hash == EXPECTED["candidate_sha256"],
    }
    print(f"Candidate Checkpoint: {'MATCH' if results['candidate_checkpoint']['match'] else 'MISMATCH'}")
    print(f"  SHA-256: {cand_hash}")

    # 2. Parent Checkpoint
    parent_p = REPO_ROOT / EXPECTED["parent_path"]
    parent_hash = sha256_file(parent_p) if parent_p.exists() else None
    parent_size = parent_p.stat().st_size if parent_p.exists() else 0
    results["parent_checkpoint"] = {
        "path": str(parent_p),
        "exists": parent_p.exists(),
        "size_bytes": parent_size,
        "sha256": parent_hash,
        "expected_sha256": EXPECTED["parent_sha256"],
        "match": parent_hash == EXPECTED["parent_sha256"],
    }
    print(f"Parent Checkpoint: {'MATCH' if results['parent_checkpoint']['match'] else 'MISMATCH'}")
    print(f"  SHA-256: {parent_hash}")

    # 3. Tokenizer Vocab & Merges & Config
    v_p = REPO_ROOT / EXPECTED["vocab_path"]
    v_hash = sha256_file(v_p) if v_p.exists() else None
    results["tokenizer_vocab"] = {
        "path": str(v_p),
        "exists": v_p.exists(),
        "sha256": v_hash,
        "expected_sha256": EXPECTED["vocab_sha256"],
        "match": v_hash == EXPECTED["vocab_sha256"],
    }
    
    m_p = REPO_ROOT / EXPECTED["merges_path"]
    m_hash = sha256_file(m_p) if m_p.exists() else None
    results["tokenizer_merges"] = {
        "path": str(m_p),
        "exists": m_p.exists(),
        "sha256": m_hash,
        "expected_sha256": EXPECTED["merges_sha256"],
        "match": m_hash == EXPECTED["merges_sha256"],
    }

    c_p = REPO_ROOT / EXPECTED["config_path"]
    c_hash = sha256_file(c_p) if c_p.exists() else None
    results["tokenizer_config"] = {
        "path": str(c_p),
        "exists": c_p.exists(),
        "sha256": c_hash,
        "expected_sha256": EXPECTED["config_sha256"],
        "match": c_hash == EXPECTED["config_sha256"],
    }
    print(f"Tokenizer Vocab: {'MATCH' if results['tokenizer_vocab']['match'] else 'MISMATCH'}")
    print(f"Tokenizer Merges: {'MATCH' if results['tokenizer_merges']['match'] else 'MISMATCH'}")
    print(f"Tokenizer Config: {'MATCH' if results['tokenizer_config']['match'] else 'MISMATCH'}")

    # 4. Canonical Baseline Hash
    torch.manual_seed(42)
    canonical_model = ChakrMicro(ModelConfig())
    canonical_model.eval()
    canonical_model_hash = compute_model_hash(canonical_model)
    results["canonical_baseline_init"] = {
        "computed_sha256": canonical_model_hash,
        "expected_sha256": EXPECTED_WEIGHT_HASH,
        "match": canonical_model_hash == EXPECTED_WEIGHT_HASH,
    }
    print(f"Canonical Baseline Init: {'MATCH' if results['canonical_baseline_init']['match'] else 'MISMATCH'}")

    # 5. Protected Runtime Session
    sess_p = REPO_ROOT / EXPECTED["runtime_session_path"]
    sess_hash = sha256_file(sess_p) if sess_p.exists() else None
    results["runtime_session"] = {
        "path": str(sess_p),
        "exists": sess_p.exists(),
        "sha256": sess_hash,
    }
    print(f"Runtime Session SHA-256: {sess_hash}")

    # 6. Load both checkpoints into ChakrMicro and inspect parameter names / shapes
    cand_ckpt = torch.load(cand_p, map_location="cpu", weights_only=False)
    parent_ckpt = torch.load(parent_p, map_location="cpu", weights_only=False)
    
    cand_state = cand_ckpt.get("model_state_dict", cand_ckpt)
    parent_state = parent_ckpt.get("model_state_dict", parent_ckpt)

    cfg = ModelConfig()
    m_cand = ChakrMicro(cfg)
    m_parent = ChakrMicro(cfg)

    cand_load = m_cand.load_state_dict(cand_state, strict=True)
    parent_load = m_parent.load_state_dict(parent_state, strict=True)

    # Compare parameter tensor shapes and compute diffs
    total_params = sum(p.numel() for p in m_cand.parameters())
    diff_params = 0
    param_l2_diff = 0.0
    for name, p_c in m_cand.named_parameters():
        p_p = m_parent.state_dict()[name]
        diff = (p_c - p_p).pow(2).sum().item()
        param_l2_diff += diff
        if not torch.equal(p_c, p_p):
            diff_params += 1
            
    param_l2_diff = param_l2_diff ** 0.5
    
    results["model_compatibility"] = {
        "total_parameters": total_params,
        "cand_strict_load": len(cand_load.missing_keys) == 0 and len(cand_load.unexpected_keys) == 0,
        "parent_strict_load": len(parent_load.missing_keys) == 0 and len(parent_load.unexpected_keys) == 0,
        "num_modified_parameter_tensors": diff_params,
        "total_parameter_tensors": len(list(m_cand.named_parameters())),
        "parameter_l2_distance": param_l2_diff,
        "candidate_metadata": {
            "step": cand_ckpt.get("step"),
            "val_loss": cand_ckpt.get("val_loss"),
            "train_loss": cand_ckpt.get("train_loss"),
            "epoch": cand_ckpt.get("epoch"),
        },
        "parent_metadata": {
            "step": parent_ckpt.get("step"),
            "val_loss": parent_ckpt.get("val_loss"),
            "train_loss": parent_ckpt.get("train_loss"),
            "epoch": parent_ckpt.get("epoch"),
        }
    }
    print(f"Strict Load: Candidate={results['model_compatibility']['cand_strict_load']}, Parent={results['model_compatibility']['parent_strict_load']}")
    print(f"L2 Distance between Candidate and Parent weights: {param_l2_diff:.6f}")
    print(f"Modified parameter tensors: {diff_params} / {len(list(m_cand.named_parameters()))}")

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_04_validation/checkpoint_identity_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Audit results written to {out_file}")

if __name__ == "__main__":
    main()
