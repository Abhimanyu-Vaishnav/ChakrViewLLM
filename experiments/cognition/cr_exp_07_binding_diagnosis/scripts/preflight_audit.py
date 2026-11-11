"""
Stage 0: Checkpoint & Implementation Preflight Script for CR-EXP-07
Verifies hashes of all 4 models, canonical baseline, tokenizer artifacts, and runtime files.
Also inspects architecture forward tensor locations.
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

CHECKPOINTS = {
    "parent": {
        "name": "Frozen Conversational Parent (Step 200)",
        "path": "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
        "expected_sha256": "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"
    },
    "candidate_pilot": {
        "name": "Conversational Pilot Candidate",
        "path": "experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt",
        "expected_sha256": "ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd"
    },
    "control_arm_best_val": {
        "name": "CR-EXP-06 Control Arm A Best Val",
        "path": "experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt",
        "expected_sha256": "729d5812c1ec02d217ef8cc342453a008ab3046a550e627b16b341c4a458c21f"
    },
    "treatment_arm_best_val": {
        "name": "CR-EXP-06 Treatment Arm B Best Val",
        "path": "experiments/cognition/cr_exp_06_variable_binding/treatment_arm/checkpoints/treatment_arm_best_val.pt",
        "expected_sha256": "2d8cea93ba8dbe06e1c714d5563ad89b18600f7a4f766fbb7cfc563c947a9d90"
    },
    "tokenizer_vocab": {
        "name": "Tokenizer Vocab",
        "path": "data/experiments/vocab_4096/vocab.json",
        "expected_sha256": EXPECTED_TOKENIZER_HASHES["vocab.json"]
    },
    "tokenizer_merges": {
        "name": "Tokenizer Merges",
        "path": "data/experiments/vocab_4096/merges.json",
        "expected_sha256": EXPECTED_TOKENIZER_HASHES["merges.json"]
    },
    "tokenizer_config": {
        "name": "Tokenizer Config",
        "path": "data/experiments/vocab_4096/config.json",
        "expected_sha256": EXPECTED_TOKENIZER_HASHES["config.json"]
    },
    "runtime_session": {
        "name": "Protected Runtime Session",
        "path": "chakrview/runtime/session.py",
        "expected_sha256": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    }
}

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== CR-EXP-07 Stage 0: Checkpoint & Preflight Audit ===")
    results = {}

    for k, item in CHECKPOINTS.items():
        p = REPO_ROOT / item["path"]
        exists = p.exists()
        h = sha256_file(p) if exists else None
        match = (h == item["expected_sha256"])
        results[k] = {
            "name": item["name"],
            "path": str(p),
            "exists": exists,
            "sha256": h,
            "expected_sha256": item["expected_sha256"],
            "match": match
        }
        print(f"  {k:25s}: {'PASS' if match else 'FAIL'} ({str(h)[:16]}...)")

    # Canonical Baseline Init
    torch.manual_seed(42)
    canon_model = ChakrMicro(ModelConfig())
    canon_model.eval()
    canon_hash = compute_model_hash(canon_model)
    canon_match = (canon_hash == EXPECTED_WEIGHT_HASH)
    results["canonical_baseline_init"] = {
        "name": "Canonical Baseline Initializer (seed=42)",
        "sha256": canon_hash,
        "expected_sha256": EXPECTED_WEIGHT_HASH,
        "match": canon_match
    }
    print(f"  canonical_baseline_init  : {'PASS' if canon_match else 'FAIL'} ({canon_hash[:16]}...)")

    # Inspect Forward Pass Hook Locations in ChakrMicro
    # model.embedding: nn.Embedding(4096, 192)
    # model.layers: ModuleList of 6 TransformerBlocks
    #   layer.norm_1: RMSNorm(192)
    #   layer.attn: Attention(192, n_heads=6)
    #   layer.norm_2: RMSNorm(192)
    #   layer.ffn: SwiGLUFFN(192 -> 512 -> 192)
    # model.final_norm: RMSNorm(192)
    # model.lm_head: Linear(192, 4096, bias=False, tied to embedding)
    hook_points = {
        "input_embedding": "model.embedding(input_ids)",
        "layers_count": len(canon_model.layers),
        "per_layer_components": [
            "norm_1 (pre-attention RMSNorm)",
            "attn (multi-head attention with RoPE)",
            "res_post_attn (residual addition: x + attn_out)",
            "norm_2 (pre-FFN RMSNorm)",
            "ffn (SwiGLU feed-forward network)",
            "res_post_ffn (residual addition: mid + ffn_out)"
        ],
        "final_norm": "model.final_norm",
        "output_projection": "model.lm_head (tied to embedding)"
    }
    results["hook_architecture_verification"] = hook_points

    all_matched = all(v.get("match", True) for v in results.values() if isinstance(v, dict) and "match" in v)
    print(f"\nPreflight Status: {'ALL MATCHES VERIFIED (Gate 0 Passed)' if all_matched else 'FAILED'}")

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_07_binding_diagnosis/checkpoint_identity_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Audit output saved to {out_file}")

    return all_matched

if __name__ == "__main__":
    if not main():
        sys.exit(1)

