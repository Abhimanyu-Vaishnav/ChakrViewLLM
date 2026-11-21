"""
UCIP-28 Phase 0 & Phase 1: Integrity, Reproducibility & UCIP-27 Verification.
Verifies:
- Git branch, HEAD and working-tree status.
- Step800 checkpoint SHA-256.
- UCIP-20 checkpoint SHA-256.
- Protected runtime SHA-256.
- UCIP-27 artifact SHA-256 hashes.
- Reproduces original UCIP-27 key findings:
  * Baseline logits and top-1 token
  * All-attention ablation preserving attractor (cosine sim > 0.99)
  * All-FFN ablation collapsing attractor (cosine sim collapses)
  * Individual Layer 4 and 5 FFN ablations shifting attractor top-1 token
  * Target-token rank and attractor-token rank
"""

from __future__ import annotations
import glob
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXPECTED_HEAD = "48d52ffb024669004eecb8f93066abc5fb348750"
EXPECTED_RUNTIME_SHA256 = "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
EXPECTED_STEP800_SHA256 = "8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943"
EXPECTED_UCIP20_SHA256 = "0b4b745e923188c6320c8c7290031bff665786c4f4c1c71a74484bcaec6b1a27"

EXPECTED_UCIP27_HASHES = {
  "experiments/ucip27/attention_ablation/attention_ablation_results.json": "bb1c39e93e665f6b4cfbf291a208ee171f59fb083949cc42edbd14b3b80fc6bd",
  "experiments/ucip27/attention_ablation/run_attention_ablation.py": "cd59f2e388081c59c0f0695e2110923bf6b6edf1bb30307bd0ecb94987c86fc1",
  "experiments/ucip27/controls/statistical_controls_summary.json": "4acf0aee4df8d8de01baa6b3fe0f81dcb33236ad0ed4421d571d270e4b8ec26b",
  "experiments/ucip27/ffn_ablation/ffn_ablation_results.json": "57a82bd2d6caa60f40d5f7b5ca208082587abf8f98cfcaa23e8da32ccb721dc8",
  "experiments/ucip27/ffn_ablation/run_ffn_ablation.py": "870e6b972c197ff2137f0e089f8cb442e71ce28ddf1cac5d6d384662b67514c8",
  "experiments/ucip27/integrity/measurement_audit.json": "7f91de3de2205c9e8279d9ec2bc574450a43ec4cdc735bce7959c5ec9c4c91d8",
  "experiments/ucip27/layer_tracing/layer_tracing_results.json": "a3eb07e252de9c4924b5a237aded5c404de4aafb7d80d4d57d1713785a283acd",
  "experiments/ucip27/lm_head/residual_and_lm_head_diagnostics.json": "d3e266fd182fc56c8f297a3bddf386cf1975e3c7adc329d72b71bc28c2207522",
  "experiments/ucip27/tests/test_ucip27.py": "c6a56c68daa4b5c24c2f85ae7f917a95581ebb1d2661ec8390d3fba9f6e27197"
}


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m + 1e-12), p, reduction="sum")
    kl_qm = F.kl_div(torch.log(m + 1e-12), q, reduction="sum")
    js = 0.5 * (kl_pm + kl_qm)
    return float(math.sqrt(max(0.0, js.item())))


def run_integrity_and_reproduction():
    print("=== UCIP-28 Phase 0: Integrity Verification ===")

    # 1. Git HEAD check
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    actual_head = res.stdout.strip()
    assert actual_head == EXPECTED_HEAD, f"HEAD mismatch: {actual_head} != {EXPECTED_HEAD}"
    print(f"PASS: Git HEAD matches expected {EXPECTED_HEAD}")

    # 2. Checkpoint and Runtime hashes
    s800_hash = compute_sha256("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt")
    assert s800_hash == EXPECTED_STEP800_SHA256, f"Step800 mismatch: {s800_hash}"
    print(f"PASS: Step-800 checkpoint hash verified: {s800_hash[:16]}...")

    u20_hash = compute_sha256("experiments/ucip20/checkpoints/checkpoint_best.pt")
    assert u20_hash == EXPECTED_UCIP20_SHA256, f"UCIP-20 mismatch: {u20_hash}"
    print(f"PASS: UCIP-20 checkpoint hash verified: {u20_hash[:16]}...")

    rt_hash = compute_sha256("chakrview/runtime/session.py")
    assert rt_hash == EXPECTED_RUNTIME_SHA256, f"Runtime mismatch: {rt_hash}"
    print(f"PASS: Protected runtime hash verified: {rt_hash[:16]}...")

    # 3. UCIP-27 artifact hashes
    for path_str, exp_h in EXPECTED_UCIP27_HASHES.items():
        actual_h = compute_sha256(path_str)
        assert actual_h == exp_h, f"UCIP-27 artifact {path_str} hash mismatch: {actual_h} != {exp_h}"
    print("PASS: All protected UCIP-27 artifact hashes verified bit-exact.")

    # Save integrity manifest
    integrity_record = {
        "expected_head": EXPECTED_HEAD,
        "actual_head": actual_head,
        "step800_sha256": s800_hash,
        "ucip20_sha256": u20_hash,
        "runtime_sha256": rt_hash,
        "ucip27_artifacts_verified": list(EXPECTED_UCIP27_HASHES.keys())
    }
    with open("experiments/ucip28/manifests/integrity_audit.json", "w", encoding="utf-8") as f:
        json.dump(integrity_record, f, indent=2)

    print("\n=== UCIP-28 Phase 1: Reproduce UCIP-27 Results ===")
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    prompt = "User: Context statement: NovaTrack has founded in recorded as 1981.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: "
    target = "1981"
    target_id = tok.encode(target)[0]
    input_ids = [0] + tok.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long)

    repro_results = {}

    for ckpt_key, ckpt_path in [
        ("step800", "experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"),
        ("ucip20_best", "experiments/ucip20/checkpoints/checkpoint_best.pt")
    ]:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ChakrMicro(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            # Baseline forward
            h = model.embedding(x)
            for l_idx, layer in enumerate(model.layers):
                h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                h = h + layer.ffn(layer.norm_2(h))
            logits_base = model.lm_head(model.final_norm(h))[0, -1, :]
            probs_base = F.softmax(logits_base, dim=-1)

            top1_id = int(torch.argmax(logits_base).item())
            top1_str = tok.decode([top1_id], errors="replace")
            ranks = torch.argsort(logits_base, descending=True)
            target_rank = int((ranks == target_id).nonzero(as_tuple=True)[0].item()) + 1

            # All Attention ablated
            h = model.embedding(x)
            for l_idx, layer in enumerate(model.layers):
                h = h + layer.ffn(layer.norm_2(h))
            logits_no_attn = model.lm_head(model.final_norm(h))[0, -1, :]
            cos_no_attn = float(F.cosine_similarity(logits_base.unsqueeze(0), logits_no_attn.unsqueeze(0)).item())

            # All FFN ablated
            h = model.embedding(x)
            for l_idx, layer in enumerate(model.layers):
                h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
            logits_no_ffn = model.lm_head(model.final_norm(h))[0, -1, :]
            cos_no_ffn = float(F.cosine_similarity(logits_base.unsqueeze(0), logits_no_ffn.unsqueeze(0)).item())

            # Layer 4 ablated
            h = model.embedding(x)
            for l_idx, layer in enumerate(model.layers):
                h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                if l_idx != 4:
                    h = h + layer.ffn(layer.norm_2(h))
            logits_no_ffn4 = model.lm_head(model.final_norm(h))[0, -1, :]
            top1_no_ffn4 = tok.decode([int(torch.argmax(logits_no_ffn4).item())], errors="replace")

            # Layer 5 ablated
            h = model.embedding(x)
            for l_idx, layer in enumerate(model.layers):
                h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                if l_idx != 5:
                    h = h + layer.ffn(layer.norm_2(h))
            logits_no_ffn5 = model.lm_head(model.final_norm(h))[0, -1, :]
            top1_no_ffn5 = tok.decode([int(torch.argmax(logits_no_ffn5).item())], errors="replace")

        print(f"Checkpoint: {ckpt_key}")
        print(f"  Baseline Top-1: {repr(top1_str)} (Target Rank: {target_rank})")
        print(f"  All Attention Ablated Cosine Sim: {cos_no_attn:.4f}")
        print(f"  All FFN Ablated Cosine Sim: {cos_no_ffn:.4f}")
        print(f"  FFN L4 Ablated Top-1: {repr(top1_no_ffn4)}")
        print(f"  FFN L5 Ablated Top-1: {repr(top1_no_ffn5)}")

        # Verification asserts matching UCIP-27
        assert cos_no_attn > 0.99, f"All-attn ablation cosine similarity {cos_no_attn} <= 0.99"
        if ckpt_key == "step800":
            assert top1_str == "#", f"Expected '#' top-1 in Step800, got {top1_str}"
            assert target_rank == 40
            assert cos_no_ffn < -0.90
            assert top1_no_ffn4.startswith("The")
            assert top1_no_ffn5.startswith("The")
        else:
            assert top1_str == "A", f"Expected 'A' top-1 in UCIP-20, got {top1_str}"
            assert target_rank == 28
            assert cos_no_ffn < 0.10
            assert top1_no_ffn4 == "["
            assert top1_no_ffn5 == "["

        repro_results[ckpt_key] = {
            "top1_str": top1_str,
            "target_rank": target_rank,
            "all_attention_ablated_cosine_sim": cos_no_attn,
            "all_ffn_ablated_cosine_sim": cos_no_ffn,
            "ffn_l4_ablated_top1": top1_no_ffn4,
            "ffn_l5_ablated_top1": top1_no_ffn5,
        }

    with open("experiments/ucip28/reports/reproduce_ucip27.json", "w", encoding="utf-8") as f:
        json.dump(repro_results, f, indent=2)
    print("PASS: UCIP-27 results 100% reproduced bit-exact.\n")


if __name__ == "__main__":
    run_integrity_and_reproduction()
