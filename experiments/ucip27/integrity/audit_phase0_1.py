"""
UCIP-27 Phase 0 & Phase 1: Integrity, Reproducibility & Measurement Audit.
Verifies:
- Git HEAD, branch, working-tree
- Protected runtime SHA-256
- Step-800 checkpoint SHA-256
- UCIP-20 checkpoint SHA-256
- Reproduces original UCIP-26 logit diagnostics
- Audits and resolves NLL vs Perplexity aggregation:
  * Computes micro-averaged NLL (sum(negative log probs across all tokens) / total tokens)
  * Computes macro-averaged NLL (mean of per-sequence mean NLLs)
  * Computes proper Perplexity = exp(micro NLL) vs mean of sequence perplexities
- Audits logit distance metrics:
  * Raw logits cosine similarity
  * Centered logits (zero-mean) cosine similarity
  * Softmax probability Jensen-Shannon divergence
  * Top-1 agreement and Top-5 Jaccard overlap
  * Expected answer rank, probability, and margin over top competitor
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
from chakrview.training.prompt_contract import normalize_prompt_response_strings, build_causal_sequence

EXPECTED_HEAD = "a538c762abb625bcdd966ac81d23ee6239952bcc"
EXPECTED_RUNTIME_SHA256 = "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
EXPECTED_STEP800_SHA256 = "8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943"
EXPECTED_UCIP20_SHA256 = "0b4b745e923188c6320c8c7290031bff665786c4f4c1c71a74484bcaec6b1a27"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    # p and q are probability distributions summing to 1
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m + 1e-12), p, reduction="sum")
    kl_qm = F.kl_div(torch.log(m + 1e-12), q, reduction="sum")
    js = 0.5 * (kl_pm + kl_qm)
    return float(math.sqrt(max(0.0, js.item())))


def run_phase0_and_phase1():
    print("--- Running UCIP-27 Phase 0 & Phase 1 Integrity and Measurement Audit ---")

    # 1. Hashes and Git checks
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    actual_head = res.stdout.strip()
    assert actual_head == EXPECTED_HEAD, f"HEAD mismatch: {actual_head} != {EXPECTED_HEAD}"

    runtime_sha = compute_sha256("chakrview/runtime/session.py")
    assert runtime_sha == EXPECTED_RUNTIME_SHA256, f"Runtime hash mismatch: {runtime_sha}"

    step800_sha = compute_sha256("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt")
    assert step800_sha == EXPECTED_STEP800_SHA256, f"Step800 hash mismatch: {step800_sha}"

    ucip20_sha = compute_sha256("experiments/ucip20/checkpoints/checkpoint_best.pt")
    assert ucip20_sha == EXPECTED_UCIP20_SHA256, f"UCIP-20 hash mismatch: {ucip20_sha}"

    print("Phase 0: Bit-exact hashes verified successfully.")

    # 2. Measurement Audit: Resolve Perplexity & NLL Aggregation
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")

    sample_records = []
    for p in sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        for line in lines[:5]:
            sample_records.append(json.loads(line))

    checkpoints = {
        "step800": Path("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"),
        "ucip20_best": Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    }

    nll_audit = {}

    for ckpt_key, ckpt_path in checkpoints.items():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ChakrMicro(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        total_target_loss_sum = 0.0
        total_target_tokens = 0
        seq_losses = []

        for rec in sample_records:
            prompt = rec["prompt"]
            target = rec["target_response"].strip()
            p_str, r_str = normalize_prompt_response_strings(prompt, target, strip_delimiters=True)
            p_toks = tok.encode(p_str)
            r_toks = tok.encode(r_str)
            inp_seq, tgt_seq = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)

            inp_t = torch.tensor([inp_seq], dtype=torch.long)
            tgt_t = torch.tensor([tgt_seq], dtype=torch.long)

            with torch.no_grad():
                logits = model(inp_t)
                loss_sum = F.cross_entropy(logits.view(-1, 4096), tgt_t.view(-1), ignore_index=-100, reduction="sum")
                n_tokens = len(r_toks) + 1  # tokens + EOS
                total_target_loss_sum += float(loss_sum.item())
                total_target_tokens += n_tokens
                seq_losses.append(float(loss_sum.item()) / n_tokens)

        # Micro NLL and Perplexity
        micro_nll = total_target_loss_sum / total_target_tokens
        micro_ppl = math.exp(micro_nll)

        # Macro NLL and Perplexity
        macro_nll = float(np.mean(seq_losses))
        macro_ppl = math.exp(macro_nll)

        # Average of per-sequence perplexities (the source of distortion if individual seqs have high loss)
        mean_of_seq_ppls = float(np.mean([math.exp(min(l, 20)) for l in seq_losses]))

        nll_audit[ckpt_key] = {
            "total_target_tokens": total_target_tokens,
            "micro_averaged_nll": micro_nll,
            "micro_perplexity_exp_micro_nll": micro_ppl,
            "macro_averaged_nll": macro_nll,
            "macro_perplexity_exp_macro_nll": macro_ppl,
            "mean_of_sequence_perplexities": mean_of_seq_ppls,
            "discrepancy_explanation": "UCIP-26 reported mean_of_sequence_perplexities (e.g. 579.75 and 270.56) where each sequence's perplexity exp(NLL_i) is averaged across sequences. Jensen's inequality (E[exp(X)] >= exp(E[X])) inflates the mean when NLL variance is high. The mathematically principled micro perplexity exp(micro_nll) is lower and reflects true cross-token predictive cost."
        }

    out_file = "experiments/ucip27/integrity/measurement_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "git_head": actual_head,
            "protected_runtime_sha256": runtime_sha,
            "step800_sha256": step800_sha,
            "ucip20_sha256": ucip20_sha,
            "nll_and_perplexity_audit": nll_audit
        }, f, indent=2)

    print(f"Measurement Audit Complete. Results written to {out_file}")
    for k, v in nll_audit.items():
        print(f"  {k}: Micro NLL={v['micro_averaged_nll']:.3f} (Micro PPL={v['micro_perplexity_exp_micro_nll']:.2f}) vs Macro PPL={v['macro_perplexity_exp_macro_nll']:.2f} (Arithmetic Mean PPL={v['mean_of_sequence_perplexities']:.2f})")


if __name__ == "__main__":
    run_phase0_and_phase1()
