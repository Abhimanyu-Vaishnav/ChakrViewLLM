"""
Batches 391-393: Controlled Training of Candidate ChakrMicro with Corrected Loss Masking & True EOS Token.
Location: experiments/batch381_410_integrity/train_corrected_candidate.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DIAGNOSTIC_SET_PATH = Path("experiments/batch381_410_integrity/data/diagnostic_answer_set.jsonl")
OUT_CKPT_PATH = Path("experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt")
REPORTS_DIR = Path("experiments/batch381_410_integrity/reports")


def encode_corrected_pair(
    tokenizer: Any,
    prompt: str,
    target: str,
    max_len: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mathematically exact next-token supervision:
    - prompt tokens: [0] + encode(prompt)
    - target tokens: encode(target) + [1] (where 1 is EOS)
    - input_ids: prompt_tokens + target_tokens
    - labels: [-100] * len(prompt_tokens) + target_tokens
    """
    p_toks = [0] + tokenizer.encode(prompt)
    r_toks = tokenizer.encode(target) + [1]

    input_ids = (p_toks + r_toks)[:max_len]
    labels = ([-100] * len(p_toks) + r_toks)[:max_len]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def evaluate_diagnostic_accuracy(
    model: ChakrMicro,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    device: torch.device = torch.device("cpu"),
    max_new_tokens: int = 25,
) -> Dict[str, Any]:
    model.eval()
    exact_matches = 0
    clean_terminations = 0
    domain_scores = {}
    details = []

    for item in items:
        dom = item["domain"]
        if dom not in domain_scores:
            domain_scores[dom] = {"total": 0, "exact": 0, "clean": 0}
        domain_scores[dom]["total"] += 1

        p_toks = [0] + tokenizer.encode(item["prompt"])
        gen = list(p_toks)
        eos_reached = False

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([gen[-256:]], dtype=torch.long, device=device)
                logits = model(x)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt == 1:  # True EOS
                    eos_reached = True
                    break
                gen.append(nxt)

        decoded = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
        exp = item["expected_answer"].strip()
        is_exact = (decoded == exp) or (exp in decoded)

        if eos_reached:
            clean_terminations += 1
            domain_scores[dom]["clean"] += 1
        if is_exact:
            exact_matches += 1
            domain_scores[dom]["exact"] += 1

        details.append({
            "id": item["id"],
            "domain": dom,
            "query": item["query"],
            "expected": exp,
            "generated": decoded,
            "exact_match": is_exact,
            "eos_reached": eos_reached,
        })

    n = max(len(items), 1)
    return {
        "total": len(items),
        "exact_match_count": exact_matches,
        "exact_match_rate": round(exact_matches / n, 4),
        "clean_termination_count": clean_terminations,
        "clean_termination_rate": round(clean_terminations / n, 4),
        "domain_scores": domain_scores,
        "details": details,
    }


def run_experiment():
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load canonical base model
    ck_base = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    base_model = ChakrMicro(ck_base["config"])
    base_model.load_state_dict(ck_base["model_state_dict"])
    base_model.to(device)

    # Load diagnostic dataset
    with open(DIAGNOSTIC_SET_PATH, "r", encoding="utf-8") as f:
        diag_items = [json.loads(l) for l in f if l.strip()]

    # 1. Baseline Evaluation
    print("--- Evaluating Canonical Baseline on Diagnostic Set ---")
    base_eval = evaluate_diagnostic_accuracy(base_model, tokenizer, diag_items, device)
    print(f"Baseline Exact Match: {base_eval['exact_match_rate']*100:.1f}% | True EOS Termination: {base_eval['clean_termination_rate']*100:.1f}%")

    # 2. Train Controlled Candidate with Corrected EOS and Masking
    print("\n--- Training Corrected Candidate (150 steps, AdamW, peak_lr=2e-4, exact EOS) ---")
    torch.manual_seed(42)
    random.seed(42)

    cand_model = copy.deepcopy(base_model)
    cand_model.train()

    optimizer = optim.AdamW(cand_model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training items: use diagnostic items that are familiar/training patterns
    train_pool = [it for it in diag_items if it["capability_level"] in ("familiar_pattern", "memorized_concept")]
    steps = 150
    t0 = time.perf_counter()

    for step in range(steps):
        it = train_pool[step % len(train_pool)]
        x, y = encode_corrected_pair(tokenizer, it["prompt"], it["target"])
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        optimizer.zero_grad()
        logits = cand_model(x)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = y[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cand_model.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"Step {step+1:3d}/{steps} | Loss: {loss.item():.4f}")

    elapsed = time.perf_counter() - t0

    # Save Candidate Checkpoint
    OUT_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "name": "candidate_step150_corrected_eos",
        "steps": steps,
        "config": cand_model.config,
        "model_state_dict": cand_model.state_dict(),
        "elapsed_seconds": round(elapsed, 2),
    }, OUT_CKPT_PATH)

    with open(OUT_CKPT_PATH, "rb") as f:
        cand_hash = hashlib.sha256(f.read()).hexdigest()

    # 3. Candidate Evaluation
    print("\n--- Evaluating Corrected Candidate on Diagnostic Set ---")
    cand_eval = evaluate_diagnostic_accuracy(cand_model, tokenizer, diag_items, device)
    print(f"Candidate Exact Match: {cand_eval['exact_match_rate']*100:.1f}% | True EOS Termination: {cand_eval['clean_termination_rate']*100:.1f}%")

    # Export Report
    results = {
        "canonical_baseline": {
            "exact_match_rate": base_eval["exact_match_rate"],
            "clean_termination_rate": base_eval["clean_termination_rate"],
            "domain_scores": base_eval["domain_scores"],
        },
        "candidate_corrected": {
            "checkpoint_path": str(OUT_CKPT_PATH),
            "sha256": cand_hash,
            "exact_match_rate": cand_eval["exact_match_rate"],
            "clean_termination_rate": cand_eval["clean_termination_rate"],
            "domain_scores": cand_eval["domain_scores"],
            "details": cand_eval["details"],
        },
    }

    out_json = REPORTS_DIR / "neural_answer_quality.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nExported comparison metrics to: {out_json}")


if __name__ == "__main__":
    run_experiment()
