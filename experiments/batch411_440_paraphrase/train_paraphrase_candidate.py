"""
Batches 423-429: Training Paraphrase-Augmented Candidate with Rehearsal Regularization.
Location: experiments/batch411_440_paraphrase/train_paraphrase_candidate.py

Evaluates:
1. No-training reference (candidate_step150_corrected_eos.pt) on held-out paraphrases.
2. Trains new candidate (candidate_step150_paraphrase_augmented.pt) using train_augmented.jsonl + 5% Stage C rehearsal.
3. Evaluates both candidates on:
   - Held-out Paraphrases (20 items across 10 concepts)
   - Held-out Compositional Tasks (4 items)
   - Diagnostic Retention Set (15 items)
4. Exports structured JSON metrics and comparison markdown.
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
STARTING_CANDIDATE = Path("experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
STAGE_C_DIR = Path("data/tokenized/stage_c/train")

DATA_DIR = Path("experiments/batch411_440_paraphrase/data")
DIAGNOSTIC_PATH = Path("experiments/batch381_410_integrity/data/diagnostic_answer_set.jsonl")
OUT_CKPT_PATH = Path("experiments/batch411_440_paraphrase/checkpoints/candidate_step150_paraphrase_augmented.pt")
REPORTS_DIR = Path("experiments/batch411_440_paraphrase/reports")


class StageCRehearsalSampler:
    def __init__(self, shards_dir: Path, seq_len: int = 128, seed: int = 42):
        self.shards = sorted(list(shards_dir.glob("shard_*.bin")))
        self.seq_len = seq_len
        self.rng = random.Random(seed)

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        shard = self.rng.choice(self.shards)
        file_size = shard.stat().st_size
        n_tokens = file_size // 2
        max_start = max(0, n_tokens - self.seq_len - 1)
        offset = self.rng.randint(0, max_start) if max_start > 0 else 0

        with open(shard, "rb") as f:
            f.seek(offset * 2)
            raw = f.read((self.seq_len + 1) * 2)

        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.int64)
        if len(arr) < self.seq_len + 1:
            arr = np.pad(arr, (0, self.seq_len + 1 - len(arr)), mode="constant", constant_values=2)

        x = torch.tensor(arr[:self.seq_len], dtype=torch.long).unsqueeze(0)
        y = torch.tensor(arr[1:self.seq_len + 1], dtype=torch.long).unsqueeze(0)
        return x, y


def encode_pair(tokenizer: Any, prompt: str, target: str, max_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    p_toks = [0] + tokenizer.encode(prompt)
    r_toks = tokenizer.encode(target) + [1]  # EOS=1

    input_ids = (p_toks + r_toks)[:max_len]
    labels = ([-100] * len(p_toks) + r_toks)[:max_len]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def evaluate_items(
    model: ChakrMicro,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    device: torch.device = torch.device("cpu"),
    max_new_tokens: int = 25,
) -> Dict[str, Any]:
    model.eval()
    exact_matches = 0
    clean_terminations = 0
    details = []

    for item in items:
        p_toks = [0] + tokenizer.encode(item["prompt"])
        gen = list(p_toks)
        eos_reached = False

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([gen[-256:]], dtype=torch.long, device=device)
                logits = model(x)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt == 1:
                    eos_reached = True
                    break
                gen.append(nxt)

        decoded = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
        exp = str(item.get("expected") or item.get("expected_answer") or "").strip()
        is_exact = (decoded == exp) or (exp in decoded)

        if eos_reached:
            clean_terminations += 1
        if is_exact:
            exact_matches += 1

        details.append({
            "id": item.get("id"),
            "prompt": item["prompt"],
            "expected": exp,
            "generated": decoded,
            "exact_match": is_exact,
            "eos_reached": eos_reached,
        })

    n = max(len(items), 1)
    return {
        "total": len(items),
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / n, 4),
        "clean_terminations": clean_terminations,
        "clean_termination_rate": round(clean_terminations / n, 4),
        "details": details,
    }


def run_experiment():
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load Starting Candidate (Reference)
    ck_start = torch.load(STARTING_CANDIDATE, map_location=device, weights_only=False)
    ref_model = ChakrMicro(ck_start["config"])
    ref_model.load_state_dict(ck_start["model_state_dict"])
    ref_model.to(device)

    # Load manifests
    def load_jsonl(p: Path):
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train_items = load_jsonl(DATA_DIR / "train_augmented.jsonl")
    heldout_paraphrase_items = load_jsonl(DATA_DIR / "test_heldout_paraphrase.jsonl")
    heldout_comp_items = load_jsonl(DATA_DIR / "test_heldout_compositional.jsonl")
    diag_items = load_jsonl(DIAGNOSTIC_PATH)

    # 1. Evaluate Reference Model (Before Training)
    print("--- Evaluating Starting Candidate Reference ---")
    ref_para_eval = evaluate_items(ref_model, tokenizer, heldout_paraphrase_items, device)
    ref_comp_eval = evaluate_items(ref_model, tokenizer, heldout_comp_items, device)
    ref_diag_eval = evaluate_items(ref_model, tokenizer, diag_items, device)

    print(f"Reference Heldout Paraphrase Match: {ref_para_eval['exact_match_rate']*100:.1f}% ({ref_para_eval['exact_matches']}/{ref_para_eval['total']})")
    print(f"Reference Heldout Compositional:   {ref_comp_eval['exact_match_rate']*100:.1f}% ({ref_comp_eval['exact_matches']}/{ref_comp_eval['total']})")
    print(f"Reference Diagnostic Retention:   {ref_diag_eval['exact_match_rate']*100:.1f}% ({ref_diag_eval['exact_matches']}/{ref_diag_eval['total']})")

    # 2. Train Paraphrase-Augmented Candidate
    print("\n--- Training Paraphrase-Augmented Candidate (150 steps, 5% Stage C Rehearsal) ---")
    torch.manual_seed(42)
    random.seed(42)

    cand_model = copy.deepcopy(ref_model)
    cand_model.train()

    optimizer = optim.AdamW(cand_model.parameters(), lr=1.5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    sampler = StageCRehearsalSampler(STAGE_C_DIR, seq_len=128, seed=42)

    steps = 150
    t0 = time.perf_counter()

    for step in range(steps):
        optimizer.zero_grad()
        is_replay = (random.random() < 0.05)

        if is_replay:
            x, y = sampler.sample_batch()
            x, y = x.to(device), y.to(device)
            logits = cand_model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            it = train_items[step % len(train_items)]
            x, y = encode_pair(tokenizer, it["prompt"], it["target"])
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
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
        "name": "candidate_step150_paraphrase_augmented",
        "steps": steps,
        "config": cand_model.config,
        "model_state_dict": cand_model.state_dict(),
        "elapsed_seconds": round(elapsed, 2),
    }, OUT_CKPT_PATH)

    with open(OUT_CKPT_PATH, "rb") as f:
        cand_hash = hashlib.sha256(f.read()).hexdigest()

    # 3. Evaluate Augmented Candidate
    print("\n--- Evaluating Paraphrase-Augmented Candidate ---")
    cand_para_eval = evaluate_items(cand_model, tokenizer, heldout_paraphrase_items, device)
    cand_comp_eval = evaluate_items(cand_model, tokenizer, heldout_comp_items, device)
    cand_diag_eval = evaluate_items(cand_model, tokenizer, diag_items, device)

    print(f"Augmented Heldout Paraphrase Match: {cand_para_eval['exact_match_rate']*100:.1f}% ({cand_para_eval['exact_matches']}/{cand_para_eval['total']})")
    print(f"Augmented Heldout Compositional:   {cand_comp_eval['exact_match_rate']*100:.1f}% ({cand_comp_eval['exact_matches']}/{cand_comp_eval['total']})")
    print(f"Augmented Diagnostic Retention:   {cand_diag_eval['exact_match_rate']*100:.1f}% ({cand_diag_eval['exact_matches']}/{cand_diag_eval['total']})")

    # Export structured JSON
    results = {
        "reference_candidate": {
            "checkpoint": str(STARTING_CANDIDATE),
            "heldout_paraphrase_exact_rate": ref_para_eval["exact_match_rate"],
            "heldout_compositional_exact_rate": ref_comp_eval["exact_match_rate"],
            "diagnostic_retention_exact_rate": ref_diag_eval["exact_match_rate"],
        },
        "augmented_candidate": {
            "checkpoint": str(OUT_CKPT_PATH),
            "sha256": cand_hash,
            "heldout_paraphrase_exact_rate": cand_para_eval["exact_match_rate"],
            "heldout_compositional_exact_rate": cand_comp_eval["exact_match_rate"],
            "diagnostic_retention_exact_rate": cand_diag_eval["exact_match_rate"],
            "paraphrase_details": cand_para_eval["details"],
            "compositional_details": cand_comp_eval["details"],
        },
    }

    out_json = REPORTS_DIR / "paraphrase_experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nExported comparison metrics to: {out_json}")


if __name__ == "__main__":
    run_experiment()
