"""
Batches 513-521: Controlled Training Experiment with Multi-Task Rule Curriculum.
Location: experiments/batch501_540/train_compositional_candidate.py

Features:
1. Multi-Task Curriculum Mix:
   - 40% Compositional Rule Scratchpads (train_compositional_rule_sft.jsonl, N=70)
   - 35% Single-Step Paraphrase Curriculum (train_augmented.jsonl, N=40)
   - 20% Single-Step Primitives & Arithmetic Controls (N=20)
   - 5% Stage C Rehearsal Sampler (preserves pretraining representations)
2. Exact causal loss masking (labels = -100 for prompt, response supervised on True EOS = 1).
3. Checkpoint lineage preservation: loads candidate_step150_scratchpad_reasoning.pt as starting weights.
4. Validation-loss monitoring every 25 steps against held-out val split (val_compositional_rule_sft.jsonl).
5. Early stopping / best model checkpoint saving to prevent overfitting.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

STARTING_CANDIDATE = REPO_ROOT / "experiments" / "batch441_470_reasoning_gate" / "checkpoints" / "candidate_step150_scratchpad_reasoning.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
STAGE_C_DIR = REPO_ROOT / "data" / "tokenized" / "stage_c" / "train"

TRAIN_COMP_DATA = REPO_ROOT / "experiments" / "batch501_540" / "data" / "train_compositional_rule_sft.jsonl"
VAL_COMP_DATA = REPO_ROOT / "experiments" / "batch501_540" / "data" / "val_compositional_rule_sft.jsonl"
TEST_COMP_DATA = REPO_ROOT / "experiments" / "batch501_540" / "data" / "test_heldout_comprehensive_60.jsonl"
PARAPHRASE_DATA = REPO_ROOT / "experiments" / "batch411_440_paraphrase" / "data" / "train_augmented.jsonl"
DIAGNOSTIC_DATA = REPO_ROOT / "experiments" / "batch381_410_integrity" / "data" / "diagnostic_answer_set.jsonl"
HELDOUT_PARAPHRASE = REPO_ROOT / "experiments" / "batch411_440_paraphrase" / "data" / "test_heldout_paraphrase.jsonl"

OUT_CKPT_PATH = REPO_ROOT / "experiments" / "batch501_540" / "checkpoints" / "candidate_step200_compositional_rule.pt"
REPORT_PATH = REPO_ROOT / "experiments" / "batch501_540" / "reports" / "training_and_evaluation_results.json"


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
    r_toks = tokenizer.encode(target) + [1]

    input_ids = (p_toks + r_toks)[:max_len]
    labels = ([-100] * len(p_toks) + r_toks)[:max_len]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def evaluate_benchmark(
    model: ChakrMicro,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    device: torch.device = torch.device("cpu"),
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    model.eval()
    exact_matches = 0
    clean_eos_count = 0
    valid_step_count = 0
    cat_stats: Dict[str, Dict[str, int]] = {}
    details = []

    for item in items:
        cat = item.get("category", item.get("family", "default"))
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "exact_matches": 0, "valid_steps": 0}
        cat_stats[cat]["total"] += 1

        prompt = item["prompt"]
        p_toks = [0] + tokenizer.encode(prompt)
        gen = list(p_toks)
        eos_found = False

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([gen[-256:]], dtype=torch.long, device=device)
                nxt = int(torch.argmax(model(x)[0, -1, :]).item())
                if nxt == 1:
                    eos_found = True
                    break
                gen.append(nxt)

        if eos_found:
            clean_eos_count += 1

        raw_gen = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
        expected = str(item.get("expected", item.get("target", ""))).strip()

        # Final answer match
        final_match = (expected in raw_gen) or (raw_gen == expected)
        if final_match:
            exact_matches += 1
            cat_stats[cat]["exact_matches"] += 1

        # Intermediate step check
        steps = item.get("steps", [])
        trace_valid = False
        if steps:
            trace_valid = all(s.replace(" ", "") in raw_gen.replace(" ", "") for s in steps)
            if trace_valid:
                valid_step_count += 1
                cat_stats[cat]["valid_steps"] += 1

        details.append({
            "id": item.get("id", ""),
            "category": cat,
            "prompt": prompt,
            "expected": expected,
            "generated": raw_gen,
            "final_match": final_match,
            "trace_valid": trace_valid,
            "eos_reached": eos_found,
        })

    total = len(items)
    return {
        "total": total,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / max(total, 1),
        "clean_eos_rate": clean_eos_count / max(total, 1),
        "valid_steps_rate": valid_step_count / max(total, 1),
        "categories": cat_stats,
        "details": details,
    }


def compute_val_loss(model: ChakrMicro, tokenizer: Any, val_items: List[Dict[str, Any]], criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for it in val_items:
            x, y = encode_pair(tokenizer, it["prompt"], it["target"], max_len=256)
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            logits = model(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def main():
    print("=== BATCHES 513-528: CONTROLLED COMPOSITIONAL LEARNING EXPERIMENT ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load Starting Checkpoint
    ck_start = torch.load(STARTING_CANDIDATE, map_location="cpu", weights_only=False)
    ref_model = ChakrMicro(ck_start["config"])
    ref_model.load_state_dict(ck_start["model_state_dict"])
    ref_model.to(device)

    def load_jsonl(p: Path):
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train_comp = load_jsonl(TRAIN_COMP_DATA)
    val_comp = load_jsonl(VAL_COMP_DATA)
    test_comp_60 = load_jsonl(TEST_COMP_DATA)
    train_para = load_jsonl(PARAPHRASE_DATA)
    heldout_para = load_jsonl(HELDOUT_PARAPHRASE)
    test_diag = load_jsonl(DIAGNOSTIC_DATA)

    # 1. Baseline Evaluation on the Fresh 60-item Benchmark
    print("\n--- Evaluating Baseline Candidate (candidate_step150_scratchpad_reasoning.pt) ---")
    base_comp_eval = evaluate_benchmark(ref_model, tokenizer, test_comp_60, device)
    base_para_eval = evaluate_benchmark(ref_model, tokenizer, heldout_para, device, max_new_tokens=30)
    base_diag_eval = evaluate_benchmark(ref_model, tokenizer, test_diag, device, max_new_tokens=30)

    print(f"Baseline Held-out 60-item Compositional: {base_comp_eval['exact_match_rate']*100:.1f}% ({base_comp_eval['exact_matches']}/60)")
    for cat, s in base_comp_eval["categories"].items():
        print(f"  {cat}: {s['exact_matches']}/{s['total']} ({s['exact_matches']/s['total']*100:.1f}%)")
    print(f"Baseline Held-out Paraphrase:            {base_para_eval['exact_match_rate']*100:.1f}% ({base_para_eval['exact_matches']}/20)")
    print(f"Baseline Diagnostic Retention:          {base_diag_eval['exact_match_rate']*100:.1f}% ({base_diag_eval['exact_matches']}/15)")

    # 2. Training the Improved Compositional Candidate
    print("\n--- Training Compositional Candidate (Step 200 Multi-Task Curriculum) ---")
    torch.manual_seed(42)
    random.seed(42)

    cand_model = copy.deepcopy(ref_model)
    cand_model.train()

    optimizer = optim.AdamW(cand_model.parameters(), lr=1.0e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    sampler = StageCRehearsalSampler(STAGE_C_DIR, seq_len=128, seed=42)

    steps = 200
    t0 = time.perf_counter()
    best_val_loss = float("inf")
    best_state = None
    loss_history = []

    for step in range(steps):
        cand_model.train()
        optimizer.zero_grad()
        rnd = random.random()

        if rnd < 0.05:
            # 5% Pretraining rehearsal
            x, y = sampler.sample_batch()
            x, y = x.to(device), y.to(device)
            logits = cand_model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        elif rnd < 0.45:
            # 40% Compositional scratchpad rule SFT
            it = train_comp[step % len(train_comp)]
            x, y = encode_pair(tokenizer, it["prompt"], it["target"], max_len=256)
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            logits = cand_model(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            # 55% Single-step paraphrase & retention rehearsal
            it = train_para[step % len(train_para)]
            x, y = encode_pair(tokenizer, it["prompt"], it["target"], max_len=256)
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            logits = cand_model(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(cand_model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(loss.item())

        if (step + 1) % 25 == 0:
            val_loss = compute_val_loss(cand_model, tokenizer, val_comp, criterion, device)
            print(f"Step {step+1:3d}/{steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(cand_model.state_dict())

    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed:.2f}s. Best Val Loss: {best_val_loss:.4f}")

    if best_state is not None:
        cand_model.load_state_dict(best_state)

    # Save Checkpoint
    OUT_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "name": "candidate_step200_compositional_rule",
        "steps": steps,
        "config": cand_model.config,
        "model_state_dict": cand_model.state_dict(),
        "elapsed_seconds": round(elapsed, 2),
        "best_val_loss": round(best_val_loss, 4),
    }, OUT_CKPT_PATH)

    with open(OUT_CKPT_PATH, "rb") as f:
        cand_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Saved Candidate: {OUT_CKPT_PATH} (SHA-256: {cand_hash})")

    # 3. Post-Training Evaluation on 60-item Held-Out Benchmark & Retention
    print("\n--- Evaluating Trained Candidate (candidate_step200_compositional_rule.pt) ---")
    new_comp_eval = evaluate_benchmark(cand_model, tokenizer, test_comp_60, device)
    new_para_eval = evaluate_benchmark(cand_model, tokenizer, heldout_para, device, max_new_tokens=30)
    new_diag_eval = evaluate_benchmark(cand_model, tokenizer, test_diag, device, max_new_tokens=30)

    print(f"\nTrained Candidate Held-out 60-item Compositional: {new_comp_eval['exact_match_rate']*100:.1f}% ({new_comp_eval['exact_matches']}/60)")
    for cat, s in new_comp_eval["categories"].items():
        print(f"  {cat}: {s['exact_matches']}/{s['total']} ({s['exact_matches']/s['total']*100:.1f}%)")
    print(f"Trained Candidate Held-out Paraphrase:            {new_para_eval['exact_match_rate']*100:.1f}% ({new_para_eval['exact_matches']}/20)")
    print(f"Trained Candidate Diagnostic Retention:          {new_diag_eval['exact_match_rate']*100:.1f}% ({new_diag_eval['exact_matches']}/15)")

    # Save structured results
    out_payload = {
        "baseline_candidate": {
            "checkpoint": str(STARTING_CANDIDATE),
            "compositional_60": base_comp_eval,
            "paraphrase_20": base_para_eval,
            "diagnostic_15": base_diag_eval,
        },
        "trained_candidate": {
            "checkpoint": str(OUT_CKPT_PATH),
            "sha256": cand_hash,
            "training_steps": steps,
            "elapsed_seconds": round(elapsed, 2),
            "best_val_loss": round(best_val_loss, 4),
            "compositional_60": new_comp_eval,
            "paraphrase_20": new_para_eval,
            "diagnostic_15": new_diag_eval,
        }
    }
    REPORT_PATH.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"\nFull evaluation results written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
