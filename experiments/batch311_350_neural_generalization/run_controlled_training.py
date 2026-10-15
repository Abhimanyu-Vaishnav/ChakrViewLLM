"""
Batches 323-330: Controlled Neural Training, Staged Step Progression & Generalization Comparison.
Location: experiments/batch311_350_neural_generalization/run_controlled_training.py

Evaluates:
- Baseline (Canonical Step 10,000 checkpoint).
- Candidate 1: 70 optimizer steps (reproducing prior milestone).
- Candidate 2: 150 optimizer steps with linear warmup (20 steps) + cosine annealing.
- Candidate 3: 300 optimizer steps with linear warmup (30 steps) + cosine annealing.

Metrics across every model:
- Train loss, Validation loss.
- Held-out Unseen Entity loss & perplexity.
- Held-out Unseen Template loss & perplexity.
- Clean `<|im_end|>` termination rate.
- Retention across English, Code, and Hindi continuations.
- Repetition n-gram ratio.
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

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DATA_DIR = Path("experiments/batch311_350_neural_generalization/data")
CKPT_DIR = Path("experiments/batch311_350_neural_generalization/checkpoints")
REPORTS_DIR = Path("experiments/batch311_350_neural_generalization/reports")

RETENTION_PROMPTS = [
    {"domain": "english", "prompt": "The development of modern science began with"},
    {"domain": "code", "prompt": "def binary_search(arr, target):"},
    {"domain": "hindi", "prompt": "भारत एक विशाल और विविधतापूर्ण"},
]


def load_canonical_model(device: torch.device = torch.device("cpu")) -> Tuple[ChakrMicro, Any]:
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    return model, tokenizer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def encode_chatml_example(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_len: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_tokens = [0] + tokenizer.encode(prompt)
    r_tokens = tokenizer.encode(response)

    input_ids = p_tokens + r_tokens
    # Target label masking: prompt tokens receive -100
    labels = [-100] * (len(p_tokens) - 1) + r_tokens + [1]

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    if len(labels) < len(input_ids):
        labels += [-100] * (len(input_ids) - len(labels))

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def evaluate_split_loss(
    model: ChakrMicro,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_active = 0

    with torch.no_grad():
        for r in records:
            x, y = encode_chatml_example(tokenizer, r["prompt"], r["response"])
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            logits = model(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()

            active_mask = (shift_labels != -100)
            n_active = active_mask.sum().item()
            if n_active > 0:
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item() * n_active
                total_active += n_active

    mean_loss = total_loss / max(total_active, 1)
    ppl = math.exp(min(mean_loss, 20.0))
    return {
        "loss": round(mean_loss, 4),
        "perplexity": round(ppl, 2),
        "active_tokens": total_active,
    }


def evaluate_generation_quality(
    model: ChakrMicro,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    device: torch.device = torch.device("cpu"),
    max_new_tokens: int = 35,
) -> Dict[str, Any]:
    model.eval()
    clean_terminations = 0
    exact_matches = 0
    samples = []

    for r in records:
        prompt_tokens = [0] + tokenizer.encode(r["prompt"])
        gen = list(prompt_tokens)
        eos_reached = False

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([gen[-256:]], dtype=torch.long, device=device)
                logits = model(x)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt in (1, 2):
                    eos_reached = True
                    break
                gen.append(nxt)

        new_tokens = gen[len(prompt_tokens):]
        decoded = tokenizer.decode(new_tokens, errors="replace").strip()
        clean_ans = decoded.replace("<|im_end|>", "").strip()
        expected = r.get("expected_answer", "").strip()

        if "<|im_end|>" in decoded or eos_reached:
            clean_terminations += 1

        is_exact = (clean_ans == expected)
        if is_exact:
            exact_matches += 1

        samples.append({
            "id": r["id"],
            "query": r["query"],
            "expected": expected,
            "generated": decoded,
            "exact_match": is_exact,
            "clean_termination": eos_reached or "<|im_end|>" in decoded,
        })

    n = max(len(records), 1)
    return {
        "total": len(records),
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / n, 4),
        "clean_terminations": clean_terminations,
        "clean_termination_rate": round(clean_terminations / n, 4),
        "samples": samples[:5],
    }


def evaluate_retention(
    model: ChakrMicro,
    tokenizer: Any,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, Any]]:
    model.eval()
    results = []

    for item in RETENTION_PROMPTS:
        p = item["prompt"]
        toks = [0] + tokenizer.encode(p)
        gen = list(toks)

        with torch.no_grad():
            for _ in range(32):
                x = torch.tensor([gen[-128:]], dtype=torch.long, device=device)
                logits = model(x)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt in (1, 2):
                    break
                gen.append(nxt)

        continuation = tokenizer.decode(gen[len(toks):], errors="replace")
        new_ids = gen[len(toks):]
        if len(new_ids) >= 3:
            ngrams = [tuple(new_ids[i:i+3]) for i in range(len(new_ids) - 2)]
            rep = 1.0 - (len(set(ngrams)) / max(len(ngrams), 1))
        else:
            rep = 0.0

        results.append({
            "domain": item["domain"],
            "prompt": p,
            "continuation": continuation.strip(),
            "repetition_3gram": round(rep, 4),
        })

    return results


def train_staged_candidate(
    base_model: ChakrMicro,
    tokenizer: Any,
    train_records: List[Dict[str, Any]],
    steps: int,
    warmup_steps: int,
    peak_lr: float = 2e-4,
    min_lr: float = 2e-5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[ChakrMicro, Dict[str, Any]]:
    model = copy.deepcopy(base_model)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_lr = min_lr + (peak_lr - min_lr) * cosine_decay
        return target_lr / peak_lr

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_history = []
    t0 = time.perf_counter()

    for step in range(steps):
        rec = train_records[step % len(train_records)]
        x, y = encode_chatml_example(tokenizer, rec["prompt"], rec["response"])
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        optimizer.zero_grad()
        logits = model(x)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = y[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(round(loss.item(), 4))

    elapsed = time.perf_counter() - t0
    return model, {
        "steps": steps,
        "warmup_steps": warmup_steps,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "train_time_sec": round(elapsed, 2),
        "loss_history_sampled": loss_history[::max(1, steps // 10)],
    }


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")

    train_recs = load_jsonl(DATA_DIR / "generalization_train.jsonl")
    val_recs = load_jsonl(DATA_DIR / "generalization_val.jsonl")
    unseen_ent_recs = load_jsonl(DATA_DIR / "heldout_unseen_entities.jsonl")
    unseen_tmpl_recs = load_jsonl(DATA_DIR / "heldout_unseen_templates.jsonl")

    print(f"Loaded: {len(train_recs)} Train, {len(val_recs)} Val, {len(unseen_ent_recs)} Unseen Ents, {len(unseen_tmpl_recs)} Unseen Tmpls.")

    # 1. Baseline Evaluation
    print("\n--- Evaluating Canonical Baseline (Step 10,000) ---")
    base_model, tokenizer = load_canonical_model(device)
    base_metrics = {
        "val_loss": evaluate_split_loss(base_model, tokenizer, val_recs, device),
        "unseen_ent_loss": evaluate_split_loss(base_model, tokenizer, unseen_ent_recs, device),
        "unseen_tmpl_loss": evaluate_split_loss(base_model, tokenizer, unseen_tmpl_recs, device),
        "gen_unseen_ent": evaluate_generation_quality(base_model, tokenizer, unseen_ent_recs, device),
        "retention": evaluate_retention(base_model, tokenizer, device),
    }
    print(f"Baseline Val Loss: {base_metrics['val_loss']['loss']} (PPL: {base_metrics['val_loss']['perplexity']})")
    print(f"Baseline Unseen Entities Loss: {base_metrics['unseen_ent_loss']['loss']} (PPL: {base_metrics['unseen_ent_loss']['perplexity']})")
    print(f"Baseline Clean Termination: {base_metrics['gen_unseen_ent']['clean_terminations']}/{base_metrics['gen_unseen_ent']['total']}")

    # Staged candidate experiments
    staged_configs = [
        {"name": "candidate_step70", "steps": 70, "warmup": 15},
        {"name": "candidate_step150", "steps": 150, "warmup": 25},
        {"name": "candidate_step300", "steps": 300, "warmup": 30},
    ]

    candidates_results = {}

    for cfg in staged_configs:
        name = cfg["name"]
        steps = cfg["steps"]
        warmup = cfg["warmup"]

        print(f"\n--- Training {name} ({steps} steps, warmup={warmup}) ---")
        cand_model, train_meta = train_staged_candidate(
            base_model, tokenizer, train_recs, steps=steps, warmup_steps=warmup, device=device
        )

        ckpt_file = CKPT_DIR / f"{name}.pt"
        torch.save({
            "name": name,
            "step": steps,
            "config": cand_model.config,
            "model_state_dict": cand_model.state_dict(),
            "train_meta": train_meta,
        }, ckpt_file)
        ckpt_sha = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()

        print(f"Evaluating {name}...")
        c_val = evaluate_split_loss(cand_model, tokenizer, val_recs, device)
        c_ent = evaluate_split_loss(cand_model, tokenizer, unseen_ent_recs, device)
        c_tmpl = evaluate_split_loss(cand_model, tokenizer, unseen_tmpl_recs, device)
        c_gen = evaluate_generation_quality(cand_model, tokenizer, unseen_ent_recs, device)
        c_ret = evaluate_retention(cand_model, tokenizer, device)

        print(f"[{name}] Val Loss: {c_val['loss']} | Unseen Ent Loss: {c_ent['loss']} | Clean Term: {c_gen['clean_terminations']}/{c_gen['total']}")

        candidates_results[name] = {
            "checkpoint": str(ckpt_file),
            "sha256": ckpt_sha,
            "steps": steps,
            "train_meta": train_meta,
            "val_loss": c_val,
            "unseen_entities_loss": c_ent,
            "unseen_templates_loss": c_tmpl,
            "generation_unseen_entities": c_gen,
            "retention": c_ret,
        }

    # Consolidated comparison
    report = {
        "milestone": "Batches 323-330: Controlled Neural Training & Generalization Comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_baseline": {
            "checkpoint": str(CANONICAL_CHECKPOINT),
            "val_loss": base_metrics["val_loss"],
            "unseen_entities_loss": base_metrics["unseen_ent_loss"],
            "unseen_templates_loss": base_metrics["unseen_tmpl_loss"],
            "generation_unseen_entities": base_metrics["gen_unseen_ent"],
            "retention": base_metrics["retention"],
        },
        "candidates": candidates_results,
    }

    out_file = REPORTS_DIR / "training_comparison.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved comprehensive training comparison to: {out_file}")


if __name__ == "__main__":
    main()
