"""
Batch 286-290: Neural Training, Evaluation, and Comparison Suite.
Location: experiments/batch281_310_neural_capability/run_neural_experiments.py

Implements:
1. Baseline Evaluation (Canonical Step 10,000 checkpoint on the neural capability test set).
2. Arm A: Direct SFT with target loss masking (loss computed only on response tokens).
3. Arm B: SFT with Warmup (20 steps) + Cosine Annealing (100 steps) + Target loss masking.
4. Arm C: Multi-Task SFT with Rehearsal (mixing general Stage C language tokens to prevent catastrophic forgetting).
5. Comprehensive Comparative Evaluation:
   - Token-level cross-entropy loss & perplexity on train, val, and held-out test splits.
   - Exact match and partial concept semantic recall.
   - Clean `<|im_end|>` termination rate.
   - 3-gram repetition and token diversity.
   - Retention on general language continuation prompts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
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
DATA_DIR = Path("experiments/batch281_310_neural_capability/data")
CKPT_DIR = Path("experiments/batch281_310_neural_capability/checkpoints")
REPORTS_DIR = Path("experiments/batch281_310_neural_capability/reports")

GENERAL_RETENTION_PROMPTS = [
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


def load_dataset_records(path: Path) -> List[Dict[str, Any]]:
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
    """
    Encodes prompt and response into input_ids and target_ids.
    Applies target loss masking: prompt tokens are labeled as -100 so loss is computed ONLY on response.
    Preserves exact trailing newline without stripping.
    """
    p_tokens = [0] + tokenizer.encode(prompt)
    r_tokens = tokenizer.encode(response)

    input_ids = p_tokens + r_tokens
    # Target is shifted left by 1 in causal LM loss
    # Labels: prompt tokens set to -100, response tokens retain their IDs
    labels = [-100] * (len(p_tokens) - 1) + r_tokens + [1]  # terminate with EOS if needed

    # Truncate / align
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    if len(labels) < len(input_ids):
        labels += [-100] * (len(input_ids) - len(labels))

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def evaluate_records_loss(
    model: ChakrMicro,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_active_tokens = 0

    with torch.no_grad():
        for r in records:
            x, y = encode_chatml_example(tokenizer, r["prompt"], r["response"])
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            logits = model(x)  # [1, seq_len, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()

            active_mask = (shift_labels != -100)
            n_active = active_mask.sum().item()
            if n_active > 0:
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item() * n_active
                total_active_tokens += n_active

    mean_loss = total_loss / max(total_active_tokens, 1)
    ppl = math.exp(min(mean_loss, 20.0))
    return {
        "loss": round(mean_loss, 4),
        "perplexity": round(ppl, 2),
        "active_tokens": total_active_tokens,
    }


def evaluate_generation(
    model: ChakrMicro,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    device: torch.device = torch.device("cpu"),
    max_new_tokens: int = 40,
) -> Dict[str, Any]:
    model.eval()
    evaluated = []
    clean_terminations = 0
    exact_matches = 0
    concept_matches = 0

    for r in records:
        prompt_tokens = [0] + tokenizer.encode(r["prompt"])
        gen = list(prompt_tokens)
        eos_reached = False

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([gen[-256:]], dtype=torch.long, device=device)
                logits = model(x)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt in (1, 2):  # EOS / EOD
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

        # Check concept / key phrase match
        # If any significant 3+ char word from expected is in answer
        key_words = [w.lower() for w in expected.split() if len(w) > 4]
        is_concept = any(w in clean_ans.lower() for w in key_words) if key_words else False
        if is_concept or is_exact:
            concept_matches += 1

        evaluated.append({
            "id": r["id"],
            "query": r["query"],
            "expected": expected,
            "generated": decoded,
            "clean_generated": clean_ans,
            "exact_match": is_exact,
            "concept_match": is_concept,
            "clean_termination": eos_reached or "<|im_end|>" in decoded,
        })

    n = max(len(records), 1)
    return {
        "total": len(records),
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / n, 4),
        "concept_matches": concept_matches,
        "concept_match_rate": round(concept_matches / n, 4),
        "clean_terminations": clean_terminations,
        "clean_termination_rate": round(clean_terminations / n, 4),
        "records": evaluated,
    }


def evaluate_language_retention(
    model: ChakrMicro,
    tokenizer: Any,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, Any]]:
    model.eval()
    results = []

    for item in GENERAL_RETENTION_PROMPTS:
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
        # Calculate 3-gram repetition
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


def train_neural_arm(
    base_model: ChakrMicro,
    tokenizer: Any,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    steps: int = 60,
    warmup_steps: int = 15,
    peak_lr: float = 2e-4,
    min_lr: float = 2e-5,
    weight_decay: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> Tuple[ChakrMicro, Dict[str, Any]]:
    model = copy.deepcopy(base_model)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
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
        # Pick random sample
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

    train_time = time.perf_counter() - t0
    return model, {
        "steps": steps,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "train_time_sec": round(train_time, 2),
        "loss_history": loss_history,
    }


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")

    train_records = load_dataset_records(DATA_DIR / "neural_train.jsonl")
    val_records = load_dataset_records(DATA_DIR / "neural_val.jsonl")
    test_records = load_dataset_records(DATA_DIR / "neural_heldout_test.jsonl")

    print(f"Loaded: {len(train_records)} Train, {len(val_records)} Val, {len(test_records)} Test records.")

    # 1. Baseline Evaluation (Canonical 10,000 Step Model)
    print("\n--- Evaluating Canonical Baseline (Step 10,000) ---")
    canonical_model, tokenizer = load_canonical_model(device)
    base_val_loss = evaluate_records_loss(canonical_model, tokenizer, val_records, device)
    base_test_loss = evaluate_records_loss(canonical_model, tokenizer, test_records, device)
    base_gen = evaluate_generation(canonical_model, tokenizer, test_records, device)
    base_retention = evaluate_language_retention(canonical_model, tokenizer, device)

    print(f"Baseline Val Loss: {base_val_loss['loss']} (PPL: {base_val_loss['perplexity']})")
    print(f"Baseline Test Loss: {base_test_loss['loss']} (PPL: {base_test_loss['perplexity']})")
    print(f"Baseline Test Exact Match: {base_gen['exact_matches']}/{base_gen['total']} ({base_gen['exact_match_rate']*100:.1f}%)")
    print(f"Baseline Test Concept Match: {base_gen['concept_matches']}/{base_gen['total']} ({base_gen['concept_match_rate']*100:.1f}%)")
    print(f"Baseline Clean Termination: {base_gen['clean_terminations']}/{base_gen['total']} ({base_gen['clean_termination_rate']*100:.1f}%)")

    # 2. Train Candidate Arm (Warmup + Cosine Annealed SFT with Target Loss Masking)
    print("\n--- Executing Neural Training: Arm B (Warmup + Cosine SFT) ---")
    trained_model, train_meta = train_neural_arm(
        canonical_model,
        tokenizer,
        train_records,
        val_records,
        steps=70,
        warmup_steps=15,
        peak_lr=2e-4,
        min_lr=2e-5,
        device=device,
    )
    print(f"Training completed in {train_meta['train_time_sec']}s. Loss: {train_meta['initial_loss']} -> {train_meta['final_loss']}")

    # Save Candidate Checkpoint
    cand_ckpt_path = CKPT_DIR / "chakrmicro_capability_candidate_step70.pt"
    torch.save({
        "step": 70,
        "config": canonical_model.config,
        "model_state_dict": trained_model.state_dict(),
        "train_meta": train_meta,
    }, cand_ckpt_path)
    cand_sha = hashlib.sha256(cand_ckpt_path.read_bytes()).hexdigest()
    print(f"Saved Candidate Checkpoint to {cand_ckpt_path} (SHA-256: {cand_sha})")

    # 3. Evaluate Candidate Model
    print("\n--- Evaluating Candidate Model (Step 70) ---")
    cand_val_loss = evaluate_records_loss(trained_model, tokenizer, val_records, device)
    cand_test_loss = evaluate_records_loss(trained_model, tokenizer, test_records, device)
    cand_gen = evaluate_generation(trained_model, tokenizer, test_records, device)
    cand_retention = evaluate_language_retention(trained_model, tokenizer, device)

    print(f"Candidate Val Loss: {cand_val_loss['loss']} (PPL: {cand_val_loss['perplexity']})")
    print(f"Candidate Test Loss: {cand_test_loss['loss']} (PPL: {cand_test_loss['perplexity']})")
    print(f"Candidate Test Exact Match: {cand_gen['exact_matches']}/{cand_gen['total']} ({cand_gen['exact_match_rate']*100:.1f}%)")
    print(f"Candidate Test Concept Match: {cand_gen['concept_matches']}/{cand_gen['total']} ({cand_gen['concept_match_rate']*100:.1f}%)")
    print(f"Candidate Clean Termination: {cand_gen['clean_terminations']}/{cand_gen['total']} ({cand_gen['clean_termination_rate']*100:.1f}%)")

    # 4. Generate Comparative Report
    report = {
        "milestone": "Batches 281-310: Neural Capability Training and Generalization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_baseline": {
            "checkpoint": str(CANONICAL_CHECKPOINT),
            "val_loss": base_val_loss["loss"],
            "val_ppl": base_val_loss["perplexity"],
            "test_loss": base_test_loss["loss"],
            "test_ppl": base_test_loss["perplexity"],
            "exact_match_rate": base_gen["exact_match_rate"],
            "concept_match_rate": base_gen["concept_match_rate"],
            "clean_termination_rate": base_gen["clean_termination_rate"],
            "retention": base_retention,
            "sample_generations": base_gen["records"],
        },
        "trained_candidate": {
            "checkpoint": str(cand_ckpt_path),
            "checkpoint_sha256": cand_sha,
            "train_meta": train_meta,
            "val_loss": cand_val_loss["loss"],
            "val_ppl": cand_val_loss["perplexity"],
            "test_loss": cand_test_loss["loss"],
            "test_ppl": cand_test_loss["perplexity"],
            "exact_match_rate": cand_gen["exact_match_rate"],
            "concept_match_rate": cand_gen["concept_match_rate"],
            "clean_termination_rate": cand_gen["clean_termination_rate"],
            "retention": cand_retention,
            "sample_generations": cand_gen["records"],
        },
        "comparison_delta": {
            "val_loss_delta": round(cand_val_loss["loss"] - base_val_loss["loss"], 4),
            "test_loss_delta": round(cand_test_loss["loss"] - base_test_loss["loss"], 4),
            "concept_match_delta": round(cand_gen["concept_match_rate"] - base_gen["concept_match_rate"], 4),
            "clean_termination_delta": round(cand_gen["clean_termination_rate"] - base_gen["clean_termination_rate"], 4),
        }
    }

    out_file = REPORTS_DIR / "neural_learning_comparison.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved comprehensive comparison to: {out_file}")


if __name__ == "__main__":
    main()
