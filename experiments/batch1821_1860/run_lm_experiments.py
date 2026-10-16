"""
Batch 1821-1860: Language Model Evaluation & Continued Pretraining Experiment Engine.
Implements:
1. Per-document cross-entropy loss, perplexity, and Top-1/Top-5 next-token accuracy.
2. Canonical Baseline (A) evaluation on held-out test corpus.
3. Controlled Continued Pretraining (B) on train_corpus.json across 3 seeds (42, 123, 999).
4. Data-matched control (C) with shuffled document ordering.
5. Transfer tasks diagnostic evaluation (Arithmetic, Code Completion, Propositions, Hinglish).
"""
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
DATA_DIR = REPO_ROOT / "experiments" / "batch1821_1860" / "data"
RESULTS_DIR = REPO_ROOT / "experiments" / "batch1821_1860" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_canonical_model() -> Tuple[ChakrMicro, Any]:
    ckpt = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, config


def evaluate_language_modeling(model: ChakrMicro, tokenizer: Any, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    model.eval()
    total_tokens = 0
    total_nll = 0.0
    top1_correct = 0
    top5_correct = 0

    per_domain_stats = {}
    per_doc_records = []

    with torch.no_grad():
        for doc in documents:
            text = doc["text"]
            domain = doc["domain"]
            doc_id = doc["doc_id"]

            token_ids = tokenizer.encode(text)
            if len(token_ids) < 2:
                continue

            input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)
            target_ids = torch.tensor([token_ids[1:]], dtype=torch.long)

            logits = model(input_ids) # [1, seq_len, vocab_size]
            vocab_size = logits.size(-1)

            loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1), reduction="sum")
            n_tokens = target_ids.numel()

            total_nll += loss.item()
            total_tokens += n_tokens

            # Top-1 and Top-5 accuracy
            preds_top5 = torch.topk(logits, k=min(5, vocab_size), dim=-1).indices # [1, seq_len, 5]
            preds_top1 = preds_top5[:, :, 0]

            matches_top1 = (preds_top1 == target_ids).sum().item()
            matches_top5 = (preds_top5 == target_ids.unsqueeze(-1)).any(dim=-1).sum().item()

            top1_correct += matches_top1
            top5_correct += matches_top5

            doc_loss = loss.item() / n_tokens
            doc_ppl = math.exp(min(doc_loss, 20.0))

            per_doc_records.append({
                "doc_id": doc_id,
                "domain": domain,
                "token_count": n_tokens,
                "loss": round(doc_loss, 4),
                "perplexity": round(doc_ppl, 2),
                "top1_acc": round(matches_top1 / n_tokens, 4),
                "top5_acc": round(matches_top5 / n_tokens, 4)
            })

            # Domain stats
            if domain not in per_domain_stats:
                per_domain_stats[domain] = {"nll": 0.0, "tokens": 0, "top1": 0, "top5": 0}
            per_domain_stats[domain]["nll"] += loss.item()
            per_domain_stats[domain]["tokens"] += n_tokens
            per_domain_stats[domain]["top1"] += matches_top1
            per_domain_stats[domain]["top5"] += matches_top5

    avg_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(min(avg_loss, 20.0))
    overall_top1 = top1_correct / total_tokens if total_tokens > 0 else 0.0
    overall_top5 = top5_correct / total_tokens if total_tokens > 0 else 0.0

    domain_summary = {}
    for d, st in per_domain_stats.items():
        d_loss = st["nll"] / st["tokens"]
        domain_summary[d] = {
            "loss": round(d_loss, 4),
            "perplexity": round(math.exp(min(d_loss, 20.0)), 2),
            "top1_acc": round(st["top1"] / st["tokens"], 4),
            "top5_acc": round(st["top5"] / st["tokens"], 4),
            "tokens": st["tokens"]
        }

    return {
        "total_tokens_evaluated": total_tokens,
        "mean_cross_entropy": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "top1_accuracy": round(overall_top1, 4),
        "top5_accuracy": round(overall_top5, 4),
        "domain_breakdown": domain_summary,
        "documents": per_doc_records
    }


def evaluate_diagnostic_transfer_tasks(model: ChakrMicro, tokenizer: Any) -> Dict[str, Any]:
    model.eval()
    tasks = [
        # Arithmetic
        {"type": "arithmetic", "prompt": "Calculate: 15 * 4 = ", "target": "60"},
        {"type": "arithmetic", "prompt": "Calculate: 99 - 33 = ", "target": "66"},
        {"type": "arithmetic", "prompt": "Calculate: 50 + 25 = ", "target": "75"},
        # Code Completion
        {"type": "code", "prompt": "def add(a, b):\n    return ", "target": "a + b"},
        {"type": "code", "prompt": "for i in range(10):\n    print(", "target": "i)"},
        # Proposition Continuation
        {"type": "propositions", "prompt": "ChakrMicro unique trainable parameter count is ", "target": "3443136"},
        {"type": "propositions", "prompt": "The safe calculator forbids python eval and ", "target": "exec"},
        # Hinglish Continuation
        {"type": "hinglish", "prompt": "ChakrView model local hardware par ", "target": "run"},
    ]

    results = []
    correct_count = 0

    with torch.no_grad():
        for t in tasks:
            prompt = t["prompt"]
            target = t["target"]

            prompt_ids = tokenizer.encode(prompt)
            target_ids = tokenizer.encode(target)

            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            decoded_next = tokenizer.decode([next_token]).strip()

            # Target first token
            target_first = tokenizer.decode([target_ids[0]]).strip() if target_ids else ""
            is_match = (decoded_next == target_first) or (target in decoded_next)
            if is_match:
                correct_count += 1

            results.append({
                "type": t["type"],
                "prompt": prompt,
                "expected": target,
                "predicted_first_token": decoded_next,
                "match": is_match
            })

    return {
        "total_tasks": len(tasks),
        "correct": correct_count,
        "transfer_accuracy": round(correct_count / len(tasks), 4),
        "task_breakdown": results
    }


def run_continued_pretraining(
    train_docs: List[Dict[str, Any]],
    val_docs: List[Dict[str, Any]],
    test_docs: List[Dict[str, Any]],
    tokenizer: Any,
    seed: int = 42,
    steps: int = 150,
    lr: float = 2e-5,
    is_shuffled_control: bool = False
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)

    model, config = load_canonical_model()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = list(train_docs)
    if is_shuffled_control:
        random.shuffle(dataset)

    train_tokens = [tokenizer.encode(d["text"]) for d in dataset]

    t0 = time.time()
    for step in range(1, steps + 1):
        # Sample mini-batch of 4 sequences
        batch_seqs = random.sample(train_tokens, min(4, len(train_tokens)))
        max_l = min(max(len(s) for s in batch_seqs), 128)

        padded_inputs = []
        padded_targets = []
        for s in batch_seqs:
            s_crop = s[:max_l]
            if len(s_crop) < 2:
                s_crop = [0, 1]
            inp = s_crop[:-1] + [2] * (max_l - 1 - len(s_crop[:-1]))
            tgt = s_crop[1:] + [-100] * (max_l - 1 - len(s_crop[1:]))
            padded_inputs.append(inp)
            padded_targets.append(tgt)

        inp_tensor = torch.tensor(padded_inputs, dtype=torch.long)
        tgt_tensor = torch.tensor(padded_targets, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(inp_tensor)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_tensor.view(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()

    wall_time = round(time.time() - t0, 2)

    # Evaluate on held-out test
    eval_metrics = evaluate_language_modeling(model, tokenizer, test_docs)
    transfer_metrics = evaluate_diagnostic_transfer_tasks(model, tokenizer)

    return {
        "seed": seed,
        "is_shuffled_control": is_shuffled_control,
        "steps": steps,
        "wall_time_sec": wall_time,
        "held_out_metrics": eval_metrics,
        "transfer_metrics": transfer_metrics,
        "trained_model": model
    }


def execute_batch1821_experiments():
    print("==================================================================")
    print("CHAKRVIEW BATCHES 1821–1860: FOUNDATIONAL LANGUAGE MODEL AUDIT")
    print("==================================================================")

    # 1. Load Tokenizer
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 2. Load Datasets
    with open(DATA_DIR / "train_corpus.json", "r", encoding="utf-8") as f:
        train_docs = json.load(f)
    with open(DATA_DIR / "val_corpus.json", "r", encoding="utf-8") as f:
        val_docs = json.load(f)
    with open(DATA_DIR / "test_held_out_corpus.json", "r", encoding="utf-8") as f:
        test_docs = json.load(f)

    print(f"Loaded: {len(train_docs)} train docs, {len(val_docs)} val docs, {len(test_docs)} held-out test docs.")

    # 3. Canonical Baseline Evaluation
    print("\n[Baseline A] Evaluating Canonical ChakrMicro Step 10,000 Checkpoint...")
    canon_model, _ = load_canonical_model()
    canon_heldout_eval = evaluate_language_modeling(canon_model, tokenizer, test_docs)
    canon_transfer_eval = evaluate_diagnostic_transfer_tasks(canon_model, tokenizer)

    print(f"  Canonical Held-Out Loss: {canon_heldout_eval['mean_cross_entropy']:.4f} (PPL: {canon_heldout_eval['perplexity']:.2f})")
    print(f"  Top-1 Next-Token Acc:   {canon_heldout_eval['top1_accuracy']*100:.2f}% | Top-5: {canon_heldout_eval['top5_accuracy']*100:.2f}%")
    print(f"  Transfer Tasks Acc:     {canon_transfer_eval['transfer_accuracy']*100:.2f}%")

    # 4. Multi-Seed Continued Pretraining (Seeds 42, 123, 999)
    seeds = [42, 123, 999]
    cpt_runs = []
    print("\n[Candidate B] Running Continued Pretraining across 3 Seeds...")
    for s in seeds:
        res = run_continued_pretraining(train_docs, val_docs, test_docs, tokenizer, seed=s, steps=120, lr=2e-5)
        # Save checkpoint
        ckpt_p = REPO_ROOT / "experiments" / "batch1821_1860" / "checkpoints" / f"chakrmicro_cpt_seed_{s}.pt"
        torch.save(res["trained_model"].state_dict(), ckpt_p)
        del res["trained_model"] # free memory
        cpt_runs.append(res)
        hm = res["held_out_metrics"]
        print(f"  Seed {s}: Held-Out Loss={hm['mean_cross_entropy']:.4f} (PPL: {hm['perplexity']:.2f}), Top-1 Acc={hm['top1_accuracy']*100:.2f}%")

    # 5. Data-Matched Shuffled Control (Candidate C)
    print("\n[Control C] Running Data-Matched Shuffled Document Order Control...")
    control_res = run_continued_pretraining(train_docs, val_docs, test_docs, tokenizer, seed=42, steps=120, lr=2e-5, is_shuffled_control=True)
    ctrl_hm = control_res["held_out_metrics"]
    del control_res["trained_model"]
    print(f"  Control C: Held-Out Loss={ctrl_hm['mean_cross_entropy']:.4f} (PPL: {ctrl_hm['perplexity']:.2f}), Top-1 Acc={ctrl_hm['top1_accuracy']*100:.2f}%")

    # Compute Summary Statistics
    losses = [r["held_out_metrics"]["mean_cross_entropy"] for r in cpt_runs]
    mean_loss = sum(losses) / len(losses)
    std_loss = math.sqrt(sum((l - mean_loss) ** 2 for l in losses) / len(losses))

    ppls = [r["held_out_metrics"]["perplexity"] for r in cpt_runs]
    mean_ppl = sum(ppls) / len(ppls)

    top1s = [r["held_out_metrics"]["top1_accuracy"] for r in cpt_runs]
    mean_top1 = sum(top1s) / len(top1s)

    experiment_artifacts = {
        "canonical_baseline": {
            "held_out_metrics": canon_heldout_eval,
            "transfer_metrics": canon_transfer_eval
        },
        "continued_pretraining_summary": {
            "mean_held_out_loss": round(mean_loss, 4),
            "std_held_out_loss": round(std_loss, 4),
            "mean_held_out_ppl": round(mean_ppl, 2),
            "mean_top1_acc": round(mean_top1, 4),
            "delta_loss_vs_canonical": round(mean_loss - canon_heldout_eval["mean_cross_entropy"], 4),
            "runs": cpt_runs
        },
        "data_matched_control": {
            "held_out_metrics": ctrl_hm,
            "transfer_metrics": control_res["transfer_metrics"]
        }
    }

    out_file = RESULTS_DIR / "LANGUAGE_MODELING_EXPERIMENT_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_artifacts, f, indent=2)

    print(f"\nSaved all experimental metrics to: {out_file}")
    print(f"Canonical Loss: {canon_heldout_eval['mean_cross_entropy']:.4f} vs CPT Mean Loss: {mean_loss:.4f} (Delta: {mean_loss - canon_heldout_eval['mean_cross_entropy']:+.4f})")

if __name__ == "__main__":
    execute_batch1821_experiments()
