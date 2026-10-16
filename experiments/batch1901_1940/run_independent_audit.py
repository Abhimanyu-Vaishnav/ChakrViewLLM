#!/usr/bin/env python3
"""
ChakrView Batches 1901–1940: Independent Audit Runner
Reproducible evaluation script for:
1. Exact replication of Batch 1861 metrics on original held-out corpus (48 docs, 12,479 tokens).
2. Clean evaluation on source-disjoint independent benchmark (88 docs, 11,110 tokens).
3. Condition D (300-step) training and persistence to disk for audit reproducibility.
4. Practical transfer evaluation across 13 diagnostic tasks.
5. Paired bootstrap statistics and per-domain dispersion.
"""

import copy
import hashlib
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
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

BATCH1861_DIR = REPO_ROOT / "experiments" / "batch1861_1900"
BATCH1901_DIR = REPO_ROOT / "experiments" / "batch1901_1940"


def load_model_from_checkpoint(ckpt_path: Path) -> Tuple[ChakrMicro, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    canon_ckpt = torch.load(CANONICAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = canon_ckpt["config"]

    model = ChakrMicro(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config


def evaluate_language_modeling(
    model: ChakrMicro,
    tokenizer: Any,
    documents: List[Dict[str, Any]],
    max_chunk_len: int = 256
) -> Dict[str, Any]:
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

            doc_nll = 0.0
            doc_tokens = 0
            doc_top1 = 0
            doc_top5 = 0

            for i in range(0, len(token_ids) - 1, max_chunk_len):
                chunk = token_ids[i:i + max_chunk_len + 1]
                if len(chunk) < 2:
                    continue
                inp = torch.tensor([chunk[:-1]], dtype=torch.long)
                tgt = torch.tensor([chunk[1:]], dtype=torch.long)

                logits = model(inp)
                vocab_size = logits.size(-1)

                loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1), reduction="sum")
                n_tok = tgt.numel()

                preds_top5 = torch.topk(logits, k=min(5, vocab_size), dim=-1).indices
                preds_top1 = preds_top5[:, :, 0]

                matches_top1 = (preds_top1 == tgt).sum().item()
                matches_top5 = (preds_top5 == tgt.unsqueeze(-1)).any(dim=-1).sum().item()

                doc_nll += loss.item()
                doc_tokens += n_tok
                doc_top1 += matches_top1
                doc_top5 += matches_top5

            total_nll += doc_nll
            total_tokens += doc_tokens
            top1_correct += doc_top1
            top5_correct += doc_top5

            avg_doc_loss = doc_nll / doc_tokens if doc_tokens > 0 else 0.0
            doc_ppl = math.exp(avg_doc_loss) if avg_doc_loss < 20 else float("inf")
            per_doc_records.append({
                "doc_id": doc_id,
                "domain": domain,
                "tokens": doc_tokens,
                "loss": round(avg_doc_loss, 4),
                "perplexity": round(doc_ppl, 2),
                "top1_acc": round(doc_top1 / doc_tokens, 4) if doc_tokens > 0 else 0.0
            })

            if domain not in per_domain_stats:
                per_domain_stats[domain] = {
                    "total_nll": 0.0,
                    "total_tokens": 0,
                    "top1_correct": 0,
                    "top5_correct": 0
                }
            per_domain_stats[domain]["total_nll"] += doc_nll
            per_domain_stats[domain]["total_tokens"] += doc_tokens
            per_domain_stats[domain]["top1_correct"] += doc_top1
            per_domain_stats[domain]["top5_correct"] += doc_top5

    mean_ce = total_nll / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(mean_ce) if mean_ce < 20 else float("inf")
    top1_acc = top1_correct / total_tokens if total_tokens > 0 else 0.0
    top5_acc = top5_correct / total_tokens if total_tokens > 0 else 0.0

    domain_breakdown = {}
    for dom, st in per_domain_stats.items():
        dom_loss = st["total_nll"] / st["total_tokens"] if st["total_tokens"] > 0 else 0.0
        dom_ppl = math.exp(dom_loss) if dom_loss < 20 else float("inf")
        domain_breakdown[dom] = {
            "loss": round(dom_loss, 4),
            "perplexity": round(dom_ppl, 2),
            "top1_acc": round(st["top1_correct"] / st["total_tokens"], 4) if st["total_tokens"] > 0 else 0.0,
            "top5_acc": round(st["top5_correct"] / st["total_tokens"], 4) if st["total_tokens"] > 0 else 0.0,
            "tokens": st["total_tokens"]
        }

    return {
        "total_tokens_evaluated": total_tokens,
        "mean_cross_entropy": round(mean_ce, 4),
        "perplexity": round(ppl, 2),
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "domain_breakdown": domain_breakdown,
        "documents": per_doc_records
    }


def evaluate_practical_transfer_suite(model: ChakrMicro, tokenizer: Any) -> Dict[str, Any]:
    model.eval()
    test_cases = [
        # Arithmetic
        {"cat": "arithmetic", "prompt": "15 + 25 = ", "expected": "40"},
        {"cat": "arithmetic", "prompt": "12 * 8 = ", "expected": "96"},
        {"cat": "arithmetic", "prompt": "100 - 35 = ", "expected": "65"},
        {"cat": "arithmetic", "prompt": "81 / 9 = ", "expected": "9"},
        # Multi-Step Arithmetic
        {"cat": "multi_step_arithmetic", "prompt": "Calculate: (10 + 5) * 4 = ", "expected": "60"},
        {"cat": "multi_step_arithmetic", "prompt": "First add 20 and 30, then multiply by 2: ", "expected": "100"},
        # Code Completion
        {"cat": "code_completion", "prompt": "def multiply(x, y):\n    return ", "expected": "x * y"},
        {"cat": "code_completion", "prompt": "class Queue:\n    def __init__(self):\n        self.items = ", "expected": "[]"},
        {"cat": "code_completion", "prompt": "import json\ndata = json.loads(", "expected": "text)"},
        # Propositions
        {"cat": "propositions", "prompt": "ChakrMicro unique trainable parameter count is ", "expected": "3443136"},
        {"cat": "propositions", "prompt": "The safe calculator strictly forbids python eval and ", "expected": "exec"},
        # Hinglish Continuation
        {"cat": "hinglish", "prompt": "ChakrView model local hardware par bina API ke ", "expected": "run"},
        {"cat": "hinglish", "prompt": "Python sandbox ke andar forbidden modules ko ", "expected": "block"},
    ]

    records = []
    correct_count = 0

    with torch.no_grad():
        for tc in test_cases:
            prompt = tc["prompt"]
            expected = tc["expected"]

            p_ids = tokenizer.encode(prompt)
            exp_ids = tokenizer.encode(expected)

            inp = torch.tensor([p_ids], dtype=torch.long)
            logits = model(inp)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            pred_tok_str = tokenizer.decode([next_tok]).strip()

            exp_tok_str = tokenizer.decode([exp_ids[0]]).strip() if exp_ids else ""
            is_match = (pred_tok_str == exp_tok_str) or (expected.startswith(pred_tok_str) and len(pred_tok_str) > 0)
            if is_match:
                correct_count += 1

            records.append({
                "category": tc["cat"],
                "prompt": prompt,
                "expected": expected,
                "predicted_first_token": pred_tok_str,
                "match": is_match
            })

    return {
        "total_transfer_tasks": len(test_cases),
        "passed": correct_count,
        "transfer_accuracy": round(correct_count / len(test_cases), 4),
        "task_records": records
    }


def train_cpt_engine(
    dataset_docs: List[Dict[str, Any]],
    tokenizer: Any,
    seed: int = 42,
    steps: int = 150,
    lr: float = 2e-5
) -> Tuple[ChakrMicro, float]:
    random.seed(seed)
    torch.manual_seed(seed)

    model, _ = load_model_from_checkpoint(CANONICAL_CHECKPOINT_PATH)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    tokenized_docs = [tokenizer.encode(d["text"]) for d in dataset_docs]

    t0 = time.time()
    for step in range(1, steps + 1):
        batch = random.sample(tokenized_docs, min(4, len(tokenized_docs)))
        max_l = min(max(len(b) for b in batch), 128)

        padded_inp = []
        padded_tgt = []
        for seq in batch:
            crop = seq[:max_l]
            if len(crop) < 2:
                crop = [0, 1]
            padded_inp.append(crop[:-1] + [2] * (max_l - 1 - len(crop[:-1])))
            padded_tgt.append(crop[1:] + [-100] * (max_l - 1 - len(crop[1:])))

        inp_t = torch.tensor(padded_inp, dtype=torch.long)
        tgt_t = torch.tensor(padded_tgt, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(inp_t)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_t.view(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()

    wall_time = round(time.time() - t0, 2)
    return model, wall_time


def compute_paired_bootstrap(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    rng = random.Random(seed)
    # Match docs by doc_id
    map_b = {r["doc_id"]: r["loss"] for r in records_b}
    paired = []
    for ra in records_a:
        did = ra["doc_id"]
        if did in map_b:
            paired.append((ra["loss"], map_b[did]))

    if not paired:
        return {}

    deltas = [b - a for a, b in paired]
    mean_delta = sum(deltas) / len(deltas)

    boot_means = []
    n = len(deltas)
    for _ in range(n_boot):
        sample = [rng.choice(deltas) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    ci_lower = boot_means[int(0.025 * n_boot)]
    ci_upper = boot_means[int(0.975 * n_boot)]

    return {
        "paired_doc_count": n,
        "mean_delta_loss": round(mean_delta, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4)
    }


def main():
    print("==================================================================")
    print("CHAKRVIEW BATCHES 1901–1940: INDEPENDENT EVALUATION AUDIT RUNNER")
    print("==================================================================")

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Load test sets
    batch1861_test_path = BATCH1861_DIR / "data" / "test_held_out_expanded_corpus.json"
    with open(batch1861_test_path, "r", encoding="utf-8") as f:
        batch1861_test_docs = json.load(f)

    independent_test_path = BATCH1901_DIR / "data" / "source_disjoint_corpus.json"
    with open(independent_test_path, "r", encoding="utf-8") as f:
        indep_test_docs = json.load(f)

    print(f"Loaded Batch 1861 original test set: {len(batch1861_test_docs)} docs.")
    print(f"Loaded Independent source-disjoint test set: {len(indep_test_docs)} docs.")

    # 2. Checkpoint registry
    checkpoints = {
        "canonical": CANONICAL_CHECKPOINT_PATH,
        "cpt_balanced_seed_42": BATCH1861_DIR / "checkpoints" / "cpt_balanced_seed_42.pt",
        "cpt_balanced_seed_123": BATCH1861_DIR / "checkpoints" / "cpt_balanced_seed_123.pt",
        "cpt_balanced_seed_999": BATCH1861_DIR / "checkpoints" / "cpt_balanced_seed_999.pt"
    }

    # 3. Train and persist Condition D (300-step) checkpoint to resolve unpersisted artifact flaw
    batch1861_train_path = BATCH1861_DIR / "data" / "train_expanded_corpus.json"
    with open(batch1861_train_path, "r", encoding="utf-8") as f:
        train_docs = json.load(f)

    cpt_300_ckpt_path = BATCH1901_DIR / "checkpoints" / "cpt_balanced_seed_42_300steps.pt"
    cpt_300_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if not cpt_300_ckpt_path.exists():
        print("\n[Condition D] Training and persisting 300-step checkpoint for audit verification...")
        m_300, wtime_300 = train_cpt_engine(train_docs, tokenizer, seed=42, steps=300, lr=2e-5)
        torch.save({
            "step": 300,
            "seed": 42,
            "model_state_dict": m_300.state_dict(),
            "wall_time_sec": wtime_300,
            "description": "Condition D Balanced CPT extended training checkpoint (300 steps)"
        }, cpt_300_ckpt_path)
        print(f"  Saved 300-step checkpoint to {cpt_300_ckpt_path}")
        del m_300

    checkpoints["cpt_balanced_seed_42_300steps"] = cpt_300_ckpt_path

    # 4. Evaluation on original Batch 1861 test set (Verification of reported reproduction)
    print("\n--- PHASE 2: Evaluating Checkpoints on Batch 1861 Original Test Set ---")
    orig_results = {}
    for name, p in checkpoints.items():
        m, _ = load_model_from_checkpoint(p)
        eval_metrics = evaluate_language_modeling(m, tokenizer, batch1861_test_docs)
        transfer_metrics = evaluate_practical_transfer_suite(m, tokenizer)
        orig_results[name] = {
            "metrics": eval_metrics,
            "transfer": transfer_metrics
        }
        print(f"  {name:30s} | CE: {eval_metrics['mean_cross_entropy']:.4f} | PPL: {eval_metrics['perplexity']:6.2f} | Top1: {eval_metrics['top1_accuracy']:.4f}")
        del m

    # 5. Evaluation on Source-Disjoint Independent Benchmark
    print("\n--- PHASE 4 & 5: Evaluating Checkpoints on Source-Disjoint Benchmark ---")
    indep_results = {}
    for name, p in checkpoints.items():
        m, _ = load_model_from_checkpoint(p)
        eval_metrics = evaluate_language_modeling(m, tokenizer, indep_test_docs)
        transfer_metrics = evaluate_practical_transfer_suite(m, tokenizer)
        indep_results[name] = {
            "metrics": eval_metrics,
            "transfer": transfer_metrics
        }
        print(f"  {name:30s} | CE: {eval_metrics['mean_cross_entropy']:.4f} | PPL: {eval_metrics['perplexity']:6.2f} | Top1: {eval_metrics['top1_accuracy']:.4f}")
        del m

    # 6. Bootstrap comparison vs Canonical on both benchmarks
    bootstrap_stats = {"original_test": {}, "independent_test": {}}
    for name in ["cpt_balanced_seed_42", "cpt_balanced_seed_123", "cpt_balanced_seed_999", "cpt_balanced_seed_42_300steps"]:
        bootstrap_stats["original_test"][name] = compute_paired_bootstrap(
            orig_results["canonical"]["metrics"]["documents"],
            orig_results[name]["metrics"]["documents"]
        )
        bootstrap_stats["independent_test"][name] = compute_paired_bootstrap(
            indep_results["canonical"]["metrics"]["documents"],
            indep_results[name]["metrics"]["documents"]
        )

    # 7. Compile Final Results Artifact
    audit_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mission": "batch1901_1940",
        "batch1861_original_reproduction": {
            "reported_claims": {
                "canonical_cross_entropy": 5.7817,
                "cpt_150step_mean_cross_entropy": 4.0067,
                "cpt_300step_cross_entropy": 3.0989
            },
            "audit_evaluated": orig_results
        },
        "source_disjoint_independent_evaluation": indep_results,
        "paired_bootstrap_differences_vs_canonical": bootstrap_stats
    }

    out_file = BATCH1901_DIR / "INDEPENDENT_EVALUATION_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nSaved complete audit evaluation results to: {out_file}")


if __name__ == "__main__":
    main()
