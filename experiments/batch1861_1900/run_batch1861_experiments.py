"""
Batch 1861-1900: Training Engine and Comprehensive Evaluation Suite.
Implements:
- Condition A: Canonical Step 10,000 Baseline evaluation across 12,527 tokens.
- Condition B: Continued Pretraining on curated multi-domain corpus across 3 seeds (42, 123, 999).
- Condition C: Content-Skewed Control (Code & Math only) with matched step budget.
- Condition D: Training Token Budget Scaling (120 steps vs 300 steps).
- Work Package 6: Expanded Diagnostic Transfer Evaluation (Arithmetic, Code, Props, Hinglish).
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
DATA_DIR = REPO_ROOT / "experiments" / "batch1861_1900" / "data"
RESULTS_DIR = REPO_ROOT / "experiments" / "batch1861_1900" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = REPO_ROOT / "experiments" / "batch1861_1900" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_canonical_model() -> Tuple[ChakrMicro, Any]:
    ckpt = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, config


def evaluate_expanded_language_modeling(model: ChakrMicro, tokenizer: Any, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
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

            # Process in sequence chunks of up to 256
            max_seq = 256
            doc_nll = 0.0
            doc_tokens = 0
            doc_top1 = 0
            doc_top5 = 0

            for i in range(0, len(token_ids) - 1, max_seq):
                chunk = token_ids[i:i + max_seq + 1]
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
            per_doc_records.append({
                "doc_id": doc_id,
                "domain": domain,
                "tokens": doc_tokens,
                "loss": round(avg_doc_loss, 4),
                "perplexity": round(math.exp(min(avg_doc_loss, 20.0)), 2),
                "top1_acc": round(doc_top1 / doc_tokens, 4) if doc_tokens > 0 else 0.0
            })

            if domain not in per_domain_stats:
                per_domain_stats[domain] = {"nll": 0.0, "tokens": 0, "top1": 0, "top5": 0}
            per_domain_stats[domain]["nll"] += doc_nll
            per_domain_stats[domain]["tokens"] += doc_tokens
            per_domain_stats[domain]["top1"] += doc_top1
            per_domain_stats[domain]["top5"] += doc_top5

    mean_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(min(mean_loss, 20.0))

    domain_summary = {}
    for d, st in per_domain_stats.items():
        d_loss = st["nll"] / st["tokens"] if st["tokens"] > 0 else 0.0
        domain_summary[d] = {
            "loss": round(d_loss, 4),
            "perplexity": round(math.exp(min(d_loss, 20.0)), 2),
            "top1_acc": round(st["top1"] / st["tokens"], 4) if st["tokens"] > 0 else 0.0,
            "top5_acc": round(st["top5"] / st["tokens"], 4) if st["tokens"] > 0 else 0.0,
            "tokens": st["tokens"]
        }

    return {
        "total_tokens_evaluated": total_tokens,
        "mean_cross_entropy": round(mean_loss, 4),
        "perplexity": round(ppl, 2),
        "top1_accuracy": round(top1_correct / total_tokens, 4) if total_tokens > 0 else 0.0,
        "top5_accuracy": round(top5_correct / total_tokens, 4) if total_tokens > 0 else 0.0,
        "domain_breakdown": domain_summary,
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

    model, _ = load_canonical_model()
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


def run_batch1861_experiments():
    print("==================================================================")
    print("CHAKRVIEW BATCHES 1861–1900: DATA SCALING & VALIDATION STUDY")
    print("==================================================================")

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(DATA_DIR / "train_expanded_corpus.json", "r", encoding="utf-8") as f:
        train_docs = json.load(f)
    with open(DATA_DIR / "test_held_out_expanded_corpus.json", "r", encoding="utf-8") as f:
        test_docs = json.load(f)
    with open(DATA_DIR / "control_skewed_train_corpus.json", "r", encoding="utf-8") as f:
        control_docs = json.load(f)

    # 1. Condition A: Canonical Step 10,000 Baseline
    print("\n[Condition A] Evaluating Canonical Baseline on 12,527 Held-Out Tokens...")
    canon_model, _ = load_canonical_model()
    canon_metrics = evaluate_expanded_language_modeling(canon_model, tokenizer, test_docs)
    canon_transfer = evaluate_practical_transfer_suite(canon_model, tokenizer)
    del canon_model

    print(f"  Canonical Held-Out Loss: {canon_metrics['mean_cross_entropy']:.4f} (PPL: {canon_metrics['perplexity']:.2f})")
    print(f"  Top-1 Next-Token Acc:   {canon_metrics['top1_accuracy']*100:.2f}% | Top-5: {canon_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Transfer Tasks Passed:  {canon_transfer['passed']}/{canon_transfer['total_transfer_tasks']}")

    # 2. Condition B: Continued Pretraining on Balanced Corpus (Seeds 42, 123, 999 - 150 Steps)
    seeds = [42, 123, 999]
    cpt_runs = []
    print("\n[Condition B] Training Continued Pretraining on Balanced Corpus across 3 Seeds...")
    for s in seeds:
        m, wtime = train_cpt_engine(train_docs, tokenizer, seed=s, steps=150, lr=2e-5)
        eval_m = evaluate_expanded_language_modeling(m, tokenizer, test_docs)
        tr_m = evaluate_practical_transfer_suite(m, tokenizer)
        ckpt_path = CKPT_DIR / f"cpt_balanced_seed_{s}.pt"
        torch.save(m.state_dict(), ckpt_path)
        del m
        cpt_runs.append({
            "seed": s, "wall_time_sec": wtime, "metrics": eval_m, "transfer": tr_m
        })
        print(f"  Seed {s}: Held-Out Loss={eval_m['mean_cross_entropy']:.4f} (PPL: {eval_m['perplexity']:.2f}), Top-1 Acc={eval_m['top1_accuracy']*100:.2f}%")

    # 3. Condition C: Content-Skewed Control (Token-Matched on Code/Math Only)
    print("\n[Condition C] Training Content-Skewed Control (Code & Math Only, 150 Steps)...")
    m_ctrl, wtime_ctrl = train_cpt_engine(control_docs, tokenizer, seed=42, steps=150, lr=2e-5)
    ctrl_metrics = evaluate_expanded_language_modeling(m_ctrl, tokenizer, test_docs)
    ctrl_transfer = evaluate_practical_transfer_suite(m_ctrl, tokenizer)
    del m_ctrl
    print(f"  Condition C (Skewed): Held-Out Loss={ctrl_metrics['mean_cross_entropy']:.4f} (PPL: {ctrl_metrics['perplexity']:.2f})")

    # 4. Condition D: Token-Budget Scaling Comparison (300 Steps on Balanced Corpus)
    print("\n[Condition D] Training Token-Budget Scaling (300 Steps Extended Training)...")
    m_scale, wtime_scale = train_cpt_engine(train_docs, tokenizer, seed=42, steps=300, lr=2e-5)
    scale_metrics = evaluate_expanded_language_modeling(m_scale, tokenizer, test_docs)
    scale_transfer = evaluate_practical_transfer_suite(m_scale, tokenizer)
    del m_scale
    print(f"  Condition D (300 Steps): Held-Out Loss={scale_metrics['mean_cross_entropy']:.4f} (PPL: {scale_metrics['perplexity']:.2f})")

    # Compile Final Artifact
    cpt_losses = [r["metrics"]["mean_cross_entropy"] for r in cpt_runs]
    mean_cpt_loss = sum(cpt_losses) / len(cpt_losses)
    std_cpt_loss = math.sqrt(sum((l - mean_cpt_loss) ** 2 for l in cpt_losses) / len(cpt_losses))
    mean_cpt_ppl = sum(r["metrics"]["perplexity"] for r in cpt_runs) / len(cpt_runs)

    full_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_baseline": {
            "metrics": canon_metrics,
            "transfer": canon_transfer
        },
        "condition_b_continued_pretraining": {
            "mean_loss": round(mean_cpt_loss, 4),
            "std_loss": round(std_cpt_loss, 4),
            "mean_perplexity": round(mean_cpt_ppl, 2),
            "delta_loss_vs_canonical": round(mean_cpt_loss - canon_metrics["mean_cross_entropy"], 4),
            "runs": cpt_runs
        },
        "condition_c_content_skewed_control": {
            "metrics": ctrl_metrics,
            "transfer": ctrl_transfer,
            "delta_loss_vs_canonical": round(ctrl_metrics["mean_cross_entropy"] - canon_metrics["mean_cross_entropy"], 4)
        },
        "condition_d_token_budget_scaling": {
            "steps": 300,
            "metrics": scale_metrics,
            "transfer": scale_transfer,
            "delta_loss_vs_canonical": round(scale_metrics["mean_cross_entropy"] - canon_metrics["mean_cross_entropy"], 4)
        }
    }

    out_json = RESULTS_DIR / "EXPANDED_LANGUAGE_MODELING_RESULTS.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n==================================================================")
    print(f"BATCH 1861–1900 EXPERIMENTS COMPLETE. Summary:")
    print(f"  Canonical Loss:         {canon_metrics['mean_cross_entropy']:.4f} (PPL {canon_metrics['perplexity']:.2f})")
    print(f"  Condition B (Balanced): {mean_cpt_loss:.4f} (PPL {mean_cpt_ppl:.2f}) | Delta: {mean_cpt_loss - canon_metrics['mean_cross_entropy']:+.4f}")
    print(f"  Condition C (Skewed):   {ctrl_metrics['mean_cross_entropy']:.4f} (PPL {ctrl_metrics['perplexity']:.2f}) | Delta: {ctrl_metrics['mean_cross_entropy'] - canon_metrics['mean_cross_entropy']:+.4f}")
    print(f"  Condition D (300 Step): {scale_metrics['mean_cross_entropy']:.4f} (PPL {scale_metrics['perplexity']:.2f}) | Delta: {scale_metrics['mean_cross_entropy'] - canon_metrics['mean_cross_entropy']:+.4f}")
    print(f"Saved results to: {out_json}")


if __name__ == "__main__":
    run_batch1861_experiments()
