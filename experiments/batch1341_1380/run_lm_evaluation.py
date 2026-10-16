"""
Reproducible evaluation harness for ChakrMicro next-token prediction and language modeling.
Location: experiments/batch1341_1380/run_lm_evaluation.py

Evaluates:
1. Uniform Random Baseline
2. Training-Set Empirical Unigram Baseline
3. Canonical ChakrMicro Neural Core (step 10,000)

Metrics:
- Token-level Cross-Entropy Loss
- Perplexity (exp(Loss))
- Top-1, Top-5, Top-10 Accuracy
- Token prediction inspection samples
- Per-category breakdown

Saves output to experiments/batch1341_1380/results/LM_EVALUATION_RESULTS.json
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
MANIFEST_PATH = Path("experiments/batch1341_1380/data/corpus_lm_split_manifest.json")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
OUT_DIR = Path("experiments/batch1341_1380/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_tokenizer():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, config


def fit_unigram_baseline(train_files: list[dict], tokenizer) -> tuple[torch.Tensor, dict]:
    """Fit an empirical unigram distribution from the training split."""
    counts = Counter()
    total_tokens = 0
    for item in train_files:
        text = (CORPUS_DIR / item["filename"]).read_text(encoding="utf-8")
        tokens = tokenizer.encode(text)
        counts.update(tokens)
        total_tokens += len(tokens)

    vocab_size = tokenizer.vocab_size
    # Add-1 Laplace smoothing to prevent log(0) for unseen tokens in train split
    smoothed_probs = torch.zeros(vocab_size, dtype=torch.float32)
    smoothed_denom = total_tokens + vocab_size
    for tid in range(vocab_size):
        smoothed_probs[tid] = (counts[tid] + 1.0) / smoothed_denom

    unigram_log_probs = torch.log(smoothed_probs)
    return unigram_log_probs, {
        "train_tokens_counted": total_tokens,
        "unique_tokens_in_train": len(counts),
        "vocab_coverage": round(len(counts) / vocab_size, 4),
        "top_5_frequent_tokens": [
            {"token_id": tid, "token_str": tokenizer.decode([tid]), "count": cnt}
            for tid, cnt in counts.most_common(5)
        ]
    }


def evaluate_sequences(model: ChakrMicro, tokenizer, unigram_log_probs: torch.Tensor, test_files: list[dict]):
    vocab_size = tokenizer.vocab_size
    uniform_loss_per_token = math.log(vocab_size)  # Exact theoretical loss: ln(4096) ≈ 8.317766

    total_target_tokens = 0
    total_model_nll = 0.0
    total_unigram_nll = 0.0

    model_top1_correct = 0
    model_top5_correct = 0
    model_top10_correct = 0

    unigram_top1_correct = 0
    unigram_top5_correct = 0
    unigram_top10_correct = 0

    # Get unigram top 10 token indices
    _, unigram_top10_indices = torch.topk(unigram_log_probs, k=10)
    unigram_top1 = unigram_top10_indices[0].item()
    unigram_top5_set = set(unigram_top10_indices[:5].tolist())
    unigram_top10_set = set(unigram_top10_indices[:10].tolist())

    per_doc_results = []
    prediction_samples = []

    for item in test_files:
        filepath = CORPUS_DIR / item["filename"]
        text = filepath.read_text(encoding="utf-8")
        tokens = tokenizer.encode(text)

        if len(tokens) < 2:
            continue

        doc_target_tokens = len(tokens) - 1
        total_target_tokens += doc_target_tokens

        # Causal sequence formulation:
        # Input: tokens[:-1]
        # Target: tokens[1:]
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)

        # 1. Unigram Evaluation on targets
        target_unigram_lp = unigram_log_probs[targets]
        doc_unigram_nll = -target_unigram_lp.sum().item()
        total_unigram_nll += doc_unigram_nll

        for t in targets.tolist():
            if t == unigram_top1:
                unigram_top1_correct += 1
            if t in unigram_top5_set:
                unigram_top5_correct += 1
            if t in unigram_top10_set:
                unigram_top10_correct += 1

        # 2. Model Evaluation
        with torch.no_grad():
            logits = model(input_ids)  # [1, T-1, vocab_size]
            logits = logits[0]  # [T-1, vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)

            # Target negative log-likelihood
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            doc_model_nll = -target_log_probs.sum().item()
            total_model_nll += doc_model_nll

            # Accuracies
            _, top10_preds = torch.topk(logits, k=10, dim=-1)
            for i, target_id in enumerate(targets.tolist()):
                pred_list = top10_preds[i].tolist()
                if target_id == pred_list[0]:
                    model_top1_correct += 1
                if target_id in pred_list[:5]:
                    model_top5_correct += 1
                if target_id in pred_list:
                    model_top10_correct += 1

                # Collect a sample of qualitative predictions
                if len(prediction_samples) < 30 and (i % 8 == 0):
                    context_snippet = tokenizer.decode(tokens[max(0, i-6):i+1])
                    actual_str = tokenizer.decode([target_id])
                    top3_tokens = [
                        {
                            "token_id": pid,
                            "token_str": tokenizer.decode([pid]),
                            "prob": round(math.exp(log_probs[i, pid].item()), 4)
                        }
                        for pid in pred_list[:3]
                    ]
                    prediction_samples.append({
                        "doc_id": item["filename"],
                        "position": i,
                        "context": context_snippet,
                        "actual_next_token_id": target_id,
                        "actual_next_token_str": actual_str,
                        "actual_token_prob": round(math.exp(log_probs[i, target_id].item()), 5),
                        "actual_in_top1": (target_id == pred_list[0]),
                        "actual_in_top5": (target_id in pred_list[:5]),
                        "actual_in_top10": (target_id in pred_list),
                        "model_top_3_predictions": top3_tokens
                    })

        per_doc_results.append({
            "doc_id": item["filename"],
            "category": item["filename"].split("_")[0],
            "tokens_evaluated": doc_target_tokens,
            "model_loss": round(doc_model_nll / doc_target_tokens, 4),
            "model_ppl": round(math.exp(doc_model_nll / doc_target_tokens), 2),
            "unigram_loss": round(doc_unigram_nll / doc_target_tokens, 4),
            "unigram_ppl": round(math.exp(doc_unigram_nll / doc_target_tokens), 2)
        })

    # Summary aggregations
    model_mean_loss = total_model_nll / total_target_tokens
    model_ppl = math.exp(model_mean_loss)
    unigram_mean_loss = total_unigram_nll / total_target_tokens
    unigram_ppl = math.exp(unigram_mean_loss)
    uniform_ppl = vocab_size

    # Per-category breakdown
    cat_summary = {}
    for r in per_doc_results:
        cat = r["category"]
        if cat not in cat_summary:
            cat_summary[cat] = {"docs": 0, "tokens": 0, "model_loss_sum": 0.0, "unigram_loss_sum": 0.0}
        cat_summary[cat]["docs"] += 1
        cat_summary[cat]["tokens"] += r["tokens_evaluated"]
        cat_summary[cat]["model_loss_sum"] += r["model_loss"] * r["tokens_evaluated"]
        cat_summary[cat]["unigram_loss_sum"] += r["unigram_loss"] * r["tokens_evaluated"]

    cat_breakdown = {}
    for cat, data in sorted(cat_summary.items()):
        m_loss = data["model_loss_sum"] / data["tokens"]
        u_loss = data["unigram_loss_sum"] / data["tokens"]
        cat_breakdown[cat] = {
            "documents_count": data["docs"],
            "tokens_count": data["tokens"],
            "model_cross_entropy": round(m_loss, 4),
            "model_perplexity": round(math.exp(m_loss), 2),
            "unigram_cross_entropy": round(u_loss, 4),
            "unigram_perplexity": round(math.exp(u_loss), 2),
        }

    return {
        "total_test_documents": len(test_files),
        "total_target_tokens_evaluated": total_target_tokens,
        "metrics_summary": {
            "uniform_random_baseline": {
                "cross_entropy_loss": round(uniform_loss_per_token, 4),
                "perplexity": round(uniform_ppl, 2),
                "top_1_accuracy": round(1.0 / vocab_size, 6),
                "top_5_accuracy": round(5.0 / vocab_size, 6),
                "top_10_accuracy": round(10.0 / vocab_size, 6),
            },
            "training_unigram_baseline": {
                "cross_entropy_loss": round(unigram_mean_loss, 4),
                "perplexity": round(unigram_ppl, 2),
                "top_1_accuracy": round(unigram_top1_correct / total_target_tokens, 4),
                "top_5_accuracy": round(unigram_top5_correct / total_target_tokens, 4),
                "top_10_accuracy": round(unigram_top10_correct / total_target_tokens, 4),
            },
            "chakrmicro_checkpoint_step_10000": {
                "cross_entropy_loss": round(model_mean_loss, 4),
                "perplexity": round(model_ppl, 2),
                "top_1_accuracy": round(model_top1_correct / total_target_tokens, 4),
                "top_5_accuracy": round(model_top5_correct / total_target_tokens, 4),
                "top_10_accuracy": round(model_top10_correct / total_target_tokens, 4),
            }
        },
        "category_breakdown": cat_breakdown,
        "per_document_results": per_doc_results,
        "prediction_samples": prediction_samples
    }


def main():
    print("Loading model and tokenizer...")
    t0 = time.perf_counter()
    model, tokenizer, config = load_model_and_tokenizer()
    load_time = time.perf_counter() - t0
    print(f"Loaded ChakrMicro (4,229,568 params) in {load_time:.2f}s")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train_files = manifest["train_split"]
    test_files = manifest["test_split"]

    print(f"Fitting unigram baseline on {len(train_files)} training files...")
    unigram_lp, unigram_meta = fit_unigram_baseline(train_files, tokenizer)
    print(f"Unigram fitted on {unigram_meta['train_tokens_counted']} tokens. Coverage: {unigram_meta['vocab_coverage']*100:.1f}%")

    print(f"Evaluating {len(test_files)} held-out technical documents...")
    t_eval = time.perf_counter()
    eval_results = evaluate_sequences(model, tokenizer, unigram_lp, test_files)
    eval_duration = time.perf_counter() - t_eval
    print(f"Evaluation complete in {eval_duration:.2f}s ({eval_results['total_target_tokens_evaluated'] / eval_duration:.1f} tokens/sec)")

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {
            "path": str(CANONICAL_CKPT),
            "step": 10000,
            "architecture": "ChakrMicro",
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "total_parameters": 4229568,
            "tied_parameters": 3443136
        },
        "dataset_split": {
            "train_documents_count": len(train_files),
            "test_documents_count": len(test_files),
            "train_tokens": unigram_meta["train_tokens_counted"],
            "test_tokens": eval_results["total_target_tokens_evaluated"]
        },
        "unigram_metadata": unigram_meta,
        "evaluation_results": eval_results
    }

    out_file = OUT_DIR / "LM_EVALUATION_RESULTS.json"
    out_file.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_file}")

    # Print summary table
    summary = eval_results["metrics_summary"]
    print("\n=========================================================================")
    print("CHAKRMICRO LANGUAGE MODELING & NEXT-TOKEN PREDICTION BENCHMARK")
    print("=========================================================================")
    print(f"{'Metric':<25} | {'Uniform Random':<16} | {'Unigram Baseline':<16} | {'ChakrMicro-10k':<16}")
    print("-" * 80)
    print(f"{'Cross-Entropy Loss':<25} | {summary['uniform_random_baseline']['cross_entropy_loss']:<16.4f} | {summary['training_unigram_baseline']['cross_entropy_loss']:<16.4f} | {summary['chakrmicro_checkpoint_step_10000']['cross_entropy_loss']:<16.4f}")
    print(f"{'Perplexity (PPL)':<25} | {summary['uniform_random_baseline']['perplexity']:<16.2f} | {summary['training_unigram_baseline']['perplexity']:<16.2f} | {summary['chakrmicro_checkpoint_step_10000']['perplexity']:<16.2f}")
    print(f"{'Top-1 Accuracy':<25} | {summary['uniform_random_baseline']['top_1_accuracy']*100:<15.3f}% | {summary['training_unigram_baseline']['top_1_accuracy']*100:<15.2f}% | {summary['chakrmicro_checkpoint_step_10000']['top_1_accuracy']*100:<15.2f}%")
    print(f"{'Top-5 Accuracy':<25} | {summary['uniform_random_baseline']['top_5_accuracy']*100:<15.3f}% | {summary['training_unigram_baseline']['top_5_accuracy']*100:<15.2f}% | {summary['chakrmicro_checkpoint_step_10000']['top_5_accuracy']*100:<15.2f}%")
    print(f"{'Top-10 Accuracy':<25} | {summary['uniform_random_baseline']['top_10_accuracy']*100:<15.3f}% | {summary['training_unigram_baseline']['top_10_accuracy']*100:<15.2f}% | {summary['chakrmicro_checkpoint_step_10000']['top_10_accuracy']*100:<15.2f}%")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
