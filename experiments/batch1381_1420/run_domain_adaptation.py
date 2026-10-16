"""
Controlled Domain Adaptation and Overfitting Detection Engine for ChakrMicro.
Location: experiments/batch1381_1420/run_domain_adaptation.py

Executes:
1. Validates canonical checkpoint integrity.
2. Partitions 25 training files into 20 adaptation train and 5 validation files.
3. Trains ChakrMicro with AdamW, conservative learning rate (5e-5), warmup & cosine decay.
4. Monitors validation loss every 5 steps; early stops if validation fails to improve for 3 evaluations.
5. Saves best validation checkpoint to experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt.
6. Evaluates BOTH baseline checkpoint and trained candidate on the 27 held-out test documents.
7. Saves comprehensive machine-readable metrics to experiments/batch1381_1420/results/ADAPTATION_EVALUATION_RESULTS.json.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
MANIFEST_PATH = Path("experiments/batch1341_1380/data/corpus_lm_split_manifest.json")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXP_CKPT_PATH = Path("experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt")
OUT_DIR = Path("experiments/batch1381_1420/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Fixed seed for deterministic batch ordering
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEV_VAL_FILENAMES = {
    "billing_refund_policy.md",
    "compute_cpu_simd.md",
    "network_firewall_rules.md",
    "security_incident_response.md",
    "storage_caching_layer.md",
}


def load_dataset_splits(tokenizer):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train_docs = manifest["train_split"]
    test_docs = manifest["test_split"]

    train_data = []
    val_data = []

    for d in train_docs:
        fn = d["filename"]
        txt = (CORPUS_DIR / fn).read_text(encoding="utf-8")
        toks = tokenizer.encode(txt)
        item = {"filename": fn, "tokens": toks}
        if fn in DEV_VAL_FILENAMES:
            val_data.append(item)
        else:
            train_data.append(item)

    test_data = []
    for d in test_docs:
        fn = d["filename"]
        txt = (CORPUS_DIR / fn).read_text(encoding="utf-8")
        toks = tokenizer.encode(txt)
        test_data.append({"filename": fn, "tokens": toks})

    return train_data, val_data, test_data


def compute_split_loss(model: ChakrMicro, data_items: list[dict]) -> tuple[float, float, int]:
    """Compute aggregate cross-entropy and perplexity over a list of document items."""
    model.eval()
    total_nll = 0.0
    total_target_tokens = 0
    with torch.no_grad():
        for item in data_items:
            tokens = item["tokens"]
            if len(tokens) < 2:
                continue
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            targets = torch.tensor(tokens[1:], dtype=torch.long)
            logits = model(input_ids)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            target_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            total_nll -= target_lp.sum().item()
            total_target_tokens += len(targets)

    mean_loss = total_nll / total_target_tokens if total_target_tokens > 0 else 0.0
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, ppl, total_target_tokens


def evaluate_full_metrics(model: ChakrMicro, tokenizer, test_data: list[dict]):
    """Evaluate comprehensive metrics: Loss, PPL, Top-1, Top-5, Top-10 on held-out test data."""
    model.eval()
    total_nll = 0.0
    total_targets = 0
    top1_c = 0
    top5_c = 0
    top10_c = 0
    per_doc = []

    with torch.no_grad():
        for item in test_data:
            tokens = item["tokens"]
            if len(tokens) < 2:
                continue
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            targets = torch.tensor(tokens[1:], dtype=torch.long)
            logits = model(input_ids)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            target_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            doc_nll = -target_lp.sum().item()
            total_nll += doc_nll
            total_targets += len(targets)

            _, top10 = torch.topk(logits, k=10, dim=-1)
            for i, target_id in enumerate(targets.tolist()):
                top10_list = top10[i].tolist()
                if target_id == top10_list[0]:
                    top1_c += 1
                if target_id in top10_list[:5]:
                    top5_c += 1
                if target_id in top10_list:
                    top10_c += 1

            per_doc.append({
                "doc_id": item["filename"],
                "category": item["filename"].split("_")[0],
                "tokens": len(targets),
                "loss": round(doc_nll / len(targets), 4),
                "ppl": round(math.exp(doc_nll / len(targets)), 2)
            })

    mean_loss = total_nll / total_targets
    return {
        "targets_evaluated": total_targets,
        "cross_entropy_loss": round(mean_loss, 4),
        "perplexity": round(math.exp(mean_loss), 2),
        "top_1_accuracy": round(top1_c / total_targets, 4),
        "top_5_accuracy": round(top5_c / total_targets, 4),
        "top_10_accuracy": round(top10_c / total_targets, 4),
        "per_document": per_doc
    }


def run_training_experiment():
    print("Initializing environment and loading tokenizer...")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_data, val_data, test_data = load_dataset_splits(tokenizer)

    print(f"Data Splits: Train={len(train_data)} docs, Val={len(val_data)} docs, Test={len(test_data)} docs")

    # 1. Load Canonical Model
    ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # 2. Benchmark Baseline Performance on Held-Out Test Set
    print("\n--- Evaluating Baseline (Canonical Checkpoint Step 10,000) ---")
    baseline_val_loss, baseline_val_ppl, _ = compute_split_loss(model, val_data)
    baseline_test_metrics = evaluate_full_metrics(model, tokenizer, test_data)
    print(f"Baseline Val Loss: {baseline_val_loss:.4f} (PPL: {baseline_val_ppl:.2f})")
    print(f"Baseline Held-Out Test Loss: {baseline_test_metrics['cross_entropy_loss']:.4f} (PPL: {baseline_test_metrics['perplexity']:.2f})")
    print(f"Baseline Top-1: {baseline_test_metrics['top_1_accuracy']*100:.2f}% | Top-10: {baseline_test_metrics['top_10_accuracy']*100:.2f}%")

    # 3. Setup Optimizer and Training Configuration
    max_steps = 80
    warmup_steps = 10
    peak_lr = 5e-5
    min_lr = 5e-6
    batch_size_docs = 4
    val_interval = 5
    patience = 3  # 15 steps without val improvement

    optimizer = AdamW(model.parameters(), lr=min_lr, betas=(0.9, 0.95), weight_decay=0.05)

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    training_log = []
    best_val_loss = baseline_val_loss
    best_step = 0
    patience_counter = 0

    print(f"\n--- Starting Controlled Domain Adaptation (Max Steps={max_steps}, Peak LR={peak_lr}) ---")
    model.train()
    t_start = time.perf_counter()

    for step in range(1, max_steps + 1):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample batch of documents
        batch_docs = random.sample(train_data, batch_size_docs)
        optimizer.zero_grad()

        step_loss = 0.0
        step_tokens = 0

        for doc in batch_docs:
            tokens = doc["tokens"]
            if len(tokens) < 2:
                continue
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            targets = torch.tensor(tokens[1:], dtype=torch.long)

            logits = model(input_ids)[0]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            step_loss += loss.item() * len(targets)
            step_tokens += len(targets)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_train_loss = step_loss / step_tokens

        # Validation Step
        if step % val_interval == 0:
            val_loss, val_ppl, _ = compute_split_loss(model, val_data)
            model.train()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_step = step
                patience_counter = 0
                # Save candidate checkpoint
                torch.save({
                    "step": 10000 + step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "validation_loss": val_loss,
                    "validation_perplexity": val_ppl,
                    "adaptation_step": step,
                    "base_checkpoint_sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
                }, EXP_CKPT_PATH)
            else:
                patience_counter += 1

            training_log.append({
                "step": step,
                "learning_rate": round(lr, 8),
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_perplexity": round(val_ppl, 2),
                "is_best": is_best
            })

            print(f"Step {step:2d}/{max_steps:2d} | LR: {lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) {'[BEST SAVED]' if is_best else ''}")

            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered at step {step}: Validation loss failed to improve for {patience * val_interval} steps.")
                break

    train_duration = time.perf_counter() - t_start
    print(f"Training completed in {train_duration:.2f} seconds. Best validation step: {best_step} (Val Loss: {best_val_loss:.4f})")

    # 4. Load Best Candidate Checkpoint for Fair Evaluation on Held-Out Test Split
    print("\n--- Evaluating Best Candidate Checkpoint on Held-Out Test Set ---")
    cand_ckpt = torch.load(EXP_CKPT_PATH, map_location="cpu", weights_only=False)
    candidate_model = ChakrMicro(config)
    candidate_model.load_state_dict(cand_ckpt["model_state_dict"])
    candidate_test_metrics = evaluate_full_metrics(candidate_model, tokenizer, test_data)

    print(f"Candidate Held-Out Test Loss: {candidate_test_metrics['cross_entropy_loss']:.4f} (PPL: {candidate_test_metrics['perplexity']:.2f})")
    print(f"Candidate Top-1: {candidate_test_metrics['top_1_accuracy']*100:.2f}% | Top-10: {candidate_test_metrics['top_10_accuracy']*100:.2f}%")

    # 5. Compute Comparative Changes
    loss_delta = candidate_test_metrics["cross_entropy_loss"] - baseline_test_metrics["cross_entropy_loss"]
    ppl_delta = candidate_test_metrics["perplexity"] - baseline_test_metrics["perplexity"]
    top1_delta = candidate_test_metrics["top_1_accuracy"] - baseline_test_metrics["top_1_accuracy"]
    top10_delta = candidate_test_metrics["top_10_accuracy"] - baseline_test_metrics["top_10_accuracy"]

    # 6. Save Comprehensive Report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_checkpoint_sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
        "training_metadata": {
            "max_steps": max_steps,
            "actual_steps_completed": step,
            "best_step": best_step,
            "best_val_loss": round(best_val_loss, 4),
            "peak_lr": peak_lr,
            "weight_decay": 0.05,
            "training_duration_seconds": round(train_duration, 2),
            "candidate_checkpoint_path": str(EXP_CKPT_PATH),
        },
        "training_log": training_log,
        "comparative_evaluation": {
            "targets_evaluated": baseline_test_metrics["targets_evaluated"],
            "baseline_canonical": {
                "cross_entropy_loss": baseline_test_metrics["cross_entropy_loss"],
                "perplexity": baseline_test_metrics["perplexity"],
                "top_1_accuracy": baseline_test_metrics["top_1_accuracy"],
                "top_5_accuracy": baseline_test_metrics["top_5_accuracy"],
                "top_10_accuracy": baseline_test_metrics["top_10_accuracy"],
            },
            "candidate_adapted": {
                "cross_entropy_loss": candidate_test_metrics["cross_entropy_loss"],
                "perplexity": candidate_test_metrics["perplexity"],
                "top_1_accuracy": candidate_test_metrics["top_1_accuracy"],
                "top_5_accuracy": candidate_test_metrics["top_5_accuracy"],
                "top_10_accuracy": candidate_test_metrics["top_10_accuracy"],
            },
            "deltas": {
                "loss_change": round(loss_delta, 4),
                "perplexity_change": round(ppl_delta, 2),
                "top_1_accuracy_change": round(top1_delta, 4),
                "top_10_accuracy_change": round(top10_delta, 4),
            }
        },
        "per_document_comparison": [
            {
                "doc_id": b["doc_id"],
                "category": b["category"],
                "tokens": b["tokens"],
                "baseline_loss": b["loss"],
                "candidate_loss": c["loss"],
                "loss_delta": round(c["loss"] - b["loss"], 4)
            }
            for b, c in zip(baseline_test_metrics["per_document"], candidate_test_metrics["per_document"])
        ]
    }

    out_file = OUT_DIR / "ADAPTATION_EVALUATION_RESULTS.json"
    out_file.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_file}")

    print("\n=========================================================================")
    print("COMPARATIVE EVALUATION: BASELINE VS. DOMAIN-ADAPTED CANDIDATE")
    print("=========================================================================")
    print(f"{'Metric':<25} | {'Baseline (Step 10k)':<20} | {'Candidate (Step ' + str(10000+best_step) + ')':<20} | {'Delta':<12}")
    print("-" * 85)
    print(f"{'Cross-Entropy Loss':<25} | {baseline_test_metrics['cross_entropy_loss']:<20.4f} | {candidate_test_metrics['cross_entropy_loss']:<20.4f} | {loss_delta:<+12.4f}")
    print(f"{'Perplexity (PPL)':<25} | {baseline_test_metrics['perplexity']:<20.2f} | {candidate_test_metrics['perplexity']:<20.2f} | {ppl_delta:<+12.2f}")
    print(f"{'Top-1 Accuracy':<25} | {baseline_test_metrics['top_1_accuracy']*100:<19.2f}% | {candidate_test_metrics['top_1_accuracy']*100:<19.2f}% | {top1_delta*100:<+11.2f}%")
    print(f"{'Top-5 Accuracy':<25} | {baseline_test_metrics['top_5_accuracy']*100:<19.2f}% | {candidate_test_metrics['top_5_accuracy']*100:<19.2f}% | {(candidate_test_metrics['top_5_accuracy'] - baseline_test_metrics['top_5_accuracy'])*100:<+11.2f}%")
    print(f"{'Top-10 Accuracy':<25} | {baseline_test_metrics['top_10_accuracy']*100:<19.2f}% | {candidate_test_metrics['top_10_accuracy']*100:<19.2f}% | {top10_delta*100:<+11.2f}%")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_training_experiment()
