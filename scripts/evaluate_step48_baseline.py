"""
Script: Evaluate Untrained Baseline Model (Step 48 Phase 3).

Establishes the exact reference point for ChakrMicro v0.1 prior to any training updates:
1. Verifies model weight hash matches canonical frozen baseline (c5571c...).
2. Evaluates baseline model on train sample, validation sample, and test sample.
3. Computes cross-entropy loss, perplexity (PPL = exp(loss)), and top-1 token accuracy.
4. Outputs results to docs/STEP_48_BASELINE_EVALUATION.json.
"""

import json
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_FILE = ROOT_DIR / "docs" / "STEP_48_BASELINE_EVALUATION.json"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss
from chakrview.training.builder import ChakrOfflineDataset

FROZEN_BASELINE_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136
TOKENIZER_CHECKSUM = "7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f"
DATASET_BASE = ROOT_DIR / "data" / "tokenized" / "stage_b"
MANIFEST_FILE = ROOT_DIR / "data" / "manifests" / "stage_b_manifest.json"


def compute_model_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def evaluate_sample(
    model: nn.Module,
    shard_dir: Path,
    sequence_length: int = 64,
    batch_size: int = 4,
    max_batches: int = 50,
) -> Dict[str, Any]:
    dataset = StreamingTokenDataset(
        shard_dir=shard_dir,
        sequence_length=sequence_length,
        loop=False,
        drop_remainder=True,
    )

    loss_fn = CausalLoss(ignore_index=2)
    losses: List[float] = []
    correct_tokens = 0
    total_tokens = 0

    batch_buffer = []
    batches_processed = 0

    model.eval()
    with torch.no_grad():
        for item in dataset:
            batch_buffer.append(item)
            if len(batch_buffer) == batch_size:
                batch = ChakrOfflineDataset.collate_fn(batch_buffer, pad_token_id=2)
                batch_buffer = []

                input_ids = batch["input_ids"]
                target_ids = batch["target_ids"]
                attention_mask = batch["attention_mask"]

                logits = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, target_ids)
                losses.append(loss.item())

                # Top-1 accuracy calculation
                preds = torch.argmax(logits, dim=-1)
                valid_mask = (target_ids != 2) & (attention_mask == 1)
                correct = ((preds == target_ids) & valid_mask).sum().item()
                valid_count = valid_mask.sum().item()

                correct_tokens += correct
                total_tokens += valid_count
                batches_processed += 1

                if batches_processed >= max_batches:
                    break

        if batch_buffer and batches_processed < max_batches:
            batch = ChakrOfflineDataset.collate_fn(batch_buffer, pad_token_id=2)
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            attention_mask = batch["attention_mask"]

            logits = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, target_ids)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=-1)
            valid_mask = (target_ids != 2) & (attention_mask == 1)
            correct_tokens += ((preds == target_ids) & valid_mask).sum().item()
            total_tokens += valid_mask.sum().item()
            batches_processed += 1

    mean_loss = float(torch.tensor(losses).mean().item()) if losses else 0.0
    ppl = math.exp(min(mean_loss, 20.0))
    accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    return {
        "batches_evaluated": batches_processed,
        "tokens_evaluated": total_tokens,
        "mean_loss": round(mean_loss, 4),
        "perplexity": round(ppl, 4),
        "top1_token_accuracy": round(accuracy, 6),
    }


def run_baseline_evaluation():
    print("=" * 72)
    print("CHAKRVIEW STEP 48: BASELINE UNTRAINED MODEL EVALUATION")
    print("=" * 72)

    # 1. Instantiate Canonical Baseline
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()

    model_hash = compute_model_hash(model)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"Model Architecture: ChakrMicro v0.1")
    print(f"Parameters:         {param_count:,}")
    print(f"Weight SHA-256:     {model_hash}")
    assert model_hash == FROZEN_BASELINE_HASH, "Baseline model hash mismatch!"
    assert param_count == EXPECTED_PARAM_COUNT, "Parameter count mismatch!"

    # 2. Manifest Hash
    hasher = hashlib.sha256()
    with open(MANIFEST_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    manifest_hash = hasher.hexdigest()

    # 3. Evaluate Splits
    print("\nEvaluating Baseline on Splits (50 batches each, seq_len=64, batch_size=4)...")
    train_res = evaluate_sample(model, DATASET_BASE / "train", sequence_length=64, batch_size=4, max_batches=50)
    print(f"  Train Sample:      Loss={train_res['mean_loss']} | PPL={train_res['perplexity']} | Acc={train_res['top1_token_accuracy']:.4%} ({train_res['tokens_evaluated']} tokens)")

    val_res = evaluate_sample(model, DATASET_BASE / "validation", sequence_length=64, batch_size=4, max_batches=50)
    print(f"  Validation Sample: Loss={val_res['mean_loss']} | PPL={val_res['perplexity']} | Acc={val_res['top1_token_accuracy']:.4%} ({val_res['tokens_evaluated']} tokens)")

    test_res = evaluate_sample(model, DATASET_BASE / "test", sequence_length=64, batch_size=4, max_batches=50)
    print(f"  Test Sample:       Loss={test_res['mean_loss']} | PPL={test_res['perplexity']} | Acc={test_res['top1_token_accuracy']:.4%} ({test_res['tokens_evaluated']} tokens)")

    report = {
        "model_name": "ChakrMicro v0.1 (Untrained Baseline)",
        "parameter_count": param_count,
        "vocabulary_size": 4096,
        "max_sequence_length": 512,
        "model_weight_sha256": model_hash,
        "tokenizer_checksum": TOKENIZER_CHECKSUM,
        "dataset_manifest_hash": manifest_hash,
        "theoretical_uniform_loss": round(-math.log(1.0 / 4096), 4),
        "theoretical_uniform_ppl": 4096.0,
        "evaluation_metrics": {
            "train_sample": train_res,
            "validation_sample": val_res,
            "test_sample": test_res,
        },
        "baseline_summary": (
            f"Untrained baseline achieves initial loss {val_res['mean_loss']:.4f} "
            f"(PPL {val_res['perplexity']:.2f}, theoretical uniform PPL 4096.00) "
            f"with random accuracy {val_res['top1_token_accuracy']:.4%}."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Baseline evaluation written to {OUTPUT_FILE}")
    print("=" * 72)
    return report


if __name__ == "__main__":
    run_baseline_evaluation()
