"""
ChakrView Production CPT Experiment Runner — Batches 2141-2180.
Executes the frozen controlled continued pretraining experiment across 3 seeds (42, 1337, 2026).
Adheres strictly to CPT_PREREGISTRATION.json:
- CPU-first hardware adaptive execution
- Context length T=256, batch_size=4, grad_accum=2 (effective batch size 8 = 2,048 tokens/step)
- Peak LR = 1.5e-4 with 20-step linear warmup and cosine decay to 1.5e-5
- 200 optimization steps per seed
- Checkpointing and validation evaluation at cadence of 25 steps
- Early stopping on equal-weighted macro-source validation cross-entropy
- Full checkpoint state serialization via cpt_checkpoint_engine (model, optimizer, scheduler, RNGs)
- Complete isolation: training never touches test split
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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.batch2101_2140.cpt_checkpoint_engine import save_cpt_checkpoint, load_cpt_checkpoint
from experiments.batch2101_2140.validation_evaluator import evaluate_validation_split

CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CORPUS_CLEANED_MANIFEST_PATH = REPO_ROOT / "experiments" / "batch2101_2140" / "CORPUS_V2_CLEANED_MANIFEST.json"
PREREGISTRATION_PATH = REPO_ROOT / "experiments" / "batch2101_2140" / "CPT_PREREGISTRATION.json"
OUTPUT_DIR = REPO_ROOT / "experiments" / "batch2141_2180" / "cpt"


def get_process_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_cpt_seed(
    seed: int,
    prereg: Dict[str, Any],
    device: str = "cpu"
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"STARTING CONTROLLED CPT RUN: SEED {seed}")
    print(f"{'='*70}")

    seed_dir = OUTPUT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Deterministic Seeding
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # 2. Load Tokenizer & Corpus
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    with open(CORPUS_CLEANED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    corpus_sha = hashlib.sha256(CORPUS_CLEANED_MANIFEST_PATH.read_bytes()).hexdigest()
    vocab_sha = hashlib.sha256((TOKENIZER_DIR / "vocab.json").read_bytes()).hexdigest()

    # Filter Training Documents Only
    train_docs = [d for d in corpus["documents"] if d["split"] == "train"]
    print(f"Loaded {len(train_docs)} training documents ({sum(d['token_count'] for d in train_docs)} tokens).")

    # Tokenize and crop/prepare training sequences
    tokenized_train: List[List[int]] = []
    for d in train_docs:
        toks = tokenizer.encode(d["text"])
        if len(toks) >= 4:
            tokenized_train.append(toks)

    # 3. Load Canonical Baseline Model
    canonical_ckpt = torch.load(CANONICAL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = canonical_ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(canonical_ckpt["model_state_dict"], strict=True)
    model.to(device)

    # 4. Hyperparameters from Preregistration
    hparams = prereg["hyperparameters"]
    context_len = hparams["context_length"]  # 256
    batch_size = hparams["batch_size_per_step"]  # 4
    grad_accum = hparams["gradient_accumulation_steps"]  # 2
    max_steps = hparams["max_steps"]  # 200
    peak_lr = hparams["peak_learning_rate"]  # 1.5e-4
    min_lr = hparams["min_learning_rate"]  # 1.5e-5
    warmup_steps = hparams["warmup_steps"]  # 20
    weight_decay = hparams["weight_decay"]  # 0.01

    val_cfg = prereg["validation_and_early_stopping"]
    cadence = val_cfg["cadence_steps"]  # 25
    patience = val_cfg["patience_checkpoints"]  # 3
    divergence_abort = val_cfg["divergence_abort_threshold_ce"]  # 0.25

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=tuple(hparams["betas"]),
        eps=hparams["eps"],
        weight_decay=weight_decay
    )
    min_ratio = min_lr / peak_lr
    scheduler = build_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio=min_ratio)

    # Baseline evaluation before training
    print("Evaluating initial baseline validation loss...")
    initial_val = evaluate_validation_split(model, tokenizer, CORPUS_CLEANED_MANIFEST_PATH, device=device)
    baseline_val_macro = initial_val["macro_source_ce"]
    baseline_val_tw = initial_val["token_weighted_ce"]
    print(f"  Initial Baseline Val Macro CE: {baseline_val_macro:.4f} (TW: {baseline_val_tw:.4f})")

    # Logging and tracking
    step_records = []
    val_records = [{
        "step": 0,
        "macro_source_ce": baseline_val_macro,
        "token_weighted_ce": baseline_val_tw,
        "per_source_metrics": initial_val["per_source_metrics"]
    }]

    best_val_macro = baseline_val_macro
    best_step = 0
    best_checkpoint_path = None
    no_improve_checkpoints = 0
    aborted_early = False
    abort_reason = None

    start_time = time.time()
    total_tokens_processed = 0
    peak_rss_mb = get_process_memory_mb()

    # Pre-index windows for deterministic batching
    rng_sample = random.Random(seed)

    model.train()
    optimizer.zero_grad()

    for step in range(1, max_steps + 1):
        step_loss = 0.0
        step_tokens = 0

        for micro_step in range(grad_accum):
            # Sample micro_batch
            batch_samples = [rng_sample.choice(tokenized_train) for _ in range(batch_size)]
            
            # Prepare sequences with causal shift
            padded_inp = []
            padded_tgt = []
            for seq in batch_samples:
                # Random window if seq longer than context_len
                if len(seq) > context_len + 1:
                    max_start = len(seq) - (context_len + 1)
                    s_idx = rng_sample.randint(0, max_start)
                    chunk = seq[s_idx : s_idx + context_len + 1]
                else:
                    chunk = seq[:context_len + 1]

                inp_ids = chunk[:-1]
                tgt_ids = chunk[1:]
                pad_len = context_len - len(inp_ids)
                
                # pad_token_id = 2, ignore_index = -100
                padded_inp.append(inp_ids + [2] * pad_len)
                padded_tgt.append(tgt_ids + [-100] * pad_len)
                step_tokens += len(tgt_ids)

            inp_tensor = torch.tensor(padded_inp, dtype=torch.long, device=device)
            tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long, device=device)

            logits = model(inp_tensor)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_tensor.view(-1), ignore_index=-100)
            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            step_loss += loss.item() / grad_accum

        # Gradient clipping & optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_tokens_processed += step_tokens
        curr_lr = scheduler.get_last_lr()[0]
        curr_rss = get_process_memory_mb()
        if curr_rss > peak_rss_mb:
            peak_rss_mb = curr_rss

        step_records.append({
            "step": step,
            "train_loss": round(step_loss, 4),
            "learning_rate": curr_lr,
            "tokens_processed": total_tokens_processed,
            "step_time_sec": 0.0
        })

        if step % 10 == 0 or step == max_steps:
            elapsed = time.time() - start_time
            print(f"  Step {step:3d}/{max_steps} | Loss: {step_loss:6.4f} | LR: {curr_lr:.2e} | Tokens: {total_tokens_processed} | RSS: {curr_rss:.1f} MB | Elapsed: {elapsed:.1f}s")

        # Validation Cadence & Early Stopping
        if step % cadence == 0 or step == max_steps:
            val_res = evaluate_validation_split(model, tokenizer, CORPUS_CLEANED_MANIFEST_PATH, device=device)
            curr_val_macro = val_res["macro_source_ce"]
            curr_val_tw = val_res["token_weighted_ce"]
            print(f"\n  >>> [Step {step} Validation] Macro CE: {curr_val_macro:.4f} (Baseline: {baseline_val_macro:.4f}) | TW CE: {curr_val_tw:.4f}")
            for sid, sm in val_res["per_source_metrics"].items():
                print(f"      - {sid:30s}: CE = {sm['token_weighted_ce']:.4f} (PPL: {sm['perplexity']:.2f})")

            val_record = {
                "step": step,
                "macro_source_ce": curr_val_macro,
                "token_weighted_ce": curr_val_tw,
                "per_source_metrics": val_res["per_source_metrics"]
            }
            val_records.append(val_record)

            # Save periodic step checkpoint
            step_ckpt_path = ckpt_dir / f"checkpoint_step_{step:03d}.pt"
            save_cpt_checkpoint(
                step_ckpt_path,
                model,
                optimizer,
                scheduler,
                global_step=step,
                seed=seed,
                config=config.__dict__ if hasattr(config, "__dict__") else config,
                corpus_manifest_sha256=corpus_sha,
                tokenizer_vocab_sha256=vocab_sha,
                sampler_state={"seed": seed, "total_tokens": total_tokens_processed}
            )

            # Check Divergence Abort
            if curr_val_macro > baseline_val_macro + divergence_abort:
                aborted_early = True
                abort_reason = f"Divergence detected: Val Macro CE {curr_val_macro:.4f} exceeded baseline + {divergence_abort}"
                print(f"  [ABORT] {abort_reason}")
                break

            # Check Improvement
            if curr_val_macro < best_val_macro:
                best_val_macro = curr_val_macro
                best_step = step
                best_checkpoint_path = step_ckpt_path
                no_improve_checkpoints = 0
                print(f"  [NEW BEST CHECKPOINT] Step {step}: Val Macro CE = {best_val_macro:.4f}")
            else:
                no_improve_checkpoints += 1
                print(f"  [PATIENCE] No improvement for {no_improve_checkpoints}/{patience} checkpoints.")
                if no_improve_checkpoints >= patience:
                    aborted_early = True
                    abort_reason = f"Early stopping triggered: {patience} consecutive checkpoints without improvement."
                    print(f"  [EARLY STOP] {abort_reason}")
                    break

            model.train()

    total_runtime = time.time() - start_time

    # Save final candidate checkpoint
    final_ckpt_path = seed_dir / "cpt_final_checkpoint.pt"
    save_cpt_checkpoint(
        final_ckpt_path,
        model,
        optimizer,
        scheduler,
        global_step=step,
        seed=seed,
        config=config.__dict__ if hasattr(config, "__dict__") else config,
        corpus_manifest_sha256=corpus_sha,
        tokenizer_vocab_sha256=vocab_sha,
        sampler_state={"seed": seed, "total_tokens": total_tokens_processed}
    )

    # Save best checkpoint symlink or copy
    best_candidate_path = seed_dir / "cpt_best_checkpoint.pt"
    if best_checkpoint_path and best_checkpoint_path.exists():
        import shutil
        shutil.copy2(best_checkpoint_path, best_candidate_path)
    else:
        import shutil
        shutil.copy2(final_ckpt_path, best_candidate_path)

    best_hash = hashlib.sha256(best_candidate_path.read_bytes()).hexdigest()
    final_hash = hashlib.sha256(final_ckpt_path.read_bytes()).hexdigest()

    # Measure disk usage of seed dir
    disk_bytes = sum(f.stat().st_size for f in seed_dir.glob("**/*") if f.is_file())
    disk_mb = disk_bytes / (1024 * 1024)

    run_summary = {
        "seed": seed,
        "status": "COMPLETED" if not aborted_early or abort_reason.startswith("Early stopping") else "ABORTED",
        "completed_steps": step,
        "max_steps": max_steps,
        "total_tokens_processed": total_tokens_processed,
        "runtime_seconds": round(total_runtime, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "disk_usage_mb": round(disk_mb, 2),
        "initial_val_macro_ce": baseline_val_macro,
        "final_train_loss": step_records[-1]["train_loss"] if step_records else None,
        "best_val_macro_ce": round(best_val_macro, 4),
        "best_step": best_step,
        "early_stopping_triggered": aborted_early,
        "abort_reason": abort_reason,
        "best_checkpoint_path": str(best_candidate_path.relative_to(REPO_ROOT)),
        "best_checkpoint_sha256": best_hash,
        "final_checkpoint_path": str(final_ckpt_path.relative_to(REPO_ROOT)),
        "final_checkpoint_sha256": final_hash,
        "validation_history": val_records,
        "training_loss_history": step_records
    }

    with open(seed_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"\nSeed {seed} finished in {total_runtime:.1f}s ({total_runtime/60:.2f}m). Best step: {best_step} (Val Macro: {best_val_macro:.4f})")
    print(f"Best checkpoint SHA-256: {best_hash}")
    return run_summary


def run_all_seeds() -> Dict[str, Any]:
    with open(PREREGISTRATION_PATH, "r", encoding="utf-8") as f:
        prereg = json.load(f)

    seeds = prereg["hyperparameters"]["random_seeds"]  # [42, 1337, 2026]
    all_summaries = {}

    for s in seeds:
        summary = run_cpt_seed(seed=s, prereg=prereg, device="cpu")
        all_summaries[f"seed_{s}"] = summary

    master_summary_path = OUTPUT_DIR / "cpt_master_summary.json"
    with open(master_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll seeds completed. Master summary written to: {master_summary_path}")
    return all_summaries


if __name__ == "__main__":
    run_all_seeds()
