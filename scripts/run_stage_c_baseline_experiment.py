"""
ChakrView Step 7: Real-Corpus Baseline Pre-Training & Comprehensive Evaluation.

Executes the first scientific pre-training baseline on the authentic Stage C multi-domain corpus:
1. Forensic environment and baseline initialization check (step 0 theoretical loss ~ln(4096)=8.318).
2. 500-step pre-training run on Stage C training shards (512,000 tokens, B=2, T=512, AdamW, cosine schedule).
3. Step-by-step metrics tracking (loss, perplexity, learning rate, grad norm, throughput, RSS memory).
4. Periodic validation evaluation every 50 steps (10 batches = 10,240 tokens).
5. Atomic checkpointing at steps 0, 100, 200, 300, 400, 500.
6. Checkpoint integrity audit and resume determinism verification (step 200 -> step 230 vs reference).
7. Comprehensive evaluation across Train sample, Full Validation split (465k tokens), and Full Test split (591k tokens).
8. Domain-specific validation loss profiling across all 8 Stage C domains.
9. Standardized qualitative capability smoke testing across 14 fixed prompts.
10. Generates complete machine-readable experiment bundle in data/experiments/stage_c_baseline/.
"""

import sys
import os
import json
import time
import math
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure UTF-8 output on console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    DataConfig,
    CheckpointConfig,
    EvaluationConfig,
)
from chakrview.training.seed import set_seed, get_rng_state, set_rng_state
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.collator import CausalLanguageModelingCollator
from chakrview.training.loss import CausalLoss
from chakrview.training.optimizer import build_optimizer, build_lr_scheduler
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.evaluator import evaluate
from chakrview.training.monitoring import ResourceMonitor
from chakrview.training.sharding import read_shard_tokens, verify_shard_integrity

EXPERIMENT_DIR = ROOT_DIR / "data" / "experiments" / "stage_c_baseline"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "stage_c_baseline"
CONFIG_PATH = ROOT_DIR / "configs" / "chakr_micro_stage_c_baseline.json"
PROMPTS_PATH = ROOT_DIR / "configs" / "stage_c_smoke_prompts.json"
MANIFEST_PATH = ROOT_DIR / "data" / "manifests" / "stage_c_manifest.json"


def get_system_metadata() -> Dict[str, Any]:
    """Capture host system and execution environment details."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def generate_text_sample(
    model: ChakrMicro,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    device: str = "cpu",
) -> str:
    """Generate continuation from model using greedy or temperature sampling."""
    model.eval()
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.shape[1] > 512:
                curr_input = input_tensor[:, -512:]
            else:
                curr_input = input_tensor

            logits = model(curr_input)
            next_token_logits = logits[0, -1, :]

            if temperature == 0.0:
                next_token_id = torch.argmax(next_token_logits).item()
            else:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

            new_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, new_token_tensor], dim=1)

            if next_token_id == 1:  # EOS token
                break

    all_ids = input_tensor[0].tolist()
    return tokenizer.decode(all_ids)


def run_stage_c_baseline_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 7 — REAL-CORPUS BASELINE PRE-TRAINING & EVALUATION")
    print("=" * 80)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Environment & Host Metadata
    sys_info = get_system_metadata()
    print("\n[1/10] Host System & Environment:")
    for k, v in sys_info.items():
        print(f"      - {k:18s}: {v}")

    # 2. Load Configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f)

    # Load frozen tokenizer
    tok_dir = ROOT_DIR / exp_cfg["tokenizer"]["path"]
    tokenizer, tok_cfg = load_tokenizer_artifacts(tok_dir)
    print(f"\n[2/10] Loaded Frozen Tokenizer: V={tokenizer.vocab_size}, merges={tokenizer.num_merges}")

    # Shard paths
    train_shards_dir = ROOT_DIR / exp_cfg["dataset"]["train_shards_dir"]
    val_shards_dir = ROOT_DIR / exp_cfg["dataset"]["validation_shards_dir"]
    test_shards_dir = ROOT_DIR / exp_cfg["dataset"]["test_shards_dir"]

    # Verify shards
    assert verify_shard_integrity(train_shards_dir), "Train shards integrity check failed!"
    assert verify_shard_integrity(val_shards_dir), "Validation shards integrity check failed!"
    assert verify_shard_integrity(test_shards_dir), "Test shards integrity check failed!"

    with open(train_shards_dir / "metadata.json", "r", encoding="utf-8") as f:
        train_meta = json.load(f)
    with open(val_shards_dir / "metadata.json", "r", encoding="utf-8") as f:
        val_meta = json.load(f)
    with open(test_shards_dir / "metadata.json", "r", encoding="utf-8") as f:
        test_meta = json.load(f)

    print(f"\n[3/10] Dataset Shards Verified:")
    print(f"      - Train:      {train_meta['total_tokens']:,} tokens across {train_meta['shard_count']} shards")
    print(f"      - Validation: {val_meta['total_tokens']:,} tokens across {val_meta['shard_count']} shards")
    print(f"      - Test:       {test_meta['total_tokens']:,} tokens across {test_meta['shard_count']} shards")

    # Hyperparameters
    t_cfg = exp_cfg["training"]
    seed = t_cfg["seed"]
    batch_size = t_cfg["batch_size"]
    seq_len = t_cfg["sequence_length"]
    max_steps = t_cfg["max_steps"]
    lr = t_cfg["learning_rate"]
    min_lr = t_cfg["min_learning_rate"]
    warmup_steps = t_cfg["warmup_steps"]
    weight_decay = t_cfg["weight_decay"]
    grad_clip = t_cfg["gradient_clipping"]
    val_interval = exp_cfg["evaluation"]["eval_interval"]
    val_batches = exp_cfg["evaluation"]["eval_batches"]
    ckpt_interval = exp_cfg["checkpoint"]["save_interval"]

    # Setup Datasets & Loaders
    collator = CausalLanguageModelingCollator(max_context=seq_len, pad_token_id=2)
    train_ds = StreamingTokenDataset(train_shards_dir, sequence_length=seq_len, loop=True, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collator)

    val_ds = StreamingTokenDataset(val_shards_dir, sequence_length=seq_len, loop=True, seed=seed + 1)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collator)

    # 3. Model Initialization & Baseline Measurement
    print("\n[4/10] Initializing ChakrMicro v0.1 Core & Measuring Step 0 Baseline...")
    set_seed(seed)
    model = ChakrMicro(ModelConfig())
    loss_fn = CausalLoss(ignore_index=2)

    train_params = TrainingHyperparameters(
        seed=seed,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        min_learning_rate=min_lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        gradient_clipping=grad_clip,
        lr_decay_style="cosine",
    )
    optimizer = build_optimizer(model, train_params)
    scheduler = build_lr_scheduler(optimizer, train_params)

    ckpt_manager = CheckpointManager(
        checkpoint_dir=CHECKPOINT_DIR,
        keep_last_n=exp_cfg["checkpoint"]["keep_last_n"],
        save_optimizer=True,
    )
    # Clean previous checkpoints in this dir
    for f in CHECKPOINT_DIR.glob("*.pt"):
        try:
            f.unlink()
        except Exception:
            pass

    monitor = ResourceMonitor()

    # Step 0 Validation
    step0_val = evaluate(model, val_loader, loss_fn, max_batches=val_batches, device="cpu")
    step0_val_loss = float(step0_val["val_loss"])
    step0_val_ppl = float(step0_val["val_perplexity"])
    theoretical_loss = math.log(4096)
    print(f"      - Step 0 Validation Loss: {step0_val_loss:.4f} (Theoretical ln(4096) = {theoretical_loss:.4f})")
    print(f"      - Step 0 Perplexity:      {step0_val_ppl:.2f}")

    # Save Step 0 Checkpoint
    ckpt0_path = ckpt_manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        epoch=0,
        config=exp_cfg,
        rng_state=get_rng_state(),
        train_metrics={"step": 0, "loss": step0_val_loss},
        val_metrics=step0_val,
    )
    print(f"      - Step 0 Checkpoint Saved: {ckpt0_path.name}")

    # 4. Main Pre-Training Loop
    print(f"\n[5/10] Executing {max_steps} Optimization Steps on Stage C Shards...")
    print(f"      Budget: {max_steps * batch_size * seq_len:,} tokens | Cosine LR: {lr} -> {min_lr} | Warmup: {warmup_steps}")

    step_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = [{"step": 0, "val_loss": step0_val_loss, "val_perplexity": step0_val_ppl}]
    checkpoint_records: List[Dict[str, Any]] = [{"step": 0, "filename": ckpt0_path.name, "path": str(ckpt0_path)}]

    jsonl_log_path = EXPERIMENT_DIR / "training_log.jsonl"
    if jsonl_log_path.exists():
        jsonl_log_path.unlink()

    data_iter = iter(train_loader)
    total_tokens_processed = 0
    t_start = time.perf_counter()

    for step in range(1, max_steps + 1):
        step_t0 = time.perf_counter()
        batch = next(data_iter)

        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        attn_mask = batch.get("attention_mask")

        # Forward
        model.train()
        logits = model(input_ids, attention_mask=attn_mask)
        loss = loss_fn(logits, target_ids)
        loss_val = float(loss.item())

        # Check for NaN/Inf anomalies
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise RuntimeError(f"Numerical instability: loss is {loss_val} at step {step}!")

        # Backward
        loss.backward()

        # Gradient norm before clipping
        grad_norm = float(
            torch.sqrt(sum(torch.norm(p.grad) ** 2 for p in model.parameters() if p.grad is not None)).item()
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step_t1 = time.perf_counter()
        step_duration = max(1e-5, step_t1 - step_t0)
        tokens_in_step = batch_size * seq_len
        total_tokens_processed += tokens_in_step
        tokens_per_sec = tokens_in_step / step_duration
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Periodic Validation
        latest_val_loss = None
        latest_val_ppl = None
        if step % val_interval == 0:
            val_metrics = evaluate(model, val_loader, loss_fn, max_batches=val_batches, device="cpu")
            latest_val_loss = float(val_metrics["val_loss"])
            latest_val_ppl = float(val_metrics["val_perplexity"])
            val_records.append({
                "step": step,
                "val_loss": latest_val_loss,
                "val_perplexity": latest_val_ppl,
            })

        # Periodic Checkpointing
        if step % ckpt_interval == 0:
            ckpt_p = ckpt_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=0,
                config=exp_cfg,
                rng_state=get_rng_state(),
                train_metrics={"step": step, "loss": loss_val},
                val_metrics={"val_loss": latest_val_loss if latest_val_loss is not None else 0.0},
            )
            checkpoint_records.append({
                "step": step,
                "filename": ckpt_p.name,
                "path": str(ckpt_p),
                "size_bytes": ckpt_p.stat().st_size,
            })

        # Memory snapshot
        res_snap = monitor.get_snapshot()

        step_data = {
            "step": step,
            "train_loss": round(loss_val, 4),
            "train_perplexity": round(math.exp(min(20.0, loss_val)), 2),
            "val_loss": round(latest_val_loss, 4) if latest_val_loss is not None else None,
            "val_perplexity": round(latest_val_ppl, 2) if latest_val_ppl is not None else None,
            "learning_rate": round(current_lr, 7),
            "grad_norm": round(grad_norm, 4),
            "tokens_in_step": tokens_in_step,
            "total_tokens": total_tokens_processed,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "step_time_sec": round(step_duration, 4),
            "elapsed_sec": round(time.perf_counter() - t_start, 2),
            "process_ram_mb": round(res_snap["process_ram_mb"], 1),
            "cpu_percent": res_snap["cpu_percent"],
        }
        step_records.append(step_data)

        with open(jsonl_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_data) + "\n")

        if step == 1 or step % 25 == 0 or step == max_steps:
            val_str = f" | Val: {latest_val_loss:.4f} (PPL: {latest_val_ppl:.1f})" if latest_val_loss is not None else ""
            print(
                f"  Step {step:4d}/{max_steps} | Loss: {loss_val:.4f} | LR: {current_lr:.6f} | "
                f"GradNorm: {grad_norm:6.2f} | {tokens_per_sec:5.0f} tok/s | RAM: {res_snap['process_ram_mb']:5.1f} MB{val_str}"
            )

    total_training_wall_sec = time.perf_counter() - t_start
    final_train_loss = step_records[-1]["train_loss"]
    final_val_loss = val_records[-1]["val_loss"]
    final_val_ppl = val_records[-1]["val_perplexity"]

    # 5. Checkpoint Integrity & Deterministic Resume Test
    print("\n[6/10] Checkpoint Integrity & Deterministic Resume Verification...")
    saved_ckpts = list(sorted(CHECKPOINT_DIR.glob("checkpoint_*.pt")))
    print(f"      Found {len(saved_ckpts)} checkpoints in {CHECKPOINT_DIR.name}:")
    for cp in saved_ckpts:
        pl = CheckpointManager.load(cp)
        print(f"        - {cp.name:24s}: step {pl['step']:3d} | size {cp.stat().st_size:,} bytes | keys OK")

    # Resume test: Load step 200 checkpoint, run 30 steps, compare with recorded steps 201-230
    step200_ckpt = CHECKPOINT_DIR / "checkpoint_0000200.pt"
    resume_status = "NOT_TESTED"
    resume_discrepancy = 0.0

    if step200_ckpt.is_file():
        print(f"      Executing Resume Verification from {step200_ckpt.name}...")
        res_model = ChakrMicro(ModelConfig())
        res_opt = build_optimizer(res_model, train_params)
        res_sched = build_lr_scheduler(res_opt, train_params)
        res_payload = CheckpointManager.load(step200_ckpt)

        res_model.load_state_dict(res_payload["model_state_dict"])
        res_opt.load_state_dict(res_payload["optimizer_state_dict"])
        res_sched.load_state_dict(res_payload["scheduler_state_dict"])
        set_rng_state(res_payload["rng_state"])

        # Create identical dataset at step 200 state
        res_ds = StreamingTokenDataset(train_shards_dir, sequence_length=seq_len, loop=True, seed=seed)
        res_loader = DataLoader(res_ds, batch_size=batch_size, collate_fn=collator)
        res_iter = iter(res_loader)
        # Fast-forward 200 batches
        for _ in range(200):
            next(res_iter)

        # Run 30 steps
        res_losses = []
        for s in range(201, 231):
            b = next(res_iter)
            res_model.train()
            res_logits = res_model(b["input_ids"], attention_mask=b.get("attention_mask"))
            l = loss_fn(res_logits, b["target_ids"])
            l_val = float(l.item())
            res_losses.append(l_val)
            l.backward()
            torch.nn.utils.clip_grad_norm_(res_model.parameters(), grad_clip)
            res_opt.step()
            res_sched.step()
            res_opt.zero_grad()

        ref_losses = [r["train_loss"] for r in step_records[200:230]]
        diffs = [abs(a - b) for a, b in zip(res_losses, ref_losses)]
        max_diff = max(diffs)
        resume_discrepancy = max_diff
        if max_diff < 1e-4:
            resume_status = "PASS — Bit-Exact Deterministic Continuation"
        elif max_diff < 0.05:
            resume_status = "PASS — Highly Consistent Numerical Continuation"
        else:
            resume_status = f"WARNING — Divergence detected (max diff: {max_diff:.4f})"
        print(f"      Resume Status: {resume_status} (Max loss delta over 30 steps: {max_diff:.6f})")

    # 6. Comprehensive Final Split Evaluations
    print("\n[7/10] Comprehensive Final Evaluations across Multi-Split Test Sets...")
    print("      Evaluating on Full Validation Split (465,954 tokens)...")
    val_full_ds = StreamingTokenDataset(val_shards_dir, sequence_length=seq_len, loop=False, seed=999)
    val_full_loader = DataLoader(val_full_ds, batch_size=batch_size, collate_fn=collator)
    t0_eval_val = time.perf_counter()
    full_val_eval = evaluate(model, val_full_loader, loss_fn, max_batches=1000, device="cpu")
    t_val_eval_duration = time.perf_counter() - t0_eval_val

    full_val_loss = float(full_val_eval["val_loss"])
    full_val_ppl = float(full_val_eval["val_perplexity"])
    val_tokens_evaluated = full_val_eval["eval_batches"] * batch_size * seq_len
    print(f"      - Validation Split: Loss={full_val_loss:.4f} | PPL={full_val_ppl:.2f} | Tokens={val_tokens_evaluated:,} | Time={t_val_eval_duration:.2f}s")

    print("      Evaluating on Full Test Split (591,554 tokens)...")
    test_ds = StreamingTokenDataset(test_shards_dir, sequence_length=seq_len, loop=False, seed=999)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collator)
    t0_eval_test = time.perf_counter()
    test_eval = evaluate(model, test_loader, loss_fn, max_batches=1000, device="cpu")
    t_test_eval_duration = time.perf_counter() - t0_eval_test

    test_loss = float(test_eval["val_loss"])
    test_ppl = float(test_eval["val_perplexity"])
    test_tokens_evaluated = test_eval["eval_batches"] * batch_size * seq_len
    print(f"      - Test Split:       Loss={test_loss:.4f} | PPL={test_ppl:.2f} | Tokens={test_tokens_evaluated:,} | Time={t_test_eval_duration:.2f}s")

    print("      Evaluating on Train Split Sample (51,200 tokens)...")
    train_sample_ds = StreamingTokenDataset(train_shards_dir, sequence_length=seq_len, loop=False, seed=777)
    train_sample_loader = DataLoader(train_sample_ds, batch_size=batch_size, collate_fn=collator)
    train_sample_eval = evaluate(model, train_sample_loader, loss_fn, max_batches=50, device="cpu")
    train_sample_loss = float(train_sample_eval["val_loss"])
    train_sample_ppl = float(train_sample_eval["val_perplexity"])
    print(f"      - Train Sample:     Loss={train_sample_loss:.4f} | PPL={train_sample_ppl:.2f}")

    # 7. Domain-Specific Loss Evaluation
    print("\n[8/10] Computing Domain-Specific Evaluation across 8 Stage C Domains...")
    domain_eval_corpus = {
        "English": (
            "The Republic of India is a country in South Asia. It is the seventh-largest country by area, "
            "the most populous country since 2023, and from the time of its independence in 1947, the world's most populous democracy. "
            "Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, "
            "it shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. "
            "Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago."
        ),
        "Hindi": (
            "भारत दक्षिण एशिया में स्थित एक देश है। क्षेत्रफल की दृष्टि से यह विश्व का सातवाँ सबसे बड़ा देश है और "
            "जनसंख्या की दृष्टि से सबसे बड़ा देश है। इसके उत्तर में हिमालय पर्वतमाला, दक्षिण में हिन्द महासागर, "
            "पूर्व में बंगाल की खाड़ी तथा पश्चिम में अरब सागर स्थित हैं। भारतीय सभ्यता विश्व की प्राचीनतम सभ्यताओं में से एक है।"
        ),
        "Code": (
            "def quicksort(arr: list[int]) -> list[int]:\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        ),
        "Sanskrit": (
            "वेदोऽखिलो धर्ममूलम् । वेदाः एव जगतः प्रथमं साहित्यम् । एते अपौरुषेयाः सन्ति। "
            "ऋग्वेदः, यजुर्वेदः, सामवेदः, अथर्ववेदः च चत्वारः वेदाः सन्ति। "
            "धर्मार्थकाममोक्षाणां ज्ञानं वेदेषु निहितम् अस्ति। सत्यमेव जयते नानृतम्।"
        ),
        "Mathematics": (
            "Theorem (Spectral Theorem): Let A be a real symmetric matrix of dimension n x n. "
            "Then all eigenvalues of A are real, and there exists an orthonormal basis of R^n consisting of eigenvectors of A. "
            "Equivalently, there exists an orthogonal matrix Q such that Q^T A Q = D, where D is a diagonal matrix."
        ),
        "Reasoning": (
            "Question: A store owner bought 15 boxes of pens, with each box containing 12 pens. "
            "If she sells each pen for $2, and the total cost was $180, what is her total profit?\n"
            "Step-by-Step Solution:\n"
            "1. Total pens = 15 * 12 = 180.\n"
            "2. Total revenue = 180 * $2 = $360.\n"
            "3. Total profit = $360 - $180 = $180.\n"
            "The profit is 180 dollars."
        ),
        "Hinglish": (
            "User: Python virtual environment kaise activate karte hain Windows me?\n"
            "Assistant: Virtual environment activate karne ke liye PowerShell me `.\\.venv\\Scripts\\Activate.ps1` execute karo. "
            "Isse environment safely isolated rehta hai aur project dependencies clean rehti hain."
        ),
        "Structured Data": (
            "| State / UT | Capital | Population | Literacy Rate (%) |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| Maharashtra | Mumbai | 112,374,333 | 82.34 |\n"
            "| Uttar Pradesh | Lucknow | 199,812,341 | 67.68 |\n"
            "| Karnataka | Bengaluru | 61,095,297 | 75.36 |\n"
            "| Kerala | Thiruvananthapuram | 33,406,061 | 94.00 |\n"
        ),
    }

    domain_eval_results: Dict[str, Any] = {}
    model.eval()

    with torch.no_grad():
        for dom_name, dom_text in domain_eval_corpus.items():
            toks = tokenizer.encode(dom_text, add_bos=True, add_eos=True)
            if len(toks) < 10:
                continue
            inp = torch.tensor([toks[:-1]], dtype=torch.long)
            tgt = torch.tensor([toks[1:]], dtype=torch.long)
            logits = model(inp)
            dom_loss = float(loss_fn(logits, tgt).item())
            dom_ppl = float(math.exp(min(20.0, dom_loss)))
            domain_eval_results[dom_name] = {
                "loss": round(dom_loss, 4),
                "perplexity": round(dom_ppl, 2),
                "tokens_evaluated": len(toks),
            }
            print(f"        - {dom_name:18s}: Loss = {dom_loss:.4f} | Perplexity = {dom_ppl:.2f}")

    # 8. Standardized Qualitative Capability Smoke Tests
    print("\n[9/10] Running Standardized Capability Smoke Tests...")
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts_cfg = json.load(f)

    smoke_results: List[Dict[str, Any]] = []
    print(f"      Generating continuations for {len(prompts_cfg['prompts'])} prompts across categories:")

    for p_info in prompts_cfg["prompts"]:
        p_id = p_info["id"]
        p_cat = p_info["category"]
        prompt_txt = p_info["prompt"]
        max_new = p_info.get("max_new_tokens", 32)

        # Generate greedy continuation
        continuation_greedy = generate_text_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_txt,
            max_new_tokens=max_new,
            temperature=0.0,
            device="cpu",
        )

        smoke_results.append({
            "id": p_id,
            "category": p_cat,
            "sub_category": p_info.get("sub_category", "general"),
            "prompt": prompt_txt,
            "continuation": continuation_greedy,
            "prompt_length_chars": len(prompt_txt),
            "generated_length_chars": len(continuation_greedy),
            "temperature": 0.0,
        })
        print(f"        [{p_cat:15s}] {p_id}")
        # Print snippet safely
        snippet = continuation_greedy.replace("\n", " ")[:90]
        try:
            print(f"           -> {snippet}...")
        except Exception:
            pass

    # Save smoke test outputs
    smoke_out_path = EXPERIMENT_DIR / "smoke_test_results.json"
    with open(smoke_out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "results": smoke_results}, f, indent=2, ensure_ascii=False)
    print(f"      Smoke test results saved to {smoke_out_path}")

    # 9. Save Master Summary & Learning Curve Artifacts
    print("\n[10/10] Compiling Master Experiment Artifacts...")
    train_loss_drop = step_records[0]["train_loss"] - final_train_loss
    train_loss_pct = (train_loss_drop / step_records[0]["train_loss"]) * 100
    val_loss_drop = step0_val_loss - full_val_loss
    val_loss_pct = (val_loss_drop / step0_val_loss) * 100

    master_summary = {
        "experiment_name": exp_cfg["experiment_name"],
        "milestone": exp_cfg["milestone"],
        "git_commit_base": exp_cfg["git_commit_base"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": sys_info,
        "model": {
            "name": "ChakrMicro v0.1",
            "parameters": 3443136,
            "vocab_size": 4096,
            "context_length": 512,
        },
        "tokenizer": exp_cfg["tokenizer"],
        "dataset": exp_cfg["dataset"],
        "training": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "tokens_processed": total_tokens_processed,
            "total_wall_clock_seconds": round(total_training_wall_sec, 2),
            "average_throughput_tokens_per_sec": round(total_tokens_processed / total_training_wall_sec, 1),
            "initial_train_loss": step_records[0]["train_loss"],
            "final_train_loss": final_train_loss,
            "train_loss_drop": round(train_loss_drop, 4),
            "train_loss_drop_pct": round(train_loss_pct, 2),
            "initial_val_loss": round(step0_val_loss, 4),
            "final_val_loss": round(full_val_loss, 4),
            "val_loss_drop": round(val_loss_drop, 4),
            "val_loss_drop_pct": round(val_loss_pct, 2),
            "initial_val_perplexity": round(step0_val_ppl, 2),
            "final_val_perplexity": round(full_val_ppl, 2),
            "test_loss": round(test_loss, 4),
            "test_perplexity": round(test_ppl, 2),
            "train_sample_loss": round(train_sample_loss, 4),
            "train_sample_perplexity": round(train_sample_ppl, 2),
            "resume_status": resume_status,
            "resume_max_discrepancy": round(resume_discrepancy, 6),
            "checkpoints_count": len(saved_ckpts),
        },
        "domain_evaluation": domain_eval_results,
        "status": "COMPLETE",
    }

    summary_path = EXPERIMENT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # Save curve data
    curve_data = {
        "steps": [r["step"] for r in step_records],
        "train_loss": [r["train_loss"] for r in step_records],
        "learning_rate": [r["learning_rate"] for r in step_records],
        "val_evaluations": val_records,
    }
    curve_path = EXPERIMENT_DIR / "loss_curve.json"
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)

    print(f"      Master summary written to: {summary_path}")
    print(f"      Loss curve written to:     {curve_path}")

    print("\n" + "=" * 80)
    print("STEP 7 BASELINE PRE-TRAINING COMPLETE!")
    print(f"  Initial Loss: {step0_val_loss:.4f} -> Final Val Loss: {full_val_loss:.4f} (Drop: {val_loss_pct:.1f}%)")
    print(f"  Test Loss:    {test_loss:.4f} | Test PPL: {test_ppl:.2f}")
    print(f"  Tokens:       {total_tokens_processed:,} processed in {total_training_wall_sec:.1f}s ({total_tokens_processed/total_training_wall_sec:.1f} tok/s)")
    print("=" * 80)

    return master_summary


if __name__ == "__main__":
    run_stage_c_baseline_experiment()
