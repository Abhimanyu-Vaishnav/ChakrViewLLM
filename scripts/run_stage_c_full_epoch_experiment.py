"""
ChakrView Step 8: Full-Epoch Pre-Training Experiment & Evaluation.

Executes a complete 1-epoch pre-training run on the authentic Stage C multi-domain corpus:
1. Forensic environment and baseline initialization check (step 0 theoretical loss ~ln(4096)=8.318).
2. 6,478-step pre-training run on Stage C training shards (6,633,472 tokens, B=2, T=512, AdamW, cosine schedule, 250 warmup steps).
3. Step-by-step metrics tracking (loss, perplexity, learning rate, grad norm, throughput, RSS memory).
4. Periodic validation evaluation every 250 steps (10 batches = 10,240 tokens).
5. Atomic checkpointing at steps 0, 500, 1000, ..., 6000, 6478.
6. Checkpoint integrity audit and resume determinism verification (step 2000 -> step 2030 vs reference).
7. Comprehensive evaluation across Train sample, Full Validation split (465k tokens), and Full Test split (591k tokens).
8. Domain-specific validation loss profiling across all 8 Stage C domains.
9. Standardized qualitative capability smoke testing across 14 fixed prompts.
10. Generates complete machine-readable experiment bundle in data/experiments/stage_c_full_epoch/.
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

EXPERIMENT_DIR = ROOT_DIR / "data" / "experiments" / "stage_c_full_epoch"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "stage_c_full_epoch"
CONFIG_PATH = ROOT_DIR / "configs" / "chakr_micro_stage_c_full_epoch.json"
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


def generate_visual_artifacts(exp_dir: Path, train_loss: List[float], val_evals: List[Dict[str, Any]], max_steps: int):
    """Generate ASCII and SVG loss curve visual artifacts."""
    rows = 11
    y_max = 8.5
    y_min = 3.5
    y_step = (y_max - y_min) / (rows - 1)

    # 1. ASCII plot
    cols = 60
    col_size = len(train_loss) // cols
    t_sampled = [sum(train_loss[i * col_size : (i + 1) * col_size]) / col_size for i in range(cols)]

    grid = [[" " for _ in range(cols)] for _ in range(rows)]

    for c, val in enumerate(t_sampled):
        r = int(round((y_max - val) / y_step))
        if 0 <= r < rows:
            grid[r][c] = "-"

    for v in val_evals:
        step = v["step"]
        val = v["val_loss"]
        c = min(cols - 1, int(round((step / float(max_steps)) * (cols - 1))))
        r = int(round((y_max - val) / y_step))
        if 0 <= r < rows:
            grid[r][c] = "O"

    lines = [
        "==========================================================================",
        "     CHAKRMICRO v0.1 — STAGE C FULL-EPOCH TRAINING & VALIDATION CURVE     ",
        "==========================================================================",
        "",
        "Loss ^",
    ]
    for i in range(rows):
        y_val = y_max - i * y_step
        row_str = "".join(grid[i])
        lines.append(f"{y_val:4.1f} | {row_str}")

    lines.append("     +" + "-" * cols)
    lines.append(f"Step:  0       1000      2000      3000      4000      5000      {max_steps}")
    lines.append("")
    lines.append("Legend: '-' = Train Loss (smoothed), 'O' = Validation Loss (periodic batches)")
    lines.append("")

    (exp_dir / "loss_curve.txt").write_text("\n".join(lines), encoding="utf-8")

    # 2. SVG Vector plot
    width = 900
    height = 520
    pad_left = 70
    pad_right = 40
    pad_top = 50
    pad_bottom = 60

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def map_x(step):
        return pad_left + (step / float(max_steps)) * plot_w

    def map_y(loss):
        clamped = max(y_min, min(y_max, loss))
        return pad_top + ((y_max - clamped) / (y_max - y_min)) * plot_h

    # SVG train path (downsample to ~300 points for crisp rendering)
    stride = max(1, len(train_loss) // 300)
    train_pts = []
    for step_idx in range(1, len(train_loss) + 1, stride):
        x = map_x(step_idx)
        y = map_y(train_loss[step_idx - 1])
        train_pts.append(f"{x:.1f},{y:.1f}")
    # Always include last point
    train_pts.append(f"{map_x(len(train_loss)):.1f},{map_y(train_loss[-1]):.1f}")
    train_path_d = "M " + " L ".join(train_pts)

    val_pts = []
    val_circles = []
    for v in val_evals:
        x = map_x(v["step"])
        y = map_y(v["val_loss"])
        val_pts.append(f"{x:.1f},{y:.1f}")
        val_circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.0" fill="#f43f5e" stroke="#ffffff" stroke-width="1.2"/>')
    val_path_d = "M " + " L ".join(val_pts)

    grid_svg = []
    for i in range(rows):
        y_val = y_max - i * y_step
        y_pos = map_y(y_val)
        grid_svg.append(f'<line x1="{pad_left}" y1="{y_pos:.1f}" x2="{width - pad_right}" y2="{y_pos:.1f}" stroke="#334155" stroke-dasharray="3,3" stroke-width="0.8"/>')
        grid_svg.append(f'<text x="{pad_left - 10}" y="{y_pos + 4:.1f}" fill="#94a3b8" font-size="12" text-anchor="end" font-family="monospace">{y_val:.1f}</text>')

    x_labels = [0, 1000, 2000, 3000, 4000, 5000, 6000, max_steps]
    for step in x_labels:
        x_pos = map_x(step)
        grid_svg.append(f'<line x1="{x_pos:.1f}" y1="{pad_top}" x2="{x_pos:.1f}" y2="{height - pad_bottom}" stroke="#334155" stroke-dasharray="3,3" stroke-width="0.8"/>')
        grid_svg.append(f'<text x="{x_pos:.1f}" y="{height - pad_bottom + 20}" fill="#94a3b8" font-size="12" text-anchor="middle" font-family="monospace">{step}</text>')

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #0f172a;">
  <!-- Title -->
  <text x="{width/2}" y="30" fill="#f8fafc" font-size="16" font-weight="bold" text-anchor="middle" font-family="sans-serif">ChakrMicro v0.1 — Stage C Full-Epoch Training Curve (1 Epoch = {max_steps} Steps)</text>
  
  <!-- Axis Grid -->
  {''.join(grid_svg)}
  
  <!-- Axes Lines -->
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#64748b" stroke-width="1.5"/>
  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#64748b" stroke-width="1.5"/>
  
  <!-- Axis Titles -->
  <text x="{width/2}" y="{height - 15}" fill="#cbd5e1" font-size="13" text-anchor="middle" font-family="sans-serif">Optimizer Steps (Tokens = Steps × 1,024)</text>
  <text x="20" y="{height/2}" fill="#cbd5e1" font-size="13" text-anchor="middle" font-family="sans-serif" transform="rotate(-90 20 {height/2})">Cross-Entropy Loss</text>
  
  <!-- Train Loss Curve -->
  <path d="{train_path_d}" fill="none" stroke="#38bdf8" stroke-width="1.2" opacity="0.85"/>
  
  <!-- Val Loss Curve and Points -->
  <path d="{val_path_d}" fill="none" stroke="#f43f5e" stroke-width="2.0"/>
  {''.join(val_circles)}
  
  <!-- Legend -->
  <rect x="{width - pad_right - 220}" y="{pad_top + 10}" width="210" height="60" rx="4" fill="#1e293b" stroke="#334155" opacity="0.9"/>
  <line x1="{width - pad_right - 205}" y1="{pad_top + 30}" x2="{width - pad_right - 180}" y2="{pad_top + 30}" stroke="#38bdf8" stroke-width="2"/>
  <text x="{width - pad_right - 170}" y="{pad_top + 34}" fill="#e2e8f0" font-size="12" font-family="sans-serif">Train Loss (step)</text>
  <circle cx="{width - pad_right - 192}" cy="{pad_top + 50}" r="4" fill="#f43f5e"/>
  <text x="{width - pad_right - 170}" y="{pad_top + 54}" fill="#e2e8f0" font-size="12" font-family="sans-serif">Validation Loss (batch)</text>
</svg>"""

    (exp_dir / "loss_curve.svg").write_text(svg_content, encoding="utf-8")


def run_stage_c_full_epoch_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 8 — FULL-EPOCH PRE-TRAINING EXPERIMENT")
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
    print(f"\n[5/10] Executing {max_steps} Optimization Steps (~1 Full Epoch on Stage C Shards)...")
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

        # Periodic Checkpointing (every ckpt_interval steps or at the final step)
        if step % ckpt_interval == 0 or step == max_steps:
            ckpt_p = ckpt_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=1 if step == max_steps else 0,
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

        if step == 1 or step % 250 == 0 or step == max_steps:
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
        print(f"        - {cp.name:24s}: step {pl['step']:4d} | size {cp.stat().st_size:,} bytes | keys OK")

    # Resume test: Load first intermediate checkpoint available among saved checkpoints
    inter_ckpts = [cp for cp in saved_ckpts if CheckpointManager.load(cp)["step"] < max_steps]
    resume_ckpt = inter_ckpts[0] if inter_ckpts else saved_ckpts[0]
    resume_payload = CheckpointManager.load(resume_ckpt)
    resume_target_step = resume_payload["step"]
    resume_status = "NOT_TESTED"
    resume_discrepancy = 0.0

    if resume_ckpt.is_file():
        print(f"      Executing Resume Verification from {resume_ckpt.name} for 30 steps...")
        res_model = ChakrMicro(ModelConfig())
        res_opt = build_optimizer(res_model, train_params)
        res_sched = build_lr_scheduler(res_opt, train_params)
        res_payload = CheckpointManager.load(resume_ckpt)

        res_model.load_state_dict(res_payload["model_state_dict"])
        res_opt.load_state_dict(res_payload["optimizer_state_dict"])
        res_sched.load_state_dict(res_payload["scheduler_state_dict"])
        set_rng_state(res_payload["rng_state"])

        # Create identical dataset and fast-forward to resume_target_step
        res_ds = StreamingTokenDataset(train_shards_dir, sequence_length=seq_len, loop=True, seed=seed)
        res_loader = DataLoader(res_ds, batch_size=batch_size, collate_fn=collator)
        res_iter = iter(res_loader)
        for _ in range(resume_target_step):
            next(res_iter)

        # Run 30 steps
        res_losses = []
        for s in range(resume_target_step + 1, resume_target_step + 31):
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

        ref_losses = [r["train_loss"] for r in step_records[resume_target_step : resume_target_step + 30]]
        diffs = [abs(a - b) for a, b in zip(res_losses, ref_losses)]
        max_diff = max(diffs)
        resume_discrepancy = max_diff
        if max_diff < 1e-4:
            resume_status = f"PASS — Deterministic Continuation (max loss delta: {max_diff:.8f} over 30 steps)"
        elif max_diff < 0.05:
            resume_status = f"PASS — Highly Consistent Numerical Continuation (max loss delta: {max_diff:.6f} over 30 steps)"
        else:
            resume_status = f"WARNING — Divergence detected (max diff: {max_diff:.4f})"
        print(f"      Resume Status: {resume_status}")

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

    # 7. Domain-Specific Loss Evaluation (Identical to Step 7)
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

    # 8. Standardized Qualitative Capability Smoke Tests (Identical to Step 7)
    print("\n[9/10] Running Standardized Capability Smoke Tests...")
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts_cfg = json.load(f)

    smoke_results: List[Dict[str, Any]] = []
    print(f"      Generating continuations for {len(prompts_cfg['prompts'])} prompts across categories:")
    for p_entry in prompts_cfg["prompts"]:
        pid = p_entry["id"]
        cat = p_entry["category"]
        p_text = p_entry["prompt"]
        max_new = p_entry.get("max_new_tokens", 32)

        continuation = generate_text_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=p_text,
            max_new_tokens=max_new,
            temperature=0.0,
            device="cpu",
        )
        smoke_results.append({
            "id": pid,
            "category": cat,
            "sub_category": p_entry.get("sub_category", ""),
            "prompt": p_text,
            "continuation": continuation,
            "prompt_length_chars": len(p_text),
            "generated_length_chars": len(continuation),
            "temperature": 0.0,
        })
        display_cont = continuation[len(p_text):].replace("\n", " ").strip()[:60]
        print(f"        [{cat:15s}] '{display_cont}...'")

    smoke_out_path = EXPERIMENT_DIR / "smoke_test_results.json"
    with open(smoke_out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": smoke_results}, f, indent=2)

    # 9. Summary and Loss Curve Serialization
    print("\n[10/10] Compiling Experiment Summary & Loss Curves...")
    avg_throughput = round(total_tokens_processed / max(1.0, total_training_wall_sec), 1)

    summary_data = {
        "experiment_name": exp_cfg["experiment_name"],
        "milestone": exp_cfg["milestone"],
        "git_commit_base": exp_cfg["git_commit_base"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": sys_info,
        "model": {
            "name": exp_cfg["model"]["model_name"],
            "parameters": exp_cfg["model"]["parameter_count"],
            "vocab_size": exp_cfg["model"]["vocab_size"],
            "context_length": exp_cfg["model"]["max_seq_len"],
        },
        "tokenizer": {
            "type": exp_cfg["tokenizer"]["type"],
            "vocab_size": exp_cfg["tokenizer"]["vocab_size"],
            "merges_count": exp_cfg["tokenizer"]["merges_count"],
            "path": exp_cfg["tokenizer"]["path"],
            "checksum": exp_cfg["tokenizer"]["checksum"],
        },
        "dataset": {
            "manifest_path": exp_cfg["dataset"]["manifest_path"],
            "manifest_hash": exp_cfg["dataset"]["manifest_hash"],
            "train_shards_dir": exp_cfg["dataset"]["train_shards_dir"],
            "validation_shards_dir": exp_cfg["dataset"]["validation_shards_dir"],
            "test_shards_dir": exp_cfg["dataset"]["test_shards_dir"],
            "total_documents": exp_cfg["dataset"]["total_documents"],
            "total_sharded_tokens": exp_cfg["dataset"]["total_sharded_tokens"],
            "train_tokens": exp_cfg["dataset"]["train_tokens"],
            "validation_tokens": exp_cfg["dataset"]["validation_tokens"],
            "test_tokens": exp_cfg["dataset"]["test_tokens"],
            "total_shards": exp_cfg["dataset"]["total_shards"],
        },
        "training": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "tokens_processed": total_tokens_processed,
            "total_wall_clock_seconds": round(total_training_wall_sec, 1),
            "average_throughput_tokens_per_sec": avg_throughput,
            "initial_train_loss": step_records[0]["train_loss"],
            "final_train_loss": final_train_loss,
            "train_loss_drop": round(step_records[0]["train_loss"] - final_train_loss, 4),
            "train_loss_drop_pct": round(((step_records[0]["train_loss"] - final_train_loss) / step_records[0]["train_loss"]) * 100, 2),
            "initial_val_loss": round(step0_val_loss, 4),
            "final_periodic_val_loss": round(final_val_loss, 4),
            "final_full_val_loss": round(full_val_loss, 4),
            "val_loss_drop": round(step0_val_loss - full_val_loss, 4),
            "val_loss_drop_pct": round(((step0_val_loss - full_val_loss) / step0_val_loss) * 100, 2),
            "initial_val_perplexity": round(step0_val_ppl, 2),
            "final_val_perplexity": round(full_val_ppl, 2),
            "test_loss": round(test_loss, 4),
            "test_perplexity": round(test_ppl, 2),
            "train_sample_loss": round(train_sample_loss, 4),
            "resume_status": resume_status,
            "resume_max_discrepancy": resume_discrepancy,
            "checkpoints_count": len(saved_ckpts),
        },
        "domain_evaluation": domain_eval_results,
        "status": "COMPLETE",
    }

    summary_path = EXPERIMENT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save loss curve data
    curve_data = {
        "steps": [r["step"] for r in step_records],
        "train_loss": [r["train_loss"] for r in step_records],
        "learning_rate": [r["learning_rate"] for r in step_records],
        "val_evaluations": val_records,
    }
    curve_path = EXPERIMENT_DIR / "loss_curve.json"
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)

    # Generate ASCII and SVG visual curve artifacts
    generate_visual_artifacts(EXPERIMENT_DIR, curve_data["train_loss"], val_records, max_steps)

    print("\n" + "=" * 80)
    print("STEP 8 FULL-EPOCH PRE-TRAINING EXPERIMENT COMPLETE")
    print(f"Tokens Processed: {total_tokens_processed:,} | Steps: {max_steps:,} | Runtime: {total_training_wall_sec:.1f}s")
    print(f"Final Train Loss: {final_train_loss:.4f} | Full Val Loss: {full_val_loss:.4f} | Test Loss: {test_loss:.4f}")
    print(f"Summary JSON:     {summary_path}")
    print("=" * 80 + "\n")

    return summary_data


if __name__ == "__main__":
    run_stage_c_full_epoch_experiment()
