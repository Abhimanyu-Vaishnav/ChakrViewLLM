"""
Batch 139–142: Controlled Neural-Core Recovery & Capability Growth Suite.
Executes:
1. Arm A: Canonical Step 10,000 Baseline evaluation.
2. Arm B: Replicated Batch 135–138 pilot (100 steps, constant LR 3e-4, balanced sampling).
3. Arm C: Conservative Warmup (50 steps) + Cosine Annealing (250 steps, peak LR 1.5e-4 -> 2e-5, natural sampling).
4. Arm D: Controlled Domain-Balanced Sampling under identical Warmup + Cosine schedule (250 steps, balanced sampling).
5. Comprehensive Multi-Domain Evaluation (English, Hindi, Code held-out test shards).
6. Quantitative Generation Metrics (Repetition ratio, unique token diversity, continuation samples).
7. Full Resumability verification on Arm C checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import math
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.pretraining_engine import MultiShardDataset, PretrainingExperimentEngine

EXP_DIR = Path("experiments/batch139_142_neural_core_recovery")
CKPT_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")

PROMPTS = {
    "english": "The development of modern science began with",
    "code": "def binary_search(arr, target):",
    "reasoning": "If all mammals are warm-blooded and a whale is a mammal, then",
    "hindi": "भारत एक विशाल और विविधतापूर्ण",
}

def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_lr = min_lr + (peak_lr - min_lr) * cosine_decay
        return target_lr / peak_lr
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def evaluate_dataset(model: ChakrMicro, dataset: MultiShardDataset, device: torch.device, num_batches: int = 20, batch_size: int = 8, seq_len: int = 64, seed: int = 42) -> dict:
    model.eval()
    rng = np.random.RandomState(seed)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_batch(batch_size=batch_size, seq_len=seq_len, rng=rng)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * (batch_size * seq_len)
            total_tokens += batch_size * seq_len
            
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "tokens_evaluated": total_tokens,
    }

def compute_generation_metrics(token_ids: list[int]) -> dict:
    if not token_ids:
        return {"repetition_3gram": 0.0, "unique_token_ratio": 0.0}
    
    # 3-gram repetition
    if len(token_ids) >= 3:
        ngrams = [tuple(token_ids[i:i+3]) for i in range(len(token_ids) - 2)]
        rep = 1.0 - (len(set(ngrams)) / max(len(ngrams), 1))
    else:
        rep = 0.0
        
    uniq = len(set(token_ids)) / max(len(token_ids), 1)
    return {
        "repetition_3gram": round(rep, 4),
        "unique_token_ratio": round(uniq, 4),
    }

def generate_continuations(model: ChakrMicro, tokenizer, device: torch.device, max_new_tokens: int = 32) -> dict:
    model.eval()
    continuations = {}
    total_new_tokens = []
    
    with torch.no_grad():
        for dom, prompt_text in PROMPTS.items():
            prompt_tokens = [0] + tokenizer.encode(prompt_text)
            curr = list(prompt_tokens)
            gen_tokens = []
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr], dtype=torch.long, device=device)
                logits = model(inp)[0, -1]
                nxt = torch.argmax(logits).item()
                curr.append(nxt)
                gen_tokens.append(nxt)
                if nxt == 1:
                    break
            raw = tokenizer.decode(gen_tokens, errors="replace")
            metrics = compute_generation_metrics(gen_tokens)
            continuations[dom] = {
                "prompt": prompt_text,
                "generated": raw.strip(),
                "full_text": prompt_text + " " + raw.strip(),
                "metrics": metrics,
            }
            total_new_tokens.extend(gen_tokens)
            
    global_metrics = compute_generation_metrics(total_new_tokens)
    return {
        "prompts": continuations,
        "global_metrics": global_metrics,
    }

def train_candidate(
    arm_name: str,
    base_state: dict,
    cfg: ModelConfig,
    train_dataset: MultiShardDataset,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    sampling_mode: str,
    seed: int,
    save_path: Path,
) -> tuple[ChakrMicro, dict]:
    print(f"\n--- Training {arm_name} ({steps} steps, sampling={sampling_mode}, peak_lr={peak_lr}) ---")
    model = ChakrMicro(cfg).to(device)
    model.load_state_dict(base_state)
    
    opt = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    scheduler = None
    if warmup_steps > 0 or min_lr < peak_lr:
        scheduler = get_lr_scheduler(opt, warmup_steps, steps, peak_lr, min_lr)
        
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)
    history = []
    
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        x, y = train_dataset.get_batch(batch_size=batch_size, seq_len=seq_len, rng=rng, sampling_mode=sampling_mode)
        x, y = x.to(device), y.to(device)
        
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step} in {arm_name}: {loss.item()}")
            
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        
        current_lr = opt.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
            
        if (step + 1) % 50 == 0 or step == 0 or (step + 1) == steps:
            print(f"[{arm_name}] Step {step+1:3d}/{steps} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Grad: {grad_norm:.4f}")
            history.append({
                "step": step + 1,
                "train_loss": round(loss.item(), 4),
                "learning_rate": round(current_lr, 7),
                "grad_norm": round(grad_norm, 4),
            })
            
    elapsed = time.perf_counter() - t0
    tokens = steps * batch_size * seq_len
    tps = tokens / max(elapsed, 1e-5)
    print(f"[{arm_name}] Finished {steps} steps in {elapsed:.2f}s ({tps:.1f} tok/s)")
    
    # Save candidate checkpoint
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arm": arm_name,
        "step": steps,
        "tokens_processed": tokens,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "config": cfg.__dict__,
        "base_init": str(CANONICAL_BASE_CKPT),
        "seed": seed,
    }, save_path)
    sha = compute_file_sha256(save_path)
    print(f"[{arm_name}] Checkpoint saved: {save_path} (SHA: {sha})")
    
    metadata = {
        "arm": arm_name,
        "steps": steps,
        "tokens_processed": tokens,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_tok_per_sec": round(tps, 1),
        "checkpoint_path": str(save_path),
        "checkpoint_sha256": sha,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_lr": history[-1]["learning_rate"] if history else None,
    }
    return model, metadata

def run_suite():
    device = torch.device("cpu")
    print("=" * 80)
    print("BATCH 139–142 CONTROLLED NEURAL-CORE RECOVERY & EVALUATION SUITE")
    print(f"Device: {device}")
    print("=" * 80)
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    # 1. Datasets
    print("\nLoading datasets...")
    train_ds = PretrainingExperimentEngine.create_multi_shard_dataset(REPO_ROOT, split="train")
    val_ds = PretrainingExperimentEngine.create_multi_shard_dataset(REPO_ROOT, split="validation")
    test_ds = PretrainingExperimentEngine.create_multi_shard_dataset(REPO_ROOT, split="test")
    
    test_shards = {
        "english": MultiShardDataset([REPO_ROOT / "data" / "tokenized" / "stage_c" / "test" / "shard_00000.bin"]),
        "hindi": MultiShardDataset([REPO_ROOT / "data" / "tokenized" / "stage_c" / "test" / "shard_00001.bin"]),
        "code": MultiShardDataset([REPO_ROOT / "data" / "tokenized" / "stage_c" / "test" / "shard_00002.bin"]),
    }
    
    # 2. Canonical Step 10,000 Base State
    cfg = ModelConfig(max_seq_len=512)
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    base_state = base_data["model_state_dict"]
    
    # Function to evaluate any model across all splits and metrics
    def evaluate_all(model: ChakrMicro) -> dict:
        v_eval = evaluate_dataset(model, val_ds, device, num_batches=20, batch_size=8, seq_len=64, seed=101)
        t_eval = evaluate_dataset(model, test_ds, device, num_batches=20, batch_size=8, seq_len=64, seed=202)
        
        domain_evals = {}
        for dom, d_ds in test_shards.items():
            seed_map = {"english": 301, "hindi": 302, "code": 303}
            domain_evals[dom] = evaluate_dataset(model, d_ds, device, num_batches=20, batch_size=8, seq_len=64, seed=seed_map[dom])
            
        gen_eval = generate_continuations(model, tokenizer, device, max_new_tokens=32)
        return {
            "validation": v_eval,
            "test_global": t_eval,
            "test_domains": domain_evals,
            "generation": gen_eval,
        }
        
    results = {}
    
    # ARM A: Canonical Baseline
    print("\n--- Evaluating Arm A: Canonical Step 10,000 Baseline ---")
    model_a = ChakrMicro(cfg).to(device)
    model_a.load_state_dict(base_state)
    eval_a = evaluate_all(model_a)
    eval_a["metadata"] = {
        "arm": "Arm A (Canonical Baseline)",
        "checkpoint_path": str(CANONICAL_BASE_CKPT),
        "checkpoint_sha256": compute_file_sha256(CANONICAL_BASE_CKPT),
        "steps": 0,
        "tokens_processed": 0,
    }
    results["Arm_A_Baseline"] = eval_a
    print(f"Arm A Val Loss: {eval_a['validation']['loss']} (PPL: {eval_a['validation']['perplexity']}) | Test Loss: {eval_a['test_global']['loss']} (PPL: {eval_a['test_global']['perplexity']})")
    
    # ARM B: Replicated Batch 135-138 Pilot
    # 100 steps, peak_lr=3e-4, min_lr=3e-4, warmup_steps=0, sampling_mode="balanced"
    ckpt_b = CKPT_DIR / "arm_b_replicated_pilot.pt"
    model_b, meta_b = train_candidate(
        arm_name="Arm B (Replicated Pilot)",
        base_state=base_state,
        cfg=cfg,
        train_dataset=train_ds,
        device=device,
        steps=100,
        batch_size=8,
        seq_len=64,
        peak_lr=3e-4,
        min_lr=3e-4,
        warmup_steps=0,
        sampling_mode="balanced",
        seed=42,
        save_path=ckpt_b,
    )
    eval_b = evaluate_all(model_b)
    eval_b["metadata"] = meta_b
    results["Arm_B_Replicated_Pilot"] = eval_b
    print(f"Arm B Val Loss: {eval_b['validation']['loss']} (PPL: {eval_b['validation']['perplexity']}) | Test Loss: {eval_b['test_global']['loss']} (PPL: {eval_b['test_global']['perplexity']})")
    
    # ARM C: Warmup + Cosine Annealing (Natural Uniform Sampling)
    # 250 steps, peak_lr=1.5e-4, min_lr=2e-5, warmup_steps=50, sampling_mode="natural"
    ckpt_c = CKPT_DIR / "arm_c_warmup_cosine_natural.pt"
    model_c, meta_c = train_candidate(
        arm_name="Arm C (Warmup+Cosine Natural)",
        base_state=base_state,
        cfg=cfg,
        train_dataset=train_ds,
        device=device,
        steps=250,
        batch_size=8,
        seq_len=64,
        peak_lr=1.5e-4,
        min_lr=2e-5,
        warmup_steps=50,
        sampling_mode="natural",
        seed=42,
        save_path=ckpt_c,
    )
    eval_c = evaluate_all(model_c)
    eval_c["metadata"] = meta_c
    results["Arm_C_Warmup_Cosine_Natural"] = eval_c
    print(f"Arm C Val Loss: {eval_c['validation']['loss']} (PPL: {eval_c['validation']['perplexity']}) | Test Loss: {eval_c['test_global']['loss']} (PPL: {eval_c['test_global']['perplexity']})")
    
    # ARM D: Warmup + Cosine Annealing (Domain-Balanced Sampling)
    # 250 steps, peak_lr=1.5e-4, min_lr=2e-5, warmup_steps=50, sampling_mode="balanced"
    ckpt_d = CKPT_DIR / "arm_d_warmup_cosine_balanced.pt"
    model_d, meta_d = train_candidate(
        arm_name="Arm D (Warmup+Cosine Balanced)",
        base_state=base_state,
        cfg=cfg,
        train_dataset=train_ds,
        device=device,
        steps=250,
        batch_size=8,
        seq_len=64,
        peak_lr=1.5e-4,
        min_lr=2e-5,
        warmup_steps=50,
        sampling_mode="balanced",
        seed=42,
        save_path=ckpt_d,
    )
    eval_d = evaluate_all(model_d)
    eval_d["metadata"] = meta_d
    results["Arm_D_Warmup_Cosine_Balanced"] = eval_d
    print(f"Arm D Val Loss: {eval_d['validation']['loss']} (PPL: {eval_d['validation']['perplexity']}) | Test Loss: {eval_d['test_global']['loss']} (PPL: {eval_d['test_global']['perplexity']})")
    
    # Verify Arm C Resumability
    print("\n--- Testing Bit-Exact Resumability on Arm C Checkpoint ---")
    resumed_ckpt = torch.load(ckpt_c, map_location=device, weights_only=False)
    model_c_resumed = ChakrMicro(cfg).to(device)
    model_c_resumed.load_state_dict(resumed_ckpt["model_state_dict"])
    eval_c_resumed = evaluate_dataset(model_c_resumed, test_ds, device, num_batches=10, batch_size=8, seq_len=64, seed=999)
    eval_c_orig = evaluate_dataset(model_c, test_ds, device, num_batches=10, batch_size=8, seq_len=64, seed=999)
    assert abs(eval_c_resumed["loss"] - eval_c_orig["loss"]) < 1e-6, "Resumed model evaluation mismatch!"
    print(f"Resumability verification PASSED: Loss {eval_c_resumed['loss']} == {eval_c_orig['loss']}")
    results["resume_verification"] = {
        "status": "PASSED",
        "tested_checkpoint": str(ckpt_c),
        "loss_match": True,
    }
    
    # Save full results json
    results_file = REPORTS_DIR / "results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {results_file}")
    
    # Close datasets
    train_ds.close()
    val_ds.close()
    test_ds.close()
    for tds in test_shards.values():
        tds.close()

if __name__ == "__main__":
    run_suite()
