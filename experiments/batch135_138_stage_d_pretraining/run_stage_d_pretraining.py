"""
Batch 135–138: Gate 1 Pipeline Smoke Test & Gate 2 Baseline Diagnostic Harness.
Executes:
1. Gate 1: 15-step pipeline smoke test with checkpoint save and resume verification.
2. Gate 2: Full baseline evaluation of Canonical Step 10,000 checkpoint on held-out multi-domain test set.
3. Gate 3: Controlled Stage D pretraining run (100 steps on multi-shard dataset with balanced domain sampling).
4. Full evaluation comparing Canonical Baseline vs Stage D Pretrained model.
"""
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

EXP_DIR = Path("experiments/batch135_138_stage_d_pretraining")
DATA_DIR = EXP_DIR / "data"
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

def evaluate_model_on_split(model: ChakrMicro, dataset: MultiShardDataset, device: torch.device, num_batches: int = 10, batch_size: int = 8, seq_len: int = 64, seed: int = 42) -> dict:
    model.eval()
    rng = np.random.RandomState(seed)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for b in range(num_batches):
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

def generate_continuations(model: ChakrMicro, tokenizer, device: torch.device, max_new_tokens: int = 32) -> dict:
    model.eval()
    continuations = {}
    with torch.no_grad():
        for dom, prompt_text in PROMPTS.items():
            prompt_tokens = [0] + tokenizer.encode(prompt_text) # add BOS
            curr = list(prompt_tokens)
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr], dtype=torch.long, device=device)
                logits = model(inp)[0, -1]
                nxt = torch.argmax(logits).item()
                curr.append(nxt)
                if nxt == 1: # EOS
                    break
            raw = tokenizer.decode(curr[len(prompt_tokens):], errors="replace")
            continuations[dom] = {
                "prompt": prompt_text,
                "generated": raw.strip(),
                "full_text": prompt_text + " " + raw.strip(),
            }
    return continuations

def run_gate1_smoke_test(repo_root: Path, device: torch.device, tokenizer) -> dict:
    print("\n" + "="*70)
    print("GATE 1: PIPELINE SMOKE TEST")
    print("="*70)
    
    smoke_ckpt_dir = CKPT_DIR / "smoke_test"
    smoke_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Train small 15-step run
    res_15 = PretrainingExperimentEngine.run_training(
        repo_root=repo_root,
        steps=15,
        batch_size=4,
        seq_len=64,
        lr=3e-4,
        seed=42,
        device=str(device),
        val_interval=5,
        checkpoint_dir=smoke_ckpt_dir,
        use_multi_shard=True,
    )
    
    ckpt_step15 = smoke_ckpt_dir / "training_checkpoint_step_15.pt"
    assert ckpt_step15.exists(), "Smoke test checkpoint step 15 missing!"
    sha15 = compute_file_sha256(ckpt_step15)
    print(f"Smoke Test Step 15 Checkpoint saved: {ckpt_step15} (SHA: {sha15[:16]}...)")
    
    # 2. Test Resume from step 15 to step 25
    res_25 = PretrainingExperimentEngine.run_training(
        repo_root=repo_root,
        steps=25,
        batch_size=4,
        seq_len=64,
        lr=3e-4,
        seed=42,
        device=str(device),
        val_interval=5,
        checkpoint_dir=smoke_ckpt_dir,
        resume_from=ckpt_step15,
        use_multi_shard=True,
    )
    
    ckpt_step25 = smoke_ckpt_dir / "training_checkpoint_step_25.pt"
    assert ckpt_step25.exists(), "Smoke test resumed checkpoint step 25 missing!"
    sha25 = compute_file_sha256(ckpt_step25)
    print(f"Resumed Smoke Test Step 25 Checkpoint saved: {ckpt_step25} (SHA: {sha25[:16]}...)")
    
    # Check that canonical checkpoint remains untouched
    canon_sha = compute_file_sha256(CANONICAL_BASE_CKPT)
    expected_canon_sha = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert canon_sha == expected_canon_sha, "CANONICAL CHECKPOINT WAS MUTATED!"
    
    smoke_report = {
        "gate": "Gate 1 Pipeline Smoke Test",
        "status": "PASSED",
        "step_15_loss": res_15.final_train_loss,
        "step_15_val_loss": res_15.final_val_loss,
        "step_25_loss": res_25.final_train_loss,
        "step_25_val_loss": res_25.final_val_loss,
        "step_15_sha256": sha15,
        "step_25_sha256": sha25,
        "canonical_baseline_intact": True,
        "canonical_baseline_sha256": canon_sha,
    }
    return smoke_report

def run_gate2_and_3(repo_root: Path, device: torch.device, tokenizer) -> dict:
    print("\n" + "="*70)
    print("GATE 2: EVALUATING CANONICAL STEP 10,000 BASELINE")
    print("="*70)
    
    val_dataset = PretrainingExperimentEngine.create_multi_shard_dataset(repo_root, split="validation")
    test_dataset = PretrainingExperimentEngine.create_multi_shard_dataset(repo_root, split="test")
    
    # Load Canonical Baseline Model
    cfg = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(cfg).to(device)
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    base_state = base_data["model_state_dict"] if "model_state_dict" in base_data else base_data
    base_model.load_state_dict(base_state)
    
    base_val_eval = evaluate_model_on_split(base_model, val_dataset, device, num_batches=15, batch_size=8, seq_len=64, seed=101)
    base_test_eval = evaluate_model_on_split(base_model, test_dataset, device, num_batches=15, batch_size=8, seq_len=64, seed=202)
    base_continuations = generate_continuations(base_model, tokenizer, device, max_new_tokens=32)
    
    print(f"Canonical Baseline Validation Loss: {base_val_eval['loss']} | PPL: {base_val_eval['perplexity']}")
    print(f"Canonical Baseline Test Loss:       {base_test_eval['loss']} | PPL: {base_test_eval['perplexity']}")
    
    print("\n" + "="*70)
    print("GATE 3: CONTROLLED STAGE D PRETRAINING RUN (100 STEPS)")
    print("="*70)
    
    stage_d_ckpt_dir = CKPT_DIR / "stage_d_pilot"
    stage_d_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # We train 100 steps from the canonical baseline checkpoint with AdamW (lr=3e-4, cosine decay)
    # on multi-shard dataset with balanced domain sampling
    trained_model = ChakrMicro(cfg).to(device)
    trained_model.load_state_dict(base_state) # Start from Step 10,000 weights
    
    train_dataset = PretrainingExperimentEngine.create_multi_shard_dataset(repo_root, split="train")
    opt = optim.AdamW(trained_model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    rng = np.random.RandomState(42)
    steps = 100
    batch_size = 8
    seq_len = 64
    history = []
    
    t0 = time.perf_counter()
    trained_model.train()
    for step in range(steps):
        x, y = train_dataset.get_batch(batch_size=batch_size, seq_len=seq_len, rng=rng, sampling_mode="balanced")
        x, y = x.to(device), y.to(device)
        
        opt.zero_grad()
        logits = trained_model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(trained_model.parameters(), 1.0).item()
        opt.step()
        
        if (step + 1) % 20 == 0 or step == 0:
            print(f"Stage D Step {step+1:3d}/{steps} | Loss: {loss.item():.4f} | Grad: {grad_norm:.4f}")
            history.append({
                "step": step + 1,
                "train_loss": round(loss.item(), 4),
                "grad_norm": round(grad_norm, 4),
            })
            
    total_time = time.perf_counter() - t0
    tokens_trained = steps * batch_size * seq_len
    throughput = tokens_trained / max(total_time, 1e-5)
    print(f"Training completed in {total_time:.2f}s ({throughput:.1f} tokens/sec, {tokens_trained} tokens)")
    
    # Save Stage D Checkpoint
    stage_d_ckpt = stage_d_ckpt_dir / "stage_d_pretrained_step100.pt"
    torch.save({
        "step": steps,
        "model_state_dict": trained_model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "config": cfg.__dict__,
        "base_init": str(CANONICAL_BASE_CKPT),
    }, stage_d_ckpt)
    stage_d_sha = compute_file_sha256(stage_d_ckpt)
    print(f"Saved Stage D Checkpoint: {stage_d_ckpt} (SHA: {stage_d_sha})")
    
    # Evaluate Stage D Checkpoint
    d_val_eval = evaluate_model_on_split(trained_model, val_dataset, device, num_batches=15, batch_size=8, seq_len=64, seed=101)
    d_test_eval = evaluate_model_on_split(trained_model, test_dataset, device, num_batches=15, batch_size=8, seq_len=64, seed=202)
    d_continuations = generate_continuations(trained_model, tokenizer, device, max_new_tokens=32)
    
    print(f"Stage D Validation Loss: {d_val_eval['loss']} | PPL: {d_val_eval['perplexity']}")
    print(f"Stage D Test Loss:       {d_test_eval['loss']} | PPL: {d_test_eval['perplexity']}")
    
    full_report = {
        "benchmark": "Batch 135-138 Stage D Pretraining Revival",
        "device": str(device),
        "steps": steps,
        "tokens_processed": tokens_trained,
        "elapsed_seconds": round(total_time, 2),
        "throughput_tok_per_sec": round(throughput, 1),
        "stage_d_checkpoint": str(stage_d_ckpt),
        "stage_d_sha256": stage_d_sha,
        "canonical_baseline": {
            "checkpoint": str(CANONICAL_BASE_CKPT),
            "sha256": compute_file_sha256(CANONICAL_BASE_CKPT),
            "validation": base_val_eval,
            "test": base_test_eval,
            "continuations": base_continuations,
        },
        "stage_d_pretrained": {
            "checkpoint": str(stage_d_ckpt),
            "sha256": stage_d_sha,
            "validation": d_val_eval,
            "test": d_test_eval,
            "continuations": d_continuations,
        },
        "comparison": {
            "val_loss_diff": round(d_val_eval["loss"] - base_val_eval["loss"], 4),
            "test_loss_diff": round(d_test_eval["loss"] - base_test_eval["loss"], 4),
            "val_ppl_diff": round(d_val_eval["perplexity"] - base_val_eval["perplexity"], 2),
            "test_ppl_diff": round(d_test_eval["perplexity"] - base_test_eval["perplexity"], 2),
        }
    }
    return full_report

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    # Adaptive Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Adaptive Hardware Device: {device}")
    
    # Execute Gate 1: Pipeline Smoke Test
    smoke_res = run_gate1_smoke_test(REPO_ROOT, device, tokenizer)
    smoke_path = REPORTS_DIR / "smoke_test_report.json"
    with open(smoke_path, "w", encoding="utf-8") as f:
        json.dump(smoke_res, f, indent=2)
    print(f"Gate 1 Report saved to {smoke_path}")
    
    # Execute Gate 2 & Gate 3: Baseline Diagnostic & Stage D Pretraining Run
    exp_res = run_gate2_and_3(REPO_ROOT, device, tokenizer)
    exp_path = REPORTS_DIR / "stage_d_experiment_results.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(exp_res, f, indent=2)
    print(f"Gate 2 & 3 Report saved to {exp_path}")

if __name__ == "__main__":
    main()
