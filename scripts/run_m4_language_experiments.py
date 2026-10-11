"""
Controlled Pretraining Experiments Runner for Milestone 4.
Executes:
1. Canonical random baseline (step 0).
2. Reproducible short baseline (30 steps, seed 42).
3. Medium training candidate (60 steps, seed 42).
4. Longer training candidate (100 steps, seed 42).
5. Multi-seed comparison (30 steps on seed 101, 202).
Outputs results with repetition, unique tokens, and generated text samples.
"""

import json
import time
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataclasses import asdict

from chakrview.training.pretraining_engine import PretrainingExperimentEngine
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
import torch
import torch.nn as nn


def run_milestone_4_experiments():
    repo_root = Path(__file__).resolve().parent.parent
    exp_dir = repo_root / "artifacts" / "milestone4_experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Running Milestone 4 Controlled Language-Learning Experiments...")
    
    # 1. Canonical Random Baseline (Step 0)
    torch.manual_seed(42)
    tok_dir = repo_root / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)
    train_tokens, val_tokens, test_tokens = PretrainingExperimentEngine.load_dataset_tokens(repo_root)

    import numpy as np
    vx, vy = PretrainingExperimentEngine.get_batch(val_tokens, batch_size=16, seq_len=64, rng=np.random.RandomState(101))
    tex, tey = PretrainingExperimentEngine.get_batch(test_tokens, batch_size=16, seq_len=64, rng=np.random.RandomState(202))

    base_model = ChakrMicro(ModelConfig())
    base_model.eval()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        base_val_loss = loss_fn(base_model(vx).reshape(-1, 4096), vy.reshape(-1)).item()
        base_test_loss = loss_fn(base_model(tex).reshape(-1, 4096), tey.reshape(-1)).item()

    # Generate samples from base_model
    prompts = ["August is", "The system", "Language learning", "ChakrView native"]
    base_samples = {}
    base_gen_ids = []
    with torch.no_grad():
        for p in prompts:
            inp = torch.tensor([tokenizer.encode(p)])
            for _ in range(16):
                nxt = base_model(inp)[:, -1, :].argmax(dim=-1, keepdim=True)
                inp = torch.cat([inp, nxt], dim=1)
            toks = inp[0].tolist()
            base_samples[p] = tokenizer.decode(toks)
            base_gen_ids.extend(toks[len(tokenizer.encode(p)):])

    base_rep = sum(1 for i in range(len(base_gen_ids) - 1) if base_gen_ids[i] == base_gen_ids[i+1]) / max(len(base_gen_ids) - 1, 1)
    base_uniq = len(set(base_gen_ids)) / max(len(base_gen_ids), 1)

    results = {
        "random_baseline": {
            "steps": 0,
            "val_loss": round(base_val_loss, 4),
            "test_loss": round(base_test_loss, 4),
            "samples": base_samples,
            "repetition_ratio": round(base_rep, 4),
            "unique_tokens_ratio": round(base_uniq, 4),
        },
        "budget_progression": {},
        "multi_seed_comparison": {},
    }

    # 2. Progression: 30 steps (reproducible baseline), 60 steps, 100 steps
    budgets = [30, 60, 100]
    for b in budgets:
        print(f"Training budget: {b} steps...")
        r = PretrainingExperimentEngine.run_training(
            repo_root=repo_root,
            steps=b,
            batch_size=8,
            seq_len=64,
            lr=5e-4,
            seed=42,
            device="cpu",
            checkpoint_dir=exp_dir,
            prompts=prompts,
        )
        results["budget_progression"][f"step_{b}"] = asdict(r)

    # 3. Multi-seed comparison (30 steps across seeds 101 and 202)
    for s in [101, 202]:
        print(f"Running seed comparison: seed {s}...")
        r_seed = PretrainingExperimentEngine.run_training(
            repo_root=repo_root,
            steps=30,
            batch_size=8,
            seq_len=64,
            lr=5e-4,
            seed=s,
            device="cpu",
            checkpoint_dir=exp_dir / f"seed_{s}",
            prompts=prompts,
        )
        results["multi_seed_comparison"][f"seed_{s}"] = asdict(r_seed)

    out_file = repo_root / "artifacts" / "milestone4_language_experiments.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Milestone 4 experiments finished. Results saved to {out_file}")


if __name__ == "__main__":
    run_milestone_4_experiments()
