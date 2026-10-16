"""
Seed replication runner for ChakrMicro domain adaptation.
Location: experiments/batch1421_1460/run_seed_replication.py

Executes:
1. Replicates the 80-step domain adaptation protocol using two additional random seeds (Seed 101 and Seed 202).
2. Starts from the canonical checkpoint (Step 10,000) using the exact same 21 train / 4 val documents.
3. Evaluates all checkpoints (Canonical, Seed 42, Seed 101, Seed 202) on the 30 independent documents.
4. Saves replication report to experiments/batch1421_1460/results/SEED_REPLICATION_RESULTS.json.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
MANIFEST_PATH = Path("experiments/batch1341_1380/data/corpus_lm_split_manifest.json")
INDEP_MANIFEST_PATH = Path("experiments/batch1421_1460/data/independent_corpus_manifest.json")
INDEP_CORPUS_DIR = Path("experiments/batch1421_1460/data/corpus")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
CKPT_OUT_DIR = Path("experiments/batch1421_1460/checkpoints")
RESULTS_OUT_DIR = Path("experiments/batch1421_1460/results")
CKPT_OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEV_VAL_FILENAMES = {
    "billing_refund_policy.md",
    "compute_cpu_simd.md",
    "network_firewall_rules.md",
    "security_incident_response.md",
    "storage_caching_layer.md",
}


def load_adaptation_data(tokenizer):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train_docs = manifest["train_split"]
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
    return train_data, val_data


def load_independent_data(tokenizer):
    manifest = json.loads(INDEP_MANIFEST_PATH.read_text(encoding="utf-8"))
    indep_data = []
    for d in manifest["documents"]:
        txt = (INDEP_CORPUS_DIR / d["filename"]).read_text(encoding="utf-8")
        toks = tokenizer.encode(txt)
        indep_data.append({"filename": d["filename"], "category": d["category"], "tokens": toks})
    return indep_data


def compute_eval_metrics(model: ChakrMicro, indep_data: list[dict]) -> dict:
    model.eval()
    total_nll = 0.0
    total_targets = 0
    top1_c = 0
    top10_c = 0
    with torch.no_grad():
        for doc in indep_data:
            toks = doc["tokens"]
            if len(toks) < 2:
                continue
            input_ids = torch.tensor([toks[:-1]], dtype=torch.long)
            targets = torch.tensor(toks[1:], dtype=torch.long)
            logits = model(input_ids)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            target_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            total_nll -= target_lp.sum().item()
            total_targets += len(targets)

            _, top10 = torch.topk(logits, k=10, dim=-1)
            for i, target_id in enumerate(targets.tolist()):
                top10_list = top10[i].tolist()
                if target_id == top10_list[0]:
                    top1_c += 1
                if target_id in top10_list:
                    top10_c += 1

    loss = total_nll / total_targets
    return {
        "cross_entropy": round(loss, 4),
        "perplexity": round(math.exp(loss), 2),
        "top_1": round(top1_c / total_targets, 4),
        "top_10": round(top10_c / total_targets, 4)
    }


def train_seed(seed: int, train_data: list[dict], val_data: list[dict], tokenizer, config, base_state_dict) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)
    
    model = ChakrMicro(config)
    model.load_state_dict(base_state_dict, strict=True)
    
    max_steps = 80
    warmup_steps = 10
    peak_lr = 5e-5
    min_lr = 5e-6
    optimizer = AdamW(model.parameters(), lr=min_lr, betas=(0.9, 0.95), weight_decay=0.05)
    
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    best_val_loss = float("inf")
    ckpt_path = CKPT_OUT_DIR / f"chakrmicro_domain_adapted_seed_{seed}.pt"
    
    model.train()
    for step in range(1, max_steps + 1):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            
        batch_docs = random.sample(train_data, 4)
        optimizer.zero_grad()
        step_loss = 0.0
        step_tokens = 0
        
        for doc in batch_docs:
            toks = doc["tokens"]
            if len(toks) < 2:
                continue
            input_ids = torch.tensor([toks[:-1]], dtype=torch.long)
            targets = torch.tensor(toks[1:], dtype=torch.long)
            logits = model(input_ids)[0]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            step_loss += loss.item() * len(targets)
            step_tokens += len(targets)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 5 == 0:
            model.eval()
            val_nll = 0.0
            val_targets = 0
            with torch.no_grad():
                for doc in val_data:
                    toks = doc["tokens"]
                    input_ids = torch.tensor([toks[:-1]], dtype=torch.long)
                    targets = torch.tensor(toks[1:], dtype=torch.long)
                    logits = model(input_ids)[0]
                    log_probs = F.log_softmax(logits, dim=-1)
                    val_nll -= log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).sum().item()
                    val_targets += len(targets)
            val_loss = val_nll / val_targets
            model.train()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step": 10000 + step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "seed": seed,
                    "val_loss": val_loss
                }, ckpt_path)

    return ckpt_path


def main():
    print("Loading artifacts for seed replication...")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_data, val_data = load_adaptation_data(tokenizer)
    indep_data = load_independent_data(tokenizer)
    
    base_ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    config = base_ckpt["config"]
    base_state_dict = base_ckpt["model_state_dict"]
    
    seeds = [101, 202]
    replication_results = {}
    
    for s in seeds:
        print(f"\n--- Training Replication Run with Seed {s} ---")
        t0 = time.perf_counter()
        ckpt_path = train_seed(s, train_data, val_data, tokenizer, config, base_state_dict)
        duration = time.perf_counter() - t0
        print(f"Seed {s} completed in {duration:.2f}s. Evaluating on independent corpus...")
        
        rep_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = ChakrMicro(config)
        m.load_state_dict(rep_ckpt["model_state_dict"])
        metrics = compute_eval_metrics(m, indep_data)
        metrics["training_time_seconds"] = round(duration, 2)
        metrics["checkpoint_path"] = str(ckpt_path)
        replication_results[f"seed_{s}"] = metrics
        print(f"Seed {s} Results -> Loss: {metrics['cross_entropy']} | PPL: {metrics['perplexity']} | Top-1: {metrics['top_1']*100:.2f}% | Top-10: {metrics['top_10']*100:.2f}%")

    out_file = RESULTS_OUT_DIR / "SEED_REPLICATION_RESULTS.json"
    out_file.write_text(json.dumps(replication_results, indent=2), encoding="utf-8")
    print(f"\nSaved seed replication results to {out_file}")


if __name__ == "__main__":
    main()
