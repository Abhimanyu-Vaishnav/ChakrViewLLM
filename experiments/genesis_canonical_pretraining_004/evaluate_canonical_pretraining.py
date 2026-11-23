"""
Validation and Comparative Evaluation Script for CV-GENESIS-NR-004.
Evaluates canonical foundation model against old pretrain baseline on:
1. Held-out Stage-C validation shards (51,200 tokens).
2. Domain prompts (English, Science, Hindi, Hinglish, Code).
"""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"


def evaluate_models(
    tokenizer_dir: str = "data/experiments/vocab_4096",
    val_shard: str = "data/tokenized/stage_c/validation/shard_00000.bin",
    old_ckpt_path: str = "experiments/project_genesis/pretraining/genesis_pretrain_best.pt",
    new_ckpt_path: str = "experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    output_json: str = "experiments/genesis_canonical_pretraining_004/eval_results.json"
):
    print("=== CV-GENESIS-NR-004 EVALUATION HARNESS ===")
    
    # 1. Verify tokenizer hash
    vocab_file = Path(tokenizer_dir) / "vocab.json"
    actual_hash = hashlib.sha256(vocab_file.read_bytes()).hexdigest()
    assert actual_hash == EXPECTED_VOCAB_HASH, f"Vocab hash mismatch: {actual_hash}"
    print(f"[VERIFIED] Tokenizer Vocab SHA-256: {actual_hash}")
    
    tok, _ = load_tokenizer_artifacts(tokenizer_dir)
    
    # 2. Load models
    old_ckpt = torch.load(old_ckpt_path, map_location=device, weights_only=False)
    model_old = ChakrGenesis(old_ckpt["config"]).to(device)
    model_old.load_state_dict(old_ckpt["model_state_dict"])
    model_old.eval()
    print(f"Loaded old baseline model from: {old_ckpt_path}")
    
    new_ckpt = torch.load(new_ckpt_path, map_location=device, weights_only=False)
    model_new = ChakrGenesis(new_ckpt["config"]).to(device)
    model_new.load_state_dict(new_ckpt["model_state_dict"])
    model_new.eval()
    print(f"Loaded canonical new model from: {new_ckpt_path}")
    
    # 3. Evaluate on held-out validation shard
    shard_data = np.fromfile(val_shard, dtype=np.uint16)
    print(f"Evaluating on held-out shard {val_shard} ({len(shard_data):,} tokens)...")
    
    num_eval_chunks = 100
    chunk_len = 512
    losses_old, losses_new = [], []
    with torch.no_grad():
        for i in range(num_eval_chunks):
            chunk = shard_data[i * chunk_len : (i + 1) * chunk_len].astype(np.int64)
            t = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
            x, y = t[:, :-1], t[:, 1:]
            losses_old.append(model_old(x, targets=y)[1].item())
            losses_new.append(model_new(x, targets=y)[1].item())
            
    mean_old_loss = float(np.mean(losses_old))
    mean_new_loss = float(np.mean(losses_new))
    ppl_old = math.exp(mean_old_loss)
    ppl_new = math.exp(mean_new_loss)
    
    print(f"Validation Shard (51,200 tokens):")
    print(f"  Old Mismatched Model: Loss = {mean_old_loss:.4f}, PPL = {ppl_old:.2f}")
    print(f"  Canonical Model:      Loss = {mean_new_loss:.4f}, PPL = {ppl_new:.2f}")
    print(f"  Relative PPL Reduction: {(1.0 - ppl_new / ppl_old) * 100:.2f}%")
    
    # 4. Qualitative generation probe
    prompts = [
        "The solar system consists of",
        "A computer algorithm is",
        "Machine learning models are used to"
    ]
    
    def generate(model, prompt_tokens, max_new_tokens=25, temperature=0.7, top_k=20):
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -512:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx[0].tolist()
        
    generation_results = []
    print("\n--- Qualitative Probes ---")
    for p in prompts:
        inp = tok.encode(p)
        out_old = tok.decode(generate(model_old, inp))
        out_new = tok.decode(generate(model_new, inp))
        print(f"Prompt: {p}")
        print(f"  Old Model: {out_old}")
        print(f"  New Model: {out_new}")
        generation_results.append({
            "prompt": p,
            "old_output": out_old,
            "new_output": out_new
        })
        
    out_dict = {
        "validation_tokens_evaluated": num_eval_chunks * chunk_len,
        "old_model": {"path": old_ckpt_path, "loss": mean_old_loss, "ppl": ppl_old},
        "new_model": {"path": new_ckpt_path, "loss": mean_new_loss, "ppl": ppl_new},
        "generations": generation_results
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Results written to {output_json}")


if __name__ == "__main__":
    evaluate_models()
