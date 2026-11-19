"""
Phase F: Domain-Specific Learning Analysis Script for UCIP-13R.
Evaluates step-000 and step-800 checkpoints on isolated domain test partitions:
- English (115 docs, 193,266 tokens)
- Hindi (72 docs, 182,574 tokens)
- Sanskrit (16 docs, 29,355 tokens)
- Python code (66 docs, 149,156 tokens)

Computes:
- Number of evaluation examples (docs & sequences)
- Number of evaluated tokens
- Average cross-entropy loss (step-000 vs step-800)
- Perplexity (step-000 vs step-800)
- Initial-to-final loss change and reduction factor
"""

import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.sharding import read_shard_tokens

DEVICE = torch.device("cpu")
EXP_DIR = ROOT / "experiments" / "ucip13"
CKPT_DIR = EXP_DIR / "checkpoints" / "primary_seed42"
OUT_FILE = ROOT / "experiments" / "ucip13r" / "evaluations" / "domain_specific_learning.json"


def evaluate_token_chunks(model, token_chunks, batch_size=16, seq_len=512, ignore_index=2):
    model.eval()
    total_loss_sum = 0.0
    total_batches = 0
    total_valid_tokens = 0
    
    # Process in batches
    num_chunks = len(token_chunks)
    with torch.no_grad():
        for b_start in range(0, num_chunks, batch_size):
            b_end = min(b_start + batch_size, num_chunks)
            batch = token_chunks[b_start:b_end]
            if len(batch) == 0:
                break
                
            input_list = [c[:-1] for c in batch]
            target_list = [c[1:] for c in batch]
            
            input_ids = torch.from_numpy(np.stack(input_list).astype(np.int64)).to(DEVICE)
            target_ids = torch.from_numpy(np.stack(target_list).astype(np.int64)).to(DEVICE)
            
            logits = model(input_ids)
            B, T, V = logits.shape
            flat_logits = logits.view(-1, V)
            flat_targets = target_ids.view(-1)
            
            valid_mask = flat_targets != ignore_index
            total_valid_tokens += valid_mask.sum().item()
            
            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="mean").item()
            total_loss_sum += loss
            total_batches += 1
            
    mean_loss = total_loss_sum / max(1, total_batches)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, ppl, total_batches, total_valid_tokens


def run_domain_evaluations():
    print("=" * 80)
    print("DOMAIN-SPECIFIC LEARNING ANALYSIS — UCIP-13R")
    print("=" * 80)
    
    with open(ROOT / "data" / "manifests" / "stage_c_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    test_docs = [d for d in manifest["documents"] if d["split"] == "test"]
    all_test_tokens = np.concatenate([
        read_shard_tokens(ROOT / f"data/tokenized/stage_c/test/shard_{i:05d}.bin")
        for i in range(3)
    ])
    
    # Slice domains
    domains = ["english", "hindi", "sanskrit", "code"]
    domain_docs = {d: [] for d in domains}
    cur_pos = 0
    
    for doc in test_docs:
        t_len = doc["token_count"] + 1
        dom = doc["domain"]
        doc_toks = all_test_tokens[cur_pos : cur_pos + t_len]
        if dom in domain_docs:
            domain_docs[dom].append(doc_toks)
        cur_pos += t_len
        
    # Load Models
    print("Loading Checkpoints (Step 0 and Step 800)...")
    m0 = ChakrMicro(ModelConfig())
    data0 = torch.load(CKPT_DIR / "checkpoint_step_000.pt", map_location="cpu", weights_only=False)
    m0.load_state_dict(data0["model_state_dict"])
    
    m800 = ChakrMicro(ModelConfig())
    data800 = torch.load(CKPT_DIR / "checkpoint_step_800.pt", map_location="cpu", weights_only=False)
    m800.load_state_dict(data800["model_state_dict"])
    
    seq_len = 512
    chunk_size = seq_len + 1
    
    results = {}
    
    for dom in domains:
        docs = domain_docs[dom]
        doc_count = len(docs)
        # Flatten domain tokens
        flat_dom_tokens = np.concatenate(docs) if docs else np.array([], dtype=np.uint16)
        raw_tokens_count = len(flat_dom_tokens)
        
        # Chunk into 513
        num_full_chunks = raw_tokens_count // chunk_size
        chunks = [
            flat_dom_tokens[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_full_chunks)
        ]
        
        print(f"\n--- Domain: {dom.upper()} ---")
        print(f"Documents: {doc_count} | Raw Tokens: {raw_tokens_count:,} | Full 513-Chunks: {len(chunks)}")
        
        # Evaluate Step 0
        loss_0, ppl_0, batches_0, tokens_0 = evaluate_token_chunks(m0, chunks)
        print(f"Step 000: Loss = {loss_0:.4f} | PPL = {ppl_0:.2f} | Tokens Evaluated = {tokens_0:,}")
        
        # Evaluate Step 800
        loss_800, ppl_800, batches_800, tokens_800 = evaluate_token_chunks(m800, chunks)
        print(f"Step 800: Loss = {loss_800:.4f} | PPL = {ppl_800:.2f} | Tokens Evaluated = {tokens_800:,}")
        
        loss_reduction = loss_0 - loss_800
        ppl_factor = ppl_0 / max(0.001, ppl_800)
        
        print(f"Improvement: Loss Delta = -{loss_reduction:.4f} | PPL Reduction Factor = {ppl_factor:.2f}x")
        
        results[dom] = {
            "document_count": doc_count,
            "raw_token_count_with_eos": raw_tokens_count,
            "evaluated_sequences_512": len(chunks),
            "evaluated_tokens": tokens_800,
            "step_000": {
                "loss": round(loss_0, 4),
                "perplexity": round(ppl_0, 2)
            },
            "step_800": {
                "loss": round(loss_800, 4),
                "perplexity": round(ppl_800, 2)
            },
            "loss_reduction": round(loss_reduction, 4),
            "perplexity_reduction_factor": round(ppl_factor, 2)
        }
        
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\nDomain Analysis successfully saved to:", OUT_FILE)


if __name__ == "__main__":
    run_domain_evaluations()
