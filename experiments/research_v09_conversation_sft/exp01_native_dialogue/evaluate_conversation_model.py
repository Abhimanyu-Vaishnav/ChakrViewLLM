"""
evaluate_conversation_model.py
ChakrView Experiment 13: Native Human Conversation Evaluation Harness.

Compares:
- Arm 1: Frozen Canonical Baseline (Step-10000)
- Arm 2: Trained Candidate Model (Step-100 best checkpoint)

Evaluates on the identical held-out Test set (60 examples across 20 categories) using pure native
autoregressive greedy decoding (T=0.0, max 40 new tokens).
Saves per-example predictions to PER_EXAMPLE_RESULTS.jsonl and summaries to BASELINE_RESULTS.json / CANDIDATE_RESULTS.json.
"""

from __future__ import annotations

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"

CANONICAL_PARENT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANDIDATE_CKPT_PATH = CKPT_DIR / "candidate_best.pt"

def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    tokens = tokenizer.encode(prompt)
    curr_ids = torch.tensor([tokens], dtype=torch.long)
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            if next_token in (1, 2):  # EOS or PAD
                break
            generated.append(next_token)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
            if curr_ids.shape[1] >= 512:
                break
                
    return tokenizer.decode(generated).strip()

def evaluate_model(model_name: str, model, tokenizer, test_data: List[Dict]) -> Tuple[Dict[str, Any], List[Dict]]:
    print(f"\nEvaluating {model_name} on {len(test_data)} held-out test examples...")
    
    records = []
    category_scores = {}
    total_relevant = 0
    total_repetitive = 0
    total_empty = 0
    
    start_t = time.time()
    
    for idx, item in enumerate(test_data):
        p_str = item["prompt"]
        target_str = item["response"]
        cat = item["category"]
        
        t0 = time.time()
        raw_output = generate_greedy(model, tokenizer, p_str, max_new_tokens=40)
        lat = time.time() - t0
        
        # Check repetition
        words = raw_output.lower().split()
        is_repetitive = False
        if len(words) >= 4:
            # check 3-gram repetition
            tri = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            if len(tri) > len(set(tri)) * 1.5:
                is_repetitive = True
                
        is_empty = len(raw_output.strip()) == 0
        
        # Domain relevance check: does the response contain key goal terms or target words?
        target_words = set(target_str.lower().split())
        target_words.discard("the")
        target_words.discard("is")
        target_words.discard("a")
        target_words.discard("to")
        
        overlap = set(words).intersection(target_words)
        is_relevant = (len(overlap) >= 1) and not is_repetitive and not is_empty
        
        if is_relevant:
            total_relevant += 1
        if is_repetitive:
            total_repetitive += 1
        if is_empty:
            total_empty += 1
            
        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "relevant": 0}
        category_scores[cat]["total"] += 1
        if is_relevant:
            category_scores[cat]["relevant"] += 1
            
        records.append({
            "model": model_name,
            "example_id": item["example_id"],
            "category": cat,
            "prompt": p_str,
            "target": target_str,
            "raw_output": raw_output,
            "is_relevant": is_relevant,
            "is_repetitive": is_repetitive,
            "is_empty": is_empty,
            "latency": round(lat, 4)
        })
        
    total_time = time.time() - start_t
    
    metrics = {
        "model_name": model_name,
        "total_examples": len(test_data),
        "relevant_count": total_relevant,
        "relevant_pct": round(total_relevant / len(test_data) * 100, 2),
        "repetitive_count": total_repetitive,
        "repetitive_pct": round(total_repetitive / len(test_data) * 100, 2),
        "empty_count": total_empty,
        "empty_pct": round(total_empty / len(test_data) * 100, 2),
        "total_time_seconds": round(total_time, 2),
        "by_category": category_scores
    }
    
    print(f"  [{model_name}] Relevant: {metrics['relevant_pct']}% | Repetitive: {metrics['repetitive_pct']}% | Empty: {metrics['empty_pct']}%")
    return metrics, records

def main():
    print("=" * 70)
    print("CHAKRVIEW EXPERIMENT 13: EVALUATE BASELINE VS CANDIDATE CONVERSATION MODEL")
    print("=" * 70)
    
    tokenizer, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)
    
    test_file = DATA_DIR / "test.jsonl"
    test_data = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
                
    device = torch.device("cpu")
    
    # 1. Evaluate Frozen Baseline
    print("\nLoading Frozen Baseline (Canonical Step-10000)...")
    baseline_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    baseline_ckpt = torch.load(CANONICAL_PARENT_PATH, map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"], strict=True)
    baseline_model.eval()
    
    baseline_metrics, baseline_records = evaluate_model("Frozen_Baseline_Step_10000", baseline_model, tokenizer, test_data)
    
    # 2. Evaluate Trained Candidate
    print("\nLoading Trained Candidate (Best Checkpoint)...")
    candidate_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    candidate_ckpt = torch.load(CANDIDATE_CKPT_PATH, map_location=device, weights_only=False)
    candidate_model.load_state_dict(candidate_ckpt["model_state_dict"], strict=True)
    candidate_model.eval()
    
    candidate_metrics, candidate_records = evaluate_model("Trained_Candidate_Best", candidate_model, tokenizer, test_data)
    
    # Save results
    (EXP_DIR / "BASELINE_RESULTS.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
    (EXP_DIR / "CANDIDATE_RESULTS.json").write_text(json.dumps(candidate_metrics, indent=2), encoding="utf-8")
    
    all_records = baseline_records + candidate_records
    with open(EXP_DIR / "PER_EXAMPLE_RESULTS.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"\nSaved metrics to BASELINE_RESULTS.json, CANDIDATE_RESULTS.json, and PER_EXAMPLE_RESULTS.jsonl")

if __name__ == "__main__":
    main()
