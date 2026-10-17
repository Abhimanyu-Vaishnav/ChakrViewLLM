"""
Independent reproduction evaluator for Batch 2221-2260.
Loads baseline and candidate checkpoints freshly, validates tokenization and manifest,
evaluates all 290 test documents, and recalculates exact metrics and confidence intervals.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CORPUS_MANIFEST_PATH = REPO_ROOT / "experiments" / "batch2101_2140" / "CORPUS_V2_CLEANED_MANIFEST.json"
OUTPUT_DIR = REPO_ROOT / "experiments" / "batch2221_2260"


def load_model(checkpoint_path: Path, canonical_config: Any, device: str = "cpu") -> ChakrMicro:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model = ChakrMicro(canonical_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def evaluate_test_set(model: ChakrMicro, tokenizer: Any, test_docs: List[Dict[str, Any]], device: str = "cpu") -> Dict[str, Any]:
    total_tokens = 0
    total_nll = 0.0
    doc_results = []
    source_stats = {}

    with torch.inference_mode():
        for doc in test_docs:
            text = doc["text"]
            did = doc["document_id"]
            sid = doc["source_id"]
            
            if sid not in source_stats:
                source_stats[sid] = {"nll": 0.0, "tokens": 0, "doc_losses": []}
                
            toks = tokenizer.encode(text)
            if len(toks) < 2:
                continue
                
            max_seq = 256
            d_tokens = 0
            d_nll = 0.0
            
            for i in range(0, len(toks) - 1, max_seq - 1):
                chunk = toks[i : i + max_seq]
                if len(chunk) < 2:
                    continue
                inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
                tgt = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
                
                logits = model(inp)
                n_tgt = tgt.numel()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="mean")
                d_nll += loss.item() * n_tgt
                d_tokens += n_tgt
                
            if d_tokens > 0:
                doc_loss = d_nll / d_tokens
                total_nll += d_nll
                total_tokens += d_tokens
                doc_results.append({
                    "doc_id": did,
                    "source_id": sid,
                    "tokens": d_tokens,
                    "loss": round(doc_loss, 4)
                })
                source_stats[sid]["nll"] += d_nll
                source_stats[sid]["tokens"] += d_tokens
                source_stats[sid]["doc_losses"].append(doc_loss)

    per_source_metrics = {}
    for sid, sdata in source_stats.items():
        stoks = sdata["tokens"]
        sce = sdata["nll"] / stoks if stoks > 0 else 0.0
        per_source_metrics[sid] = {
            "document_count": len(sdata["doc_losses"]),
            "evaluated_tokens": stoks,
            "token_weighted_ce": round(sce, 4),
            "perplexity": round(math.exp(sce), 2),
            "macro_doc_ce": round(sum(sdata["doc_losses"]) / len(sdata["doc_losses"]), 4) if sdata["doc_losses"] else 0.0
        }

    overall_tw_ce = total_nll / total_tokens if total_tokens > 0 else 0.0
    macro_source_ce = sum(m["token_weighted_ce"] for m in per_source_metrics.values()) / len(per_source_metrics) if per_source_metrics else 0.0
    macro_doc_ce = sum(d["loss"] for d in doc_results) / len(doc_results) if doc_results else 0.0

    return {
        "total_test_documents": len(doc_results),
        "total_evaluated_tokens": total_tokens,
        "token_weighted_ce": round(overall_tw_ce, 4),
        "token_weighted_perplexity": round(math.exp(overall_tw_ce), 2),
        "macro_source_ce": round(macro_source_ce, 4),
        "macro_source_perplexity": round(math.exp(macro_source_ce), 2),
        "macro_doc_ce": round(macro_doc_ce, 4),
        "per_source_metrics": per_source_metrics,
        "documents": doc_results
    }


def compute_paired_statistics(base_eval: Dict[str, Any], cand_eval: Dict[str, Any]) -> Dict[str, Any]:
    base_docs = {d["doc_id"]: d for d in base_eval["documents"]}
    cand_docs = {d["doc_id"]: d for d in cand_eval["documents"]}
    
    common_ids = sorted(list(set(base_docs.keys()) & set(cand_docs.keys())))
    diffs = []
    tokens = []
    
    for did in common_ids:
        b_loss = base_docs[did]["loss"]
        c_loss = cand_docs[did]["loss"]
        diffs.append(c_loss - b_loss)
        tokens.append(base_docs[did]["tokens"])
        
    diffs = np.array(diffs)
    tokens = np.array(tokens)
    
    mean_paired_delta = float(np.mean(diffs))
    token_weighted_delta = float(np.sum(diffs * tokens) / np.sum(tokens))
    improved_count = int(np.sum(diffs < 0))
    regressed_count = int(np.sum(diffs > 0))
    win_rate = float(improved_count / len(diffs))
    
    # Bootstrap CI
    np.random.seed(42)
    B = 10000
    boot_means = []
    for _ in range(B):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
        
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    
    return {
        "common_documents_evaluated": len(common_ids),
        "token_weighted_ce_delta": round(token_weighted_delta, 4),
        "mean_paired_doc_ce_delta": round(mean_paired_delta, 4),
        "paired_doc_bootstrap_95_ci": [round(ci_low, 4), round(ci_high, 4)],
        "documents_improved": improved_count,
        "documents_regressed": regressed_count,
        "document_win_rate_pct": round(win_rate * 100.0, 2)
    }


def run_independent_reproduction():
    print("=" * 70)
    print("RUNNING FRESH INDEPENDENT EVALUATION REPRODUCTION — BATCH 2221-2260")
    print("=" * 70)
    
    # 1. Load Tokenizer & Cleaned Corpus
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    with open(CORPUS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    test_docs = [d for d in corpus["documents"] if d["split"] == "test"]
    print(f"Loaded {len(test_docs)} test documents from {CORPUS_MANIFEST_PATH.name}")
    
    # 2. Load Canonical Baseline Configuration
    canon_ckpt = torch.load(CANONICAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    canonical_config = canon_ckpt["config"]
    
    models_to_evaluate = {
        "canonical_baseline": CANONICAL_CHECKPOINT_PATH,
        "historical_cpt_150": REPO_ROOT / "experiments" / "batch1861_1900" / "checkpoints" / "cpt_balanced_seed_42.pt",
        "historical_cpt_300": REPO_ROOT / "experiments" / "batch1901_1940" / "checkpoints" / "cpt_balanced_seed_42_300steps.pt",
        "cpt_seed_42_best": REPO_ROOT / "experiments" / "batch2141_2180" / "cpt" / "seed_42" / "cpt_best_checkpoint.pt",
        "cpt_seed_1337_best": REPO_ROOT / "experiments" / "batch2141_2180" / "cpt" / "seed_1337" / "cpt_best_checkpoint.pt",
        "cpt_seed_2026_best": REPO_ROOT / "experiments" / "batch2141_2180" / "cpt" / "seed_2026" / "cpt_best_checkpoint.pt"
    }
    
    results = {}
    for name, path in models_to_evaluate.items():
        print(f"\nEvaluating fresh: {name} ({path.name})...")
        t0 = time.perf_counter()
        model = load_model(path, canonical_config=canonical_config, device="cpu")
        eval_metrics = evaluate_test_set(model, tokenizer, test_docs, device="cpu")
        elapsed = time.perf_counter() - t0
        print(f"  Test TW-CE: {eval_metrics['token_weighted_ce']:.4f} (PPL: {eval_metrics['token_weighted_perplexity']:.2f}) | Evaluated in {elapsed:.2f}s")
        results[name] = eval_metrics

    # 3. Compute Paired Deltas against Canonical Baseline
    base_eval = results["canonical_baseline"]
    paired_comparisons = {}
    for name in ["historical_cpt_150", "historical_cpt_300", "cpt_seed_42_best", "cpt_seed_1337_best", "cpt_seed_2026_best"]:
        paired_comparisons[name] = compute_paired_statistics(base_eval, results[name])

    out_artifact = {
        "metadata": {
            "batch": "2221-2260",
            "evaluator": "Fresh Independent Reproduction Evaluator",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": "cpu",
            "total_documents_manifest": len(test_docs)
        },
        "models": results,
        "paired_comparisons_vs_baseline": paired_comparisons
    }
    
    out_path = OUTPUT_DIR / "REPRODUCED_EVALUATION.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_artifact, f, indent=2)
        
    print(f"\nFresh independent evaluation saved to: {out_path}")
    return out_artifact


if __name__ == "__main__":
    run_independent_reproduction()
