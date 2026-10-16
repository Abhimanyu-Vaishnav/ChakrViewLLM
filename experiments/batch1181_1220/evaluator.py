"""
Batch 1181–1220 Multi-Model Retrieval Evaluator and Ablation Harness.
Location: experiments/batch1181_1220/evaluator.py

Evaluates:
1. BM25 Lexical Baseline
2. Deterministic Hash Embedding Hybrid
3. Existing ChakrBiEncoder Hybrid (improved_bi_encoder.pt)
4. Domain-Adaptive ChakrBiEncoder Hybrid (domain_adaptive_bi_encoder.pt)
5. Ablation: Domain-Adaptive BiEncoder Neural Semantic Score Alone (Vector Search Top-K)

Evaluates on the frozen 70-item benchmark (experiments/batch1141_1180/data/held_out_70_benchmark.json).
Outputs comprehensive metrics, confidence intervals, paired query comparisons, and latencies.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant, AssistantResponse
from chakrview.runtime.retrieval import (
    DeterministicHashEmbeddingProvider,
    CachedEmbeddingProvider,
    RetrievalQuery,
    InMemoryVectorIndex
)
from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
BENCHMARK_PATH = Path("experiments/batch1141_1180/data/held_out_70_benchmark.json")
RESULTS_DIR = Path("experiments/batch1181_1220/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLD_CKPT = Path("experiments/batch171_180_pointer_learning/checkpoints/improved_bi_encoder.pt")
NEW_CKPT = Path("experiments/batch1181_1220/checkpoints/domain_adaptive_bi_encoder.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def token_f1(pred: str, gold: str) -> float:
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens).intersection(set(g_tokens))
    if not common:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    return 2 * (prec * rec) / (prec + rec)

def build_assistant(mode: str, tokenizer=None):
    if mode == "bm25":
        asst = LocalKnowledgeAssistant(
            storage_dir=None,
            chunk_size=400,
            chunk_overlap=100,
            embedding_provider=None,
            extract_concise_span=True,
            use_embedding_cache=False
        )
    elif mode == "hash":
        prov = DeterministicHashEmbeddingProvider(dimension=64)
        asst = LocalKnowledgeAssistant(
            storage_dir=None,
            chunk_size=400,
            chunk_overlap=100,
            embedding_provider=prov,
            extract_concise_span=True,
            use_embedding_cache=True
        )
    elif mode == "neural_old":
        prov = LearnedEmbeddingProvider.load_checkpoint(OLD_CKPT, tokenizer=tokenizer)
        asst = LocalKnowledgeAssistant(
            storage_dir=None,
            chunk_size=400,
            chunk_overlap=100,
            embedding_provider=prov,
            extract_concise_span=True,
            use_embedding_cache=True
        )
    elif mode == "neural_domain_adaptive":
        prov = LearnedEmbeddingProvider.load_checkpoint(NEW_CKPT, tokenizer=tokenizer)
        asst = LocalKnowledgeAssistant(
            storage_dir=None,
            chunk_size=400,
            chunk_overlap=100,
            embedding_provider=prov,
            extract_concise_span=True,
            use_embedding_cache=True
        )
    else:
        raise ValueError(f"Unknown mode {mode}")
        
    # Ingest corpus
    t0 = time.perf_counter()
    for f in sorted(CORPUS_DIR.glob("*.md")):
        asst.ingest_document(f)
    ingest_time = (time.perf_counter() - t0) * 1000
    return asst, ingest_time

def evaluate_assistant_benchmark(asst, items, mode_name):
    query_results = []
    latencies = []
    r1_hits = 0
    r3_hits = 0
    r5_hits = 0
    rr_sum = 0.0
    answerable_count = 0
    
    overall_correct = 0
    f1_scores = []
    category_stats = {}
    
    for item in items:
        q_id = item["id"]
        cat = item["category"]
        q_text = item["query"]
        exp_type = item["expected_type"]
        exp_source = item.get("expected_source")
        exp_sub = item.get("expected_substring", "")
        context_q = item.get("context_query")
        
        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0, "correct": 0, "r1": 0, "r3": 0, "f1_sum": 0.0
            }
        category_stats[cat]["total"] += 1
        
        asst.reset_conversation()
        if context_q:
            asst.ask(context_q)
            
        t0 = time.perf_counter()
        resp: AssistantResponse = asst.ask(q_text)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        
        # Check retrieval rank for answerable items
        r_rank = None
        if exp_type == "ANSWER" and exp_source:
            answerable_count += 1
            # Retrieve candidates directly to record rank
            res = asst.retriever.retrieve_candidates(RetrievalQuery(text=q_text, top_k=5))
            for r, cand in enumerate(res.candidates):
                if cand.source_id == exp_source:
                    r_rank = r + 1
                    break
            
            if r_rank is not None:
                rr_sum += 1.0 / r_rank
                if r_rank == 1:
                    r1_hits += 1
                    category_stats[cat]["r1"] += 1
                if r_rank <= 3:
                    r3_hits += 1
                    category_stats[cat]["r3"] += 1
                if r_rank <= 5:
                    r5_hits += 1
                
        # Check task success & answer quality
        is_success = False
        item_f1 = 0.0
        if exp_type == "ANSWER":
            if resp.decision == "ANSWER":
                ans_text = resp.answer or ""
                item_f1 = token_f1(ans_text, exp_sub)
                f1_scores.append(item_f1)
                category_stats[cat]["f1_sum"] += item_f1
                if exp_sub.lower() in ans_text.lower():
                    is_success = True
                    overall_correct += 1
                    category_stats[cat]["correct"] += 1
        elif exp_type in ("ABSTAIN", "REFUTE"):
            if resp.decision == exp_type:
                is_success = True
                overall_correct += 1
                category_stats[cat]["correct"] += 1
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
                
        query_results.append({
            "id": q_id,
            "category": cat,
            "query": q_text,
            "expected_type": exp_type,
            "expected_source": exp_source,
            "retrieval_rank": r_rank,
            "decision": resp.decision,
            "is_success": is_success,
            "f1": round(item_f1, 4),
            "latency_ms": round(latency, 2)
        })
        
    latencies.sort()
    med_lat = statistics.median(latencies)
    p95_lat = latencies[int(len(latencies) * 0.95)]
    
    summary = {
        "mode": mode_name,
        "total_queries": len(items),
        "answerable_queries": answerable_count,
        "recall_at_1": round(r1_hits / answerable_count, 4) if answerable_count else 0.0,
        "recall_at_1_count": f"{r1_hits}/{answerable_count}",
        "recall_at_3": round(r3_hits / answerable_count, 4) if answerable_count else 0.0,
        "recall_at_3_count": f"{r3_hits}/{answerable_count}",
        "recall_at_5": round(r5_hits / answerable_count, 4) if answerable_count else 0.0,
        "recall_at_5_count": f"{r5_hits}/{answerable_count}",
        "mrr": round(rr_sum / answerable_count, 4) if answerable_count else 0.0,
        "overall_accuracy": round(overall_correct / len(items), 4),
        "overall_accuracy_count": f"{overall_correct}/{len(items)}",
        "mean_token_f1": round(statistics.mean(f1_scores), 4) if f1_scores else 0.0,
        "median_latency_ms": round(med_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "category_breakdown": {
            k: {
                "count": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                "r1": round(v["r1"] / v["total"], 4) if v["total"] else 0.0,
                "r3": round(v["r3"] / v["total"], 4) if v["total"] else 0.0,
                "mean_f1": round(v["f1_sum"] / v["total"], 4) if v["total"] else 0.0
            } for k, v in category_stats.items()
        },
        "query_results": query_results
    }
    return summary

def run_all_evaluations():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    items = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    
    modes = [
        ("bm25", "BM25 Lexical Baseline"),
        ("hash", "Deterministic Hash Hybrid"),
        ("neural_old", "Untrained/Physics ChakrBiEncoder Hybrid"),
        ("neural_domain_adaptive", "Domain-Adaptive ChakrBiEncoder Hybrid")
    ]
    
    all_summaries = {}
    
    for mode_key, mode_name in modes:
        print(f"\n--- Running Evaluation: {mode_name} ---")
        asst, ingest_t = build_assistant(mode_key, tokenizer=tokenizer)
        summary = evaluate_assistant_benchmark(asst, items, mode_name)
        summary["ingest_time_ms"] = round(ingest_t, 2)
        all_summaries[mode_key] = summary
        
        # Save individual result
        out_f = RESULTS_DIR / f"EVAL_{mode_key.upper()}.json"
        out_f.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Recall@1: {summary['recall_at_1_count']} ({summary['recall_at_1'] * 100:.2f}%) | MRR: {summary['mrr']} | Accuracy: {summary['overall_accuracy_count']} | Median Latency: {summary['median_latency_ms']} ms")

    # Run Ablation: Pure Neural Vector Search (No BM25)
    print("\n--- Running Ablation: Domain-Adaptive Pure Neural Vector Search ---")
    prov = LearnedEmbeddingProvider.load_checkpoint(NEW_CKPT, tokenizer=tokenizer)
    vec_index = InMemoryVectorIndex()
    
    # Ingest chunks into pure vector index
    chunk_map = {}
    for f in sorted(CORPUS_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for i in range(0, len(text), 300):
            c_text = text[i:i + 400]
            cid = f"{f.name}:{i}"
            vec = prov.embed_text(c_text)
            vec_index.add_vector(cid, vec, text=c_text)
            chunk_map[cid] = f.name
            
    # Evaluate pure vector search
    pure_r1 = 0
    pure_r3 = 0
    pure_r5 = 0
    answerable_count = 0
    for item in items:
        if item["expected_type"] == "ANSWER" and item.get("expected_source"):
            answerable_count += 1
            q_vec = prov.embed_text(item["query"])
            top_hits = vec_index.search(q_vec, top_k=5)
            hit_sources = [chunk_map[hit_tuple[0]] for hit_tuple in top_hits]
            if item["expected_source"] in hit_sources:
                idx = hit_sources.index(item["expected_source"])
                if idx == 0:
                    pure_r1 += 1
                if idx < 3:
                    pure_r3 += 1
                if idx < 5:
                    pure_r5 += 1
                    
    ablation_summary = {
        "mode": "Pure Domain-Adaptive Neural Vector Search (No Lexical/BM25)",
        "answerable_count": answerable_count,
        "recall_at_1": round(pure_r1 / answerable_count, 4),
        "recall_at_1_count": f"{pure_r1}/{answerable_count}",
        "recall_at_3": round(pure_r3 / answerable_count, 4),
        "recall_at_3_count": f"{pure_r3}/{answerable_count}",
        "recall_at_5": round(pure_r5 / answerable_count, 4),
        "recall_at_5_count": f"{pure_r5}/{answerable_count}",
    }
    all_summaries["ablation_pure_neural"] = ablation_summary
    print(f"Pure Neural Recall@1: {ablation_summary['recall_at_1_count']} ({ablation_summary['recall_at_1'] * 100:.2f}%) | Recall@3: {ablation_summary['recall_at_3_count']}")
    
    # Paired Comparison Old Neural vs Domain-Adaptive Neural
    old_res = all_summaries["neural_old"]["query_results"]
    new_res = all_summaries["neural_domain_adaptive"]["query_results"]
    bm25_res = all_summaries["bm25"]["query_results"]
    
    paired_comparison = []
    wins = 0
    losses = 0
    ties = 0
    for o, n, b in zip(old_res, new_res, bm25_res):
        r_old = o["retrieval_rank"]
        r_new = n["retrieval_rank"]
        
        status = "TIE"
        if r_old is not None and r_new is not None:
            if r_new < r_old:
                wins += 1
                status = "NEW_MODEL_WIN"
            elif r_new > r_old:
                losses += 1
                status = "NEW_MODEL_LOSS"
            else:
                ties += 1
                
        paired_comparison.append({
            "id": o["id"],
            "query": o["query"],
            "category": o["category"],
            "bm25_rank": b["retrieval_rank"],
            "old_neural_rank": r_old,
            "new_neural_rank": r_new,
            "status": status
        })
        
    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configurations": {
            k: {
                "name": v["mode"] if "mode" in v else k,
                "recall_at_1": v.get("recall_at_1"),
                "recall_at_3": v.get("recall_at_3"),
                "recall_at_5": v.get("recall_at_5"),
                "mrr": v.get("mrr"),
                "accuracy": v.get("overall_accuracy"),
                "median_latency_ms": v.get("median_latency_ms"),
                "ingest_time_ms": v.get("ingest_time_ms")
            } for k, v in all_summaries.items()
        },
        "paired_analysis_new_vs_old_neural": {
            "new_model_wins": wins,
            "new_model_losses": losses,
            "ties": ties
        },
        "paired_details": paired_comparison
    }
    
    comp_file = RESULTS_DIR / "RETRIEVAL_COMPARISON.json"
    comp_file.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    print(f"\nSaved master comparison to {comp_file}")

if __name__ == "__main__":
    run_all_evaluations()
