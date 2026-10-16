"""
Batch 1261–1300 Reranking Experiment and Refutation Verification Runner.
Location: experiments/batch1261_1300/run_reranking_experiment.py

Evaluates:
1. Baseline BM25 (K=5, K=10, K=20)
2. Existing Domain-Adaptive Full Corpus Hybrid
3. BM25 Top-K Candidate Generation + Pure Neural Reranker (K=10, K=20)
4. BM25 Top-K + Calibrated Score Fusion Reranker (alpha=0.1, 0.2, 0.5)
5. Decision Policy Accuracy (ANSWER vs REFUTE vs ABSTAIN)

Generates machine-readable output under experiments/batch1261_1300/results/
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant, AssistantResponse
from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.runtime.retrieval import (
    BM25KnowledgeIndex,
    RetrievalQuery,
    EmbeddingProvider
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
BENCHMARK_PATH = Path("experiments/batch1141_1180/data/held_out_70_benchmark.json")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DOMAIN_ADAPTIVE_CKPT = Path("experiments/batch1181_1220/checkpoints/domain_adaptive_bi_encoder.pt")
OUT_DIR = Path("experiments/batch1261_1300/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_reranking_suite():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    provider = LearnedEmbeddingProvider.load_checkpoint(DOMAIN_ADAPTIVE_CKPT, tokenizer=tokenizer)
    bench_items = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    
    # 1. Build Base Assistant with BM25
    asst = LocalKnowledgeAssistant(
        storage_dir=None,
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider=None,
        extract_concise_span=True
    )
    for f in sorted(CORPUS_DIR.glob("*.md")):
        asst.ingest_document(f)
        
    answerable_items = [x for x in bench_items if x["expected_type"] == "ANSWER" and x.get("expected_source")]
    
    # Evaluate BM25 candidate recall at K=5, 10, 20
    k_recalls = {}
    for k_val in [5, 10, 20]:
        hits = 0
        for item in answerable_items:
            res = asst.retriever.lexical_index.search(item["query"], top_k=k_val)
            cand_sources = [Path(chunk.doc_id).name for chunk, score in res]
            if item["expected_source"] in cand_sources:
                hits += 1
        k_recalls[f"recall_at_{k_val}"] = {
            "hits": hits,
            "total": len(answerable_items),
            "ratio": round(hits / len(answerable_items), 4)
        }
        print(f"BM25 Candidate Recall at K={k_val}: {hits}/{len(answerable_items)} ({hits/len(answerable_items)*100:.2f}%)")

    # Evaluate Reranker Configurations
    def evaluate_reranker(k_depth: int, alpha: float = 1.0):
        # alpha=1.0 -> Pure dense reranker over BM25 top-K
        # alpha < 1.0 -> Score fusion: (1-alpha)*bm25_norm + alpha*dense
        r1 = 0
        r3 = 0
        r5 = 0
        rr_sum = 0.0
        latencies = []
        per_query = []
        
        for item in answerable_items:
            t0 = time.perf_counter()
            bm25_res = asst.retriever.lexical_index.search(item["query"], top_k=k_depth)
            
            if not bm25_res:
                per_query.append({"id": item["id"], "rank": None, "win_loss": "LOSS"})
                continue
                
            q_vec = provider.embed_text(item["query"])
            
            # Compute BM25 min/max for normalization
            b_scores = [score for chunk, score in bm25_res]
            min_b, max_b = min(b_scores), max(b_scores)
            range_b = (max_b - min_b) if (max_b - min_b) > 1e-6 else 1.0
            
            scored_cands = []
            for rank_idx, (chunk, score) in enumerate(bm25_res):
                d_vec = provider.embed_text(chunk.text)
                cos_sim = EmbeddingProvider.cosine_similarity(q_vec, d_vec)
                norm_b = (score - min_b) / range_b
                
                if alpha == 1.0:
                    final_score = cos_sim
                else:
                    final_score = (1.0 - alpha) * norm_b + alpha * cos_sim
                    
                scored_cands.append((final_score, -rank_idx, Path(chunk.doc_id).name, chunk))
                
            scored_cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            
            ranked_sources = [x[2] for x in scored_cands]
            m_rank = None
            if item["expected_source"] in ranked_sources:
                m_rank = ranked_sources.index(item["expected_source"]) + 1
                rr_sum += 1.0 / m_rank
                if m_rank == 1:
                    r1 += 1
                if m_rank <= 3:
                    r3 += 1
                if m_rank <= 5:
                    r5 += 1
                    
            per_query.append({
                "id": item["id"],
                "query": item["query"],
                "expected_source": item["expected_source"],
                "rank": m_rank,
                "top_source": ranked_sources[0] if ranked_sources else None
            })
            
        return {
            "k_depth": k_depth,
            "alpha": alpha,
            "recall_at_1": round(r1 / len(answerable_items), 4),
            "recall_at_1_count": f"{r1}/{len(answerable_items)}",
            "recall_at_3": round(r3 / len(answerable_items), 4),
            "recall_at_3_count": f"{r3}/{len(answerable_items)}",
            "recall_at_5": round(r5 / len(answerable_items), 4),
            "recall_at_5_count": f"{r5}/{len(answerable_items)}",
            "mrr": round(rr_sum / len(answerable_items), 4),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "p95_latency_ms": round(latencies[int(len(latencies)*0.95)], 2),
            "per_query": per_query
        }
        
    print("\nEvaluating Reranking Configurations...")
    res_pure_k10 = evaluate_reranker(k_depth=10, alpha=1.0)
    print(f"[Pure Neural Reranker K=10] R@1: {res_pure_k10['recall_at_1_count']} ({res_pure_k10['recall_at_1']*100:.2f}%) | MRR: {res_pure_k10['mrr']} | Latency: {res_pure_k10['median_latency_ms']} ms")
    
    res_pure_k20 = evaluate_reranker(k_depth=20, alpha=1.0)
    print(f"[Pure Neural Reranker K=20] R@1: {res_pure_k20['recall_at_1_count']} ({res_pure_k20['recall_at_1']*100:.2f}%) | MRR: {res_pure_k20['mrr']} | Latency: {res_pure_k20['median_latency_ms']} ms")

    res_fused_02_k10 = evaluate_reranker(k_depth=10, alpha=0.2)
    print(f"[Fused Reranker K=10, alpha=0.2] R@1: {res_fused_02_k10['recall_at_1_count']} ({res_fused_02_k10['recall_at_1']*100:.2f}%) | MRR: {res_fused_02_k10['mrr']} | Latency: {res_fused_02_k10['median_latency_ms']} ms")

    res_fused_05_k10 = evaluate_reranker(k_depth=10, alpha=0.5)
    print(f"[Fused Reranker K=10, alpha=0.5] R@1: {res_fused_05_k10['recall_at_1_count']} ({res_fused_05_k10['recall_at_1']*100:.2f}%) | MRR: {res_fused_05_k10['mrr']} | Latency: {res_fused_05_k10['median_latency_ms']} ms")

    # 3. Evaluate Decision Policy on all 70 items with refutation logic enabled
    print("\nEvaluating Decision Policy on all 70 items...")
    decision_eval = []
    decision_correct = 0
    cat_stats = {}
    
    for item in bench_items:
        cat = item["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0}
        cat_stats[cat]["total"] += 1
        
        asst.reset_conversation()
        if item.get("context_query"):
            asst.ask(item["context_query"])
            
        resp = asst.ask(item["query"])
        is_correct = (resp.decision == item["expected_type"])
        if is_correct:
            decision_correct += 1
            cat_stats[cat]["correct"] += 1
            
        decision_eval.append({
            "id": item["id"],
            "category": cat,
            "query": item["query"],
            "expected": item["expected_type"],
            "actual": resp.decision,
            "is_correct": is_correct,
            "answer_preview": resp.answer[:120]
        })
        
    print(f"Overall Decision Accuracy: {decision_correct}/{len(bench_items)} ({decision_correct/len(bench_items)*100:.2f}%)")
    for cat, v in cat_stats.items():
        print(f"  Category '{cat}': {v['correct']}/{v['total']} ({v['correct']/v['total']*100:.2f}%)")

    # Save comprehensive machine-readable report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_pool_analysis": k_recalls,
        "reranking_configurations": {
            "pure_reranker_k10": res_pure_k10,
            "pure_reranker_k20": res_pure_k20,
            "fused_reranker_k10_alpha02": res_fused_02_k10,
            "fused_reranker_k10_alpha05": res_fused_05_k10
        },
        "decision_policy_evaluation": {
            "total_queries": len(bench_items),
            "correct_decisions": decision_correct,
            "overall_decision_accuracy": round(decision_correct / len(bench_items), 4),
            "category_breakdown": cat_stats,
            "per_query_decisions": decision_eval
        }
    }
    
    out_file = OUT_DIR / "RERANKING_AND_DECISION_RESULTS.json"
    out_file.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    run_reranking_suite()
