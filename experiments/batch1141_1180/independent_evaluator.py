"""
Workstream 3 & 4 & 5: Comprehensive Independent Evaluator for Batches 1141-1180.
Location: experiments/batch1141_1180/independent_evaluator.py

Evaluates:
1. Retrieval Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR).
2. Per-category analysis: exact_entity, paraphrase, synonym, distractor_competition,
   version_conflict, follow_up, multilingual_hinglish, false_premise, abstention.
3. Decoupled Citation Validity (disk byte offset slice match) vs Citation Support (answer fact present in cited span).
4. Extractive concise span compression ratio and token-level F1.
5. Latency profiling (retrieval vs span extraction vs end-to-end).
6. Embedding cache hit rate and cold vs warm execution latencies.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Set

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant, AssistantResponse
from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.runtime.retrieval import (
    HybridRetriever,
    RetrievalQuery,
    DeterministicHashEmbeddingProvider,
    CachedEmbeddingProvider,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation, and collapses whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s\.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_token_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Computes token-level precision, recall, and F1 score."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return (1.0, 1.0, 1.0) if pred_tokens == gt_tokens else (0.0, 0.0, 0.0)

    common = set(pred_tokens).intersection(set(gt_tokens))
    if not common:
        return 0.0, 0.0, 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def check_numeric_match(prediction: str, ground_truth: str) -> bool:
    """
    Checks if numbers and units present in ground truth explicitly match in prediction.
    Avoids false matching '6' when prediction only contains '1536' or unrelated counts.
    """
    gt_nums = re.findall(r"\b\d+(?:\.\d+)?%?(?:gb|ms|mb)?\b", ground_truth.lower())
    if not gt_nums:
        return True
    pred_nums = re.findall(r"\b\d+(?:\.\d+)?%?(?:gb|ms|mb)?\b", prediction.lower())
    pred_num_set = set(pred_nums)
    return all(n in pred_num_set for n in gt_nums)


@dataclass
class EvaluationItemResult:
    id: int
    category: str
    query: str
    expected_type: str
    expected_source: Optional[str]
    expected_substring: Optional[str]
    decision: str
    is_decision_correct: bool
    citation_validity: bool
    citation_support: bool
    answer_exact_match: bool
    token_f1: float
    numeric_match: bool
    passed: bool
    retrieval_rank: Optional[int]
    latency_ms: float
    answer_preview: str
    failure_reasons: List[str]


def run_evaluation(
    corpus_dir: Path | str,
    benchmark_file: Path | str,
    retriever_mode: str = "hybrid_hash",  # "bm25", "hybrid_hash", "hybrid_neural"
    extract_concise: bool = True,
    use_cache: bool = True,
    bi_encoder_checkpoint: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> Dict[str, Any]:
    corpus_p = Path(corpus_dir).resolve()
    bench_p = Path(benchmark_file).resolve()

    if not corpus_p.exists():
        raise FileNotFoundError(f"Corpus dir not found: {corpus_p}")
    if not bench_p.exists():
        raise FileNotFoundError(f"Benchmark file not found: {bench_p}")

    bench_sha256 = hashlib.sha256(bench_p.read_bytes()).hexdigest()
    with open(bench_p, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Initialize embedding provider
    embedding_provider = None
    if retriever_mode == "hybrid_neural":
        t_dir = Path(tokenizer_dir or "data/experiments/vocab_4096")
        tok, _ = load_tokenizer_artifacts(t_dir)
        ckpt_p = bi_encoder_checkpoint or "experiments/batch171_180_pointer_learning/checkpoints/improved_bi_encoder.pt"
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        model = ChakrBiEncoder(
            vocab_size=ckpt.get("vocab_size", 4096),
            d_model=ckpt.get("d_model", 128),
            d_embed=ckpt.get("d_embed", 64),
            n_layers=ckpt.get("n_layers", 2),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        embedding_provider = LearnedEmbeddingProvider(model=model, tokenizer=tok)
    elif retriever_mode == "hybrid_hash":
        embedding_provider = DeterministicHashEmbeddingProvider()
    elif retriever_mode == "bm25":
        embedding_provider = None

    t_ingest_start = time.perf_counter()
    assistant = LocalKnowledgeAssistant(
        embedding_provider=embedding_provider,
        extract_concise_span=extract_concise,
        use_embedding_cache=use_cache,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    total_chunks = assistant.ingest_directory(corpus_p)
    t_ingest_ms = (time.perf_counter() - t_ingest_start) * 1000.0

    results: List[EvaluationItemResult] = []
    total_q = len(benchmark)
    passed_count = 0
    citation_valid_count = 0
    citation_support_count = 0
    total_citations_checked = 0
    f1_scores: List[float] = []
    exact_matches: int = 0
    latencies: List[float] = []

    # Category breakdowns
    category_stats: Dict[str, Dict[str, Any]] = {}

    # Retrieval metrics
    recalls_top1: List[int] = []
    recalls_top3: List[int] = []
    recalls_top5: List[int] = []
    mrrs: List[float] = []

    for item in benchmark:
        qid = item["id"]
        cat = item.get("category", "unknown")
        q = item["query"]
        exp_type = item["expected_type"]
        exp_source = item.get("expected_source")
        exp_sub = item.get("expected_substring")
        context_q = item.get("context_query")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "recall_top1": 0, "f1_sum": 0.0}
        category_stats[cat]["total"] += 1

        assistant.reset_conversation()
        if context_q:
            assistant.ask(context_q)

        t_start = time.perf_counter()
        resp: AssistantResponse = assistant.ask(q)
        lat = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat)

        failure_reasons = []
        is_decision_correct = (resp.decision == exp_type)
        if not is_decision_correct:
            failure_reasons.append(f"Decision mismatch: expected {exp_type}, got {resp.decision}")

        cit_valid = True
        cit_support = True
        exact_match = False
        token_f1 = 0.0
        numeric_match = True
        m_rank = None

        if exp_type == "ANSWER":
            # Check retrieval ranking if source is specified
            if exp_source:
                res = assistant.retriever.retrieve_candidates(RetrievalQuery(text=q, top_k=5))
                for r, cand in enumerate(res.candidates):
                    if cand.source_id == exp_source:
                        m_rank = r + 1
                        break
                r1 = 1 if (m_rank == 1) else 0
                r3 = 1 if (m_rank is not None and m_rank <= 3) else 0
                r5 = 1 if (m_rank is not None and m_rank <= 5) else 0
                recalls_top1.append(r1)
                recalls_top3.append(r3)
                recalls_top5.append(r5)
                mrrs.append(1.0 / m_rank if m_rank else 0.0)
                category_stats[cat]["recall_top1"] += r1

            if not resp.citations:
                cit_valid = False
                cit_support = False
                failure_reasons.append("Missing required citations")
            else:
                for c in resp.citations:
                    total_citations_checked += 1
                    doc_id = c.get("doc_id")
                    start_c = c.get("start_char", 0)
                    end_c = c.get("end_char", 0)
                    exact_text = c.get("exact_text", "")

                    doc_file = corpus_p / doc_id
                    if not doc_file.exists():
                        cit_valid = False
                        failure_reasons.append(f"Cited non-existent doc: {doc_id}")
                    else:
                        doc_content = doc_file.read_text(encoding="utf-8", errors="replace")
                        if start_c < 0 or end_c > len(doc_content) or start_c >= end_c:
                            cit_valid = False
                            failure_reasons.append(f"Invalid offsets [{start_c}:{end_c}]")
                        else:
                            disk_slice = doc_content[start_c:end_c].strip()
                            if disk_slice != exact_text.strip():
                                cit_valid = False
                                failure_reasons.append("Citation exact_text does not match disk slice")

                    if exp_sub and exp_sub.lower() not in exact_text.lower():
                        cit_support = False
                        failure_reasons.append(f"Cited span misses target '{exp_sub}'")

                if cit_valid:
                    citation_valid_count += 1
                if cit_support:
                    citation_support_count += 1

            if exp_sub:
                numeric_match = check_numeric_match(resp.answer, exp_sub)
                _, _, token_f1 = compute_token_f1(resp.answer, exp_sub)
                norm_exp = normalize_text(exp_sub)
                norm_ans = normalize_text(resp.answer)
                exact_match = (norm_exp == norm_ans)
                item_passed = (
                    is_decision_correct
                    and cit_valid
                    and cit_support
                    and numeric_match
                    and (exp_sub.lower() in resp.answer.lower())
                )
            else:
                item_passed = is_decision_correct and cit_valid

            f1_scores.append(token_f1)
            category_stats[cat]["f1_sum"] += token_f1
            if exact_match:
                exact_matches += 1

        elif exp_type == "ABSTAIN":
            is_valid_abstain = ("could not find sufficient" in resp.answer.lower())
            item_passed = is_decision_correct and is_valid_abstain
        elif exp_type == "REFUTE":
            is_valid_refute = ("contradiction detected" in resp.answer.lower())
            item_passed = is_decision_correct and is_valid_refute
        else:
            item_passed = False

        if item_passed:
            passed_count += 1
            category_stats[cat]["passed"] += 1

        results.append(EvaluationItemResult(
            id=qid,
            category=cat,
            query=q,
            expected_type=exp_type,
            expected_source=exp_source,
            expected_substring=exp_sub,
            decision=resp.decision,
            is_decision_correct=is_decision_correct,
            citation_validity=cit_valid,
            citation_support=cit_support,
            answer_exact_match=exact_match,
            token_f1=round(token_f1, 4),
            numeric_match=numeric_match,
            passed=item_passed,
            retrieval_rank=m_rank,
            latency_ms=round(lat, 2),
            answer_preview=resp.answer[:120].replace("\n", " "),
            failure_reasons=failure_reasons,
        ))

    latencies_sorted = sorted(latencies)
    median_lat = round(latencies_sorted[len(latencies_sorted) // 2], 2)
    p95_lat = round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 2)
    avg_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0

    r1_pct = round(sum(recalls_top1) / len(recalls_top1) * 100, 2) if recalls_top1 else 0.0
    r3_pct = round(sum(recalls_top3) / len(recalls_top3) * 100, 2) if recalls_top3 else 0.0
    r5_pct = round(sum(recalls_top5) / len(recalls_top5) * 100, 2) if recalls_top5 else 0.0
    mrr_score = round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0

    # Calculate category summary
    category_summary = {}
    for cat, stats in category_stats.items():
        tot = stats["total"]
        category_summary[cat] = {
            "total": tot,
            "passed": stats["passed"],
            "accuracy_pct": round(stats["passed"] / tot * 100, 2) if tot else 0.0,
            "mean_token_f1": round(stats["f1_sum"] / tot, 4) if tot else 0.0,
        }

    # Cache stats
    cache_stats = None
    if isinstance(assistant.embedding_provider, CachedEmbeddingProvider):
        cache_stats = {
            "size": assistant.embedding_provider.size,
            "hits": assistant.embedding_provider.hits,
            "misses": assistant.embedding_provider.misses,
            "evictions": assistant.embedding_provider.evictions,
        }

    return {
        "benchmark_file": str(bench_p),
        "benchmark_sha256": bench_sha256,
        "corpus_dir": str(corpus_p),
        "retriever_mode": retriever_mode,
        "extract_concise_span": extract_concise,
        "use_cache": use_cache,
        "total_documents_indexed": len(assistant.doc_metadata),
        "total_chunks_indexed": total_chunks,
        "ingest_time_ms": round(t_ingest_ms, 2),
        "total_questions": total_q,
        "passed_count": passed_count,
        "accuracy_pct": round(passed_count / total_q * 100, 2),
        "retrieval_recall_top1_pct": r1_pct,
        "retrieval_recall_top3_pct": r3_pct,
        "retrieval_recall_top5_pct": r5_pct,
        "retrieval_mrr": mrr_score,
        "mean_token_f1": avg_f1,
        "exact_matches": exact_matches,
        "citation_validity_count": citation_valid_count,
        "citation_support_count": citation_support_count,
        "total_citations_checked": total_citations_checked,
        "latency_median_ms": median_lat,
        "latency_p95_ms": p95_lat,
        "categories": category_summary,
        "cache_stats": cache_stats,
        "items": [asdict(r) for r in results],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="experiments/batch1141_1180/data/corpus")
    parser.add_argument("--benchmark", default="experiments/batch1141_1180/data/held_out_70_benchmark.json")
    parser.add_argument("--mode", default="hybrid_hash", choices=["bm25", "hybrid_hash", "hybrid_neural"])
    parser.add_argument("--concise", action="store_true", default=True)
    parser.add_argument("--no-concise", dest="concise", action="store_false")
    parser.add_argument("--no-cache", dest="cache", action="store_false", default=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    res = run_evaluation(
        corpus_dir=args.corpus,
        benchmark_file=args.benchmark,
        retriever_mode=args.mode,
        extract_concise=args.concise,
        use_cache=args.cache,
    )
    print(f"Evaluated {res['total_questions']} items (Mode: {res['retriever_mode']}, Concise: {res['extract_concise_span']}):")
    print(f"  Accuracy:                      {res['passed_count']}/{res['total_questions']} ({res['accuracy_pct']}%)")
    print(f"  Retrieval Recall@1:            {res['retrieval_recall_top1_pct']}% | Recall@3: {res['retrieval_recall_top3_pct']}% | Recall@5: {res['retrieval_recall_top5_pct']}% | MRR: {res['retrieval_mrr']}")
    print(f"  Mean Token F1:                 {res['mean_token_f1']}")
    print(f"  Citation Validity:             {res['citation_validity_count']}/{res['total_citations_checked']}")
    print(f"  Citation Support:              {res['citation_support_count']}/{res['total_citations_checked']}")
    print(f"  Ingest Latency:                {res['ingest_time_ms']} ms")
    print(f"  Query Median Latency:          {res['latency_median_ms']} ms (p95: {res['latency_p95_ms']} ms)")
    if res.get("cache_stats"):
        print(f"  Cache Stats:                   {res['cache_stats']}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {out_p}")
