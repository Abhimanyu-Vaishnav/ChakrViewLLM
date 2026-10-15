"""
Workstream A & E: Independent Evaluator for Batches 1101-1140.
Location: experiments/batch1101_1140/independent_evaluator.py

Key Capabilities:
1. Strict Evaluation Protocol:
   - Evaluates Token-level Precision, Recall, and F1.
   - Exact Normalized Answer Match.
   - Explicit Numeric and Unit Comparison (prevents single-digit substring false positives like '6' matching 800-char chunks).
   - Strict separation of Citation Validity (disk byte bounds slice check) vs. Citation Support (whether target evidence is in cited span).
   - Strict separation of Correct Abstention vs. Incorrect Abstention.
2. Supports Multiple Retrievers / Embeddings:
   - BM25 pure lexical
   - Hybrid deterministic hash
   - Hybrid learned neural bi-encoder (ChakrBiEncoder)
3. Evaluates Concise Answer Span Extraction vs. Full Chunk Dumping.
4. Produces machine-readable JSON metrics and compatibility reports.
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
from chakrview.runtime.knowledge import BM25KnowledgeIndex, KnowledgeChunk
from chakrview.runtime.retrieval import HybridRetriever, RetrievalQuery, DeterministicHashEmbeddingProvider
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
class StrictEvaluationItem:
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
    old_evaluator_passed: bool
    new_evaluator_passed: bool
    latency_ms: float
    answer_preview: str
    failure_reasons: List[str]


def run_evaluation(
    corpus_dir: Path | str,
    benchmark_file: Path | str,
    retriever_mode: str = "hybrid_hash",  # "bm25", "hybrid_hash", "hybrid_neural"
    extract_concise: bool = True,
    bi_encoder_checkpoint: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
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

    # Initialize assistant with appropriate embedding provider
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

    assistant = LocalKnowledgeAssistant(
        embedding_provider=embedding_provider,
        extract_concise_span=extract_concise,
    )
    total_chunks = assistant.ingest_directory(corpus_p)

    results: List[StrictEvaluationItem] = []
    total_q = len(benchmark)
    new_passed_count = 0
    old_passed_count = 0
    citation_valid_count = 0
    citation_support_count = 0
    total_citations_checked = 0
    f1_scores: List[float] = []
    exact_matches: int = 0
    latencies: List[float] = []

    abstention_correct = 0
    total_abstentions_expected = 0
    refutation_correct = 0
    total_refutations_expected = 0
    followup_correct = 0
    total_followups_expected = 0

    retrieval_recalls_top1: List[int] = []
    retrieval_mrrs: List[float] = []

    for item in benchmark:
        qid = item["id"]
        cat = item.get("category", "unknown")
        q = item["query"]
        exp_type = item["expected_type"]
        exp_source = item.get("expected_source")
        exp_sub = item.get("expected_substring")
        context_q = item.get("context_query")

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
            failure_reasons.append(f"Decision mismatch: exp {exp_type}, got {resp.decision}")

        cit_valid = True
        cit_support = True
        old_ans_correct = False
        new_ans_correct = False
        exact_match = False
        token_f1 = 0.0
        numeric_match = True

        if exp_type == "ANSWER":
            # Check retrieval ranking if source is known
            if exp_source:
                res = assistant.retriever.retrieve_candidates(RetrievalQuery(text=q, top_k=5))
                m_rank = None
                for r, cand in enumerate(res.candidates):
                    if cand.source_id == exp_source:
                        m_rank = r + 1
                        break
                if m_rank == 1:
                    retrieval_recalls_top1.append(1)
                else:
                    retrieval_recalls_top1.append(0)
                retrieval_mrrs.append(1.0 / m_rank if m_rank else 0.0)

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

            # Old Evaluator logic: substring inside answer
            old_ans_correct = (exp_sub.lower() in resp.answer.lower()) if exp_sub else True

            # New Strict Evaluator logic:
            if exp_sub:
                numeric_match = check_numeric_match(resp.answer, exp_sub)
                _, _, token_f1 = compute_token_f1(resp.answer, exp_sub)
                norm_exp = normalize_text(exp_sub)
                norm_ans = normalize_text(resp.answer)
                exact_match = (norm_exp == norm_ans)
                # Strict pass requires decision correct, citation valid, citation support, numeric match, and evidence retention
                new_ans_correct = is_decision_correct and cit_valid and cit_support and numeric_match and (exp_sub.lower() in resp.answer.lower())
            else:
                new_ans_correct = is_decision_correct and cit_valid

            f1_scores.append(token_f1)
            if exact_match:
                exact_matches += 1

        elif exp_type == "ABSTAIN":
            total_abstentions_expected += 1
            is_valid_abstain = ("could not find sufficient" in resp.answer.lower())
            old_ans_correct = is_decision_correct and is_valid_abstain
            new_ans_correct = is_decision_correct and is_valid_abstain
            if is_decision_correct:
                abstention_correct += 1

        elif exp_type == "REFUTE":
            total_refutations_expected += 1
            is_valid_refute = ("contradiction detected" in resp.answer.lower())
            old_ans_correct = is_decision_correct and is_valid_refute
            new_ans_correct = is_decision_correct and is_valid_refute
            if is_decision_correct:
                refutation_correct += 1

        if cat == "follow_up":
            total_followups_expected += 1
            if new_ans_correct:
                followup_correct += 1

        old_passed = is_decision_correct and old_ans_correct and (cit_valid if exp_type == "ANSWER" else True)
        new_passed = is_decision_correct and new_ans_correct

        if old_passed:
            old_passed_count += 1
        if new_passed:
            new_passed_count += 1

        results.append(StrictEvaluationItem(
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
            old_evaluator_passed=old_passed,
            new_evaluator_passed=new_passed,
            latency_ms=round(lat, 2),
            answer_preview=resp.answer[:120].replace("\n", " "),
            failure_reasons=failure_reasons,
        ))

    latencies_sorted = sorted(latencies)
    median_lat = round(latencies_sorted[len(latencies_sorted) // 2], 2)
    p95_lat = round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 2)
    avg_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
    r1 = round(sum(retrieval_recalls_top1) / len(retrieval_recalls_top1) * 100, 2) if retrieval_recalls_top1 else 0.0
    mrr = round(sum(retrieval_mrrs) / len(retrieval_mrrs), 4) if retrieval_mrrs else 0.0

    return {
        "benchmark_file": str(bench_p),
        "benchmark_sha256": bench_sha256,
        "corpus_dir": str(corpus_p),
        "retriever_mode": retriever_mode,
        "extract_concise_span": extract_concise,
        "total_chunks_indexed": total_chunks,
        "total_questions": total_q,
        "new_evaluator_passed": new_passed_count,
        "new_evaluator_accuracy_pct": round(new_passed_count / total_q * 100, 2),
        "old_evaluator_passed": old_passed_count,
        "old_evaluator_accuracy_pct": round(old_passed_count / total_q * 100, 2),
        "retrieval_recall_top1_pct": r1,
        "retrieval_mrr": mrr,
        "mean_token_f1": avg_f1,
        "exact_matches": exact_matches,
        "citation_validity_count": citation_valid_count,
        "citation_support_count": citation_support_count,
        "total_citations_checked": total_citations_checked,
        "correct_abstentions": f"{abstention_correct}/{total_abstentions_expected}",
        "correct_refutations": f"{refutation_correct}/{total_refutations_expected}",
        "correct_followups": f"{followup_correct}/{total_followups_expected}",
        "latency_median_ms": median_lat,
        "latency_p95_ms": p95_lat,
        "items": [asdict(r) for r in results],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="experiments/batch1021_1060/data/sample_corpus")
    parser.add_argument("--benchmark", default="experiments/batch1061_1100/data/generalization_eval.json")
    parser.add_argument("--mode", default="hybrid_hash", choices=["bm25", "hybrid_hash", "hybrid_neural"])
    parser.add_argument("--concise", action="store_true", default=True)
    parser.add_argument("--no-concise", dest="concise", action="store_false")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    res = run_evaluation(
        corpus_dir=args.corpus,
        benchmark_file=args.benchmark,
        retriever_mode=args.mode,
        extract_concise=args.concise,
    )
    print(f"Evaluated {res['total_questions']} items:")
    print(f"  New Strict Evaluator Accuracy: {res['new_evaluator_passed']}/{res['total_questions']} ({res['new_evaluator_accuracy_pct']}%)")
    print(f"  Old Evaluator Accuracy:        {res['old_evaluator_passed']}/{res['total_questions']} ({res['old_evaluator_accuracy_pct']}%)")
    print(f"  Retrieval Recall@1:            {res['retrieval_recall_top1_pct']}% | MRR: {res['retrieval_mrr']}")
    print(f"  Mean Token F1:                 {res['mean_token_f1']}")
    print(f"  Citation Validity:             {res['citation_validity_count']}/{res['total_citations_checked']}")
    print(f"  Citation Support:              {res['citation_support_count']}/{res['total_citations_checked']}")
    print(f"  Abstentions:                   {res['correct_abstentions']}")
    print(f"  Refutations:                   {res['correct_refutations']}")
    print(f"  Median Latency:                {res['latency_median_ms']} ms")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {out_p}")
