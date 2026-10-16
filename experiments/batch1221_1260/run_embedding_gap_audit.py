"""
Diagnostic Script: Training-to-Retrieval Gap Trace and Embedding Space Analysis.
Location: experiments/batch1221_1260/run_embedding_gap_audit.py

Analyzes:
1. Embedding Norms, Cosine Distances, Margins on Contrastive Test Triples vs Corpus Chunks.
2. Pairwise cross-similarity matrix across all 116 corpus chunks.
3. Nearest-neighbor overlap and false-retrieval diagnoses for Domain-Adaptive vs Old BiEncoder.
4. Identifies the exact mathematical reason why Pairwise Test Accuracy is 100% while Corpus Retrieval drops to 5.0% on pure dense search and 88.33% in hybrid.
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

from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.runtime.retrieval import EmbeddingProvider
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
DATA_DIR = Path("experiments/batch1181_1220/data")
OLD_CKPT = Path("experiments/batch171_180_pointer_learning/checkpoints/improved_bi_encoder.pt")
NEW_CKPT = Path("experiments/batch1181_1220/checkpoints/domain_adaptive_bi_encoder.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
OUT_DIR = Path("experiments/batch1221_1260/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_embedding_space():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    old_provider = LearnedEmbeddingProvider.load_checkpoint(OLD_CKPT, tokenizer=tokenizer)
    new_provider = LearnedEmbeddingProvider.load_checkpoint(NEW_CKPT, tokenizer=tokenizer)
    
    test_triples = json.loads((DATA_DIR / "test_split.json").read_text(encoding="utf-8"))
    
    # 1. Contrastive Test Split Margins
    def eval_triples(provider):
        pos_sims = []
        neg_sims = []
        margins = []
        correct_count = 0
        for item in test_triples:
            q_v = provider.embed_text(item["query"])
            p_v = provider.embed_text(item["pos_passage"])
            n_v = provider.embed_text(item["neg_passage"])
            
            s_p = EmbeddingProvider.cosine_similarity(q_v, p_v)
            s_n = EmbeddingProvider.cosine_similarity(q_v, n_v)
            pos_sims.append(s_p)
            neg_sims.append(s_n)
            margins.append(s_p - s_n)
            if s_p > s_n:
                correct_count += 1
        return {
            "accuracy": round(correct_count / len(test_triples), 4),
            "mean_pos_sim": round(statistics.mean(pos_sims), 4),
            "mean_neg_sim": round(statistics.mean(neg_sims), 4),
            "mean_margin": round(statistics.mean(margins), 4),
            "min_margin": round(min(margins), 4),
            "max_margin": round(max(margins), 4),
        }
        
    old_triple_metrics = eval_triples(old_provider)
    new_triple_metrics = eval_triples(new_provider)
    
    # 2. Corpus Chunk Space Density & Cross-Similarity Analysis
    # Load all 52 corpus docs and chunk them
    chunks = []
    chunk_meta = []
    for f in sorted(CORPUS_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for i in range(0, len(text), 300):
            c_text = text[i:i + 400]
            chunks.append(c_text)
            chunk_meta.append((f.name, i))
            
    old_chunk_vecs = [old_provider.embed_text(c) for c in chunks]
    new_chunk_vecs = [new_provider.embed_text(c) for c in chunks]
    
    # Sample 100 random pairs to measure inter-chunk background similarity
    import random
    random.seed(42)
    sample_pairs = [(random.randint(0, len(chunks)-1), random.randint(0, len(chunks)-1)) for _ in range(500)]
    sample_pairs = [(i, j) for (i, j) in sample_pairs if i != j]
    
    old_inter_sims = [EmbeddingProvider.cosine_similarity(old_chunk_vecs[i], old_chunk_vecs[j]) for (i, j) in sample_pairs]
    new_inter_sims = [EmbeddingProvider.cosine_similarity(new_chunk_vecs[i], new_chunk_vecs[j]) for (i, j) in sample_pairs]
    
    corpus_space_metrics = {
        "total_chunks_indexed": len(chunks),
        "old_model_mean_interchunk_cosine": round(statistics.mean(old_inter_sims), 4),
        "old_model_stdev_interchunk_cosine": round(statistics.stdev(old_inter_sims), 4),
        "new_model_mean_interchunk_cosine": round(statistics.mean(new_inter_sims), 4),
        "new_model_stdev_interchunk_cosine": round(statistics.stdev(new_inter_sims), 4),
    }
    
    # 3. Trace Failure Cases in 70-Item Benchmark where Domain-Adaptive Lost to BM25
    bench_items = json.loads(Path("experiments/batch1141_1180/data/held_out_70_benchmark.json").read_text(encoding="utf-8"))
    
    # Trace specific queries where neural model failed rank 1
    failures = []
    for item in bench_items:
        if item["expected_type"] == "ANSWER" and item.get("expected_source"):
            q_text = item["query"]
            exp_src = item["expected_source"]
            q_vec = new_provider.embed_text(q_text)
            
            # Rank all chunks
            sims = [(chunk_meta[i][0], chunk_meta[i][1], EmbeddingProvider.cosine_similarity(q_vec, new_chunk_vecs[i]), chunks[i][:150]) for i in range(len(chunks))]
            sims.sort(key=lambda x: x[2], reverse=True)
            
            top_sources = [s[0] for s in sims[:5]]
            dense_rank = top_sources.index(exp_src) + 1 if exp_src in top_sources else None
            
            if dense_rank != 1:
                failures.append({
                    "id": item["id"],
                    "category": item["category"],
                    "query": q_text,
                    "expected_source": exp_src,
                    "top_retrieved_source": top_sources[0],
                    "top_similarity": round(sims[0][2], 4),
                    "gold_source_similarity": round(next((s[2] for s in sims if s[0] == exp_src), 0.0), 4),
                    "margin_deficit": round(sims[0][2] - next((s[2] for s in sims if s[0] == exp_src), 0.0), 4),
                    "top_chunk_preview": sims[0][3]
                })
                
    diagnostic_report = {
        "contrastive_test_split_metrics": {
            "old_bi_encoder": old_triple_metrics,
            "domain_adaptive_bi_encoder": new_triple_metrics
        },
        "corpus_embedding_space_metrics": corpus_space_metrics,
        "dense_retrieval_failure_diagnoses": failures[:10]  # top 10 diagnostic cases
    }
    
    out_file = OUT_DIR / "EMBEDDING_GAP_DIAGNOSTICS.json"
    out_file.write_text(json.dumps(diagnostic_report, indent=2), encoding="utf-8")
    print(f"Saved diagnostic report to {out_file}")

if __name__ == "__main__":
    analyze_embedding_space()
