"""
Empirical Benchmark: Step 11 RAG & Domain Skill Intelligence Layer.

Measures:
1. Knowledge Ingestion Latency (document parsing, chunking, SHA-256 hashing, index insertion).
2. Lexical / BM25 Retrieval Latency (term extraction, scoring, top-k ranking).
3. Context Assembly Latency (budget allocation, boundary formatting, provenance tracking).
4. End-to-End Generation Comparison (Plain Generation vs Grounded RAG Generation).
5. Output structured JSON to docs/STEP_11_BENCHMARK_RESULTS.json.
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.context import PromptContextBuilder, ContextBudget
from chakrview.runtime.hardware import RuntimePlanner, HardwareCapabilityDetector
from chakrview.runtime.inference import InferenceSession, GenerationConfig
from chakrview.runtime.knowledge import (
    DocumentChunker,
    DocumentIngester,
    BM25KnowledgeIndex,
    LexicalRetriever,
)
from chakrview.runtime.sampling import SamplingConfig
from chakrview.runtime.skills import get_standard_skill_registry
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "stage_c_full_epoch" / "checkpoint_0006478.pt"
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
BENCHMARK_OUTPUT_PATH = ROOT_DIR / "docs" / "STEP_11_BENCHMARK_RESULTS.json"

BENCHMARK_DOCUMENTS = [
    {
        "doc_id": "doc_arch_micro",
        "title": "ChakrMicro Neural Architecture Specification",
        "content": (
            "ChakrMicro is the foundational decoder-only causal language model of ChakrView. "
            "It features exactly 3443136 parameters, 6 transformer layers, d_model 192, 6 query heads, "
            "d_head 32, d_ff 512, and a maximum context window of 512 tokens. Normalization is performed "
            "using Pre-RMSNorm with epsilon 1e-5. Positional encoding utilizes Rotary Position Embeddings (RoPE) "
            "with theta 10000. All linear projections are strictly bias-free, and token embeddings are tied to the lm_head."
        ),
    },
    {
        "doc_id": "doc_bpe_tokenizer",
        "title": "Byte-Level BPE Tokenizer Engineering",
        "content": (
            "The ChakrView tokenizer operates on raw 8-bit bytes with an exact frozen vocabulary of 4096 tokens. "
            "Special tokens are allocated at the base: BOS is 0, EOS is 1, and PAD is 2. The 256 fundamental byte values "
            "occupy token IDs 3 through 258. Exactly 3837 merges were trained deterministically using frequency tie-breaking. "
            "All Unicode sequences round-trip losslessly with zero replacement errors."
        ),
    },
    {
        "doc_id": "doc_pretraining_stage_c",
        "title": "Stage C Real-Corpus Pre-Training Report",
        "content": (
            "The Stage C pre-training corpus comprises 5534 vetted documents containing 7708601 total sharded tokens "
            "distributed across 32 binary uint16 shards. Training ran for 6478 optimizer steps completing a full epoch. "
            "The optimizer was AdamW with peak learning rate 1e-3, cosine decay to 1e-4, weight decay 0.1, and gradient clipping 1.0. "
            "Validation loss converged to 4.8062 and test loss reached 4.7828."
        ),
    },
    {
        "doc_id": "doc_kv_cache_engine",
        "title": "Persistent KV-Cache & Incremental Decoding",
        "content": (
            "Step 10 introduced an O(N) autoregressive decoding path with persistent per-layer key and value tensor caches. "
            "Rotating keys at sequence offset P eliminates redundant computation, reducing single-token generation latency "
            "from 19.5 ms to 2.8 ms per token, achieving up to 6.90x speedup over full sequence recomputation. "
            "Memory footprint remains bounded under 4.61 MB at full 512 context capacity."
        ),
    },
    {
        "doc_id": "doc_runtime_layers",
        "title": "ChakrView Modular Brain Layer Architecture",
        "content": (
            "The long-term vision separates the universal neural core from runtime orchestration, external knowledge RAG, "
            "domain skill policies, governed tools, controlled self-improvement proposals, software integrity rollback, "
            "and hardware execution planning. Instances are formed by composing Brain plus Skill plus Knowledge plus Tools."
        ),
    },
]

BENCHMARK_QUERIES = [
    "What are the parameters, layers, and context length of ChakrMicro?",
    "How does the Byte-Level BPE tokenizer handle byte values and merges?",
    "What was the training configuration and final loss for Stage C full epoch?",
    "Explain the speedup and memory usage of the persistent KV cache engine.",
    "Describe the layered architecture separating the brain core from skills and tools.",
]


def run_benchmark() -> Dict[str, Any]:
    print("=" * 72)
    print("CHAKRVIEW STEP 11 -- RAG & DOMAIN SKILL BENCHMARK")
    print("=" * 72)

    # 1. Hardware & Environment
    detector = HardwareCapabilityDetector()
    hw_profile = detector.detect()
    plan = RuntimePlanner.plan(hw_profile, requested_context_len=512)
    model_cfg = ModelConfig()
    torch.set_num_threads(plan.thread_count)

    ram_avail_mb = hw_profile.available_ram_bytes / (1024 * 1024)
    print(f"Device: {plan.device} | Threads: {plan.thread_count} | RAM: {ram_avail_mb:.1f} MB")

    # 2. Load Tokenizer & Model
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = ChakrMicro(model_cfg)

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded trained checkpoint: {CHECKPOINT_PATH.name}")
    else:
        print("Using freshly initialized ChakrMicro model.")
    model.eval()

    session = InferenceSession(model=model, tokenizer=tokenizer, execution_plan=plan)

    # 3. Benchmark Ingestion Pipeline
    print("\n--- 1. Ingestion Benchmark ---")
    chunker = DocumentChunker(chunk_size=40, chunk_overlap=10)
    ingester = DocumentIngester(chunker=chunker)
    index = BM25KnowledgeIndex("benchmark_index")

    t_ingest_start = time.perf_counter()
    all_chunks = []
    doc_ingest_records = []

    for item in BENCHMARK_DOCUMENTS:
        t0 = time.perf_counter()
        doc, chunks = ingester.ingest_text(
            text=item["content"],
            title=item["title"],
            doc_id=item["doc_id"],
        )
        t_doc = (time.perf_counter() - t0) * 1000.0
        all_chunks.extend(chunks)
        doc_ingest_records.append({
            "doc_id": doc.doc_id,
            "chunks_count": len(chunks),
            "content_bytes": len(doc.content.encode("utf-8")),
            "ingest_latency_ms": round(t_doc, 3),
        })

    t_index_start = time.perf_counter()
    added_count = index.add_chunks(all_chunks)
    t_index_ms = (time.perf_counter() - t_index_start) * 1000.0
    total_ingest_ms = (time.perf_counter() - t_ingest_start) * 1000.0

    print(f"Ingested {len(BENCHMARK_DOCUMENTS)} documents into {added_count} chunks in {total_ingest_ms:.3f} ms")
    print(f"Index Insertion Latency: {t_index_ms:.3f} ms ({t_index_ms / added_count:.3f} ms/chunk)")

    # 4. Benchmark Retrieval Latency
    print("\n--- 2. Retrieval Benchmark (BM25) ---")
    retriever = LexicalRetriever(index)
    retrieval_records = []

    for top_k in [1, 3, 5]:
        latencies = []
        for q in BENCHMARK_QUERIES:
            t0 = time.perf_counter()
            results = retriever.retrieve(q, top_k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        avg_lat = sum(latencies) / len(latencies)
        retrieval_records.append({
            "top_k": top_k,
            "avg_latency_ms": round(avg_lat, 4),
            "min_latency_ms": round(min(latencies), 4),
            "max_latency_ms": round(max(latencies), 4),
        })
        print(f"Top-K={top_k}: Avg Latency = {avg_lat:.4f} ms (Min: {min(latencies):.4f} ms, Max: {max(latencies):.4f} ms)")

    # 5. Benchmark Context Assembly Latency
    print("\n--- 3. Context Assembly Benchmark ---")
    builder = PromptContextBuilder()
    context_assembly_latencies = []

    for q in BENCHMARK_QUERIES:
        retrieved = retriever.retrieve(q, top_k=3)
        t0 = time.perf_counter()
        ctx = builder.build_prompt(
            user_query=q,
            system_prompt="You are ChakrView, an indigenous intelligent AI assistant.",
            retrieved_chunks=retrieved,
            tokenizer=tokenizer,
        )
        context_assembly_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ctx_ms = sum(context_assembly_latencies) / len(context_assembly_latencies)
    print(f"Avg Context Assembly Latency: {avg_ctx_ms:.4f} ms")

    # 6. Benchmark Plain Generation vs RAG Generation
    print("\n--- 4. End-to-End Generation Comparison (Plain vs RAG) ---")
    print(f"{'Query Index':<12} | {'Mode':<8} | {'Prompt Tok':<10} | {'Gen Tok':<8} | {'Latency (ms)':<12} | {'Throughput':<12} | {'Sources':<8}")
    print("-" * 80)

    gen_cfg = GenerationConfig(
        max_new_tokens=32,
        sampling=SamplingConfig(temperature=0.0),  # Greedy for determinism
    )

    comparison_results = []

    for idx, q in enumerate(BENCHMARK_QUERIES):
        # A) Plain Generation (no retrieval)
        t0 = time.perf_counter()
        plain_res = session.ask(
            query=q,
            knowledge=None,
            config=gen_cfg,
        )
        t_plain = (time.perf_counter() - t0) * 1000.0

        # B) RAG Generation (with BM25 retrieval)
        t0 = time.perf_counter()
        rag_res = session.ask(
            query=q,
            knowledge=index,
            config=gen_cfg,
            top_k=2,
        )
        t_rag = (time.perf_counter() - t0) * 1000.0

        print(
            f"Query {idx+1:<7} | Plain    | {plain_res.prompt_tokens_count:<10} | {len(plain_res.token_ids):<8} | "
            f"{t_plain:<12.2f} | {plain_res.metrics.throughput_tokens_per_sec:<12.1f} | {len(plain_res.sources):<8}"
        )
        print(
            f"Query {idx+1:<7} | RAG      | {rag_res.prompt_tokens_count:<10} | {len(rag_res.token_ids):<8} | "
            f"{t_rag:<12.2f} | {rag_res.metrics.throughput_tokens_per_sec:<12.1f} | {len(rag_res.sources):<8}"
        )
        print("-" * 80)

        comparison_results.append({
            "query_index": idx + 1,
            "query": q,
            "plain": {
                "prompt_tokens": plain_res.prompt_tokens_count,
                "generated_tokens": len(plain_res.token_ids),
                "total_latency_ms": round(t_plain, 2),
                "throughput_tokens_per_sec": round(plain_res.metrics.throughput_tokens_per_sec, 1),
                "sources_count": len(plain_res.sources),
            },
            "rag": {
                "prompt_tokens": rag_res.prompt_tokens_count,
                "generated_tokens": len(rag_res.token_ids),
                "total_latency_ms": round(t_rag, 2),
                "throughput_tokens_per_sec": round(rag_res.metrics.throughput_tokens_per_sec, 1),
                "sources_count": len(rag_res.sources),
                "sources": [s["doc_id"] for s in rag_res.sources],
                "skill_domain": rag_res.skill_domain,
            },
        })

    # Summary Record
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "device": plan.device,
            "thread_count": plan.thread_count,
            "ram_available_mb": round(hw_profile.available_ram_bytes / (1024 * 1024), 1),
        },
        "ingestion": {
            "total_documents": len(BENCHMARK_DOCUMENTS),
            "total_chunks": added_count,
            "total_ingest_time_ms": round(total_ingest_ms, 3),
            "avg_ingest_per_doc_ms": round(total_ingest_ms / len(BENCHMARK_DOCUMENTS), 3),
            "index_insert_time_ms": round(t_index_ms, 3),
            "documents": doc_ingest_records,
        },
        "retrieval": retrieval_records,
        "context_assembly": {
            "avg_latency_ms": round(avg_ctx_ms, 4),
            "min_latency_ms": round(min(context_assembly_latencies), 4),
            "max_latency_ms": round(max(context_assembly_latencies), 4),
        },
        "comparisons": comparison_results,
    }

    with open(BENCHMARK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBenchmark results saved to: {BENCHMARK_OUTPUT_PATH}")
    return summary


if __name__ == "__main__":
    run_benchmark()
