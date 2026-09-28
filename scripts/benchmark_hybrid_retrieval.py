"""
Empirical Benchmark: Step 13 Hybrid Memory & Semantic Retrieval Foundation.

Measures:
1. Embedding Generation Latency (single text and batch vectorization)
2. Vector Insertion Latency across corpus sizes (10, 100, 1,000 documents)
3. Vector Retrieval Latency across corpus sizes (10, 100, 1,000 documents)
4. BM25 Lexical Retrieval Latency across corpus sizes (10, 100, 1,000 documents)
5. Hybrid Retrieval Latency (BM25 + Semantic fusion) across corpus sizes
6. Working Memory Retrieval Latency across memory pool sizes (10, 50, 100 items)
7. Context Assembly Latency with unified candidates
8. End-to-End Inference Latency comparison:
   - Mode 1: Base Generation (no retrieval)
   - Mode 2: BM25 Lexical RAG
   - Mode 3: Semantic Vector RAG
   - Mode 4: Hybrid RAG
   - Mode 5: Unified Memory + Knowledge RAG

Outputs structured results to docs/STEP_13_BENCHMARK_RESULTS.json.
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
from chakrview.runtime.inference import InferenceSession, GenerationConfig
from chakrview.runtime.knowledge import (
    DocumentIngester,
    BM25KnowledgeIndex,
    LexicalRetriever,
    KnowledgeChunk,
    compute_sha256_text,
)
from chakrview.runtime.memory import (
    WorkingMemory,
    MemoryItem,
    MemoryType,
)
from chakrview.runtime.retrieval import (
    DeterministicHashEmbeddingProvider,
    InMemoryVectorIndex,
    HybridRetriever,
    UnifiedRetriever,
    RetrievalQuery,
)
from chakrview.runtime.sampling import SamplingConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "stage_c_full_epoch" / "checkpoint_0006478.pt"
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
BENCHMARK_OUTPUT_PATH = ROOT_DIR / "docs" / "STEP_13_BENCHMARK_RESULTS.json"

SAMPLE_TEXTS = [
    "ChakrMicro is an indigenous neural language model with 3443136 parameters.",
    "The architecture uses 6 transformer layers with d_model 192 and 6 attention heads.",
    "Rotary Position Embeddings are applied with base theta 10000 across 32 head dimensions.",
    "SwiGLU feed-forward networks provide non-linear feature transformation with hidden dimension 512.",
    "Pre-RMSNorm stabilizes gradient dynamics across deep transformer layers with epsilon 1e-5.",
    "Byte-Level Byte-Pair Encoding operates over a vocabulary of exactly 4096 tokens.",
    "Context budgeting guarantees that prompt and generation tokens never exceed the 512 ceiling.",
    "Working memory tracks active conversational constraints, user preferences, and transient entities.",
    "Okapi BM25 provides lexical scoring with term-frequency saturation and document-length penalty.",
    "Deterministic hash projections enable lightweight embedding generation without external ML dependencies.",
]


def generate_synthetic_corpus(num_docs: int) -> List[KnowledgeChunk]:
    """Generate deterministic synthetic knowledge chunks for scaling benchmarks."""
    chunks = []
    topics = [
        "transformer attention mechanism",
        "gradient descent optimization",
        "kv cache decoding latency",
        "rotary positional embedding",
        "working memory persistence",
        "bm25 lexical term saturation",
        "hash embedding projection",
        "context budget enforcement",
        "governed tool execution",
        "sovereign AI infrastructure",
    ]
    for i in range(num_docs):
        topic = topics[i % len(topics)]
        text = (
            f"Document {i:04d} discusses {topic} in depth. "
            f"It covers operational principles, performance trade-offs, and empirical findings. "
            f"Key metrics include latency, throughput, memory overhead, and parameter count {3443136 + i}."
        )
        c = KnowledgeChunk(
            chunk_id=f"doc_{i:04d}_c00",
            doc_id=f"doc_{i:04d}",
            chunk_index=0,
            text=text,
            token_count=max(1, int(len(text.split()) * 1.3)),
            metadata={"topic": topic, "index": i},
            chunk_hash=compute_sha256_text(text),
        )
        chunks.append(c)
    return chunks


def benchmark_embeddings(provider: DeterministicHashEmbeddingProvider, iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark embedding generation latency."""
    print("  -> Benchmarking Embedding Provider...")
    # Single text embedding latency
    t0 = time.perf_counter()
    for _ in range(iterations):
        for text in SAMPLE_TEXTS[:5]:
            provider.embed_text(text)
    total_time = time.perf_counter() - t0
    single_latency_us = (total_time / (iterations * 5)) * 1_000_000.0

    # Batch embedding latency (50 items)
    batch_texts = SAMPLE_TEXTS * 5
    t0 = time.perf_counter()
    for _ in range(iterations // 10):
        provider.embed_many(batch_texts)
    batch_time = time.perf_counter() - t0
    batch_latency_ms = (batch_time / (iterations // 10)) * 1000.0

    return {
        "dimension": provider.dimension,
        "provider_id": provider.provider_id,
        "single_text_latency_us": round(single_latency_us, 2),
        "batch_50_latency_ms": round(batch_latency_ms, 3),
        "throughput_embeddings_per_sec": int(1_000_000.0 / single_latency_us),
    }


def benchmark_corpus_scaling(
    provider: DeterministicHashEmbeddingProvider,
    corpus_sizes: List[int],
    query_text: str = "rotary positional embedding latency",
    trials: int = 100,
) -> Dict[str, Any]:
    """Benchmark vector insertion, vector retrieval, BM25 retrieval, and hybrid retrieval across corpus sizes."""
    print("  -> Benchmarking Retrieval Across Corpus Sizes...")
    scaling_results = {}

    for size in corpus_sizes:
        print(f"     Testing corpus size: {size} documents...")
        chunks = generate_synthetic_corpus(size)

        # 1. Vector Index Insertion
        vec_index = InMemoryVectorIndex(index_id=f"idx_{size}")
        t0 = time.perf_counter()
        for c in chunks:
            v = provider.embed_text(c.text)
            vec_index.add_vector(c.chunk_id, v, text=c.text, metadata=c.metadata)
        insert_time = time.perf_counter() - t0
        vector_insert_ms = insert_time * 1000.0
        insert_per_doc_us = (insert_time / size) * 1_000_000.0

        # 2. BM25 Index Insertion
        bm25_index = BM25KnowledgeIndex()
        t0 = time.perf_counter()
        bm25_index.add_chunks(chunks)
        bm25_insert_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Vector Retrieval Latency
        q_vec = provider.embed_text(query_text)
        t0 = time.perf_counter()
        for _ in range(trials):
            vec_index.search(q_vec, top_k=5)
        vec_search_us = ((time.perf_counter() - t0) / trials) * 1_000_000.0

        # 4. BM25 Retrieval Latency
        t0 = time.perf_counter()
        for _ in range(trials):
            bm25_index.search(query_text, top_k=5)
        bm25_search_us = ((time.perf_counter() - t0) / trials) * 1_000_000.0

        # 5. Hybrid Retrieval Latency (End-to-End)
        hybrid = HybridRetriever(
            lexical_index=bm25_index,
            embedding_provider=provider,
            vector_index=vec_index,
        )
        q = RetrievalQuery(text=query_text, top_k=5)
        t0 = time.perf_counter()
        for _ in range(trials):
            hybrid.retrieve_candidates(q)
        hybrid_search_us = ((time.perf_counter() - t0) / trials) * 1_000_000.0

        scaling_results[str(size)] = {
            "num_documents": size,
            "vector_insertion_total_ms": round(vector_insert_ms, 2),
            "vector_insertion_per_doc_us": round(insert_per_doc_us, 2),
            "bm25_insertion_total_ms": round(bm25_insert_ms, 2),
            "vector_search_latency_us": round(vec_search_us, 2),
            "bm25_search_latency_us": round(bm25_search_us, 2),
            "hybrid_search_latency_us": round(hybrid_search_us, 2),
            "hybrid_search_latency_ms": round(hybrid_search_us / 1000.0, 3),
        }

    return scaling_results


def benchmark_memory_retrieval(pool_sizes: List[int], trials: int = 200) -> Dict[str, Any]:
    """Benchmark working memory heuristic retrieval across memory pool depths."""
    print("  -> Benchmarking Working Memory Retrieval...")
    results = {}
    for size in pool_sizes:
        mem = WorkingMemory()
        for i in range(size):
            m_type = MemoryType.FACT if i % 2 == 0 else MemoryType.PREFERENCE
            mem.add(MemoryItem(
                memory_id=f"mem_{i:03d}",
                content=f"User stated preference {i} about code conciseness and execution style {i}.",
                memory_type=m_type,
                importance=0.5 + (i % 5) * 0.1,
            ))

        t0 = time.perf_counter()
        for _ in range(trials):
            mem.retrieve_relevant(query="code conciseness", top_k=3)
        avg_us = ((time.perf_counter() - t0) / trials) * 1_000_000.0

        results[str(size)] = {
            "pool_size": size,
            "retrieval_latency_us": round(avg_us, 2),
        }
    return results


def benchmark_context_assembly(trials: int = 500) -> Dict[str, Any]:
    """Benchmark context assembly latency with unified retrieval candidates."""
    print("  -> Benchmarking Context Assembly Latency...")
    builder = PromptContextBuilder()
    chunks = [
        KnowledgeChunk("c1", "doc1", 0, "ChakrMicro uses causal multi-head attention.", 6),
        KnowledgeChunk("c2", "doc1", 1, "Context budgeting guarantees strict 512 ceiling.", 7),
    ]
    retrieved = [(c, 0.85) for c in chunks]
    memories = [
        MemoryItem("m1", "User prefers concise python responses.", MemoryType.PREFERENCE, 0.9),
    ]

    t0 = time.perf_counter()
    for _ in range(trials):
        builder.build_prompt(
            user_query="How does ChakrMicro enforce context limits?",
            system_prompt="You are ChakrView indigenous cognitive system.",
            retrieved_chunks=retrieved,
            working_memories=memories,
        )
    assembly_us = ((time.perf_counter() - t0) / trials) * 1_000_000.0

    return {
        "assembly_latency_us": round(assembly_us, 2),
        "assembly_latency_ms": round(assembly_us / 1000.0, 4),
    }


def benchmark_end_to_end_inference(
    session: InferenceSession,
    provider: DeterministicHashEmbeddingProvider,
) -> Dict[str, Any]:
    """Compare latency across generation modes (baseline, BM25, semantic, hybrid, unified)."""
    print("  -> Benchmarking End-to-End Generation Across Modes...")
    cfg = GenerationConfig(max_new_tokens=16, sampling=SamplingConfig(temperature=0.0))
    query = "What is the parameter count of ChakrMicro?"

    # Setup Knowledge & Indices
    chunks = generate_synthetic_corpus(50)
    bm25_index = BM25KnowledgeIndex()
    bm25_index.add_chunks(chunks)

    vec_index = InMemoryVectorIndex()
    for c in chunks:
        vec_index.add_vector(c.chunk_id, provider.embed_text(c.text), text=c.text, metadata=c.metadata)

    hybrid = HybridRetriever(
        lexical_index=bm25_index,
        embedding_provider=provider,
        vector_index=vec_index,
    )

    mem = WorkingMemory()
    mem.add(MemoryItem("m1", "ChakrMicro parameters must equal 3,443,136.", MemoryType.FACT, 0.95))

    unified = UnifiedRetriever(
        knowledge_retriever=hybrid,
        working_memory=mem,
    )

    modes = {}

    # Mode 1: Base Generation (no retrieval)
    t0 = time.perf_counter()
    res1 = session.ask(query=query, config=cfg)
    m1_ms = (time.perf_counter() - t0) * 1000.0
    modes["mode_1_base_generation"] = {
        "description": "Direct ChakrMicro inference (no RAG/retrieval)",
        "latency_ms": round(m1_ms, 2),
        "tokens_generated": len(res1.token_ids),
        "knowledge_used": res1.knowledge_used,
    }

    # Mode 2: BM25 Lexical RAG
    t0 = time.perf_counter()
    res2 = session.ask(query=query, knowledge=bm25_index, config=cfg)
    m2_ms = (time.perf_counter() - t0) * 1000.0
    modes["mode_2_bm25_lexical_rag"] = {
        "description": "Inference with BM25 lexical retrieval",
        "latency_ms": round(m2_ms, 2),
        "tokens_generated": len(res2.token_ids),
        "knowledge_used": res2.knowledge_used,
    }

    # Mode 3: Semantic Vector RAG
    # Hybrid with semantic_weight=1.0, lexical_weight=0.0
    semantic_retriever = HybridRetriever(
        lexical_index=bm25_index,
        embedding_provider=provider,
        vector_index=vec_index,
        lexical_weight=0.0,
        semantic_weight=1.0,
    )
    t0 = time.perf_counter()
    res3 = session.ask(query=query, hybrid_retriever=semantic_retriever, config=cfg)
    m3_ms = (time.perf_counter() - t0) * 1000.0
    modes["mode_3_semantic_vector_rag"] = {
        "description": "Inference with semantic vector retrieval (hash reference)",
        "latency_ms": round(m3_ms, 2),
        "tokens_generated": len(res3.token_ids),
        "knowledge_used": res3.knowledge_used,
    }

    # Mode 4: Hybrid RAG
    t0 = time.perf_counter()
    res4 = session.ask(query=query, hybrid_retriever=hybrid, config=cfg)
    m4_ms = (time.perf_counter() - t0) * 1000.0
    modes["mode_4_hybrid_rag"] = {
        "description": "Inference with normalized hybrid score fusion (BM25 + Semantic)",
        "latency_ms": round(m4_ms, 2),
        "tokens_generated": len(res4.token_ids),
        "knowledge_used": res4.knowledge_used,
    }

    # Mode 5: Unified Memory + Knowledge RAG (Chat turn)
    t0 = time.perf_counter()
    res5 = session.chat(
        session_id="bench_sess_unified",
        user_text=query,
        config=cfg,
        unified_retriever=unified,
    )
    m5_ms = (time.perf_counter() - t0) * 1000.0
    modes["mode_5_unified_memory_knowledge"] = {
        "description": "Multi-turn chat with unified working memory + hybrid knowledge",
        "latency_ms": round(m5_ms, 2),
        "tokens_generated": len(res5.token_ids),
        "memories_used": len(res5.working_memories_used),
        "turns_in_context": res5.turns_in_context,
    }

    return modes


def main() -> None:
    print("=" * 70)
    print("  CHAKRVIEW STEP 13: HYBRID RETRIEVAL BENCHMARK SUITE")
    print("=" * 70)

    # 1. Initialize tokenizer and session
    print("Loading Tokenizer and Model...")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = ChakrMicro(ModelConfig())
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {CHECKPOINT_PATH.name}")
    else:
        print("Using initialized model.")
    model.eval()

    session = InferenceSession(model=model, tokenizer=tokenizer)
    provider = DeterministicHashEmbeddingProvider(dimension=64)

    # 2. Run Benchmarks
    embedding_metrics = benchmark_embeddings(provider)
    corpus_scaling_metrics = benchmark_corpus_scaling(provider, corpus_sizes=[10, 100, 1000])
    memory_metrics = benchmark_memory_retrieval(pool_sizes=[10, 50, 100])
    context_assembly_metrics = benchmark_context_assembly()
    e2e_modes_metrics = benchmark_end_to_end_inference(session, provider)

    # 3. Compile report
    report = {
        "step": "Step 13 - Hybrid Memory & Semantic Retrieval Foundation",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "max_context_length": model.config.max_seq_len,
        "embedding_metrics": embedding_metrics,
        "corpus_scaling_metrics": corpus_scaling_metrics,
        "memory_retrieval_metrics": memory_metrics,
        "context_assembly_metrics": context_assembly_metrics,
        "end_to_end_modes_latency": e2e_modes_metrics,
    }

    BENCHMARK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Benchmark completed successfully!")
    print(f"Results saved to: {BENCHMARK_OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
