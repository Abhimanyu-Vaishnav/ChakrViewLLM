"""
Step 45 Evaluation Harness: Neural Generation & Engine Reliability Benchmark.

Runs a standardized, deterministic test set across 6 prompt categories to evaluate
engine behavior, boundary enforcement, numerical safety, and decoding determinism:
1. Factual Continuation
2. Open Continuation
3. Structured Reasoning Substrate
4. Multilingual (Hindi-English / Devanagari Byte-Level BPE)
5. Boundary Cases (Near-Context Ceiling)
6. Security & Negative Edge Cases

CRITICAL NOTICE:
This evaluation harness strictly measures ENGINE RELIABILITY, REPRODUCIBILITY,
and EXECUTION CONTRACT CONFORMANCE. It does NOT claim or measure linguistic
intelligence, factual accuracy, or production reasoning capability.
"""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.inference import GenerationConfig, StopReason
from chakrview.runtime.pipeline import (
    InferenceEngine,
    InferenceRequest,
    ModelIdentity,
)
from chakrview.runtime.sampling import SamplingConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


EVALUATION_PROMPTS = [
    {
        "id": "factual_01",
        "category": "factual_continuation",
        "prompt": "ChakrMicro is an indigenous decoder-only transformer with 3,443,136 parameters and 6 layers.",
        "max_new_tokens": 16,
        "temperature": 0.0,
    },
    {
        "id": "open_01",
        "category": "open_continuation",
        "prompt": "In distributed sovereign artificial intelligence systems, local policy superiority ensures that",
        "max_new_tokens": 20,
        "temperature": 0.0,
    },
    {
        "id": "reasoning_01",
        "category": "simple_reasoning",
        "prompt": "Rule 1: All messages require Ed25519 signatures. Rule 2: Message M has no signature. Conclusion:",
        "max_new_tokens": 16,
        "temperature": 0.0,
    },
    {
        "id": "multilingual_01",
        "category": "multilingual_hindi_english",
        "prompt": "ChakrView ek bharatiya swadeshi AI architecture hai jisme neural core",
        "max_new_tokens": 16,
        "temperature": 0.0,
    },
    {
        "id": "boundary_01",
        "category": "context_boundary_long_prompt",
        "prompt": "Architectural boundary test repeating tokens to reach significant sequence length. " * 15,
        "max_new_tokens": 12,
        "temperature": 0.0,
    },
    {
        "id": "stochastic_seed_01",
        "category": "stochastic_seeded_reproducibility",
        "prompt": "Exploration of novel hypotheses under stochastic temperature sampling",
        "max_new_tokens": 16,
        "temperature": 0.7,
        "seed": 4242,
    },
]


def run_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("ChakrView Step 45: Neural Generation & Engine Reliability Evaluation Harness")
    print("=" * 80)
    print("[NOTICE] Evaluating inference engine behavior and contract conformance.")
    print("=" * 80)

    root_dir = Path(__file__).resolve().parent.parent
    tok_dir = root_dir / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()

    engine = InferenceEngine(model=model, tokenizer=tokenizer)

    results: List[Dict[str, Any]] = []

    for test_case in EVALUATION_PROMPTS:
        t_id = test_case["id"]
        cat = test_case["category"]
        prompt = test_case["prompt"]
        max_new = test_case["max_new_tokens"]
        temp = test_case["temperature"]
        seed = test_case.get("seed")

        sampling_cfg = SamplingConfig(
            temperature=temp,
            seed=seed,
            top_k=50 if temp > 0.0 else 0,
            repetition_penalty=1.1,
        )
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new,
            sampling=sampling_cfg,
        )
        req = InferenceRequest(
            prompt=prompt,
            generation_config=gen_cfg,
            truncate_if_overflow=True,
        )

        res = engine.execute(req)

        record = {
            "test_id": t_id,
            "category": cat,
            "prompt_preview": prompt[:60] + ("..." if len(prompt) > 60 else ""),
            "prompt_token_count": res.input_token_count,
            "generated_token_count": res.output_token_count,
            "total_token_count": res.total_token_count,
            "stop_reason": res.stop_reason.value,
            "latency_ms": round(res.latency_ms, 2),
            "generated_text_preview": res.text[:60] + ("..." if len(res.text) > 60 else ""),
            "generated_token_ids": res.token_ids[:8],
            "weight_hash_verified": res.weight_hash_verified,
            "reproducibility": res.reproducibility,
        }
        results.append(record)

        print(
            f"  [{t_id:<20}] "
            f"category={cat:<25} "
            f"in={res.input_token_count:3d} "
            f"out={res.output_token_count:3d} "
            f"lat={res.latency_ms:6.2f}ms "
            f"stop={res.stop_reason.value:<10} "
            f"dW={res.weight_hash_verified}"
        )

    summary = {
        "evaluation_suite": "Step 45: Neural Generation & Engine Reliability Harness",
        "timestamp": time.time(),
        "model_identity": ModelIdentity().to_dict(),
        "total_test_cases": len(results),
        "all_weight_hashes_verified": all(r["weight_hash_verified"] for r in results),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 2),
        "test_results": results,
    }

    out_file = root_dir / "docs" / "STEP_45_EVALUATION_REPORT.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(f"Evaluation report written to: {out_file}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    run_evaluation()
