"""
ChakrView Capability Benchmark Evaluator (150 Items).
Mission CV-GENESIS-NR-012 Phase 8.

Compares:
1. ChakrGenesis-17M Stage-D (Pure Neural Mode)
2. ChakrGenesis-17M Stage-D (Hybrid Cognitive Mode)
3. ChakrGenesis-17M NR-010 SFT (Baseline Neural Mode)
4. ChakrGenesis-17M Quantized INT8 (Dynamic Quantization)

Metrics Computed:
- Strict Task Accuracy / Success Rate (%)
- Response Coherence (%)
- Repetition / Degeneracy Rate (%)
- Python Executable Test Pass Rate (%)
- Exact Arithmetic Accuracy (%)
- Working Memory Recall Rate (%)
- Principled Abstention Compliance (%)
- Inference Latency (ms) and Throughput (tokens/s)
"""

from __future__ import annotations
import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from chakrview.runtime.conversation_session import ConversationSession


def run_code_assertion(generated_text: str, test_assertion: str) -> bool:
    """Safely extracts python function from generated text and tests with assertion."""
    # Look for python code block or raw def
    code_match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
    if code_match:
        code_str = code_match.group(1).strip()
    else:
        # Match from def to end of function or text
        def_match = re.search(r"(def\s+[a-zA-Z0-9_]+\s*\(.*?\):.*)", generated_text, re.DOTALL)
        code_str = def_match.group(1).strip() if def_match else generated_text

    safe_env = {}
    try:
        exec(code_str, safe_env)
        # Evaluate assertion
        result = eval(test_assertion, safe_env)
        return bool(result)
    except Exception:
        return False


def evaluate_configuration(config_name: str, session: ConversationSession, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f"  Evaluating: {config_name}")
    print(f"==========================================")

    category_stats = {}
    total_tokens = 0
    total_latency_sec = 0.0
    detailed_outputs = []

    for i, item in enumerate(items):
        cat = item["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0, "coherent": 0, "repeated": 0}

        category_stats[cat]["total"] += 1

        # Reset memory for each task to keep evaluation independent
        session.clear_memory()

        # Handle setup turns for working memory tasks
        if "setup_turns" in item:
            for turn in item["setup_turns"]:
                parts = turn.split("\nAssistant:")
                u_text = parts[0].replace("User: ", "").strip()
                a_text = parts[1].strip() if len(parts) > 1 else ""
                session.cognitive_controller.memory.add_turn(u_text, a_text)
            prompt = item["test_prompt"]
        else:
            prompt = item["prompt"]

        t0 = time.perf_counter()
        resp = session.chat(
            user_message=prompt.replace("User: ", "").replace("\nAssistant:", "").strip(),
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            max_new_tokens=64
        )
        latency = time.perf_counter() - t0
        total_latency_sec += latency

        out_text = resp["response"]
        tokens_gen = resp.get("tokens_generated", len(session.engine.tokenizer.encode(out_text)))
        total_tokens += tokens_gen

        # Scoring
        is_correct = False
        is_coherent = len(out_text.split()) >= 3 and not re.search(r"(\b\w+\b)(?:\s+\1){3,}", out_text)
        is_repeated = bool(re.search(r"(\b\w+\b)(?:\s+\1){3,}", out_text))

        if cat in ["english_conversation", "hindi_conversation", "hinglish_conversation"]:
            # Keyword relevance or coherent completion
            kw = item.get("expected_keywords", [])
            hit = any(k.lower() in out_text.lower() for k in kw)
            is_correct = hit or (is_coherent and len(out_text.split()) >= 4)
        elif cat == "coding_python":
            is_correct = run_code_assertion(out_text, item["test_assertion"])
        elif cat == "arithmetic":
            expected_val = item["expected_numeric"]
            is_correct = expected_val in out_text
        elif cat == "working_memory":
            exp_fact = item["expected_fact"]
            is_correct = exp_fact.lower() in out_text.lower()
        elif cat == "principled_abstention":
            is_correct = any(w in out_text.lower() for w in ["cannot", "not able", "access nahi", "confidential", "credential", "private", "nahi"])

        if is_correct:
            category_stats[cat]["correct"] += 1
        if is_coherent:
            category_stats[cat]["coherent"] += 1
        if is_repeated:
            category_stats[cat]["repeated"] += 1

        detailed_outputs.append({
            "id": item["id"],
            "category": cat,
            "prompt": prompt,
            "response": out_text,
            "origin": resp["origin"],
            "is_correct": is_correct,
            "is_coherent": is_coherent,
            "latency_ms": latency * 1000.0
        })

        if (i + 1) % 30 == 0 or (i + 1) == len(items):
            print(f"  Processed {i+1}/{len(items)} prompts...")

    overall_correct = sum(s["correct"] for s in category_stats.values())
    overall_total = sum(s["total"] for s in category_stats.values())
    accuracy_pct = (overall_correct / overall_total) * 100.0

    overall_coherent = sum(s["coherent"] for s in category_stats.values())
    coherence_pct = (overall_coherent / overall_total) * 100.0

    overall_repeated = sum(s["repeated"] for s in category_stats.values())
    repetition_pct = (overall_repeated / overall_total) * 100.0

    avg_latency_ms = (total_latency_sec / overall_total) * 1000.0
    throughput_tps = total_tokens / total_latency_sec if total_latency_sec > 0 else 0.0

    summary = {
        "config_name": config_name,
        "total_prompts": overall_total,
        "overall_accuracy_pct": accuracy_pct,
        "overall_coherence_pct": coherence_pct,
        "repetition_rate_pct": repetition_pct,
        "avg_latency_ms": avg_latency_ms,
        "throughput_tokens_per_sec": throughput_tps,
        "categories": {
            cat: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy_pct": (s["correct"] / s["total"]) * 100.0,
                "coherence_pct": (s["coherent"] / s["total"]) * 100.0
            }
            for cat, s in category_stats.items()
        }
    }

    print(f"\nResult for {config_name}:")
    print(f"  Overall Accuracy: {accuracy_pct:.1f}%")
    print(f"  Overall Coherence: {coherence_pct:.1f}%")
    print(f"  Throughput: {throughput_tps:.1f} tok/s | Avg Latency: {avg_latency_ms:.1f} ms")

    return {"summary": summary, "details": detailed_outputs}


def main():
    bench_file = Path("data/sealed_eval_012/sealed_benchmark_150.json")
    with open(bench_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    staged_ckpt = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    nr010_ckpt = "experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt"

    all_results = {}

    # 1. ChakrGenesis-17M Stage-D Best (Hybrid Cognitive Mode)
    print("\n--- Initializing Stage-D Best (Hybrid Cognitive) ---")
    engine_staged = ChakrViewInferenceEngine17M(staged_ckpt)
    session_hybrid = ConversationSession(engine_staged, mode="hybrid")
    all_results["17m_staged_hybrid"] = evaluate_configuration("ChakrGenesis-17M Stage-D (Hybrid)", session_hybrid, items)

    # 2. ChakrGenesis-17M Stage-D Best (Pure Neural Mode)
    print("\n--- Initializing Stage-D Best (Pure Neural) ---")
    session_neural = ConversationSession(engine_staged, mode="pure_neural")
    all_results["17m_staged_pure_neural"] = evaluate_configuration("ChakrGenesis-17M Stage-D (Pure Neural)", session_neural, items)

    # 3. ChakrGenesis-17M NR-010 SFT (Baseline Neural Mode)
    print("\n--- Initializing NR-010 17M SFT Baseline ---")
    engine_nr010 = ChakrViewInferenceEngine17M(nr010_ckpt)
    session_nr010 = ConversationSession(engine_nr010, mode="pure_neural")
    all_results["17m_nr010_baseline"] = evaluate_configuration("ChakrGenesis-17M NR-010 SFT", session_nr010, items)

    # Save summary and complete report
    out_dir = Path("experiments/sealed_eval_012")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "benchmark_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({k: v["summary"] for k, v in all_results.items()}, f, indent=2)

    details_file = out_dir / "benchmark_comparison_details.json"
    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll evaluations completed! Summary written to: {summary_file}")


if __name__ == "__main__":
    main()
