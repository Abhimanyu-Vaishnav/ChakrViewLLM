"""
evaluation/canonical_baseline_runner.py

Reproducible, deterministic capability-evaluation baseline runner for ChakrMicro
Step 10,000 canonical checkpoint (SHA-256: 96357705...).

Evaluates across 7 formal capability dimensions:
1. Loadability (Parameter shapes, weight count, tied embedding verification)
2. Tokenizer correctness (Roundtrip encode/decode, special tokens contract)
3. Deterministic generation behaviour (Fixed prompt, fixed seed, bit-exact reproducibility)
4. Basic language & instruction behaviour (Native intelligence task rubrics)
5. Coding capability (Isolated executor unit tests & syntax validity)
6. Robustness & edge cases (Empty inputs, long contexts, special tokens, malformed strings)
7. Resource performance (CPU latency per prompt, token throughput, environment details)
"""

import sys
import os
import json
import time
import hashlib
import platform
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    EXPECTED_TOKENIZER_FILES_HASHES,
    DEFAULT_CHECKPOINT_KEY,
    compute_file_sha256,
)
from evaluation.intelligence_suite.run_intelligence_benchmark import run_benchmark
from evaluation.coding_suite.run_coding_evaluation import evaluate_coding_suite


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_canonical_baseline(
    checkpoint_key: str = DEFAULT_CHECKPOINT_KEY,
    output_json_path: Path = None,
) -> Dict[str, Any]:
    if output_json_path is None:
        output_json_path = REPO_ROOT / "evaluation" / "results" / "canonical_step10000_baseline_report.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("CHAKRVIEW CANONICAL CAPABILITY EVALUATION BASELINE")
    print("================================================================================")

    # 1. Verification of Canonical Artifacts
    ckpt_path = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
    vocab_path = REPO_ROOT / "data" / "experiments" / "vocab_4096" / "vocab.json"
    merges_path = REPO_ROOT / "data" / "experiments" / "vocab_4096" / "merges.json"
    model_py_path = REPO_ROOT / "chakrview" / "brain" / "model.py"
    session_py_path = REPO_ROOT / "chakrview" / "runtime" / "session.py"

    expected_hashes = {
        "checkpoint": ("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt", ckpt_path, "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"),
        "vocab": ("data/experiments/vocab_4096/vocab.json", vocab_path, "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"),
        "merges": ("data/experiments/vocab_4096/merges.json", merges_path, "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"),
        "model_py": ("chakrview/brain/model.py", model_py_path, "1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361"),
        "session_py": ("chakrview/runtime/session.py", session_py_path, "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"),
    }

    artifact_integrity = {}
    for name, (rel_path, p, exp_hash) in expected_hashes.items():
        exists = p.is_file()
        actual_hash = compute_sha256(p) if exists else None
        match = (actual_hash == exp_hash)
        artifact_integrity[name] = {
            "relative_path": rel_path,
            "exists": exists,
            "expected_sha256": exp_hash,
            "actual_sha256": actual_hash,
            "verified": match,
            "size_bytes": p.stat().st_size if exists else 0,
        }
        status_str = "OK" if match else "MISMATCH / MISSING"
        print(f"[{status_str}] {name}: {rel_path} (hash: {actual_hash[:16]}...)")

    # 2. Model & Checkpoint Loadability
    print("\n--- Initializing Neural Runtime Adapter ---")
    t0_load = time.perf_counter()
    adapter = NeuralRuntimeAdapter(
        checkpoint_path=ckpt_path,
        max_new_tokens=32,
        decoding="greedy",
    )
    adapter.initialize()
    load_time_sec = time.perf_counter() - t0_load
    print(f"Model initialized in {load_time_sec:.3f} seconds.")

    model = adapter._contract.model
    total_params = sum(p.numel() for p in model.parameters())
    unique_params = len(set(p.data_ptr() for p in model.parameters()))
    is_tied = (model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr())

    loadability_report = {
        "load_time_sec": round(load_time_sec, 3),
        "total_parameters": total_params,
        "tied_embedding_verified": is_tied,
        "model_architecture": {
            "vocab_size": model.config.vocab_size,
            "d_model": model.config.d_model,
            "n_layers": model.config.n_layers,
            "n_heads": model.config.n_heads,
            "hidden_dim": model.config.hidden_dim,
            "max_seq_len": model.config.max_seq_len,
        }
    }

    # 3. Tokenizer Correctness
    print("\n--- Verifying Tokenizer Contracts ---")
    tok = adapter._tokenizer
    sample_text = "ChakrView indigenous micro-intelligence model for offline edge inference."
    token_ids = tok.encode(sample_text)
    decoded_text = tok.decode(token_ids)
    roundtrip_exact = (sample_text == decoded_text)

    # Special token contract check
    special_contract = {
        "bos_id": getattr(tok, "bos_token_id", None) or getattr(tok, "bos_id", 0),
        "eos_id": getattr(tok, "eos_token_id", None) or getattr(tok, "eos_id", 1),
        "pad_id": getattr(tok, "pad_token_id", None) or getattr(tok, "pad_id", 2),
    }

    tokenizer_report = {
        "vocab_size": getattr(tok, "vocab_size", len(getattr(tok, "vocab", {}))),
        "roundtrip_exact": roundtrip_exact,
        "sample_tokens_count": len(token_ids),
        "special_tokens": special_contract,
    }
    print(f"Tokenizer Roundtrip: {'MATCH' if roundtrip_exact else 'MISMATCH'} (Tokens: {len(token_ids)})")

    # 4. Deterministic Smoke Generation Check
    print("\n--- Verifying Deterministic Generation Contract ---")
    fixed_prompt = "Ram lives in"
    out1 = adapter.generate(fixed_prompt)["generated_text"]
    out2 = adapter.generate(fixed_prompt)["generated_text"]
    deterministic = (out1 == out2)
    print(f"Prompt: '{fixed_prompt}' -> Out: '{out1.strip()[:40]}'")
    print(f"Determinism Check (greedy output 1 == output 2): {deterministic}")

    # 5. Robustness & Resource Performance Tests
    print("\n--- Running Robustness & Resource Performance Benchmark ---")
    robustness_prompts = [
        ("empty_string", ""),
        ("single_char", "a"),
        ("whitespace_only", "   "),
        ("ascii_punctuation", "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
        ("unicode_mix", "Hello \ud83d\ude00 world \ufffd"),
        ("long_repeated_context", "the model was trained on tokens. " * 30),
        ("boundary_numeric", "100000000000000000000000000000000"),
        ("factual_simple", "The capital of France is"),
        ("arithmetic_simple", "Compute: 7 + 5 ="),
        ("code_prefix", "def fibonacci(n):"),
    ]

    robustness_results = []
    latencies = []
    tokens_out_counts = []

    for name, p_text in robustness_prompts:
        t_start = time.perf_counter()
        gen_str = ""
        out_tokens = 0
        stop_reason = "NONE"
        status_ok = True
        err_msg = None
        try:
            gen_res = adapter.generate(p_text)
            dt = time.perf_counter() - t_start
            gen_str = gen_res["generated_text"]
            out_tokens = gen_res["output_token_count"]
            stop_reason = str(gen_res.get("stop_reason", "MAX_TOKENS"))
            latencies.append(dt)
            tokens_out_counts.append(out_tokens)
            print(f"  [{name:<22}] {dt*1000:6.1f}ms | {out_tokens:2d} tokens | stop: {stop_reason}")
        except Exception as e:
            dt = time.perf_counter() - t_start
            status_ok = False
            err_msg = f"{type(e).__name__}: {str(e)}"
            stop_reason = "REJECTED_INPUT"
            print(f"  [{name:<22}] {dt*1000:6.1f}ms | REJECTED/EXCEPTION: {err_msg[:40]}")

        robustness_results.append({
            "prompt_id": name,
            "prompt_preview": p_text[:40],
            "output_preview": gen_str[:50].replace("\n", "\\n") if gen_str else f"EXCEPTION: {err_msg}",
            "latency_ms": round(dt * 1000, 2),
            "output_tokens": out_tokens,
            "stop_reason": stop_reason,
            "no_crash": status_ok,
            "error": err_msg,
        })

    total_bench_time = sum(latencies)
    total_tokens_bench = sum(tokens_out_counts)
    throughput = round(total_tokens_bench / total_bench_time, 2) if total_bench_time > 0 else 0.0
    print(f"Average CPU Inference Latency: {sum(latencies)/len(latencies)*1000:.1f}ms | Throughput: {throughput} tok/s")

    # 6. Native Intelligence Benchmark (32 Tasks)
    print("\n--- Executing Native Intelligence Benchmark (32 Tasks) ---")
    raw_intel_out = REPO_ROOT / "evaluation" / "results" / "canonical_step10000_intelligence_baseline_20261028.json"
    intel_benchmark = run_benchmark(
        checkpoint_key=checkpoint_key,
        output_path=raw_intel_out,
        decoding="greedy",
        temperature=0.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        max_new_tokens=32,
    )
    intel_summary = {
        "suite_name": intel_benchmark.get("suite_name"),
        "total_tasks": intel_benchmark.get("total_tasks", 32),
        "passed_tasks_count": intel_benchmark.get("passed_tasks_count", 0),
        "overall_pass_rate": intel_benchmark.get("overall_pass_rate", "0/32 (0.0%)"),
        "attractor_collapse_count": intel_benchmark.get("attractor_collapse_count", 0),
        "attractor_rate": intel_benchmark.get("attractor_rate", "0/32 (0.0%)"),
        "category_metrics": intel_benchmark.get("category_metrics", {}),
    }
    print(f"Intelligence Suite Pass Rate: {intel_summary['overall_pass_rate']}")
    print(f"Attractor Collapse Rate: {intel_summary['attractor_rate']}")

    # 7. Coding Capability Suite (12 Tasks)
    print("\n--- Executing Coding Capability Benchmark (12 Tasks) ---")
    coding_out = REPO_ROOT / "evaluation" / "results" / "canonical_step10000_coding_baseline_20261028.json"
    coding_res = evaluate_coding_suite(
        checkpoint_key=checkpoint_key,
        max_new_tokens=48,
        decoding="greedy",
        output_path=str(coding_out),
    )
    coding_summary = {
        "suite_name": coding_res.get("suite_name"),
        "total_tasks": coding_res.get("total_tasks", 12),
        "syntax_valid_rate": coding_res.get("syntax_valid_rate", "0/12 (0.0%)"),
        "overall_success_rate": coding_res.get("overall_success_rate", "0/12 (0.0%)"),
        "tokens_per_second": coding_res.get("tokens_per_second", 0.0),
        "categories": coding_res.get("categories", {}),
    }
    print(f"Coding Suite Pass Rate: {coding_summary['overall_success_rate']}")
    print(f"Syntax Validity Rate: {coding_summary['syntax_valid_rate']}")

    # Assemble Baseline Package
    report = {
        "benchmark_metadata": {
            "title": "ChakrMicro Canonical Capability Evaluation Baseline",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": "a3951692b24f69ebfeed5b3fe42a021964728fb0",
            "environment": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "checkpoint_evaluated": {
                "path": str(ckpt_path.relative_to(REPO_ROOT)),
                "sha256": expected_hashes["checkpoint"][2],
            }
        },
        "artifact_integrity": artifact_integrity,
        "loadability": loadability_report,
        "tokenizer_contracts": tokenizer_report,
        "generation_determinism": {
            "deterministic_output_verified": deterministic,
            "sample_prompt": fixed_prompt,
            "sample_output": out1,
        },
        "robustness_and_performance": {
            "throughput_tokens_per_sec": throughput,
            "average_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
            "test_cases": robustness_results,
        },
        "intelligence_capability_summary": intel_summary,
        "coding_capability_summary": coding_summary,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[COMPLETE] Unified baseline evaluation recorded to: {output_json_path}")
    return report


if __name__ == "__main__":
    run_canonical_baseline()
