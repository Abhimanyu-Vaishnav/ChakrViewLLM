#!/usr/bin/env python3
"""
Tracked, Reproducible Neural Generation Capability Benchmark Harness.

Location: evaluation/neural_benchmark_harness.py

Key Capabilities:
1. Loads explicit model configurations or checkpoints (canonical random baseline, Stage C 9k, Stage D 500).
2. Validates checkpoint compatibility and architecture invariants fail-closed.
3. Evaluates models against a fixed, versioned prompt suite (evaluation/benchmark_prompts_v1.0.json).
4. Strictly measures pure neural autoregressive forward passes (LocalModelRuntime / InferenceEngine).
   Never falls back to symbolic templates or rules.
5. Computes precise metrics:
   - actual generated tokens (measured chunk by chunk)
   - latency (wall-clock total and time-to-first-token TTFT)
   - actual tokens per second (actual generated tokens / elapsed seconds)
   - repeated n-grams (2-gram, 3-gram, 4-gram repetition ratios)
   - unique token ratio (distinct generated tokens / total generated tokens)
   - EOS termination rate (genuine EOS vs MAX_TOKENS limit)
6. Outputs structured JSON reports and clean Markdown summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.inference import GenerationConfig, StopReason
from chakrview.runtime.local_runtime import LocalModelRuntime, EXPECTED_PARAM_COUNT
from chakrview.runtime.pipeline import InferenceEngine
from chakrview.runtime.sampling import SamplingConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = ROOT_DIR / "evaluation" / "benchmark_prompts_v1.0.json"
DEFAULT_TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"


def compute_ngram_repetition_ratio(token_ids: List[int], n: int) -> float:
    """Computes the ratio of repeated n-grams to total n-grams in the token sequence."""
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def compute_unique_token_ratio(token_ids: List[int]) -> float:
    """Computes the ratio of unique token IDs to total tokens generated."""
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def file_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PromptGenerationResult:
    prompt_id: str
    category: str
    prompt_text: str
    generated_text: str
    generated_token_ids: List[int]
    generated_token_count: int
    ttft_seconds: float
    total_latency_seconds: float
    tokens_per_second: float
    stop_reason: str
    unique_token_ratio: float
    repeated_2gram_ratio: float
    repeated_3gram_ratio: float
    repeated_4gram_ratio: float


@dataclass
class BenchmarkSuiteResult:
    benchmark_name: str
    checkpoint_label: str
    checkpoint_path: Optional[str]
    checkpoint_file_sha256: Optional[str]
    model_weight_hash: str
    parameter_count: int
    device: str
    cpu_thread_count: int
    temperature: float
    max_new_tokens: int
    total_prompts: int
    total_tokens_generated: int
    mean_latency_ms: float
    median_latency_ms: float
    mean_ttft_ms: float
    mean_throughput_tps: float
    eos_termination_count: int
    max_tokens_termination_count: int
    mean_unique_token_ratio: float
    mean_repeated_4gram_ratio: float
    system_info: Dict[str, Any]
    prompt_results: List[PromptGenerationResult]


class NeuralBenchmarkHarness:
    """Unified runner for evaluating neural checkpoints on CPU."""

    def __init__(
        self,
        checkpoint_label: str,
        checkpoint_path: Optional[Path] = None,
        tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR,
        cpu_threads: int = 2,
        seed: int = 42,
    ) -> None:
        self.checkpoint_label = checkpoint_label
        self.checkpoint_path = checkpoint_path
        self.tokenizer_dir = tokenizer_dir
        self.cpu_threads = cpu_threads
        self.seed = seed

        # Configure PyTorch CPU threads
        torch.set_num_threads(cpu_threads)
        torch.manual_seed(seed)

        # Load tokenizer
        if not self.tokenizer_dir.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_dir}")
        self.tokenizer, _ = load_tokenizer_artifacts(self.tokenizer_dir)

        # Initialize model
        self.model = ChakrMicro(ModelConfig())
        self.checkpoint_sha256: Optional[str] = None

        if checkpoint_path is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            self.checkpoint_sha256 = file_sha256(checkpoint_path)
            ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt_data.get("model_state_dict") or ckpt_data.get("model")
            if state_dict is None:
                raise ValueError(f"Checkpoint at {checkpoint_path} missing state_dict!")
            self.model.load_state_dict(state_dict, strict=True)
        else:
            # Random baseline seed 42
            pass

        self.model.eval()

        # Build runtime & engine
        self.runtime = LocalModelRuntime.from_default(tokenizer_dir=self.tokenizer_dir, torch_seed=seed)
        self.runtime.model = self.model
        self.runtime.engine.model = self.model
        self.model_weight_hash = self.runtime.engine.compute_weight_hash()
        self.runtime.expected_weight_hash = self.model_weight_hash
        self.runtime.engine.expected_weight_hash = self.model_weight_hash

    def run_benchmark(
        self,
        prompts_path: Path = DEFAULT_PROMPTS_PATH,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        warmup_runs: int = 2,
    ) -> BenchmarkSuiteResult:
        """Executes the benchmark suite over the configured model."""
        if not prompts_path.is_file():
            raise FileNotFoundError(f"Prompt suite not found: {prompts_path}")

        with open(prompts_path, "r", encoding="utf-8") as f:
            suite_data = json.load(f)
        prompts = suite_data.get("prompts", [])
        if not prompts:
            raise ValueError(f"No prompts found in {prompts_path}")

        # Warm-up phase (discarded)
        gen_cfg = GenerationConfig(
            max_new_tokens=16,
            sampling=SamplingConfig(temperature=temperature),
        )
        for _ in range(warmup_runs):
            list(self.runtime.stream_generate("Warmup prompt for inference initialization.", generation_config=gen_cfg))

        prompt_results: List[PromptGenerationResult] = []
        latencies_ms = []
        ttfts_ms = []
        throughputs = []
        eos_count = 0
        max_tokens_count = 0

        eval_gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            sampling=SamplingConfig(temperature=temperature),
        )

        for p_item in prompts:
            pid = p_item["id"]
            cat = p_item["category"]
            prompt_str = p_item["prompt"]

            tokens_emitted: List[str] = []
            tokens_emitted_ids: List[int] = []
            ttft_time: Optional[float] = None
            start_time = time.perf_counter()

            for chunk in self.runtime.stream_generate(prompt_str, generation_config=eval_gen_cfg):
                if ttft_time is None:
                    ttft_time = time.perf_counter() - start_time
                tokens_emitted.append(chunk.token_text)
                tokens_emitted_ids.append(chunk.token_id)
                last_stop_reason = chunk.stop_reason

            end_time = time.perf_counter()
            total_duration = end_time - start_time
            ttft_sec = ttft_time if ttft_time is not None else total_duration

            act_token_count = len(tokens_emitted_ids)
            tps = act_token_count / total_duration if total_duration > 0 else 0.0

            stop_str = str(last_stop_reason.value) if last_stop_reason else "unknown"
            if last_stop_reason == StopReason.EOS:
                eos_count += 1
            else:
                max_tokens_count += 1

            u_ratio = compute_unique_token_ratio(tokens_emitted_ids)
            rep_2 = compute_ngram_repetition_ratio(tokens_emitted_ids, 2)
            rep_3 = compute_ngram_repetition_ratio(tokens_emitted_ids, 3)
            rep_4 = compute_ngram_repetition_ratio(tokens_emitted_ids, 4)

            latencies_ms.append(total_duration * 1000.0)
            ttfts_ms.append(ttft_sec * 1000.0)
            throughputs.append(tps)

            prompt_results.append(
                PromptGenerationResult(
                    prompt_id=pid,
                    category=cat,
                    prompt_text=prompt_str,
                    generated_text="".join(tokens_emitted),
                    generated_token_ids=tokens_emitted_ids,
                    generated_token_count=act_token_count,
                    ttft_seconds=ttft_sec,
                    total_latency_seconds=total_duration,
                    tokens_per_second=tps,
                    stop_reason=stop_str,
                    unique_token_ratio=u_ratio,
                    repeated_2gram_ratio=rep_2,
                    repeated_3gram_ratio=rep_3,
                    repeated_4gram_ratio=rep_4,
                )
            )

        mean_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        sorted_lat = sorted(latencies_ms)
        median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        mean_ttft = sum(ttfts_ms) / len(ttfts_ms) if ttfts_ms else 0.0
        mean_tps = sum(throughputs) / len(throughputs) if throughputs else 0.0
        mean_unique = sum(r.unique_token_ratio for r in prompt_results) / len(prompt_results) if prompt_results else 0.0
        mean_rep4 = sum(r.repeated_4gram_ratio for r in prompt_results) / len(prompt_results) if prompt_results else 0.0

        sys_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cpu_count_logical": os.cpu_count(),
        }

        return BenchmarkSuiteResult(
            benchmark_name="ChakrView Frozen Neural Generation Benchmark v1.0",
            checkpoint_label=self.checkpoint_label,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            checkpoint_file_sha256=self.checkpoint_sha256,
            model_weight_hash=self.model_weight_hash,
            parameter_count=EXPECTED_PARAM_COUNT,
            device="cpu",
            cpu_thread_count=self.cpu_threads,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            total_prompts=len(prompts),
            total_tokens_generated=sum(r.generated_token_count for r in prompt_results),
            mean_latency_ms=mean_lat,
            median_latency_ms=median_lat,
            mean_ttft_ms=mean_ttft,
            mean_throughput_tps=mean_tps,
            eos_termination_count=eos_count,
            max_tokens_termination_count=max_tokens_count,
            mean_unique_token_ratio=mean_unique,
            mean_repeated_4gram_ratio=mean_rep4,
            system_info=sys_info,
            prompt_results=prompt_results,
        )


def save_benchmark_results(res: BenchmarkSuiteResult, out_json: Path, out_md: Path):
    """Saves machine-readable JSON and human-readable Markdown."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2)

    # Markdown format
    md_lines = [
        f"# Benchmark Report: {res.checkpoint_label}",
        f"- **Model**: ChakrMicro ({res.parameter_count:,} parameters)",
        f"- **Weight Hash**: `{res.model_weight_hash}`",
        f"- **Checkpoint Path**: `{res.checkpoint_path or 'None (Random Init)'}`",
        f"- **Checkpoint SHA-256**: `{res.checkpoint_file_sha256 or 'N/A'}`",
        f"- **Device / Threads**: CPU ({res.cpu_thread_count} threads)",
        f"- **Prompts Evaluated**: {res.total_prompts}",
        f"- **Total Generated Tokens**: {res.total_tokens_generated}",
        f"- **Mean Latency**: {res.mean_latency_ms:.2f} ms (Median: {res.median_latency_ms:.2f} ms)",
        f"- **Mean TTFT**: {res.mean_ttft_ms:.2f} ms",
        f"- **Mean Throughput**: {res.mean_throughput_tps:.2f} tok/s",
        f"- **Mean Unique-Token Ratio**: {res.mean_unique_token_ratio:.3f}",
        f"- **Mean 4-Gram Repetition Ratio**: {res.mean_repeated_4gram_ratio:.3f}",
        f"- **Termination**: {res.eos_termination_count} EOS / {res.max_tokens_termination_count} Token Limit",
        "",
        "## Prompt Generation Samples",
        "",
        "| ID | Category | Prompt | Generated Output (First 60 chars) | TPS | Unique Ratio | 4-Gram Rep |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
    ]

    for p in res.prompt_results:
        clean_out = p.generated_text.replace("\n", " ").replace("|", "\\|")[:60]
        md_lines.append(
            f"| `{p.prompt_id}` | {p.category} | {p.prompt_text[:30]}... | `{clean_out}` | {p.tokens_per_second:.1f} | {p.unique_token_ratio:.2f} | {p.repeated_4gram_ratio:.2f} |"
        )

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
