#!/usr/bin/env python3
"""
Tracked, Reproducible Neural Generation Capability Benchmark Harness v2.0.

Location: evaluation/neural_benchmark_harness.py

Key Capabilities:
1. Loads explicit model configurations or checkpoints (canonical random baseline, Stage C 9k, Stage D 500).
2. Validates checkpoint compatibility and architecture invariants fail-closed.
3. Evaluates models against a fixed, versioned prompt suite (v1.0 or v2.0).
4. Strictly measures pure neural autoregressive forward passes (LocalModelRuntime / InferenceEngine).
   Never falls back to symbolic templates or rules.
5. Computes precise metrics:
   - actual generated tokens (measured suffix chunk by chunk, strictly excluding prompt tokens)
   - latency (wall-clock total and time-to-first-token TTFT)
   - actual tokens per second (actual generated tokens / elapsed seconds)
   - repeated n-grams (2-gram, 3-gram, 4-gram repetition ratios over generated tokens)
   - unique token ratio (distinct generated tokens / total generated tokens)
   - EOS termination rate (genuine EOS vs MAX_TOKENS limit)
   - deterministic task checking (for arithmetic, factual recall, logic, and code syntax)
6. Outputs structured JSON reports and clean Markdown summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
DEFAULT_PROMPTS_PATH = ROOT_DIR / "evaluation" / "benchmark_prompts_v2.0.json"
DEFAULT_PROMPTS_V1_PATH = ROOT_DIR / "evaluation" / "benchmark_prompts_v1.0.json"
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


import ast
import re

EVALUATOR_VERSION = "2.1.0"


def evaluate_deterministic_match(
    generated_text: str,
    eval_type: str,
    eval_keys: List[str],
) -> Optional[bool]:
    """
    Evaluates objective checkability (v2.1.0).
    - Uses regex word boundary checks (?<!\\w)...(?!\\w) to prevent accidental substring false positives
      (e.g., matching '27' in '127' or 'Corbett' in 'unCorbett').
    - Uses AST parsing for Python syntax checks, preventing broken syntax with matching substrings from passing.
    - Preserves case-insensitive checking for semantic keys while respecting formatting invariants.
    Returns True/False for deterministic types, or None for open/human review types.
    """
    if not eval_keys or eval_type == "human_rubric":
        return None

    clean_gen = generated_text.strip()
    if not clean_gen:
        return False

    if eval_type in ("deterministic_exact_or_substring", "deterministic_substring"):
        # Match with word boundaries to avoid false substring collisions
        for k in eval_keys:
            # If key contains symbols (e.g. operators or punctuation), escape safely
            escaped = re.escape(k.strip())
            pattern = rf"(?<!\w){escaped}(?!\w)"
            if re.search(pattern, clean_gen, re.IGNORECASE):
                return True
        return False

    elif eval_type == "deterministic_syntax_check":
        # Check if code snippet contains Python function definition, attempt AST parse
        if any(k.startswith("def ") or "return " in k for k in eval_keys):
            try:
                ast.parse(clean_gen)
            except SyntaxError:
                return False

        # Validate presence of required syntactic substrings
        for k in eval_keys:
            if k not in clean_gen and k.lower() not in clean_gen.lower():
                return False
        return True

    return None


def classify_failure_mode(
    generated_text: str,
    domain: str,
    is_correct: Optional[bool],
) -> str:
    """
    Assigns an evidence-supported, auditable failure category (v2.1.0).
    Categories:
    - correct_deterministic: Answer verified correct by deterministic evaluator.
    - human_judgment_required: Open-ended task requiring semantic human rubric evaluation.
    - degeneration / character_loop: Repeating ungrounded punctuation or characters.
    - template_leakage / memorization: Canned arithmetic scratchpad or prompt leakage.
    - repetition / loop_collapse: Autoregressive n-gram loop attractor.
    - arithmetic_error / ungrounded: Calculation failure or unrelated numbers.
    - instruction_noncompliance: Ignored formatting or logic instructions.
    - syntax_error / non_code: Invalid code syntax or non-code text for coding prompts.
    - unsupported_claim / non_grounded: Information absent from supplied source passage.
    - irrelevant_response: Off-topic generation.
    """
    if is_correct is True:
        return "correct_deterministic"

    clean = generated_text.strip()
    if not clean:
        return "empty_output / early_eos"

    if clean.startswith("?") or clean.startswith(":") or clean.startswith("="):
        return "degeneration / character_loop"

    if "<<" in clean or "The answer is 200" in clean or "So he has 200*2=" in clean or "There are 50*2=" in clean:
        return "template_leakage / memorization"

    if (
        "The Great of the Great" in clean
        or "the first of the first" in clean
        or "the group of the group" in clean
        or "number of the number" in clean
        or "| \n|" in clean
        or "* [http" in clean
    ):
        return "repetition / loop_collapse"

    if domain == "arithmetic_logic":
        return "arithmetic_error / ungrounded"

    if domain == "algorithmic_reasoning":
        return "instruction_noncompliance"

    if domain == "safe_coding_structure":
        return "syntax_error / non_code"

    if domain in ("passage_factual_recall", "passage_explanation_summarization"):
        return "unsupported_claim / non_grounded"

    if domain == "open_continuation":
        return "human_judgment_required"

    return "irrelevant_response"


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
    expected_answer: Optional[str] = None
    evaluation_type: Optional[str] = None
    eval_keys: Optional[List[str]] = None
    is_correct_deterministic: Optional[bool] = None
    human_rubric: Optional[str] = None
    failure_category: Optional[str] = None



@dataclass
class BenchmarkSuiteResult:
    benchmark_name: str
    suite_version: str
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
    deterministic_evaluated_count: int
    deterministic_passed_count: int
    deterministic_accuracy: float
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
        suite_version = suite_data.get("suite_version", "1.0.0")
        prompts = suite_data.get("prompts", [])
        if not prompts:
            raise ValueError(f"No prompts found in {prompts_path}")

        # Warm-up phase (discarded from metrics)
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

        det_eval_count = 0
        det_pass_count = 0

        for p_item in prompts:
            pid = p_item["id"]
            cat = p_item.get("domain") or p_item.get("category", "general")
            prompt_str = p_item["prompt"]
            exp_ans = p_item.get("expected_answer") or p_item.get("expected_topic")
            eval_type = p_item.get("evaluation_type")
            eval_keys = p_item.get("eval_keys")
            rubric = p_item.get("human_rubric")

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

            gen_text = "".join(tokens_emitted)

            # Deterministic check if applicable
            is_correct = evaluate_deterministic_match(gen_text, eval_type or "", eval_keys or [])
            if is_correct is not None:
                det_eval_count += 1
                if is_correct:
                    det_pass_count += 1

            fail_cat = classify_failure_mode(gen_text, cat, is_correct)

            prompt_results.append(
                PromptGenerationResult(
                    prompt_id=pid,
                    category=cat,
                    prompt_text=prompt_str,
                    generated_text=gen_text,
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
                    expected_answer=exp_ans,
                    evaluation_type=eval_type,
                    eval_keys=eval_keys,
                    is_correct_deterministic=is_correct,
                    human_rubric=rubric,
                    failure_category=fail_cat,
                )
            )

        mean_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        sorted_lat = sorted(latencies_ms)
        median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        mean_ttft = sum(ttfts_ms) / len(ttfts_ms) if ttfts_ms else 0.0
        mean_tps = sum(throughputs) / len(throughputs) if throughputs else 0.0
        mean_unique = sum(r.unique_token_ratio for r in prompt_results) / len(prompt_results) if prompt_results else 0.0
        mean_rep4 = sum(r.repeated_4gram_ratio for r in prompt_results) / len(prompt_results) if prompt_results else 0.0
        det_acc = (det_pass_count / det_eval_count) if det_eval_count > 0 else 0.0

        sys_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cpu_count_logical": os.cpu_count(),
        }

        return BenchmarkSuiteResult(
            benchmark_name="ChakrView Frozen Neural Generation Benchmark v2.0",
            suite_version=suite_version,
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
            deterministic_evaluated_count=det_eval_count,
            deterministic_passed_count=det_pass_count,
            deterministic_accuracy=det_acc,
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
        f"- **Suite Version**: `{res.suite_version}`",
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
        f"- **Deterministic Task Accuracy**: {res.deterministic_passed_count}/{res.deterministic_evaluated_count} ({res.deterministic_accuracy:.1%})",
        "",
        "## Prompt Generation Samples",
        "",
        "| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |",
        "| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |",
    ]

    for p in res.prompt_results:
        clean_out = p.generated_text.replace("\n", " ").replace("|", "\\|")[:50]
        corr_str = "N/A" if p.is_correct_deterministic is None else ("PASS" if p.is_correct_deterministic else "FAIL")
        f_cat = p.failure_category or "unclassified"
        md_lines.append(
            f"| `{p.prompt_id}` | {p.category} | {p.prompt_text[:25]}... | `{clean_out}` | {corr_str} | `{f_cat}` | {p.tokens_per_second:.1f} | {p.unique_token_ratio:.2f} | {p.repeated_4gram_ratio:.2f} |"
        )

    out_md.write_text("\n".join(md_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="ChakrView Frozen Neural Capability Benchmark Harness")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--label", type=str, default="RANDOM_INITIALIZATION", help="Label for the model")
    parser.add_argument("--prompts", type=str, default=str(DEFAULT_PROMPTS_PATH), help="Path to prompts JSON")
    parser.add_argument("--out-json", type=str, required=True, help="Path for output JSON")
    parser.add_argument("--out-md", type=str, required=True, help="Path for output Markdown")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--cpu-threads", type=int, default=2, help="Number of CPU threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Number of warmup inference passes")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    harness = NeuralBenchmarkHarness(
        checkpoint_label=args.label,
        checkpoint_path=ckpt_path,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )

    results = harness.run_benchmark(
        prompts_path=Path(args.prompts),
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup_runs=args.warmup_runs,
    )

    save_benchmark_results(results, Path(args.out_json), Path(args.out_md))
    print(f"Benchmark finished successfully. Results saved to {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
