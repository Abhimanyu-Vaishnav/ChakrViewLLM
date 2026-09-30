"""
ChakrView Interactive Model Evaluation Runtime (Step 50).

Provides interactive session management, model comparison, structured evaluation
probes, raw token-level diagnostics, deterministic generation controls, and session
logging for experimentally trained ChakrMicro models.

Core Invariants Enforced:
1. ΔW = 0: Neither trained model nor frozen baseline weights are ever mutated.
2. CPU-first execution: No CUDA/GPU or external dependencies.
3. Separation of observation and interpretation: Objective statistical measurements.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.inference import GenerationConfig, StopReason
from chakrview.runtime.sampling import SamplingConfig
from chakrview.runtime.pipeline import (
    InferenceEngine,
    EXPECTED_WEIGHT_HASH,
    EXPECTED_VOCAB_SIZE,
    MAX_CONTEXT_WINDOW,
    validate_checkpoint_compatibility,
)
from chakrview.runtime.session import LocalModelSession, ConversationRole, ConversationTurn
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
EXPECTED_TOKENIZER_CHECKSUM = "7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f"
DEFAULT_TRAINED_CHECKPOINT = (
    ROOT_DIR / "artifacts" / "step48" / "run_seed_42" / "checkpoint_0000100.pt"
)
SESSIONS_LOG_DIR = ROOT_DIR / "artifacts" / "step50" / "sessions"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_hash(model: nn.Module) -> str:
    """Compute SHA-256 digest of named model parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def instantiate_frozen_baseline(config: Optional[ModelConfig] = None) -> ChakrMicro:
    """Instantiate canonical frozen baseline model (hash: c5571c...)."""
    cfg = config or ModelConfig()
    torch.manual_seed(42)
    model = ChakrMicro(cfg)
    model.eval()
    baseline_hash = compute_model_hash(model)
    if baseline_hash != EXPECTED_WEIGHT_HASH:
        raise RuntimeError(
            f"Frozen baseline SHA-256 mismatch: {baseline_hash} != {EXPECTED_WEIGHT_HASH}"
        )
    return model


def load_trained_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Optional[ModelConfig] = None,
    allow_baseline: bool = False,
) -> Tuple[ChakrMicro, Dict[str, Any], str]:
    """
    Safely load an experimental checkpoint, verify integrity, parameter count,
    finite weights, and check that it is not the frozen baseline unless allowed.

    Returns:
        (model, checkpoint_payload, weight_hash)
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    cfg = config or ModelConfig()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Validate compatibility via runtime pipeline
    validate_checkpoint_compatibility(checkpoint, expected_config=cfg)

    model = ChakrMicro(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Check finite weights
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise ValueError(f"Non-finite weights (NaN or Inf) detected in tensor: {name}")

    weight_hash = compute_model_hash(model)

    if not allow_baseline and weight_hash == EXPECTED_WEIGHT_HASH:
        raise ValueError(
            "Selected checkpoint has identical hash to frozen baseline (no training updates)."
        )

    return model, checkpoint, weight_hash


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generation Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResponseMetrics:
    """Objective quantitative metrics for a single generation."""
    length: int
    token_diversity: float
    repetition_unigram: float
    repetition_bigram: float
    repetition_trigram: float
    eos_occurred: bool
    throughput_tokens_per_sec: float
    latency_ms_per_token: float
    elapsed_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GenerationMetricsCalculator:
    """Calculates objective sequence diversity, repetition, and timing metrics."""

    @staticmethod
    def calculate(
        token_ids: List[int],
        elapsed_sec: float,
        eos_token_id: int = 1,
    ) -> ResponseMetrics:
        n = len(token_ids)
        eos_occurred = (eos_token_id in token_ids)

        if n == 0:
            return ResponseMetrics(
                length=0,
                token_diversity=0.0,
                repetition_unigram=0.0,
                repetition_bigram=0.0,
                repetition_trigram=0.0,
                eos_occurred=eos_occurred,
                throughput_tokens_per_sec=0.0,
                latency_ms_per_token=0.0,
                elapsed_sec=elapsed_sec,
            )

        # Unigrams
        unique_unigrams = len(set(token_ids))
        token_diversity = unique_unigrams / float(n)
        repetition_unigram = 1.0 - (unique_unigrams / float(n))

        # Bigrams
        if n >= 2:
            bigrams = [tuple(token_ids[i : i + 2]) for i in range(n - 1)]
            unique_bigrams = len(set(bigrams))
            repetition_bigram = 1.0 - (unique_bigrams / float(len(bigrams)))
        else:
            repetition_bigram = 0.0

        # Trigrams
        if n >= 3:
            trigrams = [tuple(token_ids[i : i + 3]) for i in range(n - 2)]
            unique_trigrams = len(set(trigrams))
            repetition_trigram = 1.0 - (unique_trigrams / float(len(trigrams)))
        else:
            repetition_trigram = 0.0

        throughput = (n / elapsed_sec) if elapsed_sec > 0 else 0.0
        latency_ms_per_tok = (elapsed_sec * 1000.0 / n) if n > 0 else 0.0

        return ResponseMetrics(
            length=n,
            token_diversity=round(token_diversity, 4),
            repetition_unigram=round(repetition_unigram, 4),
            repetition_bigram=round(repetition_bigram, 4),
            repetition_trigram=round(repetition_trigram, 4),
            eos_occurred=eos_occurred,
            throughput_tokens_per_sec=round(throughput, 2),
            latency_ms_per_token=round(latency_ms_per_tok, 2),
            elapsed_sec=round(elapsed_sec, 4),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Raw Token Diagnostics & Information Divergence
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawTokenDiagnostic:
    """Step-level probability inspection data for a single generated token."""
    step: int
    token_id: int
    token_text: str
    probability: float
    rank: int
    top5_alternatives: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_cosine_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Cosine distance between two 1D probability distributions."""
    dot = torch.sum(p * q).item()
    norm_p = torch.norm(p, p=2).item()
    norm_q = torch.norm(q, p=2).item()
    if norm_p == 0.0 or norm_q == 0.0:
        return 0.0
    sim = dot / (norm_p * norm_q)
    return max(0.0, min(1.0, 1.0 - sim))


def compute_js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two 1D probability distributions."""
    p_safe = p + eps
    p_safe = p_safe / p_safe.sum()
    q_safe = q + eps
    q_safe = q_safe / q_safe.sum()
    m = 0.5 * (p_safe + q_safe)
    kl_pm = torch.sum(p_safe * torch.log(p_safe / m)).item()
    kl_qm = torch.sum(q_safe * torch.log(q_safe / m)).item()
    return max(0.0, 0.5 * (kl_pm + kl_qm))


def compute_topk_overlap(p: torch.Tensor, q: torch.Tensor, k: int = 5) -> float:
    """Jaccard overlap of top-k token sets."""
    top_p = set(torch.topk(p, k).indices.tolist())
    top_q = set(torch.topk(q, k).indices.tolist())
    intersection = len(top_p & top_q)
    union = len(top_p | top_q)
    return (intersection / float(union)) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Comparator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """Side-by-side comparison between frozen baseline and trained checkpoint."""
    prompt: str
    baseline_text: str
    trained_text: str
    baseline_tokens: List[int]
    trained_tokens: List[int]
    baseline_metrics: ResponseMetrics
    trained_metrics: ResponseMetrics
    seed: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "baseline_text": self.baseline_text,
            "trained_text": self.trained_text,
            "baseline_tokens": self.baseline_tokens,
            "trained_tokens": self.trained_tokens,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "trained_metrics": self.trained_metrics.to_dict(),
            "seed": self.seed,
        }


class ModelComparator:
    """Executes side-by-side generation using identical seed and sampling configuration."""

    def __init__(
        self,
        trained_model: ChakrMicro,
        frozen_baseline: ChakrMicro,
        tokenizer: BPETokenizer,
    ) -> None:
        self.trained_model = trained_model
        self.frozen_baseline = frozen_baseline
        self.tokenizer = tokenizer

    def compare(
        self,
        prompt: str,
        gen_config: GenerationConfig,
        seed: Optional[int] = 42,
    ) -> ComparisonResult:
        """Run identical prompt through both models with identical RNG seed."""
        # 1. Baseline generation
        if seed is not None:
            torch.manual_seed(seed)
        t0 = time.perf_counter()
        base_tokens, base_text = self._generate_tokens(
            self.frozen_baseline, prompt, gen_config, seed=seed
        )
        t_base = time.perf_counter() - t0
        base_metrics = GenerationMetricsCalculator.calculate(base_tokens, t_base)

        # 2. Trained generation
        if seed is not None:
            torch.manual_seed(seed)
        t0 = time.perf_counter()
        trained_tokens, trained_text = self._generate_tokens(
            self.trained_model, prompt, gen_config, seed=seed
        )
        t_trained = time.perf_counter() - t0
        trained_metrics = GenerationMetricsCalculator.calculate(trained_tokens, t_trained)

        return ComparisonResult(
            prompt=prompt,
            baseline_text=base_text,
            trained_text=trained_text,
            baseline_tokens=base_tokens,
            trained_tokens=trained_tokens,
            baseline_metrics=base_metrics,
            trained_metrics=trained_metrics,
            seed=seed,
        )

    def _generate_tokens(
        self,
        model: ChakrMicro,
        prompt: str,
        config: GenerationConfig,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], str]:
        if seed is not None:
            torch.manual_seed(seed)
        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        curr_tokens = list(prompt_tokens)
        generated: List[int] = []

        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                if len(curr_tokens) >= 512:
                    break
                x = torch.tensor([curr_tokens], dtype=torch.long)
                logits = model(x)[:, -1, :]

                # Apply repetition penalty
                sampling = config.sampling
                if sampling.repetition_penalty > 1.0:
                    for prev_id in set(curr_tokens):
                        if logits[0, prev_id] > 0:
                            logits[0, prev_id] /= sampling.repetition_penalty
                        else:
                            logits[0, prev_id] *= sampling.repetition_penalty

                if sampling.temperature <= 1e-4:
                    next_token = int(torch.argmax(logits, dim=-1).item())
                else:
                    scaled = logits / sampling.temperature
                    if sampling.top_k > 0:
                        top_vals, _ = torch.topk(scaled, min(sampling.top_k, scaled.size(-1)))
                        scaled[scaled < top_vals[:, [-1]]] = -float("Inf")
                    probs = F.softmax(scaled, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())

                generated.append(next_token)
                curr_tokens.append(next_token)

                if next_token == 1 and len(generated) >= config.min_new_tokens:
                    break

        text = self.tokenizer.decode(generated, skip_special_tokens=True, errors="replace")
        return generated, text


# ─────────────────────────────────────────────────────────────────────────────
# 5. Structured Probe Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class StructuredProbeEvaluator:
    """Executes objective evaluation suite from step50_interactive_probes.json."""

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        fixtures_path: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.fixtures_path = fixtures_path or (
            ROOT_DIR / "tests" / "fixtures" / "step50_interactive_probes.json"
        )

    def load_probes(self) -> Dict[str, Any]:
        with open(self.fixtures_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_all_probes(self) -> Dict[str, Any]:
        """Run all structured probe categories and return objective measurements."""
        data = self.load_probes()
        results: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "probes": {},
        }

        # Probe A: Basic Language
        results["probes"]["probe_a"] = self.eval_probe_a(data.get("probe_a_basic_language", []))

        # Probe B: Context Change
        results["probes"]["probe_b"] = self.eval_probe_b(data.get("probe_b_context_change", []))

        # Probe C: Controlled Memory
        results["probes"]["probe_c"] = self.eval_probe_c(data.get("probe_c_controlled_memory", []))

        # Probe D: Negative Context Control
        results["probes"]["probe_d"] = self.eval_probe_d(data.get("probe_d_negative_control", []))

        # Probe E: Simple Pattern
        results["probes"]["probe_e"] = self.eval_probe_e(data.get("probe_e_simple_pattern", []))

        return results

    def eval_probe_a(self, prompts: List[str]) -> List[Dict[str, Any]]:
        records = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            x = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(x)[:, -1, :]
                probs = F.softmax(logits, dim=-1)[0]
                top5_vals, top5_ids = torch.topk(probs, 5)

            top1_id = int(top5_ids[0].item())
            top1_prob = float(top5_vals[0].item())
            top1_str = self.tokenizer.decode([top1_id], skip_special_tokens=False, errors="replace")

            top5_items = [
                {
                    "rank": i + 1,
                    "token_id": int(tid.item()),
                    "token_str": self.tokenizer.decode([int(tid.item())], skip_special_tokens=False, errors="replace"),
                    "probability": round(float(val.item()), 4),
                }
                for i, (val, tid) in enumerate(zip(top5_vals, top5_ids))
            ]

            # Continuation (greedy 10 tokens)
            curr = list(tokens)
            cont_tokens = []
            for _ in range(10):
                if len(curr) >= 512:
                    break
                xx = torch.tensor([curr], dtype=torch.long)
                with torch.no_grad():
                    nxt = int(torch.argmax(self.model(xx)[:, -1, :], dim=-1).item())
                cont_tokens.append(nxt)
                curr.append(nxt)
                if nxt == 1:
                    break
            cont_text = self.tokenizer.decode(cont_tokens, skip_special_tokens=True, errors="replace")

            records.append({
                "prompt": prompt,
                "top1_token": top1_str,
                "top1_probability": round(top1_prob, 4),
                "top5": top5_items,
                "continuation": cont_text,
            })
        return records

    def eval_probe_b(self, pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        records = []
        for pair in pairs:
            p1 = pair["prompt_1"]
            p2 = pair["prompt_2"]

            tok1 = self.tokenizer.encode(p1, add_bos=True, add_eos=False)
            tok2 = self.tokenizer.encode(p2, add_bos=True, add_eos=False)

            with torch.no_grad():
                probs1 = F.softmax(self.model(torch.tensor([tok1]))[:, -1, :], dim=-1)[0]
                probs2 = F.softmax(self.model(torch.tensor([tok2]))[:, -1, :], dim=-1)[0]

            js_div = compute_js_divergence(probs1, probs2)
            cos_dist = compute_cosine_distance(probs1, probs2)
            topk_over = compute_topk_overlap(probs1, probs2, k=5)

            records.append({
                "prompt_1": p1,
                "prompt_2": p2,
                "js_divergence": round(js_div, 4),
                "cosine_distance": round(cos_dist, 4),
                "top5_overlap": round(topk_over, 4),
            })
        return records

    def eval_probe_c(self, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = []
        filler_token_str = " The quick brown fox jumped."
        for spec in specs:
            v1 = spec["target_value_1"]
            v2 = spec["target_value_2"]
            distances = spec.get("distances", [8, 16, 32, 64, 128])
            prefix = spec.get("base_prefix", "The secret color is ")
            suffix = spec.get("query_suffix", "The secret color is")

            for dist in distances:
                filler_toks = self.tokenizer.encode(filler_token_str, add_bos=False, add_eos=False)
                # Repeat to match distance
                repeats = max(1, dist // max(1, len(filler_toks)))
                filler = (filler_token_str * repeats)[: dist * 4]

                prompt1 = f"{prefix}{v1}. {filler} {suffix}"
                prompt2 = f"{prefix}{v2}. {filler} {suffix}"

                tok1 = self.tokenizer.encode(prompt1, add_bos=True, add_eos=False)
                tok2 = self.tokenizer.encode(prompt2, add_bos=True, add_eos=False)

                if len(tok1) > 512 or len(tok2) > 512:
                    continue

                with torch.no_grad():
                    probs1 = F.softmax(self.model(torch.tensor([tok1]))[:, -1, :], dim=-1)[0]
                    probs2 = F.softmax(self.model(torch.tensor([tok2]))[:, -1, :], dim=-1)[0]

                js = compute_js_divergence(probs1, probs2)
                cos = compute_cosine_distance(probs1, probs2)

                records.append({
                    "distance_tokens": dist,
                    "target_1": v1,
                    "target_2": v2,
                    "js_divergence": round(js, 4),
                    "cosine_distance": round(cos, 4),
                })
        return records

    def eval_probe_d(self, controls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        records = []
        for ctrl in controls:
            clean = ctrl["clean_prompt"]
            corrupted = ctrl["corrupted_prompt"]

            tok_clean = self.tokenizer.encode(clean, add_bos=True, add_eos=False)
            tok_corr = self.tokenizer.encode(corrupted, add_bos=True, add_eos=False)

            with torch.no_grad():
                probs_clean = F.softmax(self.model(torch.tensor([tok_clean]))[:, -1, :], dim=-1)[0]
                probs_corr = F.softmax(self.model(torch.tensor([tok_corr]))[:, -1, :], dim=-1)[0]

            js = compute_js_divergence(probs_clean, probs_corr)
            cos = compute_cosine_distance(probs_clean, probs_corr)

            records.append({
                "clean_prompt": clean,
                "corrupted_prompt": corrupted,
                "js_divergence": round(js, 4),
                "cosine_distance": round(cos, 4),
            })
        return records

    def eval_probe_e(self, patterns: List[str]) -> List[Dict[str, Any]]:
        records = []
        for pat in patterns:
            tokens = self.tokenizer.encode(pat, add_bos=True, add_eos=False)
            with torch.no_grad():
                probs = F.softmax(self.model(torch.tensor([tokens]))[:, -1, :], dim=-1)[0]
                top5_vals, top5_ids = torch.topk(probs, 5)

            top5 = [
                {
                    "token_id": int(tid.item()),
                    "token_str": self.tokenizer.decode([int(tid.item())], skip_special_tokens=False, errors="replace"),
                    "probability": round(float(val.item()), 4),
                }
                for val, tid in zip(top5_vals, top5_ids)
            ]
            records.append({
                "pattern": pat,
                "top5": top5,
            })
        return records


# ─────────────────────────────────────────────────────────────────────────────
# 6. Interactive Model Session Coordinator
# ─────────────────────────────────────────────────────────────────────────────

class InteractiveModelSessionCoordinator:
    """
    Coordinates interactive dialogue, CLI command routing, deterministic generation,
    model comparison, raw probability inspection, and session metric tracking.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        tokenizer_dir: Optional[Path] = None,
        deterministic: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_TRAINED_CHECKPOINT)
        self.tokenizer_dir = tokenizer_dir or DEFAULT_TOKENIZER_DIR
        self.tokenizer, _ = load_tokenizer_artifacts(self.tokenizer_dir)

        # Load trained checkpoint
        self.trained_model, self.checkpoint_meta, self.trained_hash = load_trained_checkpoint(
            self.checkpoint_path, allow_baseline=False
        )

        # Load frozen baseline
        self.baseline_model = instantiate_frozen_baseline()
        self.baseline_hash = EXPECTED_WEIGHT_HASH

        # Multi-turn session manager (Step 46 architecture)
        self.session = LocalModelSession(
            session_id=f"session_{int(time.time())}",
            tenant_id="developer_eval",
            max_context=MAX_CONTEXT_WINDOW,
        )

        # Settings
        self.deterministic = deterministic
        self.seed = seed
        self.raw_diagnostics = False
        self.gen_config = GenerationConfig(
            max_new_tokens=48,
            min_new_tokens=1,
            sampling=SamplingConfig(
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                seed=seed,
            ),
        )

        # Comparator & Probe Evaluator
        self.comparator = ModelComparator(
            trained_model=self.trained_model,
            frozen_baseline=self.baseline_model,
            tokenizer=self.tokenizer,
        )
        self.probe_evaluator = StructuredProbeEvaluator(
            model=self.trained_model,
            tokenizer=self.tokenizer,
        )

        # Session History and Metrics Tracking
        self.history: List[Dict[str, Any]] = []
        self.total_prompts: int = 0
        self.total_generated_tokens: int = 0
        self.total_latencies: List[float] = []
        self.total_eos_count: int = 0
        self.unigram_repetitions: List[float] = []

    def get_info(self) -> Dict[str, Any]:
        """Return model and session configuration parameters."""
        step = self.checkpoint_meta.get("step", "N/A")
        param_count = sum(p.numel() for p in self.trained_model.parameters())
        tok_chk = self.checkpoint_meta.get("tokenizer_checksum", EXPECTED_TOKENIZER_CHECKSUM)
        return {
            "checkpoint": str(self.checkpoint_path),
            "training_steps": step,
            "parameters": param_count,
            "vocabulary": EXPECTED_VOCAB_SIZE,
            "context_length": MAX_CONTEXT_WINDOW,
            "tokenizer_checksum": tok_chk,
            "model_weight_hash": self.trained_hash,
            "baseline_hash": self.baseline_hash,
            "deterministic": self.deterministic,
            "seed": self.seed,
            "raw_diagnostics": self.raw_diagnostics,
        }

    def reset_context(self) -> None:
        """Clear conversation turns."""
        self.session.clear()

    def get_context_stats(self) -> Dict[str, Any]:
        """Return turns, current token count, and remaining context budget."""
        turns_count = len(self.session.turns)
        total_tokens = sum(t.token_count for t in self.session.turns)
        remaining = max(0, MAX_CONTEXT_WINDOW - self.gen_config.max_new_tokens - total_tokens)
        return {
            "turns": turns_count,
            "token_count": total_tokens,
            "remaining_context_budget": remaining,
        }

    def get_session_stats(self) -> Dict[str, Any]:
        """Return cumulative session statistics."""
        avg_latency = (
            sum(self.total_latencies) / len(self.total_latencies)
            if self.total_latencies
            else 0.0
        )
        avg_throughput = (
            (self.total_generated_tokens / (sum(self.total_latencies) / 1000.0))
            if sum(self.total_latencies) > 0
            else 0.0
        )
        avg_rep = (
            sum(self.unigram_repetitions) / len(self.unigram_repetitions)
            if self.unigram_repetitions
            else 0.0
        )
        return {
            "prompts": self.total_prompts,
            "generated_tokens": self.total_generated_tokens,
            "average_latency_ms": round(avg_latency, 2),
            "average_tokens_per_sec": round(avg_throughput, 2),
            "eos_count": self.total_eos_count,
            "repetition_ratio": round(avg_rep, 4),
        }

    def toggle_raw(self) -> bool:
        """Toggle raw token diagnostics display."""
        self.raw_diagnostics = not self.raw_diagnostics
        return self.raw_diagnostics

    def set_seed(self, val: Union[int, str]) -> Optional[int]:
        """Set or randomize generation seed."""
        if isinstance(val, str) and val.lower() == "random":
            self.deterministic = False
            self.seed = None
            self.gen_config.sampling.seed = None
            return None
        seed_int = int(val)
        self.deterministic = True
        self.seed = seed_int
        self.gen_config.sampling.seed = seed_int
        return seed_int

    def generate_turn(
        self,
        user_input: str,
        collect_raw: bool = False,
    ) -> Tuple[str, ResponseMetrics, List[RawTokenDiagnostic]]:
        """
        Generate response turn, update multi-turn context, and calculate metrics.
        """
        # Set seed if deterministic
        if self.deterministic and self.seed is not None:
            torch.manual_seed(self.seed)

        # Build multi-turn prompt from history
        final_prompt, _, active_turns = self.session.build_prompt_package(
            user_input=user_input,
            tokenizer=self.tokenizer,
            max_new_tokens=self.gen_config.max_new_tokens,
            retrieve_memory=False,
        )

        prompt_tokens = self.tokenizer.encode(final_prompt, add_bos=True, add_eos=False)
        curr_tokens = list(prompt_tokens)
        generated_tokens: List[int] = []
        raw_diagnostics_list: List[RawTokenDiagnostic] = []

        t0 = time.perf_counter()
        with torch.no_grad():
            for step_idx in range(self.gen_config.max_new_tokens):
                if len(curr_tokens) >= MAX_CONTEXT_WINDOW:
                    break

                x = torch.tensor([curr_tokens], dtype=torch.long)
                logits = self.trained_model(x)[:, -1, :]

                # Probability calculation over raw vocabulary
                base_probs = F.softmax(logits, dim=-1)[0]

                # Sampling logic
                sampling = self.gen_config.sampling
                if sampling.repetition_penalty > 1.0:
                    for prev_id in set(curr_tokens):
                        if logits[0, prev_id] > 0:
                            logits[0, prev_id] /= sampling.repetition_penalty
                        else:
                            logits[0, prev_id] *= sampling.repetition_penalty

                if sampling.temperature <= 1e-4:
                    next_token = int(torch.argmax(logits, dim=-1).item())
                else:
                    scaled = logits / sampling.temperature
                    if sampling.top_k > 0:
                        top_vals, _ = torch.topk(scaled, min(sampling.top_k, scaled.size(-1)))
                        scaled[scaled < top_vals[:, [-1]]] = -float("Inf")
                    probs = F.softmax(scaled, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())

                # Collect diagnostics if requested or enabled
                if collect_raw or self.raw_diagnostics:
                    token_prob = float(base_probs[next_token].item())
                    rank = int((base_probs > base_probs[next_token]).sum().item()) + 1
                    top5_vals, top5_ids = torch.topk(base_probs, 5)
                    top5_alts = [
                        {
                            "rank": r + 1,
                            "token_id": int(tid.item()),
                            "token_str": self.tokenizer.decode([int(tid.item())], skip_special_tokens=False, errors="replace"),
                            "probability": round(float(v.item()), 4),
                        }
                        for r, (v, tid) in enumerate(zip(top5_vals, top5_ids))
                    ]
                    tok_text = self.tokenizer.decode([next_token], skip_special_tokens=False, errors="replace")
                    raw_diagnostics_list.append(
                        RawTokenDiagnostic(
                            step=step_idx + 1,
                            token_id=next_token,
                            token_text=tok_text,
                            probability=round(token_prob, 4),
                            rank=rank,
                            top5_alternatives=top5_alts,
                        )
                    )

                generated_tokens.append(next_token)
                curr_tokens.append(next_token)

                if next_token == 1 and len(generated_tokens) >= self.gen_config.min_new_tokens:
                    break

        elapsed_sec = time.perf_counter() - t0
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True, errors="replace").strip()
        metrics = GenerationMetricsCalculator.calculate(generated_tokens, elapsed_sec)

        # Update multi-turn history
        user_tok_count = len(self.tokenizer.encode(user_input, add_bos=False, add_eos=False))
        self.session.add_turn(ConversationRole.USER, user_input, user_tok_count)
        self.session.add_turn(ConversationRole.ASSISTANT, response_text, len(generated_tokens))

        # Update session stats
        self.total_prompts += 1
        self.total_generated_tokens += len(generated_tokens)
        self.total_latencies.append(elapsed_sec * 1000.0)
        if metrics.eos_occurred:
            self.total_eos_count += 1
        self.unigram_repetitions.append(metrics.repetition_unigram)

        # Record to history log
        self.history.append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "user_input": user_input,
            "response": response_text,
            "token_ids": generated_tokens,
            "metrics": metrics.to_dict(),
            "raw_diagnostics": [d.to_dict() for d in raw_diagnostics_list],
        })

        return response_text, metrics, raw_diagnostics_list

    def compare_prompt(self, prompt: str) -> ComparisonResult:
        """Compare trained model against frozen baseline on prompt."""
        return self.comparator.compare(
            prompt=prompt,
            gen_config=self.gen_config,
            seed=self.seed if self.deterministic else None,
        )

    def run_probes(self) -> Dict[str, Any]:
        """Execute full structured probe suite."""
        return self.probe_evaluator.run_all_probes()

    def save_session_log(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save structured session log to artifacts/step50/sessions/."""
        SESSIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        if filepath is None:
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            save_path = SESSIONS_LOG_DIR / f"session_{ts}.json"
        else:
            save_path = Path(filepath)

        payload = {
            "session_id": self.session.session_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_info": self.get_info(),
            "generation_config": self.gen_config.to_dict(),
            "session_stats": self.get_session_stats(),
            "history": self.history,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return save_path
