"""
CAND_01 Benchmark Evaluation Harness.
Specification: docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.3_REVIEWED.md
Authoritative Errata: docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md

GOVERNANCE RESTRICTIONS:
- Strictly operates on SYNTHETIC logits and synthetic fixtures in this phase.
- DOES NOT load or evaluate canonical model checkpoints.
- Clearly distinguishes between teacher-forced first-token accuracy (A_FT) and
  free autoregressive generation completion with clean termination (A_FC).
- Calculates exact target accuracy, target loss, target perplexity, and baseline probability.
- Integrates with chakrview.eval.statistical_bootstrap for uncertainty estimation.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chakrview.eval.statistical_bootstrap import (
    ordinary_nonparametric_bootstrap,
    one_way_clustered_bootstrap,
    provisional_two_way_cluster_bootstrap,
    BootstrapResult,
    EstimatorStatus,
    DEFAULT_NUM_BOOTSTRAP_REPLICATES,
    DEFAULT_CONFIDENCE_LEVEL,
)

EOS_TOKEN_ID = 1  # Standard <EOS> ID per tokenizer schema


@dataclass
class SampleEvaluationResult:
    """Evaluation result for an individual sample record."""
    record_id: str
    task_family: str
    partition: str
    query_key_id: int
    target_val_id: int
    target_token_str: str

    # Teacher-Forced Diagnostic Metrics
    target_logit: float
    target_log_prob: float
    target_loss: float
    target_perplexity: float
    target_rank: int
    prior_unconditional_prob: float
    contextual_gain: float

    # Autoregressive Generation Metrics
    generated_token_ids: List[int]
    top1_predicted_id: int
    top1_predicted_str: str
    is_first_token_match: bool  # A_FT (Primary Metric per Evaluator Spec v1.3 §3.1)
    is_full_completion_match: bool  # A_FC (Co-Primary Metric per Evaluator Spec v1.3 §3.2)
    evaluation_mode: str = "SYNTHETIC_MOCK_ONLY"
    attention_allocation: Optional[float] = None  # Optional interface; None in standard logit mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_family": self.task_family,
            "partition": self.partition,
            "query_key_id": self.query_key_id,
            "target_val_id": self.target_val_id,
            "target_token_str": self.target_token_str,
            "target_logit": float(self.target_logit),
            "target_log_prob": float(self.target_log_prob),
            "target_loss": float(self.target_loss),
            "target_perplexity": float(self.target_perplexity),
            "target_rank": int(self.target_rank),
            "prior_unconditional_prob": float(self.prior_unconditional_prob),
            "contextual_gain": float(self.contextual_gain),
            "generated_token_ids": [int(t) for t in self.generated_token_ids],
            "top1_predicted_id": self.top1_predicted_id,
            "top1_predicted_str": self.top1_predicted_str,
            "is_first_token_match": self.is_first_token_match,
            "is_full_completion_match": self.is_full_completion_match,
            "evaluation_mode": self.evaluation_mode,
            "attention_allocation": self.attention_allocation,
        }


class SyntheticModelMock:
    """
    Mock inference engine that returns simulated logits and autoregressive tokens.
    Allows testing evaluation harness, loss calculations, exact match logic,
    and statistical bootstrap integration without loading neural model weights.
    """
    def __init__(
        self,
        vocab_size: int = 4096,
        target_copy_bias: float = 0.85,
        clean_termination_rate: float = 0.90,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.target_copy_bias = target_copy_bias
        self.clean_termination_rate = clean_termination_rate
        self.rng = np.random.default_rng(seed)

    def predict_target_logits(
        self,
        target_id: int,
        context_ids: List[int],
    ) -> np.ndarray:
        """
        Generates simulated logit vector at the target position.
        With probability `target_copy_bias`, the correct target token receives the top logit.
        """
        logits = self.rng.normal(0, 1.0, size=self.vocab_size)
        if self.rng.random() < self.target_copy_bias:
            logits[target_id] = np.max(logits) + self.rng.uniform(1.0, 3.0)
        else:
            # Distractor or random error
            distractor_id = self.rng.choice(context_ids) if context_ids else self.rng.integers(0, self.vocab_size)
            logits[distractor_id] = np.max(logits) + self.rng.uniform(1.0, 3.0)
        return logits

    def generate_completion(
        self,
        first_token_id: int,
        target_id: int,
    ) -> List[int]:
        """
        Simulates autoregressive sequence emission starting from first_token_id.
        Per Evaluator Spec v1.3 §2.2, clean completion is [target_token_id, EOS_TOKEN_ID].
        """
        tokens = [int(first_token_id)]
        if first_token_id == target_id:
            if self.rng.random() < self.clean_termination_rate:
                tokens.append(int(EOS_TOKEN_ID))  # Clean termination
            else:
                # Syntax echo / trailing repetition failure
                tokens.extend([int(self.rng.integers(10, 50)), int(EOS_TOKEN_ID)])
        else:
            # Hallucination / failure continuation
            tokens.extend([int(self.rng.integers(10, 50)), int(EOS_TOKEN_ID)])
        return tokens


class CAND01EvaluatorHarness:
    """
    Evaluation harness for CAND_01 datasets.
    Operates on JSONL fixtures, computes metrics, and runs statistical bootstrapping.
    """
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size

    def evaluate_sample(
        self,
        record: Dict[str, Any],
        logits: np.ndarray,
        generated_tokens: List[int],
        unconditional_prior_prob: float = 1.0 / 4096.0,
    ) -> SampleEvaluationResult:
        """
        Computes sample-level metrics decoupling teacher-forced logits from autoregressive metrics.
        """
        assert logits.ndim == 1 and len(logits) == self.vocab_size, (
            f"Expected 1D logit vector of size {self.vocab_size}, got {logits.shape}"
        )

        target_id = record["target_token_id"]
        query_key_id = record["query_key_id"]
        target_str = record["target_token"]

        # Numerically stable log-softmax
        max_logit = np.max(logits)
        log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
        log_probs = logits - log_sum_exp

        # Pre-softmax rank of target token (Evaluator Spec v1.3 §3.3)
        target_logit_val = logits[target_id]
        target_rank = 1 + int(np.sum(logits > target_logit_val))

        target_log_prob = float(log_probs[target_id])
        target_prob = float(np.exp(target_log_prob))
        target_loss = float(-target_log_prob)
        target_ppl = float(np.exp(min(20.0, target_loss)))  # Clip exponent to avoid overflow

        contextual_gain = target_prob - unconditional_prior_prob

        # Autoregressive generation metrics:
        # A_FT: generated_tokens[0] == target_token_id
        # A_FC: generated_tokens[0] == target_token_id and generated_tokens[1] == EOS and len == 2
        first_token = generated_tokens[0] if len(generated_tokens) > 0 else -1
        is_first_match = bool(first_token == target_id)

        is_full_match = bool(
            len(generated_tokens) == 2 and
            generated_tokens[0] == target_id and
            generated_tokens[1] == EOS_TOKEN_ID
        )

        return SampleEvaluationResult(
            record_id=record["record_id"],
            task_family=record["task_family"],
            partition=record["partition"],
            query_key_id=query_key_id,
            target_val_id=target_id,
            target_token_str=target_str,
            target_logit=float(target_logit_val),
            target_log_prob=target_log_prob,
            target_loss=target_loss,
            target_perplexity=target_ppl,
            target_rank=target_rank,
            prior_unconditional_prob=unconditional_prior_prob,
            contextual_gain=contextual_gain,
            generated_token_ids=generated_tokens,
            top1_predicted_id=first_token,
            top1_predicted_str=f"pred_{first_token}",
            is_first_token_match=is_first_match,
            is_full_completion_match=is_full_match,
            evaluation_mode="SYNTHETIC_MOCK_ONLY",
            attention_allocation=None,
        )

    def evaluate_dataset(
        self,
        records: List[Dict[str, Any]],
        model_mock: SyntheticModelMock,
        bootstrap_replicates: int = DEFAULT_NUM_BOOTSTRAP_REPLICATES,
        random_seed: int = 999042,
    ) -> Dict[str, Any]:
        """
        Evaluates records, reporting A_FT, A_FC, loss, PPL, and statistical bootstraps.
        """
        if not records:
            return {
                "num_samples": 0,
                "aggregate_metrics": {
                    "first_token_exact_accuracy_A_FT": 0.0,
                    "full_completion_accuracy_A_FC": 0.0,
                    "mean_cross_entropy_loss": 0.0,
                    "mean_target_perplexity": 0.0,
                    "mean_target_rank": 0.0,
                },
                "statistical_uncertainty": {},
                "sample_results": [],
            }

        sample_results: List[SampleEvaluationResult] = []

        for rec in records:
            ctx_ids = []
            for k, v in rec.get("context_pair_ids", []):
                ctx_ids.extend([k, v])

            logits = model_mock.predict_target_logits(
                target_id=rec["target_token_id"],
                context_ids=ctx_ids,
            )
            top1_id = int(np.argmax(logits))
            gen_tokens = model_mock.generate_completion(
                first_token_id=top1_id,
                target_id=rec["target_token_id"],
            )

            sample_res = self.evaluate_sample(rec, logits, gen_tokens)
            sample_results.append(sample_res)

        a_ft_arr = np.array([1.0 if s.is_first_token_match else 0.0 for s in sample_results], dtype=float)
        a_fc_arr = np.array([1.0 if s.is_full_completion_match else 0.0 for s in sample_results], dtype=float)
        losses = np.array([s.target_loss for s in sample_results], dtype=float)
        ppls = np.array([s.target_perplexity for s in sample_results], dtype=float)
        ranks = np.array([s.target_rank for s in sample_results], dtype=float)
        key_ids = np.array([s.query_key_id for s in sample_results], dtype=int)
        val_ids = np.array([s.target_val_id for s in sample_results], dtype=int)

        # Bootstraps computed on primary metric A_FT
        ord_boot = ordinary_nonparametric_bootstrap(
            a_ft_arr,
            num_replicates=bootstrap_replicates,
            random_seed=random_seed,
        )
        oneway_boot = one_way_clustered_bootstrap(
            a_ft_arr,
            cluster_ids=val_ids,
            num_replicates=bootstrap_replicates,
            random_seed=random_seed,
        )
        twoway_boot = provisional_two_way_cluster_bootstrap(
            a_ft_arr,
            key_ids=key_ids,
            val_ids=val_ids,
            num_replicates=bootstrap_replicates,
            random_seed=random_seed,
        )

        return {
            "num_samples": len(sample_results),
            "aggregate_metrics": {
                "first_token_exact_accuracy_A_FT": float(np.mean(a_ft_arr)),
                "full_completion_accuracy_A_FC": float(np.mean(a_fc_arr)),
                "mean_cross_entropy_loss": float(np.mean(losses)),
                "mean_target_perplexity": float(np.mean(ppls)),
                "mean_target_rank": float(np.mean(ranks)),
            },
            "statistical_uncertainty_A_FT": {
                "ordinary_bootstrap": ord_boot.to_dict(),
                "one_way_clustered_bootstrap": oneway_boot.to_dict(),
                "provisional_two_way_bootstrap": twoway_boot.to_dict(),
            },
            "sample_results": [s.to_dict() for s in sample_results],
        }


def parse_args():
    parser = argparse.ArgumentParser(description="CAND_01 Benchmark Evaluator Harness")
    parser.add_argument("--fixture-file", type=str, required=True, help="Path to synthetic jsonl evaluation fixture")
    parser.add_argument("--output-report", type=str, default="evaluation_report.json", help="Path to write json evaluation report")
    parser.add_argument("--replicates", type=int, default=200, help="Number of bootstrap replicates")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="FORBIDDEN: Attempt to load real model weights")
    return parser.parse_args()


def main():
    args = parse_args()

    # Hard boundary check: Real checkpoints cannot be evaluated in this phase
    if args.checkpoint_path is not None or (args.checkpoint_path and os.path.exists(args.checkpoint_path)):
        sys.stderr.write(
            "FATAL GOVERNANCE BREACH: Canonical model checkpoints cannot be evaluated in isolated implementation phase.\n"
            "This harness is restricted to synthetic models and synthetic evaluation fixtures.\n"
        )
        sys.exit(1)

    if not os.path.exists(args.fixture_file):
        sys.stderr.write(f"Fixture file not found: {args.fixture_file}\n")
        sys.exit(1)

    print("=================================================================")
    print("CAND_01 SYNTHETIC EVALUATION HARNESS")
    print(f"Fixture: {args.fixture_file} | Bootstrap Replicates: {args.replicates}")
    print("=================================================================")

    records = []
    with open(args.fixture_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from fixture.")

    mock_model = SyntheticModelMock(vocab_size=4096, target_copy_bias=0.80, seed=42)
    harness = CAND01EvaluatorHarness(vocab_size=4096)

    results = harness.evaluate_dataset(
        records=records,
        model_mock=mock_model,
        bootstrap_replicates=args.replicates,
    )

    print("\nEvaluation Results:")
    print(f"  First-Token Accuracy (A_FT): {results['aggregate_metrics']['first_token_exact_accuracy_A_FT']:.2%}")
    print(f"  Full Completion (A_FC):      {results['aggregate_metrics']['full_completion_accuracy_A_FC']:.2%}")
    print(f"  Mean Target Loss:            {results['aggregate_metrics']['mean_cross_entropy_loss']:.4f}")
    print(f"  Mean Perplexity:             {results['aggregate_metrics']['mean_target_perplexity']:.2f}")
    print(f"  Mean Target Rank:            {results['aggregate_metrics']['mean_target_rank']:.2f}")

    print("\nBootstrap Uncertainty Diagnostics (A_FT):")
    for key, b_res in results["statistical_uncertainty_A_FT"].items():
        print(f"  [{key}]")
        print(f"    Status:     {b_res['status']}")
        print(f"    Point Est:  {b_res['point_estimate']:.4f}")
        print(f"    95% CI:     [{b_res['ci_lower']:.4f}, {b_res['ci_upper']:.4f}]")
        print(f"    Degen Rate: {b_res['degenerate_rate']:.2%}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote evaluation report to: {args.output_report}")
    print("SUCCESS: Evaluation harness completed cleanly in isolated synthetic mode.")


if __name__ == "__main__":
    main()
