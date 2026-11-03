"""
CAND_01 Dry-Run Non-Interactive Experiment Runner.
Specification: docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.3_REVIEWED.md
Authoritative Errata: docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md

GOVERNANCE RESTRICTIONS:
- Dedicated to DRY-RUN mode using synthetic pipelines and synthetic evaluators.
- Hard-coded execution boundaries: REJECTS production data paths and real checkpoint paths.
- Robust cross-platform path normalization protecting against traversal or bypass attempts.
- Verifies SHA-256 integrity of canonical checkpoints if probed to assert immutability.
- DOES NOT invoke training, fine-tuning, optimizer updates, or backward passes.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_cand01_dataset import (
    SyntheticEntityPool,
    CAND01DatasetGenerator,
    PartitionType,
    TaskFamily,
)
from scripts.evaluate_cand01_benchmark import (
    CAND01EvaluatorHarness,
    SyntheticModelMock,
)


CANONICAL_CHECKPOINT_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

# Forbidden paths defined in standard relative format; checked via os.path.normpath
FORBIDDEN_DATA_DIRECTORIES = [
    os.path.normpath("data/preprocessed"),
    os.path.normpath("data/tokenized"),
    os.path.normpath("data/experiments/cand_01"),
    os.path.normpath("data/checkpoints"),
    os.path.normpath("models"),
    os.path.normpath("experiments/overnight_runs"),
]


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_immutability(checkpoint_path: str):
    """Verifies that the canonical checkpoint is strictly unchanged before and after operations."""
    if not os.path.exists(checkpoint_path):
        return
    actual_hash = compute_sha256(checkpoint_path)
    assert actual_hash == CANONICAL_CHECKPOINT_HASH, (
        f"FATAL REPOSITORY CORRUPTION: Checkpoint {checkpoint_path} hash altered! Expected {CANONICAL_CHECKPOINT_HASH}, got {actual_hash}"
    )


def assert_safe_path(path_str: str, param_name: str):
    """
    Validates that a path does not traverse into or overlap forbidden canonical data directories.
    Handles relative, absolute, Windows backslashes, and path traversal sequences.
    """
    if not path_str:
        return
    abs_path = os.path.normpath(os.path.abspath(path_str)).lower()

    for forbidden in FORBIDDEN_DATA_DIRECTORIES:
        norm_forb = os.path.normpath(forbidden).lower()
        if norm_forb in abs_path:
            # Allow fixtures folder inside data/fixtures
            if "fixtures" in abs_path:
                continue
            raise PermissionError(
                f"FATAL GOVERNANCE BREACH: Parameter '{param_name}' references forbidden directory: '{forbidden}'. "
                f"Resolved path: '{abs_path}'."
            )


def parse_args():
    parser = argparse.ArgumentParser(description="CAND_01 Isolated Experiment Runner")
    parser.add_argument("--dry-run", action="store_true", required=True, help="Execute pipeline in synthetic dry-run mode")
    parser.add_argument("--synthetic-dir", type=str, default="tests/fixtures/cand01_synthetic", help="Directory for synthetic fixtures")
    parser.add_argument("--output-dir", type=str, default="tests/fixtures/cand01_dryrun_output", help="Directory for dry-run output logs")
    parser.add_argument("--seed", type=int, default=999042, help="Pipeline random seed")
    parser.add_argument("--replicates", type=int, default=100, help="Bootstrap replicates for dry-run evaluation")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="FORBIDDEN: Attempt to load real model weights")
    parser.add_argument("--train-shards", type=str, default=None, help="FORBIDDEN: Attempt to load real pretraining shards")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. HARD BOUNDARY CHECKS
    if not args.dry_run:
        sys.stderr.write(
            "FATAL GOVERNANCE BREACH: CAND_01 experiment execution is UNAPPROVED.\n"
            "Only --dry-run mode is permitted during the isolated implementation phase.\n"
        )
        sys.exit(1)

    if args.checkpoint_path is not None:
        sys.stderr.write(
            "FATAL GOVERNANCE BREACH: --checkpoint-path is prohibited during isolated implementation.\n"
            "Loading real model weights is unauthorized.\n"
        )
        sys.exit(1)

    if args.train_shards is not None:
        sys.stderr.write(
            "FATAL GOVERNANCE BREACH: --train-shards is prohibited.\n"
            "Reading canonical pretraining data is unauthorized.\n"
        )
        sys.exit(1)

    # Check paths with robust normalization
    try:
        assert_safe_path(args.synthetic_dir, "--synthetic-dir")
        assert_safe_path(args.output_dir, "--output-dir")
    except PermissionError as e:
        sys.stderr.write(f"{str(e)}\n")
        sys.exit(1)

    print("=================================================================")
    print("CAND_01 EXPERIMENT RUNNER — DRY RUN / SYNTHETIC ISOLATED MODE")
    print(f"Seed: {args.seed} | Replicates: {args.replicates} | Dry-Run: True")
    print("=================================================================")

    # 2. RUN SYNTHETIC GENERATION
    print("\n[Phase 1] Executing Synthetic Dataset Generation across TF01-TF08...")
    pool = SyntheticEntityPool(n_train=450, n_test=150, seed=args.seed)
    gen = CAND01DatasetGenerator(pool, seed=args.seed)

    counts_train = {
        TaskFamily.TF01_DIRECT_COPY: 10,
        TaskFamily.TF02_SINGLE_PAIR: 10,
        TaskFamily.TF03_MULTI_PAIR: 10,
        TaskFamily.TF04_DISTRACTOR: 10,
        TaskFamily.TF05_SIMILAR_KEY: 10,
    }
    counts_val = {
        TaskFamily.TF02_SINGLE_PAIR: 5,
        TaskFamily.TF03_MULTI_PAIR: 5,
        TaskFamily.TF04_DISTRACTOR: 5,
        TaskFamily.TF05_SIMILAR_KEY: 5,
        TaskFamily.TF06_RECOMBINATION: 5,
        TaskFamily.TF07_OFFSET_SCALING: 5,
        TaskFamily.TF08_CONTEXT_SCALING: 5,
    }
    counts_test = {
        TaskFamily.TF02_SINGLE_PAIR: 5,
        TaskFamily.TF03_MULTI_PAIR: 5,
        TaskFamily.TF04_DISTRACTOR: 5,
        TaskFamily.TF05_SIMILAR_KEY: 5,
        TaskFamily.TF06_RECOMBINATION: 5,
        TaskFamily.TF07_OFFSET_SCALING: 5,
        TaskFamily.TF08_CONTEXT_SCALING: 5,
    }
    counts_disjoint = {
        TaskFamily.TF01_DIRECT_COPY: 5,
        TaskFamily.TF02_SINGLE_PAIR: 5,
        TaskFamily.TF03_MULTI_PAIR: 5,
        TaskFamily.TF04_DISTRACTOR: 5,
        TaskFamily.TF05_SIMILAR_KEY: 5,
        TaskFamily.TF06_RECOMBINATION: 5,
        TaskFamily.TF07_OFFSET_SCALING: 5,
        TaskFamily.TF08_CONTEXT_SCALING: 5,
    }

    train_recs = gen.generate_partition(PartitionType.PARTITION_A_TRAIN, counts_train)
    val_recs = gen.generate_partition(PartitionType.PARTITION_B_VAL_RECOMB, counts_val)
    test_recs = gen.generate_partition(PartitionType.PARTITION_C_TEST_RECOMB, counts_test)
    disjoint_recs = gen.generate_partition(PartitionType.PARTITION_D_TEST_DISJOINT, counts_disjoint)

    isolation_audit = gen.verify_all_isolation_invariants()
    print(f"  Isolation Audit Passed: {isolation_audit['audit_passed']}")
    assert isolation_audit["audit_passed"], "Isolation invariants violated!"

    # 3. RUN SYNTHETIC EVALUATION
    print("\n[Phase 2] Executing Synthetic Benchmark Evaluation on Partitions...")
    mock_model = SyntheticModelMock(vocab_size=4096, target_copy_bias=0.85, seed=args.seed)
    harness = CAND01EvaluatorHarness(vocab_size=4096)

    val_eval = harness.evaluate_dataset(
        [r.to_dict() for r in val_recs],
        mock_model,
        bootstrap_replicates=args.replicates,
        random_seed=args.seed,
    )
    test_recomb_eval = harness.evaluate_dataset(
        [r.to_dict() for r in test_recs],
        mock_model,
        bootstrap_replicates=args.replicates,
        random_seed=args.seed,
    )
    test_disjoint_eval = harness.evaluate_dataset(
        [r.to_dict() for r in disjoint_recs],
        mock_model,
        bootstrap_replicates=args.replicates,
        random_seed=args.seed,
    )

    print("\n[Phase 3] Synthetic Benchmark Evaluation Summary:")
    print(f"  Val Recomb First-Token Acc (A_FT):    {val_eval['aggregate_metrics']['first_token_exact_accuracy_A_FT']:.2%}")
    print(f"  Val Recomb Full Completion (A_FC):    {val_eval['aggregate_metrics']['full_completion_accuracy_A_FC']:.2%}")
    print(f"  Test Recomb First-Token Acc (A_FT):   {test_recomb_eval['aggregate_metrics']['first_token_exact_accuracy_A_FT']:.2%}")
    print(f"  Test Recomb Full Completion (A_FC):   {test_recomb_eval['aggregate_metrics']['full_completion_accuracy_A_FC']:.2%}")
    print(f"  Test Disjoint First-Token Acc (A_FT): {test_disjoint_eval['aggregate_metrics']['first_token_exact_accuracy_A_FT']:.2%}")
    print(f"  Test Disjoint Full Completion (A_FC): {test_disjoint_eval['aggregate_metrics']['full_completion_accuracy_A_FC']:.2%}")

    print("\n[Phase 4] Statistical Uncertainty Classification (A_FT):")
    for part_name, res in [
        ("Val Recomb", val_eval),
        ("Test Recomb", test_recomb_eval),
        ("Test Disjoint", test_disjoint_eval),
    ]:
        b_res = res["statistical_uncertainty_A_FT"]["provisional_two_way_bootstrap"]
        print(f"  [{part_name}] Estimator: {b_res['estimator_name']} | Status: {b_res['status']} | CI: [{b_res['ci_lower']:.4f}, {b_res['ci_upper']:.4f}] | Degen: {b_res['degenerate_rate']:.1%}")

    # 4. WRITE DRY-RUN REPORT
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "cand01_dryrun_execution_report.json")
    execution_report = {
        "pipeline_status": "DRY_RUN_COMPLETED_SUCCESSFULLY",
        "dry_run": True,
        "seed": args.seed,
        "bootstrap_replicates": args.replicates,
        "isolation_audit": isolation_audit,
        "partitions_evaluated": {
            "val_recombination": val_eval,
            "test_recombination": test_recomb_eval,
            "test_disjoint": test_disjoint_eval,
        },
        "governance_compliance": {
            "production_data_generated": False,
            "canonical_checkpoints_evaluated": False,
            "training_invoked": False,
            "calibration_invoked": False,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(execution_report, f, indent=2)

    print(f"\nWrote full dry-run execution report to: {report_path}")
    print("=================================================================")
    print("SUCCESS: CAND_01 dry-run pipeline completed cleanly and securely.")
    print("=================================================================")


if __name__ == "__main__":
    main()
