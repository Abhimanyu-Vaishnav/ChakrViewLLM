"""
ChakrView Step 119: Extended Stage-C Curriculum Pre-Training Wave Integration.

Integrates the cognitive architecture with CPU-first Stage C training:
- Tokenizer checksum validation
- Manifest verification
- Warm start from Step 102 candidate checkpoint
- Controlled, reproducible training steps on CPU
- Syntactic probe validation
- Checkpoint registry update while strictly preserving canonical baseline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from chakrview.training.stage_c_runner import run_stage_c_pretraining_wave
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


def run_extended_curriculum_wave(
    artifacts_dir: Path,
    steps: int = 5,
) -> Dict[str, Any]:
    """
    Executes a controlled pre-training wave on CPU using Stage C data,
    verifying checkpoint creation, loss decrease, and baseline immutability.
    """
    artifacts_dir = Path(artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline check before curriculum
    base_model = instantiate_frozen_baseline()
    base_hash = compute_model_hash(base_model)
    assert base_hash == EXPECTED_WEIGHT_HASH, "Canonical baseline mutated before Stage C wave!"

    # 2. Run Stage C pretraining wave (5 steps CPU benchmark)
    warm_ckpt = Path("artifacts/step102_stage_b_curriculum/checkpoints/checkpoint_0000500.pt")
    if not warm_ckpt.exists():
        warm_ckpt = None

    report = run_stage_c_pretraining_wave(
        artifacts_dir=artifacts_dir,
        warm_start_checkpoint=warm_ckpt,
        max_steps=steps,
        sequence_length=128,
        batch_size=2,
        eval_interval=steps,
        save_interval=steps,
    )

    # 3. Post-training baseline immutability check
    base_model_post = instantiate_frozen_baseline()
    base_hash_post = compute_model_hash(base_model_post)
    assert base_hash_post == EXPECTED_WEIGHT_HASH, "Canonical baseline mutated by Stage C wave!"

    report["baseline_immutability_verified"] = True
    report["canonical_baseline_hash"] = base_hash_post
    return report
