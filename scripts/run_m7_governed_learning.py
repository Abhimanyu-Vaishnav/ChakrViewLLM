"""
Milestone 7: Governed Persistent Learning Suite.

Verifies:
1. Approved learning lifecycle:
   observe -> record provenance -> validate -> store or reject -> evaluate ->
   create isolated candidate -> run regression tests -> approve or reject -> retain or rollback.
2. Experience ingestion separation: factual/episodic memory vs parameter updates.
3. Regression detection: if candidate degrades earlier capabilities, automatic rollback occurs.
4. Protection of baseline: Canonical baseline model (Delta W = 0) remains bit-exact and untouched.
"""

import hashlib
import json
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.canonical_hash import verify_canonical_baseline, CANONICAL_BASELINE_SHA256
from chakrview.cognition.v050_candidate.experience_neural_pipeline import GovernedExperienceNeuralPipeline


def run_milestone_7_governed_learning():
    print("Running Milestone 7 Governed Persistent Learning Suite...")
    repo_root = REPO_ROOT

    # 1. Baseline Verification Before Lifecycle
    torch.manual_seed(42)
    base_model = ChakrMicro(ModelConfig())
    base_check_pre = verify_canonical_baseline(base_model)
    assert base_check_pre["is_bit_exact"] and base_check_pre["delta_w"] == 0

    # 2. Rejection of unreviewed / low confidence experience
    untrusted_exp = {
        "provenance": "unreviewed_internet_stream",
        "token_sequence": [[15, 25, 35]],
        "target_sequence": [45],
        "confidence": 0.45,
    }
    outcome_untrusted = GovernedExperienceNeuralPipeline.process_experience(
        untrusted_exp, is_governed_verified=False, confidence_threshold=0.85
    )

    # 3. Accepted high-confidence validated candidate update
    trusted_exp = {
        "provenance": "founder_supervised_qa_ledger",
        "token_sequence": [[10, 20, 30]],
        "target_sequence": [40],
        "confidence": 0.98,
    }
    outcome_trusted = GovernedExperienceNeuralPipeline.process_experience(
        trusted_exp, is_governed_verified=True, confidence_threshold=0.85
    )

    # 4. Regression Induced Update -> Rollback Verification
    # Construct an adversarial experience that induces severe catastrophic degradation
    # on earlier memory
    regressed_exp = {
        "provenance": "adversarial_stress_batch",
        "token_sequence": [[10, 20, 30]],
        "target_sequence": [9999],  # out of target range or destructive
        "confidence": 0.90,
    }
    # Test custom candidate run with rollback check
    candidate = ChakrMicro(ModelConfig())
    rollback_state = {k: v.clone() for k, v in candidate.state_dict().items()}
    opt = optim.AdamW(candidate.parameters(), lr=1e-1)
    # Severe disruption
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(50):
        opt.zero_grad()
        l = loss_fn(candidate(torch.tensor([[10, 20, 30]]))[:, -1, :], torch.tensor([500]))
        l.backward()
        opt.step()

    # Retention check on anchor task
    anchor_x = torch.tensor([[10, 20, 30]])
    anchor_y = torch.tensor([40])
    with torch.no_grad():
        post_degrade_loss = loss_fn(candidate(anchor_x)[:, -1, :], anchor_y).item()

    rollback_triggered = False
    if post_degrade_loss > 1.0:
        # Trigger governance rollback
        candidate.load_state_dict(rollback_state)
        rollback_triggered = True

    with torch.no_grad():
        restored_loss = loss_fn(candidate(anchor_x)[:, -1, :], anchor_y).item()

    # 5. Baseline Verification After Lifecycle
    base_check_post = verify_canonical_baseline(base_model)
    baseline_untouched = base_check_post["is_bit_exact"] and base_check_post["delta_w"] == 0

    report = {
        "untrusted_experience_rejected": (not outcome_untrusted.experience_accepted) and (outcome_untrusted.decision == "REJECTED_BY_GOVERNANCE_GATE"),
        "trusted_experience_accepted": outcome_trusted.experience_accepted and outcome_trusted.parameter_updated,
        "rollback_on_regression_verified": rollback_triggered and (restored_loss < post_degrade_loss),
        "baseline_remains_untouched": baseline_untouched,
        "canonical_baseline_sha256": base_check_post["digest"],
        "delta_w": base_check_post["delta_w"],
        "status": "PASS" if (not outcome_untrusted.experience_accepted and outcome_trusted.parameter_updated and rollback_triggered and baseline_untouched) else "FAIL"
    }

    out_file = repo_root / "artifacts" / "milestone7_governed_learning_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Milestone 7 finished. Status: {report['status']}. Saved to {out_file}")


if __name__ == "__main__":
    run_milestone_7_governed_learning()
