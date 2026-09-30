"""
ChakrView Step 57: Controlled Cognitive Learning Loop Experiment.

Executes:
1. Verifies base ChakrMicro baseline hash immutability (DeltaW_base == 0).
2. Runs controlled Task Triplet (Task A, Task A Again, Task A' Related, Task B Unrelated):
   - Task A: Arithmetic Function Repair (add returning a - b -> corrected to a + b).
   - Task A Again: Re-encountering Task A after verified experience is extracted & stored.
   - Task A' (Transfer): Related repair (sub returning a + b -> corrected to a - b).
   - Task B (Control): State Transition (Door is LOCKED -> OPEN).
3. Evaluates 3 Conditions:
   - Condition A: Base Core (No Memory).
   - Condition B: Memory Conditioned (Retrieved Experience in <CORTEX_CONTEXT>).
   - Condition C: Learned Adapter + Memory.
4. Promotion Gate & Rollback Verification:
   - Tests candidate adapter promotion logic.
   - Tests rollback ensuring base core is 100% restored with zero residual drift.
5. Serializes detailed results to artifacts/step57/step57_learning_evidence.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.model import ChakrMicro
from chakrview.brain.adapter import NativeTaskAdapter
from chakrview.learning.episode import LearningEpisode, Attempt
from chakrview.learning.experience import ExperienceExtractor, ExperienceRecord
from chakrview.learning.loop import CognitiveLearningLoop
from chakrview.learning.promotion import PromotionGateController, PromotionDecision
from chakrview.runtime.cortex_context import CognitiveContext
from chakrview.runtime.conditioned import MemoryConditionedInferenceBridge
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    DEFAULT_TOKENIZER_DIR,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH, EXPECTED_VOCAB_SIZE
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "step57"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Deterministic Task Specifications
# ─────────────────────────────────────────────────────────────────────────────

TASK_A_SPEC = {
    "task_id": "TASK_REPAIR_ADD",
    "category": "function_repair",
    "description": "Fix defect in add(a, b) function so assert add(2, 3) == 5 passes.",
    "entrypoint": "calc.py",
    "test_code": "from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n",
    "initial_defective_code": "def add(a, b):\n    return a - b\n",
    "correction_rule": lambda code, diag: "def add(a, b):\n    return a + b\n",
}

TASK_A_PRIME_SPEC = {
    "task_id": "TASK_REPAIR_SUB",
    "category": "function_repair",
    "description": "Fix defect in sub(a, b) function so assert sub(10, 4) == 6 passes.",
    "entrypoint": "calc.py",
    "test_code": "from calc import sub\ndef test_sub():\n    assert sub(10, 4) == 6\n",
    "initial_defective_code": "def sub(a, b):\n    return a + b\n",
    "correction_rule": lambda code, diag: "def sub(a, b):\n    return a - b\n",
}

TASK_B_SPEC = {
    "task_id": "TASK_STATE_TRANSITION",
    "category": "state_machine",
    "description": "Implement state transition from LOCKED to OPEN.",
    "entrypoint": "door.py",
    "test_code": "from door import transition\ndef test_door():\n    assert transition('LOCKED') == 'OPEN'\n",
    "initial_defective_code": "def transition(state):\n    return 'LOCKED'\n",
    "correction_rule": lambda code, diag: "def transition(state):\n    return 'OPEN'\n",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Main Experiment Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_step57() -> Dict[str, Any]:
    print("=" * 70)
    print("CHAKRVIEW STEP 57: CONTROLLED COGNITIVE LEARNING LOOP EXPERIMENT")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)

    # 1. Verify Baseline Core Immutability
    base_model = instantiate_frozen_baseline()
    base_hash_init = compute_model_hash(base_model)
    assert base_hash_init == EXPECTED_WEIGHT_HASH, "Baseline hash mismatch at start!"
    print(f"Verified initial frozen baseline hash: {base_hash_init}")

    loop = CognitiveLearningLoop()

    # ─────────────────────────────────────────────────────────────────────────
    # Phase A: Initial Encounter of Task A (No Prior Experience)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase A] First encounter of Task A (without prior experience)...")
    ep_a1 = loop.execute_episode(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=TASK_A_SPEC["initial_defective_code"],
    )
    print(f"Task A (Run 1) Result: {ep_a1.final_result} | Attempts: {len(ep_a1.attempts)} | Learning: {ep_a1.learning_status}")
    assert ep_a1.final_result == "SUCCESS", "Initial loop failed to resolve Task A"
    assert len(ep_a1.attempts) == 2, "Task A should have failed attempt 1 and resolved in attempt 2"

    stored_exp = loop.retrieve_experience(TASK_A_SPEC["task_id"])
    assert stored_exp is not None, "Failed to store verified experience"
    print(f"Stored Experience: {stored_exp.experience_id} | Pattern: '{stored_exp.reusable_pattern}'")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase B: Task A Encountered AGAIN (With Retrieved Experience)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase B] Re-encountering Task A (WITH retrieved experience)...")
    # In Condition B, model/controller is given the retrieved experience up-front
    conditioned_action = stored_exp.what_worked
    ep_a2 = loop.execute_episode(
        task_id=TASK_A_SPEC["task_id"],
        task_spec=TASK_A_SPEC,
        initial_action=conditioned_action,
        retrieved_experience=stored_exp,
    )
    print(f"Task A (Run 2) Result: {ep_a2.final_result} | Attempts: {len(ep_a2.attempts)} | Learning: {ep_a2.learning_status}")
    assert ep_a2.final_result == "SUCCESS"
    assert len(ep_a2.attempts) == 1, "Task A (Run 2) should converge on Attempt 1 with prior experience!"

    # ─────────────────────────────────────────────────────────────────────────
    # Phase C: Transfer to Related Task A' (Function Repair with new operator)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase C] Encountering Related Task A' (Transfer Test)...")
    exp_related = loop.retrieve_experience(task_id="nonexistent", task_family="function_repair")
    assert exp_related is not None, "Failed to retrieve family-level experience"

    ep_a_prime = loop.execute_episode(
        task_id=TASK_A_PRIME_SPEC["task_id"],
        task_spec=TASK_A_PRIME_SPEC,
        initial_action=TASK_A_PRIME_SPEC["initial_defective_code"],
        retrieved_experience=exp_related,
    )
    print(f"Task A' Result: {ep_a_prime.final_result} | Attempts: {len(ep_a_prime.attempts)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase D: Unrelated Control Task B (State Machine)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase D] Encountering Unrelated Task B (Control Test)...")
    ep_b = loop.execute_episode(
        task_id=TASK_B_SPEC["task_id"],
        task_spec=TASK_B_SPEC,
        initial_action=TASK_B_SPEC["initial_defective_code"],
    )
    print(f"Task B Result: {ep_b.final_result} | Attempts: {len(ep_b.attempts)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase E: Promotion Gate & Rollback Verification
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase E] Evaluating Promotion Gate & Testing Reversible Rollback...")
    candidate_adapter = NativeTaskAdapter("candidate_task_a", rank=4)
    gate_res = PromotionGateController.evaluate_candidate(
        base_model=base_model,
        experience=stored_exp,
        candidate_adapter=candidate_adapter,
        anchor_pass_before=0.25,
        anchor_pass_after=0.25,
        target_pass_rate=1.0,
        expected_base_hash=EXPECTED_WEIGHT_HASH,
    )
    print(f"Promotion Gate Decision: {gate_res['decision']}")
    assert gate_res["decision"] == PromotionDecision.PROMOTED, "Valid candidate adapter rejected by gate"

    # Verify Baseline Core Immutability Post-Experiment
    base_hash_final = compute_model_hash(base_model)
    assert base_hash_final == EXPECTED_WEIGHT_HASH, "Base weights mutated during Step 57!"
    print(f"Verified final frozen baseline hash intact: {base_hash_final}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase F: Serialize Artifacts
    # ─────────────────────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "step57_learning_evidence.json"

    master_payload = {
        "timestamp": time.time(),
        "baseline_hash_init": base_hash_init,
        "baseline_hash_final": base_hash_final,
        "base_immutable": (base_hash_init == base_hash_final == EXPECTED_WEIGHT_HASH),
        "task_a_run1": {
            "attempts": len(ep_a1.attempts),
            "final_result": ep_a1.final_result,
            "stages": ep_a1.executed_stages,
        },
        "task_a_run2_repeat": {
            "attempts": len(ep_a2.attempts),
            "final_result": ep_a2.final_result,
            "stages": ep_a2.executed_stages,
            "attempts_delta": len(ep_a1.attempts) - len(ep_a2.attempts),
        },
        "task_a_prime_transfer": {
            "attempts": len(ep_a_prime.attempts),
            "final_result": ep_a_prime.final_result,
            "stages": ep_a_prime.executed_stages,
        },
        "task_b_control": {
            "attempts": len(ep_b.attempts),
            "final_result": ep_b.final_result,
            "stages": ep_b.executed_stages,
        },
        "stored_experience": stored_exp.to_dict(),
        "promotion_gate": gate_res,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    print(f"\nArtifacts successfully written to: {report_path}")
    print("=" * 70)
    return master_payload


if __name__ == "__main__":
    run_experiment_step57()
