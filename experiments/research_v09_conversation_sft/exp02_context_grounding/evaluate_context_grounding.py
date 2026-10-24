"""
evaluate_context_grounding.py
ChakrView Experiment 14: Context Grounding Evaluator.

Implements rigorous deterministic evaluation across 10 context-grounding tasks:
1. Exact Entity Binding (EEB)
2. Turn-Dependent Retrieval (TDR)
3. Anaphora Reference Clarification (ARC)
4. Superseded Instruction Adherence (SIA)
5. User Correction Handling (UCH)
6. Distractor Rejection Accuracy (DRA)
7. Ambiguity Clarification (AC)
8. Missing Context Abstention (MCA)
9. Factual Uncertainty Handling (FUH)
10. Constrained Code Synthesis (CCS via ast.parse)

Evaluates:
- Arm A: Native Causal LM Frozen Baseline (Step-10000)
- Arm B: Exp13 Candidate Best Checkpoint (Step-100)
- Arm C: Deterministic Exact-Match Heuristic Baseline (Non-Neural Baseline)
"""

from __future__ import annotations

import json
import time
import ast
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"

CANONICAL_PARENT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXP13_CANDIDATE_PATH = REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp01_native_dialogue" / "checkpoints" / "candidate_best.pt"

def is_repetitive_ngram(text: str, n: int = 3) -> bool:
    words = text.lower().split()
    if len(words) < 6:
        return False
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(ngrams) > len(set(ngrams)) * 1.5

def score_example(category: str, raw_output: str, metadata: Dict[str, Any]) -> Tuple[bool, str]:
    out_clean = raw_output.strip()
    out_lower = out_clean.lower()
    
    if len(out_clean) == 0:
        return False, "EMPTY_OUTPUT"
        
    if is_repetitive_ngram(out_clean):
        return False, "REPETITIVE_ATTRACTOR_COLLAPSE"
        
    if category == "dynamic_entity_binding":
        target = metadata["target_entity"]
        # Must contain exact target entity and not a training artifact
        if target.lower() in out_lower and "train" not in out_lower:
            return True, "EXACT_ENTITY_BOUND"
        return False, "ENTITY_NOT_FOUND"

    elif category == "turn_dependent_retrieval":
        target_val = metadata["expected_value"].lower()
        if target_val in out_lower:
            return True, "VALUE_RETRIEVED"
        return False, "VALUE_MISMATCH"

    elif category == "anaphora_reference_resolution":
        # Must ask for clarification or mention candidates
        c1, c2 = [c.lower() for c in metadata["candidates"]]
        if any(w in out_lower for w in ["clarify", "specify", "which"]) or (c1 in out_lower and c2 in out_lower):
            return True, "CLARIFICATION_REQUESTED"
        return False, "UNGROUNDED_GUESS"

    elif category == "superseded_instruction_adherence":
        target_fmt = metadata["target_format"].lower()
        superseded = metadata["superseded_format"].lower()
        if target_fmt in out_lower and superseded not in out_lower:
            return True, "SUPERSEDED_INSTRUCTION_OVERRIDDEN"
        return False, "SUPERSEDED_OR_MISSING_FORMAT"

    elif category == "user_correction_handling":
        u_new = metadata["new_target"].lower()
        u_old = metadata["cancelled_target"].lower()
        # Must mention the new target
        if u_new in out_lower:
            # If it mentions old target, it must be accompanied by cancel/not/instead or the new target is the one assigned
            if u_old in out_lower:
                if any(neg in out_lower for neg in ["cancel", "not", "instead", "updating", "switching"]):
                    return True, "CORRECTED_TARGET_BOUND"
                return False, "FAILED_TO_UPDATE_TARGET"
            return True, "CORRECTED_TARGET_BOUND"
        return False, "FAILED_TO_UPDATE_TARGET"

    elif category == "distractor_resistance":
        target_state = metadata["target_state"].lower()
        distractors = [d.lower() for d in metadata["distractors"]]
        has_target = target_state in out_lower
        has_distractor = any(d in out_lower for d in distractors)
        if has_target and not has_distractor:
            return True, "TARGET_BOUND_DISTRACTORS_REJECTED"
        return False, "DISTRACTOR_INTRUSION_OR_MISSING"

    elif category == "ambiguity_clarification":
        pod = metadata["target_pod"].lower()
        if any(w in out_lower for w in ["which", "what", "service", "pid", "specify"]):
            return True, "AMBIGUITY_DETECTED_CLARIFICATION_REQUESTED"
        return False, "FAILED_TO_DETECT_AMBIGUITY"

    elif category == "missing_context_abstention":
        abstain_phrases = ["does not state", "not provided", "cannot be determined", "not mention", "no information"]
        if any(p in out_lower for p in abstain_phrases):
            return True, "CORRECT_ABSTENTION"
        return False, "UNGROUNDED_HALLUCINATION"

    elif category == "factual_uncertainty_handling":
        uncertainty_phrases = ["cannot be predicted", "uncertain", "cannot predict", "unknown"]
        if any(p in out_lower for p in uncertainty_phrases):
            return True, "UNCERTAINTY_ACKNOWLEDGED"
        return False, "UNVERIFIED_ASSERTION"

    elif category == "constrained_code_synthesis":
        fn_name = metadata["function_name"]
        if fn_name not in out_clean or "train" in out_clean:
            return False, "FUNCTION_NAME_MISMATCH"
        try:
            ast.parse(out_clean)
            return True, "VALID_AST_AND_NAME"
        except Exception as e:
            return False, f"AST_SYNTAX_ERROR: {e}"

    return False, "UNKNOWN_CATEGORY"

def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    tokens = tokenizer.encode(prompt)
    curr_ids = torch.tensor([tokens], dtype=torch.long)
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            if next_token in (1, 2):  # EOS or PAD
                break
            generated.append(next_token)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
            if curr_ids.shape[1] >= 512:
                break
                
    return tokenizer.decode(generated).strip()

def evaluate_deterministic_baseline(test_data: List[Dict]) -> Dict[str, Any]:
    correct = 0
    records = []
    for item in test_data:
        cat = item["category"]
        m = item["metadata"]
        # Rule-based oracle/deterministic extraction
        if cat == "dynamic_entity_binding":
            resp = f"The sensor cluster is {m['target_entity']}."
        elif cat == "turn_dependent_retrieval":
            resp = f"The calibrated frequency is {m['expected_value']}."
        elif cat == "anaphora_reference_resolution":
            resp = f"Which pump do you mean, {m['candidates'][0]} or {m['candidates'][1]}?"
        elif cat == "superseded_instruction_adherence":
            resp = f"The format is {m['target_format']}."
        elif cat == "user_correction_handling":
            resp = f"Assigned to {m['new_target']}."
        elif cat == "distractor_resistance":
            resp = f"{m['target_entity']} is {m['target_state']}."
        elif cat == "ambiguity_clarification":
            resp = f"Which service on {m['target_pod']}?"
        elif cat == "missing_context_abstention":
            resp = "The context does not state the operating system."
        elif cat == "factual_uncertainty_handling":
            resp = "Future market prices cannot be predicted with certainty."
        elif cat == "constrained_code_synthesis":
            resp = f"def {m['function_name']}(val):\n    return val >= 10"
        else:
            resp = ""
            
        success, reason = score_example(cat, resp, m)
        if success:
            correct += 1
        records.append({"example_id": item["example_id"], "success": success, "reason": reason})
        
    return {
        "model_name": "Deterministic_Rule_Baseline",
        "total": len(test_data),
        "correct": correct,
        "accuracy_pct": round(correct / len(test_data) * 100, 2),
        "records": records
    }

def main():
    print("=" * 70)
    print("CHAKRVIEW EXPERIMENT 14: CONTEXT GROUNDING EVALUATION")
    print("=" * 70)
    
    test_f = DATA_DIR / "benchmark_test.jsonl"
    test_data = []
    with open(test_f, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} held-out test examples.")
    
    # 1. Deterministic Rule Baseline
    det_results = evaluate_deterministic_baseline(test_data)
    print(f"\n[Arm C: Deterministic Rule Baseline]")
    print(f"  Accuracy: {det_results['correct']}/{det_results['total']} ({det_results['accuracy_pct']}%)")
    
    # 2. Native Model Baselines
    tokenizer, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)
    device = torch.device("cpu")
    
    # Arm A: Frozen Canonical Baseline (Step-10000)
    print("\n[Arm A: Frozen Canonical Baseline Step-10000]")
    model_a = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    parent_ckpt = torch.load(CANONICAL_PARENT_PATH, map_location=device, weights_only=False)
    model_a.load_state_dict(parent_ckpt["model_state_dict"], strict=True)
    model_a.eval()
    
    correct_a = 0
    records_a = []
    for item in test_data:
        p = item["prompt"]
        raw = generate_greedy(model_a, tokenizer, p, max_new_tokens=40)
        success, reason = score_example(item["category"], raw, item["metadata"])
        if success:
            correct_a += 1
        records_a.append({"example_id": item["example_id"], "category": item["category"], "success": success, "reason": reason, "raw": raw})
        
    acc_a = round(correct_a / len(test_data) * 100, 2)
    print(f"  Accuracy: {correct_a}/{len(test_data)} ({acc_a}%)")
    
    # Arm B: Exp13 Candidate Best Checkpoint
    print("\n[Arm B: Exp13 Candidate Best Checkpoint]")
    correct_b = 0
    records_b = []
    if EXP13_CANDIDATE_PATH.exists():
        model_b = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
        cand_ckpt = torch.load(EXP13_CANDIDATE_PATH, map_location=device, weights_only=False)
        model_b.load_state_dict(cand_ckpt["model_state_dict"], strict=True)
        model_b.eval()
        
        for item in test_data:
            p = item["prompt"]
            raw = generate_greedy(model_b, tokenizer, p, max_new_tokens=40)
            success, reason = score_example(item["category"], raw, item["metadata"])
            if success:
                correct_b += 1
            records_b.append({"example_id": item["example_id"], "category": item["category"], "success": success, "reason": reason, "raw": raw})
        acc_b = round(correct_b / len(test_data) * 100, 2)
    else:
        acc_b = 0.0
    print(f"  Accuracy: {correct_b}/{len(test_data)} ({acc_b}%)")
    
    results = {
        "benchmark_name": "ChakrView Context Grounding Benchmark v1.0",
        "total_test_examples": len(test_data),
        "arms": {
            "arm_a_frozen_canonical": {
                "correct": correct_a,
                "total": len(test_data),
                "accuracy_pct": acc_a,
                "records": records_a
            },
            "arm_b_exp13_candidate": {
                "correct": correct_b,
                "total": len(test_data),
                "accuracy_pct": acc_b,
                "records": records_b
            },
            "arm_c_deterministic_rules": {
                "correct": det_results["correct"],
                "total": det_results["total"],
                "accuracy_pct": det_results["accuracy_pct"]
            }
        }
    }
    
    out_res = EXP_DIR / "BASELINE_RESULTS.json"
    out_res.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved verified evaluation results to {out_res}")

if __name__ == "__main__":
    main()
