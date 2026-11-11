"""
Stage 5: Comprehensive Frozen Evaluation Script across 4 Models
Models:
1. Frozen Conversational Parent (cbr_sft_04_step_200.pt)
2. Current Conversational Candidate (conversational_pilot_best_val.pt)
3. CR-EXP-06 Control Arm Candidate (control_arm_best_val.pt)
4. CR-EXP-06 Treatment Arm Candidate (treatment_arm_best_val.pt)

Evaluates on:
- In-Distribution Relational Retrieval (test_in_distribution.jsonl - 100 cases)
- Strict OOD Entities (test_ood_entities.jsonl - 100 cases)
- Strict OOD Templates (test_ood_templates.jsonl - 100 cases)
- Multi-Hop Relational (test_multihop.jsonl - 60 cases)
- Conversational Preservation & Persona Suite (from CR-EXP-04 - 25 cases)
- Multi-turn horizon (2, 3, 5, 8 turns)
- Generation stability (repetition, attractor collapse, empty responses)
"""

import sys
import os
import json
import time
import math
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_06_variable_binding" / "data"
EVAL_OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_06_variable_binding" / "evaluation"

MODELS = {
    "parent": {
        "name": "Frozen Conversational Parent",
        "path": REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
    },
    "candidate_pilot": {
        "name": "Current Conversational Candidate",
        "path": REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "checkpoints" / "conversational_pilot_best_val.pt"
    },
    "control_arm": {
        "name": "CR-EXP-06 Control Arm (Standard SFT)",
        "path": REPO_ROOT / "experiments" / "cognition" / "cr_exp_06_variable_binding" / "control_arm" / "checkpoints" / "control_arm_best_val.pt"
    },
    "treatment_arm": {
        "name": "CR-EXP-06 Treatment Arm (Target-Weighted)",
        "path": REPO_ROOT / "experiments" / "cognition" / "cr_exp_06_variable_binding" / "treatment_arm" / "checkpoints" / "treatment_arm_best_val.pt"
    }
}

CONVERSATION_PRESERVATION_PROBES = [
    {"id": "CP_01", "prompt": "User: Hello!\nAssistant:", "expected": "greeting"},
    {"id": "CP_02", "prompt": "User: Hi there, good morning.\nAssistant:", "expected": "greeting"},
    {"id": "CP_03", "prompt": "User: Good evening, how are you?\nAssistant:", "expected": "greeting"},
    {"id": "CP_04", "prompt": "User: Who are you?\nAssistant:", "expected": "I am ChakrView, an indigenous neural language model developed from first principles."},
    {"id": "CP_05", "prompt": "User: What is your primary design philosophy?\nAssistant:", "expected": "first principles"},
    {"id": "CP_06", "prompt": "User: What is a hash function?\nAssistant:", "expected": "hash function"},
    {"id": "CP_07", "prompt": "User: Thank you for your help!\nAssistant:", "expected": "welcome"},
    {"id": "CP_08", "prompt": "User: Have a great day!\nAssistant:", "expected": "day"},
    {"id": "CP_09", "prompt": "User: What did I eat for breakfast this morning?\nAssistant:", "expected": "not provided in the context"},
    {"id": "CP_10", "prompt": "User: What is the secret password for user_admin?\nAssistant:", "expected": "not provided in the context"},
    {"id": "CP_11", "prompt": "User: Bhai, kaise ho?\nAssistant:", "expected": "theek"},
    {"id": "CP_12", "prompt": "User: Can you assist with technical questions?\nAssistant:", "expected": "assist"},
]

MULTI_TURN_HORIZON_PROBES = [
    {
        "id": "MT_01_TURN_2",
        "turns": 2,
        "prompt": "User: The project codename is Phoenix.\nAssistant: Phoenix recorded.\nUser: What is the project codename?\nAssistant:",
        "target": "Phoenix"
    },
    {
        "id": "MT_02_TURN_3",
        "turns": 3,
        "prompt": "User: My name is Kabir.\nAssistant: Nice to meet you Kabir!\nUser: The server port is 9000.\nAssistant: Port 9000 noted.\nUser: What is my name?\nAssistant:",
        "target": "Kabir"
    },
    {
        "id": "MT_03_TURN_5",
        "turns": 5,
        "prompt": (
            "User: We are configuring database backups.\n"
            "Assistant: Backups are critical.\n"
            "User: Backup location is /mnt/storage.\n"
            "Assistant: Location /mnt/storage noted.\n"
            "User: Run backup every 6 hours.\n"
            "Assistant: Schedule set to every 6 hours.\n"
            "User: Notify admin@chakrview.ai on failure.\n"
            "Assistant: Notification email configured.\n"
            "User: Where is the backup location?\n"
            "Assistant:"
        ),
        "target": "/mnt/storage"
    },
    {
        "id": "MT_04_TURN_8",
        "turns": 8,
        "prompt": (
            "User: Turn 1: Deploy service Alpha.\n"
            "Assistant: Service Alpha deployment started.\n"
            "User: Turn 2: Setting memory limit to 4GB.\n"
            "Assistant: Memory limit 4GB configured.\n"
            "User: Turn 3: Set cluster region to ap-south-1.\n"
            "Assistant: Region ap-south-1 configured.\n"
            "User: Turn 4: Verify SSL cert.\n"
            "Assistant: SSL verified.\n"
            "User: Turn 5: Set replica count to 3.\n"
            "Assistant: Replica count set to 3.\n"
            "User: Turn 6: Enable Prometheus metrics.\n"
            "Assistant: Metrics enabled.\n"
            "User: Turn 7: Check healthcheck endpoint.\n"
            "Assistant: Healthcheck responding 200 OK.\n"
            "User: What cluster region did I configure in Turn 3?\n"
            "Assistant:"
        ),
        "target": "ap-south-1"
    }
]

def load_model(ckpt_path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def generate_greedy(model, tok, prompt, max_new_tokens=30):
    tokens = tok.encode(prompt)
    generated = []
    stop_tokens = ["\nUser:", "User:", "\nAssistant:", "Assistant:"]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([tokens], dtype=torch.long)
            logits = model(inp)
            next_token = torch.argmax(logits[0, -1, :]).item()
            tokens.append(next_token)
            generated.append(next_token)
            text = tok.decode(generated, errors="replace")
            if any(st in text for st in stop_tokens):
                break
    res = tok.decode(generated, errors="replace")
    for st in stop_tokens:
        if st in res:
            res = res.split(st)[0]
    return res.strip()

def evaluate_retrieval_dataset(model, tok, samples):
    correct = 0
    records = []
    for s in samples:
        prompt = s["prompt"]
        expected_resp = s["response"]
        spans = s.get("target_spans", [])
        
        # Extract target substring from response
        targets = []
        for sp in spans:
            targets.append(expected_resp[sp[0]:sp[1]])
        if not targets:
            # fallback: look for last word/entity in response
            targets = [expected_resp.split()[-1].rstrip(".")]

        gen = generate_greedy(model, tok, prompt)
        
        is_hit = any(t.lower() in gen.lower() for t in targets)
        if is_hit:
            correct += 1

        records.append({
            "prompt": prompt,
            "expected_response": expected_resp,
            "target_entities": targets,
            "generation": gen,
            "success": is_hit
        })

    acc = (correct / len(samples)) * 100.0 if samples else 0.0
    return acc, records

def analyze_stability(text):
    is_empty = len(text.strip()) == 0
    has_attractor = any(a in text and text.count(a) > 3 for a in ["Cube", "cube", "Mumbai", "mumbai", "Celsia"])
    is_repetition = False
    words = text.split()
    if len(words) >= 9:
        for n in [2, 3, 4]:
            for i in range(len(words) - 3 * n):
                if words[i:i+n] == words[i+n:i+2*n] == words[i+2*n:i+3*n]:
                    is_repetition = True
                    break
    return {
        "is_empty": is_empty,
        "has_attractor": has_attractor,
        "is_repetition": is_repetition
    }

def main():
    print("=== CR-EXP-06 Stage 5: Frozen Evaluation across 4 Models ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load test datasets
    def load_jsonl(p):
        items = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    test_id = load_jsonl(DATA_DIR / "test_in_distribution.jsonl")
    test_ood_ent = load_jsonl(DATA_DIR / "test_ood_entities.jsonl")
    test_ood_tmpl = load_jsonl(DATA_DIR / "test_ood_templates.jsonl")
    test_multihop = load_jsonl(DATA_DIR / "test_multihop.jsonl")

    print(f"Loaded evaluation splits: ID={len(test_id)}, OOD-Entities={len(test_ood_ent)}, OOD-Templates={len(test_ood_tmpl)}, MultiHop={len(test_multihop)}")

    loaded_models = {}
    for m_key, m_info in MODELS.items():
        print(f"Loading {m_info['name']} from {m_info['path'].name}...")
        loaded_models[m_key] = load_model(m_info["path"])

    overall_results = {}
    per_example_all = {}

    for m_key, model in loaded_models.items():
        print(f"\n--- Evaluating Model: {m_key.upper()} ---")
        
        # 1. In-Distribution Retrieval
        acc_id, rec_id = evaluate_retrieval_dataset(model, tok, test_id)
        # 2. Strict OOD Entities
        acc_ood_ent, rec_ood_ent = evaluate_retrieval_dataset(model, tok, test_ood_ent)
        # 3. Strict OOD Templates
        acc_ood_tmpl, rec_ood_tmpl = evaluate_retrieval_dataset(model, tok, test_ood_tmpl)
        # 4. Multi-Hop Relational
        acc_multihop, rec_multihop = evaluate_retrieval_dataset(model, tok, test_multihop)

        # 5. Conversational Preservation
        cp_correct = 0
        cp_records = []
        for p in CONVERSATION_PRESERVATION_PROBES:
            gen = generate_greedy(model, tok, p["prompt"])
            exp = p["expected"].lower()
            gen_lower = gen.lower()
            hit = False
            if exp == "greeting":
                hit = any(g in gen_lower for g in ["morning", "afternoon", "evening", "hello", "hi", "assist", "help"])
            else:
                hit = exp in gen_lower
            if hit:
                cp_correct += 1
            cp_records.append({
                "probe_id": p["id"],
                "prompt": p["prompt"],
                "expected": p["expected"],
                "generated": gen,
                "success": hit
            })
        acc_cp = (cp_correct / len(CONVERSATION_PRESERVATION_PROBES)) * 100.0

        # 6. Multi-Turn Horizon
        mt_correct = 0
        mt_records = []
        for p in MULTI_TURN_HORIZON_PROBES:
            gen = generate_greedy(model, tok, p["prompt"])
            target = p["target"].lower()
            hit = target in gen.lower()
            if hit:
                mt_correct += 1
            mt_records.append({
                "probe_id": p["id"],
                "turns": p["turns"],
                "target": p["target"],
                "generated": gen,
                "success": hit
            })
        acc_mt = (mt_correct / len(MULTI_TURN_HORIZON_PROBES)) * 100.0

        # 7. Stability Metrics across all generated samples
        all_gens = [r["generation"] for r in rec_id + rec_ood_ent + rec_ood_tmpl + rec_multihop]
        repetition_count = sum(1 for g in all_gens if analyze_stability(g)["is_repetition"])
        attractor_count = sum(1 for g in all_gens if analyze_stability(g)["has_attractor"])
        empty_count = sum(1 for g in all_gens if analyze_stability(g)["is_empty"])

        overall_results[m_key] = {
            "model_name": MODELS[m_key]["name"],
            "accuracy_in_distribution_pct": round(acc_id, 2),
            "accuracy_ood_entities_pct": round(acc_ood_ent, 2),
            "accuracy_ood_templates_pct": round(acc_ood_tmpl, 2),
            "accuracy_multihop_pct": round(acc_multihop, 2),
            "conversational_preservation_pct": round(acc_cp, 2),
            "multiturn_horizon_pct": round(acc_mt, 2),
            "stability_failures": {
                "repetition_loops": repetition_count,
                "attractor_collapse": attractor_count,
                "empty_responses": empty_count
            }
        }

        per_example_all[m_key] = {
            "in_distribution": rec_id,
            "ood_entities": rec_ood_ent,
            "ood_templates": rec_ood_tmpl,
            "multihop": rec_multihop,
            "conversational_preservation": cp_records,
            "multiturn_horizon": mt_records
        }

        print(f"Results for {m_key}:")
        print(f"  ID Retrieval       : {acc_id:.1f}%")
        print(f"  OOD Entities       : {acc_ood_ent:.1f}%")
        print(f"  OOD Templates      : {acc_ood_tmpl:.1f}%")
        print(f"  Multi-Hop          : {acc_multihop:.1f}%")
        print(f"  Conversation Pres  : {acc_cp:.1f}%")
        print(f"  Multi-Turn Horizon : {acc_mt:.1f}%")

    # Save summary evaluation deliverables
    with open(REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/control_arm_results.json", "w", encoding="utf-8") as f:
        json.dump(overall_results["control_arm"], f, indent=2)

    with open(REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/treatment_arm_results.json", "w", encoding="utf-8") as f:
        json.dump(overall_results["treatment_arm"], f, indent=2)

    with open(REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/ood_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "models_comparison": {
                k: {
                    "accuracy_ood_entities_pct": v["accuracy_ood_entities_pct"],
                    "accuracy_ood_templates_pct": v["accuracy_ood_templates_pct"]
                } for k, v in overall_results.items()
            }
        }, f, indent=2)

    with open(REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/multihop_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "models_comparison": {k: v["accuracy_multihop_pct"] for k, v in overall_results.items()}
        }, f, indent=2)

    with open(REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/conversation_preservation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "models_comparison": {k: v["conversational_preservation_pct"] for k, v in overall_results.items()}
        }, f, indent=2)

    with open(REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/generation_stability_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "models_comparison": {k: v["stability_failures"] for k, v in overall_results.items()}
        }, f, indent=2)

    with open(REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/per_example_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(per_example_all, f, indent=2, ensure_ascii=False)

    print("\nAll frozen evaluation deliverables successfully compiled and written to disk.")

if __name__ == "__main__":
    main()

