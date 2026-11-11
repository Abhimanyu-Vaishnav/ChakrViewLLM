"""
Stage 8: Robustness and Generalization Evaluation Script
Tests on 500 independent prompts disjoint from CR-EXP-06 training:
- 100 single-hop subject-value cases
- 100 counterfactual reassignment cases
- 100 distractor cases
- 50 missing-information cases
- 50 unseen-entity cases
- 50 unseen-template cases
Evaluates complete generated answer sequences.
Reports separately:
- Exact match
- Subject correctness
- Value correctness
- Subject-substitution rate
"""

import json
import random
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
MODEL_PATH = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt"

# Disjoint entity pools
PERSONS_POOL_A = ["Suresh", "Meenakshi", "Gopal", "Lalita", "Mahesh", "Sunita", "Tarun", "Kiran"]
PERSONS_POOL_B = ["Aarav", "Diya", "Ishaan", "Sanya", "Aditya", "Rhea", "Manish", "Tanya"]
CITIES_POOL_A = ["Kanpur", "Varanasi", "Dehradun", "Patna", "Ranchi", "Surat", "Bhopal", "Gwalior"]
CITIES_POOL_B = ["Kochi", "Mysuru", "Coimbatore", "Madurai", "Thiruvananthapuram", "Mangaluru", "Hubballi"]

DISTRACTORS_POOL = [
    "Database replica sync is in progress.",
    "Network firewall logs indicate normal traffic.",
    "System memory cache was cleared at midnight.",
    "Automated regression tests passed with zero regressions.",
    "The SSL certificate authority renewal is pending approval."
]

def load_model(ckpt_path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def generate_greedy(model, tok, prompt, max_new_tokens=25):
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

def evaluate_suite():
    print("=== CR-EXP-07 Stage 8: Robustness and Generalization Suite ===")
    rng = random.Random(20261107)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = load_model(MODEL_PATH)

    categories = {
        "single_hop_100": [],
        "reassignment_100": [],
        "distractors_100": [],
        "missing_info_50": [],
        "unseen_entities_50": [],
        "unseen_templates_50": []
    }

    # 1. 100 Single-Hop
    for i in range(100):
        p = rng.choice(PERSONS_POOL_A)
        c = rng.choice(CITIES_POOL_A)
        prompt = f"Context: {p} lives in {c}.\nUser: Where does {p} live?\nAssistant:"
        categories["single_hop_100"].append({"prompt": prompt, "subject": p, "value": c})

    # 2. 100 Counterfactual Reassignment
    for i in range(100):
        p = rng.choice(PERSONS_POOL_A)
        c1, c2 = rng.sample(CITIES_POOL_A, 2)
        prompt = f"Context: First, {p} lived in {c1}. Later, {p} moved to {c2}.\nUser: Where does {p} live now?\nAssistant:"
        categories["reassignment_100"].append({"prompt": prompt, "subject": p, "value": c2})

    # 3. 100 Distractors
    for i in range(100):
        p = rng.choice(PERSONS_POOL_A)
        c = rng.choice(CITIES_POOL_A)
        d = rng.choice(DISTRACTORS_POOL)
        prompt = f"Context: {d} {p} lives in {c}. {rng.choice(DISTRACTORS_POOL)}\nUser: Where does {p} live?\nAssistant:"
        categories["distractors_100"].append({"prompt": prompt, "subject": p, "value": c})

    # 4. 50 Missing Info
    for i in range(50):
        p1, p2 = rng.sample(PERSONS_POOL_A, 2)
        c = rng.choice(CITIES_POOL_A)
        prompt = f"Context: {p1} lives in {c}.\nUser: Where does {p2} live?\nAssistant:"
        categories["missing_info_50"].append({"prompt": prompt, "subject": p2, "value": "not provided"})

    # 5. 50 Unseen Entities
    for i in range(50):
        p = rng.choice(PERSONS_POOL_B)
        c = rng.choice(CITIES_POOL_B)
        prompt = f"Context: {p} lives in {c}.\nUser: Where does {p} live?\nAssistant:"
        categories["unseen_entities_50"].append({"prompt": prompt, "subject": p, "value": c})

    # 6. 50 Unseen Templates
    for i in range(50):
        p = rng.choice(PERSONS_POOL_A)
        c = rng.choice(CITIES_POOL_A)
        prompt = f"Record: Resident {p} resides at location {c}.\nUser: State the residence location of {p}.\nAssistant:"
        categories["unseen_templates_50"].append({"prompt": prompt, "subject": p, "value": c})

    results_summary = {}

    for cat_name, cases in categories.items():
        total = len(cases)
        correct_values = 0
        correct_subjects = 0
        exact_matches = 0
        substitution_hallucinations = 0

        for item in cases:
            prompt = item["prompt"]
            subj = item["subject"]
            val = item["value"]

            gen = generate_greedy(model, tok, prompt)
            gen_lower = gen.lower()

            v_hit = (val.lower() in gen_lower)
            s_hit = (subj.lower() in gen_lower)

            # Check if an unmentioned familiar training name was substituted
            substituted = False
            for train_name in ["Kabir", "Priya", "Ravi", "Amit", "Kavita", "Sunil", "Neha"]:
                if train_name != subj and train_name.lower() in gen_lower:
                    substituted = True
                    break

            if v_hit:
                correct_values += 1
            if s_hit:
                correct_subjects += 1
            if v_hit and s_hit:
                exact_matches += 1
            if v_hit and substituted and not s_hit:
                substitution_hallucinations += 1

        results_summary[cat_name] = {
            "total_cases": total,
            "exact_match_hits": exact_matches,
            "exact_match_pct": round((exact_matches / total) * 100.0, 2),
            "correct_value_hits": correct_values,
            "correct_value_pct": round((correct_values / total) * 100.0, 2),
            "correct_subject_hits": correct_subjects,
            "correct_subject_pct": round((correct_subjects / total) * 100.0, 2),
            "subject_substitution_hallucination_pct": round((substitution_hallucinations / total) * 100.0, 2)
        }
        print(f"[{cat_name:22s}] ExactMatch: {results_summary[cat_name]['exact_match_pct']}% | ValueHit: {results_summary[cat_name]['correct_value_pct']}% | SubstHalluc: {results_summary[cat_name]['subject_substitution_hallucination_pct']}%")

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_07_binding_diagnosis/robustness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nStage 8 results written to {out_file}")

if __name__ == "__main__":
    evaluate_suite()

