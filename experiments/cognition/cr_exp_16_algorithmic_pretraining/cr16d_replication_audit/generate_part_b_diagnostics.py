"""Part B Diagnostic Suite:
1. G1 Counterfactual Paired Inversion: 200 matched pairs (400 probes).
   - In each pair (Probe A, Probe B), the key is identical but the assigned value changes.
   - Evaluates: Both correct, A correct only, B correct only, Both incorrect, Invariant output rate.
2. Position Permutation Test: 100 matched pairs (200 probes).
   - In each pair, multiple facts are presented in swapped orders.
   - Evaluates whether fact ordering disrupts retrieval.
3. Conflicting Distractor Interventions: 100 probes.
   - Distractors with shared values or conflicting values placed at varying distances from query.
4. G2 Unseen Value Logit Diagnostic:
   - For unseen-value probes, inspect teacher-forced loss, top-5 logits, and whether the correct token is in top-k.
"""
import sys
import json
import random
import torch
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

# Vocabulary pools from CR-16C / CR-16D
TRAIN_KEYS = ["color", "fruit", "animal", "city", "country", "vehicle", "sport", "flower", "profession", "weather"]
TRAIN_VALS = [
    "Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Silver", "Gold",
    "Apple", "Banana", "Orange", "Grape", "Mango", "Lemon", "Berry", "Peach",
    "Lion", "Tiger", "Bear", "Wolf", "Eagle", "Hawk", "Shark", "Whale",
    "Paris", "London", "Tokyo", "Berlin", "Rome", "Madrid", "Delhi", "Cairo",
    "France", "Japan", "Brazil", "Canada", "India", "Italy", "Spain", "Germany",
    "Car", "Truck", "Bus", "Train", "Plane", "Boat", "Bike", "Ship",
    "Soccer", "Tennis", "Cricket", "Golf", "Hockey", "Rugby", "Boxing", "Judo",
    "Rose", "Tulip", "Daisy", "Lily", "Lotus", "Orchid", "Iris", "Violet",
    "Doctor", "Teacher", "Lawyer", "Pilot", "Nurse", "Chef", "Artist", "Judge",
    "Rain", "Snow", "Storm", "Wind", "Cloud", "Sun", "Frost", "Fog"
]
OOD_KEYS = ["season", "planet", "element", "metal", "gem", "weapon", "fabric", "beverage", "tree", "continent"]
OOD_VALS = [
    "Spring", "Summer", "Autumn", "Winter",
    "Mars", "Venus", "Jupiter", "Saturn", "Mercury",
    "Fire", "Water", "Earth", "Air",
    "Iron", "Copper", "Bronze", "Steel",
    "Diamond", "Ruby", "Emerald", "Sapphire",
    "Sword", "Shield", "Bow", "Spear",
    "Cotton", "Silk", "Wool", "Linen",
    "Coffee", "Tea", "Juice", "Milk",
    "Oak", "Pine", "Maple", "Cedar",
    "Asia", "Africa", "Europe"
]

def build_part_b_diagnostics():
    rng = random.Random(42)
    
    # 1. G1 Counterfactual Paired Interventions (200 pairs = 400 probes)
    counterfactual_pairs = []
    for i in range(200):
        key = rng.choice(TRAIN_KEYS)
        v1, v2 = rng.sample(TRAIN_VALS, 2)
        # Template: Store: {key} = {val}. Query: {key}.\nAnswer: 
        pA = f"Store: {key} = {v1}. Query: {key}.\nAnswer: "
        pB = f"Store: {key} = {v2}. Query: {key}.\nAnswer: "
        counterfactual_pairs.append({
            "pair_id": f"CF_PAIR_{i+1:03d}",
            "key": key,
            "probe_A": {"prompt": pA, "expected": v1},
            "probe_B": {"prompt": pB, "expected": v2}
        })
        
    # 2. Position Permutations (100 pairs = 200 probes)
    position_pairs = []
    for i in range(100):
        k1, k2 = rng.sample(TRAIN_KEYS, 2)
        v1, v2 = rng.sample(TRAIN_VALS, 2)
        # Target is k1
        orig_prompt = f"Store: {k1} = {v1}. Store: {k2} = {v2}. Query: {k1}.\nAnswer: "
        perm_prompt = f"Store: {k2} = {v2}. Store: {k1} = {v1}. Query: {k1}.\nAnswer: "
        position_pairs.append({
            "pair_id": f"PERM_PAIR_{i+1:03d}",
            "target_key": k1,
            "expected": v1,
            "distractor_key": k2,
            "distractor_val": v2,
            "orig_prompt": orig_prompt,
            "perm_prompt": perm_prompt
        })
        
    # 3. Distractor Interventions (100 probes)
    distractor_probes = []
    for i in range(100):
        target_k = rng.choice(TRAIN_KEYS)
        target_v = rng.choice(TRAIN_VALS)
        other_keys = [k for k in TRAIN_KEYS if k != target_k]
        dist_k = rng.choice(other_keys)
        # Distractor types:
        # A: Distractor contains same value but different key
        # B: Conflicting distractor placed right before query
        if i % 2 == 0:
            prompt = f"Store: {target_k} = {target_v}. Store: {dist_k} = {target_v}. Query: {target_k}.\nAnswer: "
            dtype = "shared_value"
        else:
            other_v = rng.choice([v for v in TRAIN_VALS if v != target_v])
            prompt = f"Store: {target_k} = {target_v}. Store: {dist_k} = {other_v}. Query: {target_k}.\nAnswer: "
            dtype = "adjacent_distractor"
        distractor_probes.append({
            "id": f"DISTRACTOR_{i+1:03d}",
            "type": dtype,
            "target_key": target_k,
            "target_val": target_v,
            "prompt": prompt,
            "expected": target_v
        })
        
    out_dir = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cr16dr_counterfactual_pairs_200.json", "w") as f:
        json.dump(counterfactual_pairs, f, indent=2)
    with open(out_dir / "cr16dr_position_pairs_100.json", "w") as f:
        json.dump(position_pairs, f, indent=2)
    with open(out_dir / "cr16dr_distractor_probes_100.json", "w") as f:
        json.dump(distractor_probes, f, indent=2)
    print("Generated Part B diagnostic benchmark files successfully.")

if __name__ == "__main__":
    build_part_b_diagnostics()

