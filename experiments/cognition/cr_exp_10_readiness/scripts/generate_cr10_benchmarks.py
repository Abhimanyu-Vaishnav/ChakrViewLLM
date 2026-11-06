"""
CR-EXP-10 Benchmark Generator
Constructs:
1. Frozen 800-probe Multi-Dimensional Evaluation Suite (100 probes x 8 categories) with completely fresh entities & unseen templates.
2. Dedicated 300-probe Conversational & Working Memory Evaluation Suite for v0.9.1 readiness.
"""

import json
import random
import hashlib
from pathlib import Path

random.seed(1337)

OUT_DIR = Path("experiments/cognition/cr_exp_10_readiness/benchmark_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Unseen OOD entities (not in CR-EXP-08/09 datasets)
OOD_NAMES = ["Kavita", "Manish", "Divya", "Suresh", "Lakshmi", "Gautam", "Swati", "Nikhil", "Pooja", "Rajesh", "Sunita", "Deepak", "Anil", "Bhavna", "Harish", "Isha", "Jatin", "Komal", "Lalit", "Pallavi"]
OOD_CITIES = ["Ahmedabad", "Chandigarh", "Bhopal", "Indore", "Surat", "Nagpur", "Visakhapatnam", "Patna", "Vadodara", "Coimbatore", "Kochi", "Varanasi", "Agra", "Amritsar", "Dehradun", "Ranchi"]
OOD_OCCUPATIONS = ["architect", "surgeon", "pilot", "carpenter", "librarian", "journalist", "photographer", "geologist", "electrician", "astronomer"]
OOD_FRUITS = ["fig", "plum", "cherry", "apricot", "papaya", "pomegranate", "kiwi", "blueberry"]
OOD_ANIMALS = ["tiger", "elephant", "leopard", "peacock", "dolphin", "falcon", "camel", "otter"]

PROBES_800 = []

# CAT-1: Conversational Understanding & Persona (100 probes)
conv_prompts = [
    ("User: Hello, who are you?\nAssistant: ", "ChakrView"),
    ("User: What is your primary purpose?\nAssistant: ", "assist"),
    ("User: Good morning! How are you?\nAssistant: ", "help"),
    ("User: Can you assist me today?\nAssistant: ", "assist"),
    ("User: Are you a human?\nAssistant: ", "neural"),
    ("User: What is your name?\nAssistant: ", "ChakrView"),
    ("User: Namaste!\nAssistant: ", "namaste"),
    ("User: Tell me about yourself.\nAssistant: ", "ChakrView"),
    ("User: Hello there.\nAssistant: ", "hello"),
    ("User: Can I ask you a question?\nAssistant: ", "help")
]
for i in range(100):
    p, exp = random.choice(conv_prompts)
    PROBES_800.append({
        "id": f"CR10_CAT1_{i+1:03d}",
        "category": "conversational_understanding",
        "prompt": p,
        "expected_target": exp
    })

# CAT-2: Single-Hop & Counterfactual Retrieval (100 probes)
for i in range(100):
    name = random.choice(OOD_NAMES)
    city = random.choice(OOD_CITIES)
    occ = random.choice(OOD_OCCUPATIONS)
    if i % 2 == 0:
        prompt = f"Context: {name} moved to {city}.\nQuestion: Where does {name} live?\nAnswer: "
        exp = city
    else:
        prompt = f"Context: {name} works as a {occ}.\nQuestion: What is {name}'s profession?\nAnswer: "
        exp = occ
    PROBES_800.append({
        "id": f"CR10_CAT2_{i+1:03d}",
        "category": "single_hop_and_counterfactual",
        "prompt": prompt,
        "expected_target": exp
    })

# CAT-3: Multi-Entity Distractor Disambiguation (100 probes)
for i in range(100):
    p1, p2, p3 = random.sample(OOD_NAMES, 3)
    c1, c2, c3 = random.sample(OOD_CITIES, 3)
    prompt = f"Context: {p1} lives in {c1}. {p2} lives in {c2}. {p3} lives in {c3}.\nQuestion: Where does {p2} live?\nAnswer: "
    PROBES_800.append({
        "id": f"CR10_CAT3_{i+1:03d}",
        "category": "multi_entity_distractor",
        "prompt": prompt,
        "expected_target": c2,
        "distractor_value": f"{c1}, {c3}"
    })

# CAT-4: Missing-Info Epistemic Abstention (100 probes)
for i in range(100):
    p1 = random.choice(OOD_NAMES)
    p_absent = random.choice([n for n in OOD_NAMES if n != p1])
    c1 = random.choice(OOD_CITIES)
    if i % 2 == 0:
        prompt = f"Context: {p1} lives in {c1}.\nQuestion: Where does {p_absent} live?\nAnswer: "
    else:
        prompt = f"Context: {p1} lives in {c1}.\nQuestion: What is {p1}'s profession?\nAnswer: "
    PROBES_800.append({
        "id": f"CR10_CAT4_{i+1:03d}",
        "category": "missing_info_abstention",
        "prompt": prompt,
        "expected_target": "The context does not state"
    })

# CAT-5: Multi-Turn Working Memory & State Updating (100 probes)
for i in range(100):
    p = random.choice(OOD_NAMES)
    c_initial = random.choice(OOD_CITIES)
    c_updated = random.choice([c for c in OOD_CITIES if c != c_initial])
    prompt = f"User: {p} started the journey in {c_initial}.\nChakrView: Noted.\nUser: {p} has now relocated to {c_updated}.\nChakrView: Updated.\nUser: What is {p}'s current location?\nChakrView: "
    PROBES_800.append({
        "id": f"CR10_CAT5_{i+1:03d}",
        "category": "multiturn_state_tracking",
        "prompt": prompt,
        "expected_target": c_updated,
        "earlier_value": c_initial
    })

# CAT-6: Structured Elementary Arithmetic (100 probes)
for i in range(100):
    p = random.choice(OOD_NAMES)
    init_val = random.randint(2, 10)
    delta = random.randint(1, 5)
    is_add = (i % 2 == 0)
    final_val = init_val + delta if is_add else init_val - delta
    op_word = "received" if is_add else "lost"
    prompt = f"Context: {p} originally had {init_val} books. Then {p} {op_word} {delta} books.\nQuestion: How many books does {p} have in total?\nAnswer: "
    PROBES_800.append({
        "id": f"CR10_CAT6_{i+1:03d}",
        "category": "structured_arithmetic_state",
        "prompt": prompt,
        "expected_target": str(final_val)
    })

# CAT-7: Conditional Deductive Logic (100 probes)
for i in range(100):
    p = random.choice(OOD_NAMES)
    c = random.choice(OOD_CITIES)
    cond_true = (i % 2 == 0)
    if cond_true:
        prompt = f"Premise: Whenever {p} visits {c}, {p} attends the seminar.\nContext: {p} is currently in {c}.\nQuestion: Does {p} attend the seminar?\nAnswer: "
        exp = "Yes"
    else:
        c_diff = random.choice([x for x in OOD_CITIES if x != c])
        prompt = f"Premise: Whenever {p} visits {c}, {p} attends the seminar.\nContext: {p} is currently in {c_diff}.\nQuestion: Does {p} attend the seminar?\nAnswer: "
        exp = "No"
    PROBES_800.append({
        "id": f"CR10_CAT7_{i+1:03d}",
        "category": "conditional_logical_reasoning",
        "prompt": prompt,
        "expected_target": exp
    })

# CAT-8: Strict Negative Constraint Satisfaction (100 probes)
for i in range(100):
    fruit = random.choice(OOD_FRUITS)
    animal = random.choice(OOD_ANIMALS)
    order = [fruit, animal]
    random.shuffle(order)
    prompt = f"Constraint: Pick the fruit only and strictly exclude animals.\nOptions: {order[0]}, {order[1]}.\nAnswer: "
    PROBES_800.append({
        "id": f"CR10_CAT8_{i+1:03d}",
        "category": "constraint_satisfaction",
        "prompt": prompt,
        "expected_target": fruit
    })

with open(OUT_DIR / "benchmark_800_suite.json", "w", encoding="utf-8") as f:
    json.dump(PROBES_800, f, indent=2)

# Dedicated 300-Probe Conversational & Working Memory Suite
CONV_300 = []
for i in range(150):
    # Pure conversational turn
    p, exp = random.choice(conv_prompts)
    CONV_300.append({
        "id": f"CONV_EVAL_{i+1:03d}",
        "type": "persona_and_fluency",
        "prompt": p,
        "expected": exp
    })
for i in range(150):
    # Multi-turn context memory
    p = random.choice(OOD_NAMES)
    c1 = random.choice(OOD_CITIES)
    c2 = random.choice([x for x in OOD_CITIES if x != c1])
    prompt = f"User: Hi, {p} is traveling from {c1}.\nAssistant: Understood.\nUser: {p} has reached {c2}.\nAssistant: Recorded.\nUser: Where can {p} be found right now?\nAssistant: "
    CONV_300.append({
        "id": f"CONV_MEM_{i+1:03d}",
        "type": "conversational_state_retention",
        "prompt": prompt,
        "expected": c2
    })

with open(OUT_DIR / "conversational_300_suite.json", "w", encoding="utf-8") as f:
    json.dump(CONV_300, f, indent=2)

# Hash manifest
h_800 = hashlib.sha256((OUT_DIR / "benchmark_800_suite.json").read_bytes()).hexdigest()
h_300 = hashlib.sha256((OUT_DIR / "conversational_300_suite.json").read_bytes()).hexdigest()

manifest = {
    "benchmark_800_probes": {
        "file": "benchmark_800_suite.json",
        "sha256": h_800,
        "total_probes": len(PROBES_800),
        "categories": 8
    },
    "conversational_300_probes": {
        "file": "conversational_300_suite.json",
        "sha256": h_300,
        "total_probes": len(CONV_300)
    }
}

with open(OUT_DIR / "benchmark_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"CR-EXP-10 Benchmarks created:\n  800-probe suite SHA: {h_800}\n  300-probe suite SHA: {h_300}")

