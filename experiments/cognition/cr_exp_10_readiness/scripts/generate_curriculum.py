"""
CR-EXP-10 Curriculum Dataset Generator
Creates two balanced, controlled training sets:
- Arm A (Balanced Generalist Curriculum): 3,200 samples across all 8 dimensions with 20% conversational anchoring.
- Arm B (Targeted Gap-Closure Curriculum): 3,200 samples with focused weight on diagnosed weaknesses (Negative Constraints, Multi-Entity Distractors, Predicate Abstention, and Modus Ponens).
"""

import json
import random
from pathlib import Path

random.seed(2026)

DATA_DIR = Path("experiments/cognition/cr_exp_10_readiness/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Training entity pool (disjoint from OOD benchmark)
TRAIN_NAMES = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Neha", "Kabir", "Meera", "Aditya", "Tara", "Karan", "Simran", "Arjun", "Pooja", "Dev", "Ira", "Kunal", "Riya", "Varun", "Tanvi"]
TRAIN_CITIES = ["Delhi", "Mumbai", "Jaipur", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Lucknow", "Goa", "Shimla", "Mysuru", "Udaipur", "Thiruvananthapuram"]
TRAIN_OCCUPATIONS = ["doctor", "engineer", "teacher", "artist", "lawyer", "scientist", "chef", "writer", "pilot", "musician"]
TRAIN_FRUITS = ["apple", "banana", "mango", "orange", "grapes", "guava", "peach", "pear"]
TRAIN_COLORS = ["red", "blue", "green", "yellow", "purple", "white", "black", "orange"]

def generate_sample(cat):
    if cat == "conversational":
        prompts = [
            ("User: Hello, who are you?\nAssistant: ", "I am ChakrView, an indigenous neural intelligence being developed from scratch."),
            ("User: Can you assist me today?\nAssistant: ", "Yes, I am here to help you. What would you like to explore or discuss?"),
            ("User: What is your primary objective?\nAssistant: ", "My objective is to assist users with natural conversation, reasoning, and factual retrieval."),
            ("User: Are you a biological human?\nAssistant: ", "No, I am ChakrView, a local neural language model trained from scratch."),
            ("User: Good evening!\nAssistant: ", "Good evening! I hope your day went well. How can I assist you right now?")
        ]
        p, c = random.choice(prompts)
        return {"type": "conversational", "prompt": p, "completion": c, "is_relational": False, "weight": 1.0}
        
    elif cat == "single_hop":
        p = random.choice(TRAIN_NAMES)
        c = random.choice(TRAIN_CITIES)
        prompt = f"Context: {p} lives in {c}.\nQuestion: Where does {p} live?\nAnswer: "
        return {"type": "single_hop", "prompt": prompt, "completion": c, "is_relational": True, "weight": 2.0}
        
    elif cat == "distractor":
        p1, p2, p3 = random.sample(TRAIN_NAMES, 3)
        c1, c2, c3 = random.sample(TRAIN_CITIES, 3)
        prompt = f"Context: {p1} is in {c1}. {p2} is in {c2}. {p3} is in {c3}.\nQuestion: Where is {p2} currently located?\nAnswer: "
        return {"type": "distractor", "prompt": prompt, "completion": c2, "is_relational": True, "weight": 2.5}
        
    elif cat == "abstention":
        p1 = random.choice(TRAIN_NAMES)
        p_absent = random.choice([n for n in TRAIN_NAMES if n != p1])
        c1 = random.choice(TRAIN_CITIES)
        if random.random() > 0.5:
            prompt = f"Context: {p1} lives in {c1}.\nQuestion: Where does {p_absent} live?\nAnswer: "
            comp = "The context does not state where that person lives."
        else:
            prompt = f"Context: {p1} lives in {c1}.\nQuestion: What is {p1}'s profession?\nAnswer: "
            comp = f"The context does not state {p1}'s profession."
        return {"type": "abstention", "prompt": prompt, "completion": comp, "is_relational": True, "weight": 2.0}
        
    elif cat == "multiturn":
        p = random.choice(TRAIN_NAMES)
        c1 = random.choice(TRAIN_CITIES)
        c2 = random.choice([x for x in TRAIN_CITIES if x != c1])
        prompt = f"User: {p} was residing in {c1}.\nChakrView: Understood.\nUser: {p} has recently moved to {c2}.\nChakrView: Recorded.\nUser: Where is {p} now?\nChakrView: "
        comp = f"{p} is currently in {c2}."
        return {"type": "multiturn", "prompt": prompt, "completion": comp, "is_relational": True, "weight": 2.0}
        
    elif cat == "arithmetic":
        p = random.choice(TRAIN_NAMES)
        n = random.randint(3, 12)
        d = random.randint(1, 5)
        is_add = random.random() > 0.5
        res = n + d if is_add else n - d
        verb = "acquired" if is_add else "spent"
        prompt = f"Context: {p} held {n} coins. Later, {p} {verb} {d} coins.\nQuestion: How many coins does {p} have in total?\nAnswer: "
        return {"type": "arithmetic", "prompt": prompt, "completion": str(res), "is_relational": True, "weight": 2.0}
        
    elif cat == "logic":
        p = random.choice(TRAIN_NAMES)
        c = random.choice(TRAIN_CITIES)
        is_true = random.random() > 0.5
        if is_true:
            prompt = f"Rule: If {p} visits {c}, then {p} meets the mentor.\nContext: {p} is currently in {c}.\nQuestion: Does {p} meet the mentor?\nAnswer: "
            comp = "Yes"
        else:
            c_diff = random.choice([x for x in TRAIN_CITIES if x != c])
            prompt = f"Rule: If {p} visits {c}, then {p} meets the mentor.\nContext: {p} is currently in {c_diff}.\nQuestion: Does {p} meet the mentor?\nAnswer: "
            comp = "No"
        return {"type": "logic", "prompt": prompt, "completion": comp, "is_relational": True, "weight": 2.5}
        
    elif cat == "constraint":
        fruit = random.choice(TRAIN_FRUITS)
        color = random.choice(TRAIN_COLORS)
        options = [fruit, color]
        random.shuffle(options)
        prompt = f"Constraint: Select the fruit and omit colors.\nOptions: {options[0]}, {options[1]}.\nAnswer: "
        return {"type": "constraint", "prompt": prompt, "completion": fruit, "is_relational": True, "weight": 3.0}

# Generate Arm A (Uniform Balanced Curriculum)
arm_a_samples = []
for _ in range(3200):
    cat = random.choice(["conversational", "single_hop", "distractor", "abstention", "multiturn", "arithmetic", "logic", "constraint"])
    arm_a_samples.append(generate_sample(cat))
random.shuffle(arm_a_samples)

# Generate Arm B (Targeted Gap-Closure Curriculum: 25% Constraint, 20% Distractor, 20% Logic, 15% Abstention, 10% Conversational, 10% Memory)
arm_b_samples = []
weights_b = [
    ("constraint", 800),
    ("distractor", 650),
    ("logic", 650),
    ("abstention", 500),
    ("conversational", 300),
    ("multiturn", 300)
]
for cat, count in weights_b:
    for _ in range(count):
        arm_b_samples.append(generate_sample(cat))
random.shuffle(arm_b_samples)

# Generate Validation Set (300 samples)
val_samples = []
for _ in range(300):
    cat = random.choice(["conversational", "single_hop", "distractor", "abstention", "multiturn", "arithmetic", "logic", "constraint"])
    val_samples.append(generate_sample(cat))

with open(DATA_DIR / "train_arm_a.json", "w", encoding="utf-8") as f:
    json.dump(arm_a_samples, f, indent=2)

with open(DATA_DIR / "train_arm_b.json", "w", encoding="utf-8") as f:
    json.dump(arm_b_samples, f, indent=2)

with open(DATA_DIR / "val.json", "w", encoding="utf-8") as f:
    json.dump(val_samples, f, indent=2)

print(f"Datasets generated:\n  Arm A: {len(arm_a_samples)} samples\n  Arm B: {len(arm_b_samples)} samples\n  Val:   {len(val_samples)} samples")

