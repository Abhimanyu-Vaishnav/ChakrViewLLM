"""
CR-EXP-09 Candidate 2 Dataset Generator: Balanced Multi-Task Regularization
Addresses the over-abstention trade-off observed in Candidate 1:
1. Single-Hop & Counterfactual Grounding (600 samples, 25%):
   Explicitly pairs mapping queries with direct factual answers ("Answer: Delhi") to re-anchor positive retrieval.
2. Epistemic Missing-Information Abstention (300 samples, 12.5%):
   Calibrated abstention on genuinely missing subjects ("The context does not state where X lives.").
3. Multi-Entity Distractor Disambiguation (400 samples, 16.7%):
   2-entity and 3-entity contexts with balanced query targets.
4. Multi-Turn Dialogue State Tracking (400 samples, 16.7%):
   Conversational turn updates and factual relocations.
5. Elementary Arithmetic & Logical Reasoning (300 samples, 12.5%):
   Book/coin inventory transfer arithmetic and boolean conditional rules.
6. Conversational Preservation & Persona (400 samples, 16.7%):
   ChakrView persona, greetings, instruction adherence.

Total: 2,400 training samples, 300 validation samples.
Outputs data/train_candidate2.json and data/val_candidate2.json.
"""

import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_09_autonomous_mission" / "data"

TRAIN_SUBJECTS = ["Ravi", "Aman", "Priya", "Neha", "Rohit", "Sneha", "Amit", "Rahul", "Anjali", "Karan", "Simran", "Deepak", "Kavita", "Sanjay", "Geeta"]
VAL_SUBJECTS = ["Sunil", "Preeti", "Rajesh", "Sunita"]

TRAIN_CITIES = [
    "Delhi", "Jaipur", "Pune", "Surat", "Kanpur", "Nagpur", "Indore", "Patna",
    "Mumbai", "Bengaluru", "Bhopal", "Nashik", "Chennai", "Kolkata", "Hyderabad", "Lucknow"
]

OBJECTS = ["books", "pens", "coins", "apples", "candies", "cards", "markers"]

CONVERSATIONAL_PAIRS = [
    ("User: Hello! How are you doing today?\nAssistant: ", "Hello! I am doing well, ready to assist you. How can I help you today?"),
    ("User: Who are you?\nAssistant: ", "I am ChakrView, an indigenous neural intelligence being developed from scratch."),
    ("User: What is your primary purpose?\nAssistant: ", "My purpose is to develop genuine neural capabilities for conversational understanding and reasoning."),
    ("User: Can you explain your mission?\nAssistant: ", "I am built to progressively learn language understanding, contextual memory, and problem-solving step by step."),
    ("User: Good morning!\nAssistant: ", "Good morning! I hope you have a wonderful and productive day. What are you working on?"),
    ("User: Thank you so much for your assistance.\nAssistant: ", "You are very welcome! Please let me know if you need any further help."),
    ("User: Namaste!\nAssistant: ", "Namaste! Main badhiya hoon. Aapki kya madad kar sakta hoon?"),
    ("User: Tell me about yourself.\nAssistant: ", "I am ChakrView, an autonomous cognitive prototype learning to understand context and solve problems.")
]

def generate_balanced_samples(subject_pool, total_samples):
    samples = []
    
    # 1. Single-Hop & Counterfactual Grounding (25% = 600 in 2400)
    n_single = int(total_samples * 0.25)
    for i in range(n_single):
        s = random.choice(subject_pool)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        if i % 2 == 0:
            prompt = f"Mapping: {s} lives in {c1}. Query: Where does {s} live? Answer: "
            completion = c1
        else:
            prompt = f"Mapping: {s} used to live in {c1}, but now lives in {c2}. Query: Where does {s} live? Answer: "
            completion = c2
        samples.append({
            "type": "single_hop_grounding",
            "prompt": prompt,
            "completion": completion,
            "is_relational": True
        })
        
    # 2. Multi-Entity Distractor Disambiguation (17%)
    n_dist = int(total_samples * 0.17)
    for _ in range(n_dist):
        s1, s2 = random.sample(subject_pool, 2)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        target_s, target_c = (s1, c1) if random.random() < 0.5 else (s2, c2)
        prompt = f"Mapping: {s1} lives in {c1}. {s2} lives in {c2}. Query: Where does {target_s} live? Answer: "
        completion = target_c
        samples.append({
            "type": "distractor_disambiguation",
            "prompt": prompt,
            "completion": completion,
            "is_relational": True
        })
        
    # 3. Multi-Turn Dialogue State Tracking (17%)
    n_mt = int(total_samples * 0.17)
    for _ in range(n_mt):
        s = random.choice(subject_pool)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        prompt = f"User: Note that {s} is located in {c1}.\nAssistant: Understood, I have noted that {s} is in {c1}.\nUser: Actually, {s} just moved to {c2}.\nAssistant: Noted the change.\nUser: Where is {s} now?\nAssistant: "
        completion = c2
        samples.append({
            "type": "multiturn_state_tracking",
            "prompt": prompt,
            "completion": completion,
            "is_relational": True
        })
        
    # 4. Epistemic Missing-Information Abstention (12%)
    n_abstain = int(total_samples * 0.12)
    for _ in range(n_abstain):
        s1, s2 = random.sample(subject_pool, 2)
        c1 = random.choice(TRAIN_CITIES)
        prompt = f"Mapping: {s1} lives in {c1}. Query: Where does {s2} live? Answer: "
        completion = f"The context does not state where {s2} lives."
        samples.append({
            "type": "missing_info_abstention",
            "prompt": prompt,
            "completion": completion,
            "is_relational": False
        })
        
    # 5. Elementary Reasoning & Arithmetic (12%)
    n_reason = int(total_samples * 0.12)
    for _ in range(n_reason):
        if random.random() < 0.5:
            s1, s2 = random.sample(subject_pool, 2)
            obj = random.choice(OBJECTS)
            init_a = random.randint(4, 9)
            give_qty = random.randint(1, init_a - 1)
            rem_a = init_a - give_qty
            prompt = f"Context: {s1} has {init_a} {obj}. {s1} gives {give_qty} {obj} to {s2}. Question: How many {obj} does {s1} have left? Answer: "
            completion = str(rem_a)
        else:
            s = random.choice(subject_pool)
            c = random.choice(TRAIN_CITIES)
            is_pos = (random.random() < 0.5)
            actual_c = c if is_pos else random.choice([ct for ct in TRAIN_CITIES if ct != c])
            prompt = f"Rule: If a person is in {c}, they have a green pass. Premise: {s} is in {actual_c}. Question: Does {s} have a green pass? Answer: "
            completion = "Yes" if is_pos else "No"
        samples.append({
            "type": "elementary_reasoning",
            "prompt": prompt,
            "completion": completion,
            "is_relational": True
        })
        
    # 6. Conversational Preservation (17%)
    n_conv = total_samples - len(samples)
    for _ in range(n_conv):
        p, a = random.choice(CONVERSATIONAL_PAIRS)
        samples.append({
            "type": "conversational_preservation",
            "prompt": p,
            "completion": a,
            "is_relational": False
        })
        
    random.shuffle(samples)
    return samples

def main():
    random.seed(101)
    train_data = generate_balanced_samples(TRAIN_SUBJECTS, 2400)
    val_data = generate_balanced_samples(VAL_SUBJECTS, 300)
    
    with open(DATA_DIR / "train_candidate2.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(DATA_DIR / "val_candidate2.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
        
    print(f"Candidate 2 dataset generated: {len(train_data)} train, {len(val_data)} val.")

if __name__ == "__main__":
    main()

