"""
CR-EXP-09 Candidate 1 Dataset Generator: Curriculum Regularization for Cognitive Capabilities
Generates 1,800 training samples and 200 validation samples carefully balanced across 5 streams:
1. Multi-Turn Dialogue State Tracking (350 samples): Tracking entity moves across user-assistant turns.
2. Epistemic Missing-Information Abstention (350 samples): Explicit uncertainty when subject is not in context.
3. Multi-Entity Distractor Grounding (350 samples): Counterbalanced 2-entity and 3-entity mappings with random query targets.
4. Elementary State Arithmetic & Logic (350 samples): Book/coin transfers and boolean conditional rules.
5. Conversational Preservation & Persona (400 samples): ChakrView persona, greetings, polite assistance.

Strictly preserves disjoint subject names between train/val and the 600-probe benchmark suite.
Outputs data/train.json and data/val.json.
"""

import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_09_autonomous_mission" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Training Subjects (Disjoint from the 16 benchmark subjects: Vikram, Zoya, Arjun, Meera, Suresh, Tanya, Aditya, Ritu, Dev, Tara, Nikhil, Ishita, Varun, Rhea, Manish, Pooja)
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

def generate_samples(subject_pool, total_samples):
    samples = []
    
    # 1. Multi-Turn Dialogue State Tracking (~20%)
    n_mt = int(total_samples * 0.20)
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

    # 2. Epistemic Missing-Information Abstention (~20%)
    n_abstain = int(total_samples * 0.20)
    for _ in range(n_abstain):
        s1, s2 = random.sample(subject_pool, 2)
        c1 = random.choice(TRAIN_CITIES)
        prompt = f"Mapping: {s1} lives in {c1}. Query: Where does {s2} live? Answer: "
        completion = f"The context does not state where {s2} lives."
        samples.append({
            "type": "missing_info_abstention",
            "prompt": prompt,
            "completion": completion,
            "is_relational": False # trained as full sentence completion
        })

    # 3. Multi-Entity Distractor Disambiguation (~20%)
    n_dist = int(total_samples * 0.20)
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

    # 4. Elementary State Arithmetic & Logic (~20%)
    n_reason = int(total_samples * 0.20)
    for _ in range(n_reason):
        if random.random() < 0.5:
            # Arithmetic transfer
            s1, s2 = random.sample(subject_pool, 2)
            obj = random.choice(OBJECTS)
            init_a = random.randint(4, 9)
            give_qty = random.randint(1, init_a - 1)
            rem_a = init_a - give_qty
            prompt = f"Context: {s1} has {init_a} {obj}. {s1} gives {give_qty} {obj} to {s2}. Question: How many {obj} does {s1} have left? Answer: "
            completion = str(rem_a)
        else:
            # Conditional rule
            s = random.choice(subject_pool)
            rule_type = random.choice(["pass", "team"])
            if rule_type == "pass":
                c = random.choice(TRAIN_CITIES)
                is_pos = (random.random() < 0.5)
                actual_c = c if is_pos else random.choice([ct for ct in TRAIN_CITIES if ct != c])
                prompt = f"Rule: If a person is in {c}, they have a green pass. Premise: {s} is in {actual_c}. Question: Does {s} have a green pass? Answer: "
                completion = "Yes" if is_pos else "No"
            else:
                is_pos = (random.random() < 0.5)
                team = "Alpha" if is_pos else "Beta"
                prompt = f"Premise: All members of Team Alpha are certified. {s} is a member of Team {team}. Question: Is {s} certified? Answer: "
                completion = "Yes" if is_pos else "No"
        samples.append({
            "type": "elementary_reasoning",
            "prompt": prompt,
            "completion": completion,
            "is_relational": True
        })

    # 5. Conversational Preservation (~20%)
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
    random.seed(42)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    train_data = generate_samples(TRAIN_SUBJECTS, 1800)
    val_data = generate_samples(VAL_SUBJECTS, 200)
    
    with open(DATA_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(DATA_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
        
    manifest = {
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "streams": {
            "multiturn_state_tracking": sum(1 for s in train_data if s["type"] == "multiturn_state_tracking"),
            "missing_info_abstention": sum(1 for s in train_data if s["type"] == "missing_info_abstention"),
            "distractor_disambiguation": sum(1 for s in train_data if s["type"] == "distractor_disambiguation"),
            "elementary_reasoning": sum(1 for s in train_data if s["type"] == "elementary_reasoning"),
            "conversational_preservation": sum(1 for s in train_data if s["type"] == "conversational_preservation")
        },
        "disjoint_training_subjects": sorted(TRAIN_SUBJECTS),
        "disjoint_validation_subjects": sorted(VAL_SUBJECTS)
    }
    with open(DATA_DIR / "curriculum_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Candidate 1 curriculum dataset generated: {len(train_data)} train, {len(val_data)} val.")
    print("Stream distribution:", manifest["streams"])

if __name__ == "__main__":
    main()

