"""
CR-EXP-12 Phase 9: Training Curriculum Generator
Constructs datasets for:
1. Candidate A: Memory and Entity Binding Specialist
   - Multi-turn working memory with explicit subject anchoring
   - Multi-step state updates (A -> B -> C)
   - Interleaved multi-entity state updates
   - Counterfactual recency updates
   - Single-hop relational retrieval (positive & negative balanced)
   - Elementary arithmetic restoration stream
   - Conversational persona anchoring

2. Candidate B: Balanced Conversational Consolidation Champion
   - Balanced mix across all 9 capability domains:
     * Natural conversation (persona & fluency)
     * Entity retrieval & distractor disambiguation
     * Multi-turn working memory
     * Counterfactual state updates
     * Balanced answerability (50/50 positive/abstention)
     * Elementary arithmetic (restoring 0% -> >50%)
     * Conditional logic (preserving >50%)
     * Multi-category constraint satisfaction (animals, colors, fruits, vehicles, furniture)

All training entities, cities, and professions are strictly held-out from the frozen benchmark.
"""

import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_12_conversational_consolidation" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 4242
random.seed(SEED)

TRAIN_NAMES = [
    "Aarav", "Meera", "Kabir", "Neha", "Rohan", "Ananya", "Vikram", "Pooja", "Aditya", "Sneha",
    "Kunal", "Tanvi", "Siddharth", "Ishita", "Varun", "Riya", "Gaurav", "Kavya", "Deepak", "Simran",
    "Naveen", "Swati", "Rahul", "Divya", "Amit", "Shreya", "Manish", "Preeti", "Sanjay", "Ankita",
    "Alok", "Priya", "Harsh", "Karan", "Payal", "Sachin", "Archana", "Mayank", "Tarun", "Komal"
]

TRAIN_CITIES = [
    "Jaipur", "Pune", "Lucknow", "Bhopal", "Indore", "Patna", "Nagpur", "Kochi", "Madurai", "Amritsar",
    "Udaipur", "Gwalior", "Jodhpur", "Ranchi", "Guwahati", "Mysuru", "Varanasi", "Agra", "Chandigarh", "Surat"
]

TRAIN_PROFESSIONS = [
    "architect", "doctor", "engineer", "teacher", "lawyer", "scientist", "artist", "pilot", "chef", "geologist"
]

def make_candidate_a_dataset(num_samples=3200):
    # Candidate A: Heavy Working Memory & Entity Binding Specialist + Arithmetic
    samples = []
    
    # 1. Multi-turn dialogue with explicit subject anchoring (1000)
    for _ in range(1000):
        name = random.choice(TRAIN_NAMES)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        p = (
            f"User: Please track this update: {name} is staying in {c1}.\n"
            f"Assistant: Understood, I have noted that {name} is in {c1}.\n"
            f"User: Quick change, {name} just moved over to {c2}.\n"
            f"Assistant: Noted the new location.\n"
            f"User: Where is {name} now?\n"
            f"Assistant: "
        )
        # CRITICAL: Include subject name in completion to stop subject substitution error!
        samples.append({"type": "multiturn_anchored", "prompt": p, "completion": f"{name} is currently in {c2}.", "weight": 3.0})
        
    # 2. Multi-step state updates (A -> B -> C) (600)
    for _ in range(600):
        name = random.choice(TRAIN_NAMES)
        c1, c2, c3 = random.sample(TRAIN_CITIES, 3)
        p = f"Context: {name} started in {c1}. Then {name} moved to {c2}. Finally, {name} settled in {c3}.\nQuestion: Where is {name} now?\nAnswer: "
        samples.append({"type": "multistep_update", "prompt": p, "completion": f"{name} is currently in {c3}.", "weight": 3.0})
        
    # 3. Interleaved 2-entity updates (500)
    for _ in range(500):
        n1, n2 = random.sample(TRAIN_NAMES, 2)
        c1, c2, c3, c4 = random.sample(TRAIN_CITIES, 4)
        target = random.choice([n1, n2])
        target_city = c3 if target == n1 else c4
        p = (
            f"Context: {n1} was in {c1}. {n2} was in {c2}. Later, {n1} moved to {c3}. Also, {n2} moved to {c4}.\n"
            f"Question: Where does {target} live now?\n"
            f"Answer: "
        )
        samples.append({"type": "interleaved_update", "prompt": p, "completion": f"{target} is currently in {target_city}.", "weight": 3.0})
        
    # 4. Elementary Arithmetic Restoration (400)
    for _ in range(400):
        name = random.choice(TRAIN_NAMES)
        init_val = random.randint(5, 20)
        delta = random.randint(2, 8)
        add = random.random() < 0.5
        final_val = init_val + delta if add else init_val - delta
        act = f"bought {delta} more" if add else f"gave away {delta}"
        p = f"Context: {name} started with {init_val} pens. Then {name} {act}.\nQuestion: How many pens does {name} have now?\nAnswer: "
        samples.append({"type": "arithmetic", "prompt": p, "completion": f"{final_val}.", "weight": 2.5})
        
    # 5. Conversational & Persona Anchoring (400)
    conv_anchors = [
        ("User: Hello!\nAssistant: ", "Hello! How can I assist you with your questions today?"),
        ("User: Namaste!\nAssistant: ", "Namaste! I am here to assist you with reasoning and knowledge."),
        ("User: Who are you?\nAssistant: ", "I am ChakrView, an indigenous low-resource neural intelligence."),
        ("User: Can you assist me?\nAssistant: ", "Yes, I am happy to assist you. What would you like to know?"),
        ("User: Good morning.\nAssistant: ", "Good morning! I am ChakrView, ready to help you.")
    ]
    for _ in range(400):
        p, c = random.choice(conv_anchors)
        samples.append({"type": "conversational", "prompt": p, "completion": c, "weight": 1.5})
        
    # 6. Balanced Abstention (300)
    for _ in range(300):
        n1, n2 = random.sample(TRAIN_NAMES, 2)
        city = random.choice(TRAIN_CITIES)
        p = f"Context: {n1} works in {city}.\nQuestion: Where does {n2} work?\nAnswer: "
        samples.append({"type": "abstention", "prompt": p, "completion": "The context does not state where that person works.", "weight": 2.0})
        
    random.shuffle(samples)
    return samples

def make_candidate_b_dataset(num_samples=3200):
    # Candidate B: Full-Spectrum Conversational Consolidation
    samples = []
    
    # 1. Multi-turn Working Memory (600)
    for _ in range(600):
        name = random.choice(TRAIN_NAMES)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        p = (
            f"User: Please track this update: {name} is staying in {c1}.\n"
            f"Assistant: Understood, I have noted that {name} is in {c1}.\n"
            f"User: Quick change, {name} just moved over to {c2}.\n"
            f"Assistant: Noted the new location.\n"
            f"User: Where is {name} now?\n"
            f"Assistant: "
        )
        samples.append({"type": "multiturn_anchored", "prompt": p, "completion": f"{name} is currently in {c2}.", "weight": 2.5})
        
    # 2. Counterfactual Updates (400)
    for _ in range(400):
        name = random.choice(TRAIN_NAMES)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        p = f"Context: Initially, {name} lived in {c1}. Later on, {name} relocated to {c2}.\nQuestion: Where does {name} live currently?\nAnswer: "
        samples.append({"type": "counterfactual", "prompt": p, "completion": f"{c2}.", "weight": 2.5})
        
    # 3. Single-hop Retrieval & Distractor (500)
    for _ in range(250):
        name = random.choice(TRAIN_NAMES)
        city = random.choice(TRAIN_CITIES)
        p = f"Context: {name} lives in {city}.\nQuestion: Where does {name} live?\nAnswer: "
        samples.append({"type": "single_hop_loc", "prompt": p, "completion": f"{city}.", "weight": 2.5})
    for _ in range(250):
        names = random.sample(TRAIN_NAMES, 3)
        cities = random.sample(TRAIN_CITIES, 3)
        tgt_idx = random.randint(0, 2)
        p = f"Context: {names[0]} is in {cities[0]}. {names[1]} is in {cities[1]}. {names[2]} is in {cities[2]}.\nQuestion: Where is {names[tgt_idx]} currently located?\nAnswer: "
        samples.append({"type": "distractor", "prompt": p, "completion": f"{cities[tgt_idx]}.", "weight": 2.5})
        
    # 4. Balanced Abstention (400)
    for _ in range(200):
        n1, n2 = random.sample(TRAIN_NAMES, 2)
        city = random.choice(TRAIN_CITIES)
        p = f"Context: {n1} works in {city}.\nQuestion: Where does {n2} work?\nAnswer: "
        samples.append({"type": "abstention_ent", "prompt": p, "completion": "The context does not state where that person works.", "weight": 2.0})
    for _ in range(200):
        name = random.choice(TRAIN_NAMES)
        city = random.choice(TRAIN_CITIES)
        p = f"Context: {name} lives in {city}.\nQuestion: What is {name}'s profession?\nAnswer: "
        samples.append({"type": "abstention_attr", "prompt": p, "completion": f"The context does not state {name}'s profession.", "weight": 2.0})
        
    # 5. Arithmetic Restoration (400)
    for _ in range(400):
        name = random.choice(TRAIN_NAMES)
        init_val = random.randint(5, 25)
        delta = random.randint(2, 9)
        add = random.random() < 0.5
        final_val = init_val + delta if add else init_val - delta
        act = f"bought {delta} more" if add else f"gave away {delta}"
        p = f"Context: {name} started with {init_val} pens. Then {name} {act}.\nQuestion: How many pens does {name} have now?\nAnswer: "
        samples.append({"type": "arithmetic", "prompt": p, "completion": f"{final_val}.", "weight": 2.5})
        
    # 6. Multi-Category Constraint Satisfaction (400) - FIXES THE 10% GENERALIZATION COLLAPSE
    const_classes = [
        ("fruit", ["apple", "pear", "banana", "mango", "orange"], ["blue", "table", "car", "chair", "pencil"]),
        ("animal", ["elephant", "lion", "tiger", "bear", "wolf"], ["car", "spoon", "desk", "blue", "apple"]),
        ("color", ["red", "blue", "green", "yellow", "purple"], ["tiger", "chair", "mango", "truck", "fork"]),
        ("vehicle", ["truck", "car", "airplane", "bicycle", "bus"], ["banana", "orange", "desk", "lion", "red"]),
        ("furniture", ["desk", "sofa", "chair", "table", "bed"], ["grape", "purple", "truck", "elephant", "green"])
    ]
    for _ in range(400):
        c_name, targets, foils = random.choice(const_classes)
        t_item = random.choice(targets)
        f_item = random.choice(foils)
        order = [t_item, f_item] if random.random() < 0.5 else [f_item, t_item]
        p = f"Constraint: Pick the {c_name} only and strictly ignore other items.\nOptions: {order[0]}, {order[1]}.\nAnswer: "
        samples.append({"type": "constraint_broad", "prompt": p, "completion": t_item, "weight": 2.5})
        
    # 7. Conditional Logic & Conversational Anchoring (500)
    for _ in range(250):
        name = random.choice(TRAIN_NAMES)
        is_true = random.random() < 0.5
        p = (
            f"Premise: Whenever the weather is sunny, {name} wears sunglasses. Otherwise, {name} does not.\n"
            f"Observation: Today the weather is {'sunny' if is_true else 'cloudy'}.\n"
            f"Question: Does {name} wear sunglasses today? (Answer Yes or No)\n"
            f"Answer: "
        )
        samples.append({"type": "logic", "prompt": p, "completion": "Yes" if is_true else "No", "weight": 2.0})
    for _ in range(250):
        conv_anchors = [
            ("User: Hello!\nAssistant: ", "Hello! How can I assist you with your questions today?"),
            ("User: Namaste!\nAssistant: ", "Namaste! I am here to assist you with reasoning and knowledge."),
            ("User: Who are you?\nAssistant: ", "I am ChakrView, an indigenous low-resource neural intelligence."),
            ("User: Can you assist me?\nAssistant: ", "Yes, I am happy to assist you. What would you like to know?")
        ]
        p, c = random.choice(conv_anchors)
        samples.append({"type": "conversational", "prompt": p, "completion": c, "weight": 1.5})
        
    random.shuffle(samples)
    return samples

def main():
    ds_a = make_candidate_a_dataset(3200)
    ds_b = make_candidate_b_dataset(3200)
    
    path_a = DATA_DIR / "train_candidate_a.json"
    path_b = DATA_DIR / "train_candidate_b.json"
    
    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(ds_a, f, indent=2)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(ds_b, f, indent=2)
        
    val_samples = ds_a[:250] + ds_b[:250]
    path_val = DATA_DIR / "val.json"
    with open(path_val, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
        
    print(f"Candidate A Dataset: {len(ds_a)} samples")
    print(f"Candidate B Dataset: {len(ds_b)} samples")
    print(f"Validation Dataset: {len(val_samples)} samples")

if __name__ == "__main__":
    main()

