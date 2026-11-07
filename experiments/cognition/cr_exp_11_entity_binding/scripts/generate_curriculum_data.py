"""
CR-EXP-11 Phase 7: Dataset Curriculum Generator
Builds training datasets for:
1. Candidate A: Balanced Conversational Anchoring
   - Natural conversation (anchoring persona)
   - Single-hop entity binding (both location and profession)
   - Multi-entity distractor retrieval
   - Multi-turn state updates
   - Counterfactual reassignment
   - Missing-information epistemic discrimination (strictly paired 50/50 with positive examples)
   - Basic reasoning & constraint satisfaction

2. Candidate B: Structured Entity-State Curriculum
   - Heavy emphasis on subject-attribute-value representation
   - Variable name permutation across multiple turns
   - Multi-step state updates (A -> B -> C)
   - Distractor-controlled interleaved updates
   - Natural language variations & paraphrases

All training entities, cities, and professions are STRICTLY DISJOINT from the frozen CR-11 benchmark.
"""

import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_11_entity_binding" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1337
random.seed(SEED)

# Training Entities (Strictly disjoint from benchmark names)
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

def make_candidate_a_dataset(num_samples=3000):
    samples = []
    
    # 1. Natural conversation (500)
    conv_templates = [
        ("User: Hello!\nAssistant: ", "Hello! How can I assist you with your questions today?"),
        ("User: Good morning.\nAssistant: ", "Good morning! I am ChakrView, ready to help you."),
        ("User: Namaste!\nAssistant: ", "Namaste! I am here to assist you with reasoning and knowledge."),
        ("User: Who are you?\nAssistant: ", "I am ChakrView, an indigenous low-resource neural intelligence."),
        ("User: Can you assist me?\nAssistant: ", "Yes, I am happy to assist you. What would you like to know?"),
        ("User: Hello ChakrView.\nAssistant: ", "Hello! It is a pleasure to assist you today."),
        ("User: Can you answer my query?\nAssistant: ", "Certainly, please share your context or question."),
        ("User: Good afternoon.\nAssistant: ", "Good afternoon! How may I be of service to you?")
    ]
    for _ in range(500):
        p, c = random.choice(conv_templates)
        samples.append({"type": "conversational", "prompt": p, "completion": c, "weight": 1.5})
        
    # 2. Single-hop positive retrieval (500) - CRITICAL: Resolves the 0% retrieval collapse
    for _ in range(250):
        name = random.choice(TRAIN_NAMES)
        city = random.choice(TRAIN_CITIES)
        p = f"Context: {name} lives in {city}.\nQuestion: Where does {name} live?\nAnswer: "
        samples.append({"type": "single_hop_loc", "prompt": p, "completion": f"{city}.", "weight": 2.5})
    for _ in range(250):
        name = random.choice(TRAIN_NAMES)
        prof = random.choice(TRAIN_PROFESSIONS)
        p = f"Context: {name} works as a {prof}.\nQuestion: What is {name}'s profession?\nAnswer: "
        samples.append({"type": "single_hop_prof", "prompt": p, "completion": f"{prof}.", "weight": 2.5})
        
    # 3. Missing-information abstention (500) - Balanced 50/50 with positive single-hop
    for _ in range(250):
        n1 = random.choice(TRAIN_NAMES)
        n2 = random.choice([n for n in TRAIN_NAMES if n != n1])
        city = random.choice(TRAIN_CITIES)
        p = f"Context: {n1} lives in {city}.\nQuestion: Where does {n2} live?\nAnswer: "
        samples.append({"type": "abstention_ent", "prompt": p, "completion": "The context does not state where that person lives.", "weight": 2.0})
    for _ in range(250):
        name = random.choice(TRAIN_NAMES)
        city = random.choice(TRAIN_CITIES)
        p = f"Context: {name} lives in {city}.\nQuestion: What is {name}'s profession?\nAnswer: "
        samples.append({"type": "abstention_attr", "prompt": p, "completion": f"The context does not state {name}'s profession.", "weight": 2.0})
        
    # 4. Multi-entity distractor disambiguation (500)
    for _ in range(500):
        names = random.sample(TRAIN_NAMES, 3)
        cities = random.sample(TRAIN_CITIES, 3)
        target_idx = random.randint(0, 2)
        target_name = names[target_idx]
        target_city = cities[target_idx]
        p = f"Context: {names[0]} is in {cities[0]}. {names[1]} is in {cities[1]}. {names[2]} is in {cities[2]}.\nQuestion: Where is {target_name} currently located?\nAnswer: "
        samples.append({"type": "distractor", "prompt": p, "completion": f"{target_city}.", "weight": 2.5})
        
    # 5. Multi-turn state updates & Counterfactuals (600)
    for _ in range(300):
        name = random.choice(TRAIN_NAMES)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        p = (
            f"User: Please note that {name} is located in {c1}.\n"
            f"Assistant: Understood, I have noted that {name} is in {c1}.\n"
            f"User: Actually, {name} just moved to {c2}.\n"
            f"Assistant: Noted the change.\n"
            f"User: Where is {name} now?\n"
            f"Assistant: "
        )
        samples.append({"type": "multiturn_update", "prompt": p, "completion": f"{name} is currently in {c2}.", "weight": 2.5})
    for _ in range(300):
        name = random.choice(TRAIN_NAMES)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        p = f"Context: Initially, {name} lived in {c1}. Later on, {name} relocated to {c2}.\nQuestion: Where does {name} live currently?\nAnswer: "
        samples.append({"type": "counterfactual", "prompt": p, "completion": f"{c2}.", "weight": 2.5})
        
    # 6. Logic and Constraint (400)
    for _ in range(200):
        name = random.choice(TRAIN_NAMES)
        is_true = random.choice([True, False])
        p = (
            f"Premise: Whenever the weather is sunny, {name} wears sunglasses. Otherwise, {name} does not.\n"
            f"Observation: Today the weather is {'sunny' if is_true else 'cloudy'}.\n"
            f"Question: Does {name} wear sunglasses today? (Answer Yes or No)\n"
            f"Answer: "
        )
        samples.append({"type": "logic", "prompt": p, "completion": "Yes" if is_true else "No", "weight": 2.0})
    for _ in range(200):
        fruits = ["apple", "banana", "orange", "mango", "grape"]
        non_fruits = ["blue", "green", "chair", "stone", "table"]
        f_item = random.choice(fruits)
        nf_item = random.choice(non_fruits)
        order = [f_item, nf_item] if random.random() < 0.5 else [nf_item, f_item]
        p = f"Constraint: Pick the fruit only and strictly ignore other items.\nOptions: {order[0]}, {order[1]}.\nAnswer: "
        samples.append({"type": "constraint", "prompt": p, "completion": f_item, "weight": 2.0})
        
    random.shuffle(samples)
    return samples

def make_candidate_b_dataset(num_samples=3000):
    # Candidate B emphasizes deep variable binding, multi-step updates (A -> B -> C), and paraphrases
    samples = []
    
    # 1. Multi-step updates (A -> B -> C) (800)
    for _ in range(800):
        name = random.choice(TRAIN_NAMES)
        c1, c2, c3 = random.sample(TRAIN_CITIES, 3)
        style = random.choice([0, 1])
        if style == 0:
            p = f"Context: {name} started in {c1}. Then {name} moved to {c2}. Finally, {name} settled in {c3}.\nQuestion: Where is {name} now?\nAnswer: "
            samples.append({"type": "multistep_cxt", "prompt": p, "completion": f"{c3}.", "weight": 3.0})
        else:
            p = (
                f"User: {name} arrived in {c1}.\n"
                f"Assistant: Recorded.\n"
                f"User: {name} traveled to {c2}.\n"
                f"Assistant: Updated.\n"
                f"User: Now {name} has reached {c3}.\n"
                f"Assistant: Noted.\n"
                f"User: Where can {name} be found right now?\n"
                f"Assistant: "
            )
            samples.append({"type": "multistep_dialogue", "prompt": p, "completion": f"{name} is currently in {c3}.", "weight": 3.0})
            
    # 2. Interleaved 2-entity updates (700)
    for _ in range(700):
        n1, n2 = random.sample(TRAIN_NAMES, 2)
        c1, c2, c3, c4 = random.sample(TRAIN_CITIES, 4)
        target_entity = random.choice([n1, n2])
        target_city = c3 if target_entity == n1 else c4
        p = (
            f"Context: {n1} was in {c1}. {n2} was in {c2}. Later, {n1} moved to {c3}. Also, {n2} moved to {c4}.\n"
            f"Question: Where does {target_entity} live now?\n"
            f"Answer: "
        )
        samples.append({"type": "interleaved_update", "prompt": p, "completion": f"{target_city}.", "weight": 3.0})
        
    # 3. Positive single-hop & distractor with diverse phrasing (700)
    verbs = ["resides in", "is based in", "works from", "is located in", "lives in"]
    for _ in range(400):
        name = random.choice(TRAIN_NAMES)
        city = random.choice(TRAIN_CITIES)
        v = random.choice(verbs)
        p = f"Context: {name} {v} {city}.\nQuestion: Where does {name} stay?\nAnswer: "
        samples.append({"type": "paraphrased_retrieval", "prompt": p, "completion": f"{city}.", "weight": 2.5})
    for _ in range(300):
        names = random.sample(TRAIN_NAMES, 3)
        cities = random.sample(TRAIN_CITIES, 3)
        tgt_idx = random.randint(0, 2)
        p = f"Context: {names[0]} works in {cities[0]}. {names[1]} works in {cities[1]}. {names[2]} works in {cities[2]}.\nQuestion: Where does {names[tgt_idx]} work?\nAnswer: "
        samples.append({"type": "distractor_work", "prompt": p, "completion": f"{cities[tgt_idx]}.", "weight": 2.5})
        
    # 4. Balanced Abstention (400)
    for _ in range(400):
        n1, n2 = random.sample(TRAIN_NAMES, 2)
        city = random.choice(TRAIN_CITIES)
        p = f"Context: {n1} works in {city}.\nQuestion: Where does {n2} work?\nAnswer: "
        samples.append({"type": "abstention_balanced", "prompt": p, "completion": "The context does not state where that person works.", "weight": 2.0})
        
    # 5. Conversational Anchoring (400)
    conv_anchors = [
        ("User: Hello!\nAssistant: ", "Hello! How can I assist you with your questions today?"),
        ("User: Namaste!\nAssistant: ", "Namaste! I am here to assist you with reasoning and knowledge."),
        ("User: Who are you?\nAssistant: ", "I am ChakrView, an indigenous low-resource neural intelligence."),
        ("User: Can you help me?\nAssistant: ", "Yes, I am happy to assist you. What would you like to know?")
    ]
    for _ in range(400):
        p, c = random.choice(conv_anchors)
        samples.append({"type": "conversational", "prompt": p, "completion": c, "weight": 1.5})
        
    random.shuffle(samples)
    return samples

def main():
    ds_a = make_candidate_a_dataset(3000)
    ds_b = make_candidate_b_dataset(3000)
    
    path_a = DATA_DIR / "train_candidate_a.json"
    path_b = DATA_DIR / "train_candidate_b.json"
    
    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(ds_a, f, indent=2)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(ds_b, f, indent=2)
        
    # Validation set (500 items, mixed)
    val_samples = ds_a[:250] + ds_b[:250]
    path_val = DATA_DIR / "val.json"
    with open(path_val, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
        
    print(f"Generated train_candidate_a.json: {len(ds_a)} samples")
    print(f"Generated train_candidate_b.json: {len(ds_b)} samples")
    print(f"Generated val.json: {len(val_samples)} samples")

if __name__ == "__main__":
    main()

