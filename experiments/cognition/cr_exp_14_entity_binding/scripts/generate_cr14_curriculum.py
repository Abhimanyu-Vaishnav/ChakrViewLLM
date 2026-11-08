"""
CR-EXP-14: Curriculum Generation Script
Generates disjoint training datasets for Candidate A, Candidate B, Candidate C, and Validation.
Guarantees ZERO overlap with the 1,200 sealed test probes:
- Strictly disjoint entities
- Strictly disjoint locations
- Diverse conversational phrasing
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_14_entity_binding" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Sealed Benchmark sets to strictly exclude
BENCH_INFO_PATH = DATA_DIR / "benchmark_entities_locations.json"
with open(BENCH_INFO_PATH, "r", encoding="utf-8") as f:
    bench_info = json.load(f)

BENCH_ENTITIES = set(bench_info["benchmark_entities"])
BENCH_LOCATIONS = set(bench_info["benchmark_locations"])

# Training candidate entity names (60 names, guaranteed disjoint from benchmark)
TRAIN_NAMES = [
    "Aaditya", "Aakash", "Aalok", "Aarohi", "Aarti", "Aditi", "Alok", "Amita", "Amitabh", "Amrita",
    "Anand", "Aniket", "Anil", "Anita", "Anjali", "Ankush", "Anshu", "Anurag", "Aparna", "Archna",
    "Ashish", "Ashok", "Avinash", "Barkha", "Bhavesh", "Bhupendra", "Chhaya", "Chinmay", "Deepak", "Deven",
    "Dharmesh", "Dinesh", "Divya", "Ganesh", "Gaurav", "Geeta", "Hemant", "Ishaan", "Jayesh", "Jitendra",
    "Kalpesh", "Kamal", "Kavita", "Kishore", "Kunal", "Madhuri", "Manoj", "Mayur", "Meera", "Mukesh",
    "Naresh", "Neelam", "Nilesh", "Nirmal", "Pallavi", "Pankaj", "Parul", "Pradeep", "Pramod", "Pratap"
]

# Validation entity names (20 names, guaranteed disjoint from benchmark & train)
VAL_NAMES = [
    "Prashant", "Pratibha", "Preeti", "Ragini", "Rajani", "Rajeev", "Rajesh", "Rakesh", "Ramesh", "Rashmi",
    "Ravi", "Reena", "Rekha", "Rishi", "Rohini", "Rohit", "Rupal", "Sachin", "Sadhana", "Sameer"
]

# Training locations (35 cities, guaranteed disjoint from benchmark)
TRAIN_CITIES = [
    "Agartala", "Ahmedabad", "Aizawl", "Aligarh", "Allahabad", "Ambala", "Amritsar", "Aurangabad",
    "Bengaluru", "Bhopal", "Bhubaneswar", "Bilaspur", "Chandigarh", "Chennai", "Coimbatore", "Dehradun",
    "Delhi", "Faridabad", "Ghaziabad", "Gorakhpur", "Gwalior", "Hyderabad", "Indore", "Jabalpur",
    "Jaipur", "Jalandhar", "Jamshedpur", "Jodhpur", "Kanpur", "Kochi", "Kolkata", "Kota",
    "Lucknow", "Madurai", "Meerut"
]

# Validation locations (12 cities, guaranteed disjoint from benchmark & train)
VAL_CITIES = [
    "Moradabad", "Mumbai", "Mysuru", "Nagpur", "Nashik", "Noida", "Panaji", "Patna",
    "Pondicherry", "Pune", "Raipur", "Ranchi"
]

# Verification of disjoint sets
assert len(set(TRAIN_NAMES).intersection(BENCH_ENTITIES)) == 0, f"Train names overlap with Benchmark: {set(TRAIN_NAMES).intersection(BENCH_ENTITIES)}"
assert len(set(VAL_NAMES).intersection(BENCH_ENTITIES)) == 0, f"Val names overlap with Benchmark: {set(VAL_NAMES).intersection(BENCH_ENTITIES)}"
assert len(set(TRAIN_NAMES).intersection(set(VAL_NAMES))) == 0, f"Train names overlap with Val names: {set(TRAIN_NAMES).intersection(set(VAL_NAMES))}"

assert len(set(TRAIN_CITIES).intersection(BENCH_LOCATIONS)) == 0, f"Train cities overlap with Benchmark: {set(TRAIN_CITIES).intersection(BENCH_LOCATIONS)}"
assert len(set(VAL_CITIES).intersection(BENCH_LOCATIONS)) == 0, f"Val cities overlap with Benchmark: {set(VAL_CITIES).intersection(BENCH_LOCATIONS)}"
assert len(set(TRAIN_CITIES).intersection(set(VAL_CITIES))) == 0, f"Train cities overlap with Val cities: {set(TRAIN_CITIES).intersection(set(VAL_CITIES))}"

print(f"Safety check passed. Train Entities: {len(TRAIN_NAMES)}, Val Entities: {len(VAL_NAMES)}")
print(f"Train Locations: {len(TRAIN_CITIES)}, Val Locations: {len(VAL_CITIES)}")

random.seed(42)

STATEMENT_TEMPLATES = [
    "{e} lives in {c}.",
    "{e} is currently residing in {c}.",
    "{e} is located in {c}.",
    "{e} is staying in {c}.",
    "{e} has moved to {c}.",
    "{e} relocated to {c} recently."
]

UPDATE_TEMPLATES = [
    "{e} moved to {c}.",
    "{e} has now relocated to {c}.",
    "{e} changed location to {c}.",
    "{e} is now staying in {c}.",
    "Later, {e} traveled and settled in {c}."
]

QUESTION_TEMPLATES = [
    "Where is {e} currently located?",
    "Where does {e} live now?",
    "What is the current city of {e}?",
    "Where can {e} be found at present?",
    "In which city is {e} now?"
]

DISTRACTOR_STATEMENTS = [
    "The weather has been quite pleasant today.",
    "The quarterly reports have been processed successfully.",
    "A team meeting is scheduled for tomorrow afternoon.",
    "Public transportation options have expanded recently.",
    "Several improvements were made to local infrastructure."
]

def make_turn_dialogue(updates, query_entity, expected_city, use_delimiters=False, distractor=None):
    turns = []
    t_idx = 1
    for item in updates:
        e, c = item
        if use_delimiters:
            stmt = f"<ent>{e}</ent> is currently in <val>{c}</val>."
        else:
            tmpl = random.choice(STATEMENT_TEMPLATES if t_idx == 1 else UPDATE_TEMPLATES)
            stmt = tmpl.format(e=e, c=c)
        turns.append(f"Turn {t_idx}:\nUser: {stmt}")
        t_idx += 1
        
    if distractor:
        turns.append(f"Turn {t_idx}:\nUser: {distractor}")
        t_idx += 1
        
    q_tmpl = random.choice(QUESTION_TEMPLATES)
    if use_delimiters:
        q_text = f"Where is <ent>{query_entity}</ent> located?"
        ans_text = f"<ent>{query_entity}</ent> is currently in <val>{expected_city}</val>."
    else:
        q_text = q_tmpl.format(e=query_entity)
        ans_text = f"{query_entity} is currently in {expected_city}."
        
    turns.append(f"Turn {t_idx}:\nUser: {q_text}\nAssistant: {ans_text}")
    return "\n\n".join(turns)

def generate_dataset_a(n=600):
    # Candidate A: Explicit Subject-Value Conditioning (Single and multi updates, focused on subject repetition)
    samples = []
    for _ in range(n):
        num_entities = random.choice([1, 2])
        ents = random.sample(TRAIN_NAMES, num_entities)
        updates = []
        state = {}
        for e in ents:
            c = random.choice(TRAIN_CITIES)
            updates.append((e, c))
            state[e] = c
        if random.random() < 0.5:
            e_up = random.choice(ents)
            c2 = random.choice([c for c in TRAIN_CITIES if c != state[e_up]])
            updates.append((e_up, c2))
            state[e_up] = c2
        q_ent = random.choice(ents)
        text = make_turn_dialogue(updates, q_ent, state[q_ent], use_delimiters=False)
        samples.append({"text": text})
    return samples

def generate_dataset_b(n=600):
    # Candidate B: Interleaved State-Update Curriculum (Complex multi-entity, distractors, counterfactuals)
    samples = []
    for _ in range(n):
        num_entities = random.choice([2, 3, 4])
        ents = random.sample(TRAIN_NAMES, num_entities)
        updates = []
        state = {}
        for e in ents:
            c = random.choice(TRAIN_CITIES)
            updates.append((e, c))
            state[e] = c
        num_extra_updates = random.randint(1, 3)
        for _ in range(num_extra_updates):
            e_up = random.choice(ents)
            c2 = random.choice([c for c in TRAIN_CITIES if c != state[e_up]])
            updates.append((e_up, c2))
            state[e_up] = c2
        dist = random.choice(DISTRACTOR_STATEMENTS) if random.random() < 0.4 else None
        q_ent = random.choice(ents)
        text = make_turn_dialogue(updates, q_ent, state[q_ent], use_delimiters=False, distractor=dist)
        samples.append({"text": text})
    return samples

def generate_dataset_c(n=600):
    # Candidate C: Structured Entity Delimiters (<ent>...</ent>, <val>...</val>)
    samples = []
    for _ in range(n):
        num_entities = random.choice([2, 3])
        ents = random.sample(TRAIN_NAMES, num_entities)
        updates = []
        state = {}
        for e in ents:
            c = random.choice(TRAIN_CITIES)
            updates.append((e, c))
            state[e] = c
        num_extra_updates = random.randint(1, 2)
        for _ in range(num_extra_updates):
            e_up = random.choice(ents)
            c2 = random.choice([c for c in TRAIN_CITIES if c != state[e_up]])
            updates.append((e_up, c2))
            state[e_up] = c2
        dist = random.choice(DISTRACTOR_STATEMENTS) if random.random() < 0.3 else None
        q_ent = random.choice(ents)
        text = make_turn_dialogue(updates, q_ent, state[q_ent], use_delimiters=True, distractor=dist)
        samples.append({"text": text})
    return samples

def generate_val_dataset(n=150):
    # Validation split using strictly held-out val entities and cities
    samples = []
    for _ in range(n):
        num_entities = random.choice([1, 2, 3])
        ents = random.sample(VAL_NAMES, num_entities)
        updates = []
        state = {}
        for e in ents:
            c = random.choice(VAL_CITIES)
            updates.append((e, c))
            state[e] = c
        if random.random() < 0.5:
            e_up = random.choice(ents)
            c2 = random.choice([c for c in VAL_CITIES if c != state[e_up]])
            updates.append((e_up, c2))
            state[e_up] = c2
        q_ent = random.choice(ents)
        text = make_turn_dialogue(updates, q_ent, state[q_ent], use_delimiters=False)
        samples.append({"text": text})
    return samples

def main():
    print("Generating Candidate A training dataset (600 samples)...")
    data_a = generate_dataset_a(600)
    with open(DATA_DIR / "train_candidate_a.json", "w", encoding="utf-8") as f:
        json.dump(data_a, f, indent=2)

    print("Generating Candidate B training dataset (600 samples)...")
    data_b = generate_dataset_b(600)
    with open(DATA_DIR / "train_candidate_b.json", "w", encoding="utf-8") as f:
        json.dump(data_b, f, indent=2)

    print("Generating Candidate C training dataset (600 samples)...")
    data_c = generate_dataset_c(600)
    with open(DATA_DIR / "train_candidate_c.json", "w", encoding="utf-8") as f:
        json.dump(data_c, f, indent=2)

    print("Generating Validation dataset (150 samples)...")
    data_val = generate_val_dataset(150)
    with open(DATA_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(data_val, f, indent=2)

    print("Curriculum generation complete. All datasets saved safely to data/")

if __name__ == "__main__":
    main()
