"""
CR-EXP-15: Training Curriculum Generation Script
Generates controlled, comparable curricula across 4 experimental arms:
- Candidate A: Standard language-model control (dialogues without targeted binding enforcement)
- Candidate B: Entity copying and selection curriculum
- Candidate C: Static subject-value binding curriculum
- Candidate D: Dynamic state updates and interleaved pointer-routing curriculum
- Validation: Held-out validation split
All entity names and locations are strictly disjoint from the sealed test benchmark!
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_15_architecture_binding" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 40 Test entities (from sealed benchmark) to strictly exclude
BENCH_ENTITIES = {
    "Apurva", "Badri", "Bhargav", "Chaitanya", "Devendra", "Digvijay", "Eshwar", "Fateh", "Gautam", "Govind",
    "Harish", "Himanshu", "Indrajit", "Jaideep", "Jawahar", "Kalyan", "Kuldeep", "Lalit", "Madhav", "Manvendra",
    "Mukesh", "Naveen", "Niraj", "Padma", "Pankhuri", "Pranav", "Raghav", "Rajnath", "Ranveer", "Sanjay",
    "Sarvesh", "Shashank", "Siddharth", "Somesh", "Tarun", "Umesh", "Vaibhav", "Varun", "Vikas", "Yashwant"
}

# 25 Test locations to strictly exclude
BENCH_LOCATIONS = {
    "Alwar", "Bareilly", "Belgaum", "Calicut", "Durgapur", "Faridabad", "Gorakhpur", "Haridwar",
    "Imphal", "Jamnagar", "Kakinada", "Kollam", "Malda", "Muzaffarpur", "Nellore", "Pathankot",
    "Rajahmundry", "Rourkela", "Saharanpur", "Thanjavur", "Thrissur", "Udaipur", "Ujjain", "Vellore", "Warangal"
}

# Training entity names (50 names, strictly disjoint from benchmark)
TRAIN_NAMES = [
    "Aaditya", "Aakash", "Aalok", "Aarohi", "Aarti", "Aditi", "Alok", "Amita", "Amitabh", "Amrita",
    "Anand", "Aniket", "Anil", "Anita", "Anjali", "Ankush", "Anshu", "Anurag", "Aparna", "Archna",
    "Ashish", "Ashok", "Avinash", "Barkha", "Bhavesh", "Bhupendra", "Chhaya", "Chinmay", "Deepak", "Deven",
    "Dharmesh", "Dinesh", "Divya", "Ganesh", "Gaurav", "Geeta", "Hemant", "Ishaan", "Jayesh", "Jitendra",
    "Kalpesh", "Kamal", "Kavita", "Kishore", "Kunal", "Madhuri", "Manoj", "Mayur", "Meera", "Pradeep"
]

# Validation entity names (15 names, strictly disjoint from benchmark and train)
VAL_NAMES = [
    "Prashant", "Pratibha", "Preeti", "Ragini", "Rajani", "Rajeev", "Rajesh", "Rakesh", "Ramesh", "Rashmi",
    "Ravi", "Reena", "Rekha", "Rishi", "Rohini"
]

# Training locations (30 cities, strictly disjoint from benchmark)
TRAIN_CITIES = [
    "Agartala", "Ahmedabad", "Aizawl", "Aligarh", "Allahabad", "Ambala", "Amritsar", "Aurangabad",
    "Bengaluru", "Bhopal", "Bhubaneswar", "Bilaspur", "Chandigarh", "Chennai", "Coimbatore", "Dehradun",
    "Delhi", "Gwalior", "Hyderabad", "Indore", "Jabalpur", "Jaipur", "Jalandhar", "Jamshedpur",
    "Jodhpur", "Kanpur", "Kochi", "Kolkata", "Kota", "Lucknow"
]

# Validation locations (10 cities, strictly disjoint from benchmark and train)
VAL_CITIES = [
    "Madurai", "Meerut", "Moradabad", "Mumbai", "Mysuru", "Nagpur", "Nashik", "Noida", "Panaji", "Patna"
]

# Strict assertion checks
assert len(set(TRAIN_NAMES).intersection(BENCH_ENTITIES)) == 0, "Train names overlap benchmark!"
assert len(set(VAL_NAMES).intersection(BENCH_ENTITIES)) == 0, "Val names overlap benchmark!"
assert len(set(TRAIN_NAMES).intersection(set(VAL_NAMES))) == 0, "Train names overlap val!"

assert len(set(TRAIN_CITIES).intersection(BENCH_LOCATIONS)) == 0, "Train cities overlap benchmark!"
assert len(set(VAL_CITIES).intersection(BENCH_LOCATIONS)) == 0, "Val cities overlap benchmark!"
assert len(set(TRAIN_CITIES).intersection(set(VAL_CITIES))) == 0, "Train cities overlap val!"

random.seed(4215)

def gen_candidate_a(n=500):
    # Candidate A: Standard Conversational Control
    # Conversational chit-chat, facts, descriptions without copy/binding tasks
    samples = []
    templates = [
        "User: How is the weather in {c} today?\nAssistant: The weather in {c} is typically pleasant with moderate humidity.",
        "User: Tell me about public transportation in {c}.\nAssistant: {c} offers bus transit and commuter rail connecting major suburban zones.",
        "User: What industries are prominent around {c}?\nAssistant: Manufacturing, commerce, and service technology thrive around {c}.",
        "User: {e} mentioned that the library is open on weekends.\nAssistant: Yes, weekend library hours facilitate community learning."
    ]
    for _ in range(n):
        tmpl = random.choice(templates)
        e = random.choice(TRAIN_NAMES)
        c = random.choice(TRAIN_CITIES)
        samples.append({"text": tmpl.format(e=e, c=c)})
    return samples

def gen_candidate_b(n=500):
    # Candidate B: Entity Copying and Entity Selection Curriculum
    samples = []
    for _ in range(n):
        if random.random() < 0.5:
            # Copying task
            e = random.choice(TRAIN_NAMES)
            tmpl = random.choice([
                "User: Repeat the exact name: {e}.\nAssistant: {e}",
                "User: State the name again: {e}.\nAssistant: {e}",
                "User: Echo the following person: {e}.\nAssistant: {e}"
            ])
            samples.append({"text": tmpl.format(e=e)})
        else:
            # Selection task
            e1, e2 = random.sample(TRAIN_NAMES, 2)
            first = random.random() < 0.5
            target = e1 if first else e2
            q_text = "Who was mentioned first in that sentence?" if first else "Who was mentioned second in that sentence?"
            text = f"Between {e1} and {e2}, two individuals were discussed.\nQuestion: {q_text}\nAnswer: {target}"
            samples.append({"text": text})
    return samples

def gen_candidate_c(n=500):
    # Candidate C: Static Subject-Value Binding Curriculum
    samples = []
    for _ in range(n):
        e1, e2 = random.sample(TRAIN_NAMES, 2)
        c1, c2 = random.sample(TRAIN_CITIES, 2)
        target = random.choice([e1, e2])
        c_target = c1 if target == e1 else c2
        tmpl = random.choice([
            "Context: {e1} lives in {c1}. {e2} lives in {c2}.\nQuestion: Where is {target} located?\nAnswer: {target} is in {c_target}.",
            "Context: Currently, {e1} resides in {c1} while {e2} resides in {c2}.\nQuestion: Where does {target} live?\nAnswer: {target} is currently in {c_target}."
        ])
        samples.append({"text": tmpl.format(e1=e1, c1=c1, e2=e2, c2=c2, target=target, c_target=c_target)})
    return samples

def gen_candidate_d(n=500):
    # Candidate D: Dynamic State Updates & Interleaved Pointer-Routing Curriculum
    samples = []
    for _ in range(n):
        e1, e2 = random.sample(TRAIN_NAMES, 2)
        c1, c2, c3, c4 = random.sample(TRAIN_CITIES, 4)
        q_target = random.choice([e1, e2])
        c_target = c3 if q_target == e1 else c4
        text = (
            f"Turn 1:\nUser: {e1} lives in {c1}.\n\n"
            f"Turn 2:\nUser: {e2} lives in {c2}.\n\n"
            f"Turn 3:\nUser: {e1} moved to {c3}.\n\n"
            f"Turn 4:\nUser: {e2} moved to {c4}.\n\n"
            f"Turn 5:\nUser: Where is {q_target} currently located?\nAssistant: {q_target} is currently in {c_target}."
        )
        samples.append({"text": text})
    return samples

def gen_val_data(n=100):
    samples = []
    for _ in range(n):
        e1, e2 = random.sample(VAL_NAMES, 2)
        c1, c2, c3 = random.sample(VAL_CITIES, 3)
        target = random.choice([e1, e2])
        c_target = c3 if target == e1 else c2
        text = (
            f"Turn 1:\nUser: {e1} lives in {c1}.\n\n"
            f"Turn 2:\nUser: {e2} lives in {c2}.\n\n"
            f"Turn 3:\nUser: {e1} moved to {c3}.\n\n"
            f"Turn 4:\nUser: Where is {target} located?\nAssistant: {target} is in {c_target}."
        )
        samples.append({"text": text})
    return samples

def main():
    print("Generating CR-15 Training Curricula (500 samples each)...")
    data_a = gen_candidate_a(500)
    data_b = gen_candidate_b(500)
    data_c = gen_candidate_c(500)
    data_d = gen_candidate_d(500)
    val_data = gen_val_data(100)

    with open(DATA_DIR / "train_candidate_a.json", "w", encoding="utf-8") as f:
        json.dump(data_a, f, indent=2)
    with open(DATA_DIR / "train_candidate_b.json", "w", encoding="utf-8") as f:
        json.dump(data_b, f, indent=2)
    with open(DATA_DIR / "train_candidate_c.json", "w", encoding="utf-8") as f:
        json.dump(data_c, f, indent=2)
    with open(DATA_DIR / "train_candidate_d.json", "w", encoding="utf-8") as f:
        json.dump(data_d, f, indent=2)
    with open(DATA_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    print("Curricula generation complete.")

if __name__ == "__main__":
    main()
