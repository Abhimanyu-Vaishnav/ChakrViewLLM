import json
import random

random.seed(42)

with open("experiments/cognition/cr_exp_09_autonomous_mission/data/train_candidate2.json", "r", encoding="utf-8") as f:
    train_c2 = json.load(f)

extra_samples = []

entities = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Neha", "Kabir", "Meera", "Aditya", "Tara", "Karan", "Simran", "Arjun", "Pooja"]
cities = ["Delhi", "Mumbai", "Jaipur", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Lucknow", "Goa"]
fruits = ["apple", "banana", "mango", "orange", "grapes", "peach", "guava", "pineapple"]
colors = ["red", "blue", "green", "yellow", "purple", "white", "black"]

# 1. 300 Conditional Logic Samples
for _ in range(300):
    p1 = random.choice(entities)
    c1 = random.choice(cities)
    r = random.choice([True, False])
    if r:
        prompt = f"Rule: If {p1} lives in {c1}, then {p1} visits the market.\nContext: {p1} lives in {c1}.\nQuestion: Does {p1} visit the market?\nAnswer: "
        comp = "Yes"
    else:
        c2 = random.choice([c for c in cities if c != c1])
        prompt = f"Rule: If {p1} lives in {c1}, then {p1} visits the market.\nContext: {p1} lives in {c2}.\nQuestion: Does {p1} visit the market?\nAnswer: "
        comp = "No"
    extra_samples.append({
        "type": "conditional_logic",
        "prompt": prompt,
        "completion": comp,
        "is_relational": True
    })

# 2. 150 Constraint Satisfaction Samples
for _ in range(150):
    target = random.choice(fruits)
    non_target = random.choice(colors)
    prompt = f"Constraint: Select the fruit and omit colors.\nOptions: {target}, {non_target}.\nAnswer: "
    comp = target
    extra_samples.append({
        "type": "constraint_satisfaction",
        "prompt": prompt,
        "completion": comp,
        "is_relational": True
    })

# 3. 150 Multi-turn & Single-hop Reinforcement Samples
for _ in range(150):
    p = random.choice(entities)
    c_old = random.choice(cities)
    c_new = random.choice([c for c in cities if c != c_old])
    prompt = f"User: {p} was in {c_old}.\nChakrView: Understood.\nUser: Later, {p} moved to {c_new}.\nChakrView: Recorded.\nUser: Where is {p} now?\nChakrView: "
    comp = f"{p} is currently in {c_new}."
    extra_samples.append({
        "type": "multiturn_reinforcement",
        "prompt": prompt,
        "completion": comp,
        "is_relational": True
    })

train_c3 = train_c2 + extra_samples
random.shuffle(train_c3)

with open("experiments/cognition/cr_exp_09_autonomous_mission/data/train_candidate3.json", "w", encoding="utf-8") as f:
    json.dump(train_c3, f, indent=2)

print(f"Candidate 3 training set created: {len(train_c3)} samples (C2: {len(train_c2)} + Extra: {len(extra_samples)})")
