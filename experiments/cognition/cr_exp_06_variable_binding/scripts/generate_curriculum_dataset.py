"""
Stage 2: Deterministic Relational Curriculum Dataset Generator
Generates:
- Train: ~1,500 samples
- Validation: ~200 samples
- Test In-Distribution: 100 samples
- Test OOD Entities: 100 samples (completely disjoint entity pool)
- Test OOD Templates: 100 samples (completely disjoint relational phrasing)
- Test Multi-Hop: 60 samples (strictly two-hop relational reasoning)
Includes explicit character-span metadata: `target_spans: [[start, end], ...]` in the response
for Treatment Arm B loss weighting.
"""

import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_06_variable_binding" / "data"

# Deterministic Seed
SEED = 20261106

# --- ENTITY POOLS ---
# In-distribution entity pool (Train, Val, Test-ID)
ID_PERSONS = ["Ravi", "Priya", "Amit", "Kavita", "Sunil", "Neha", "Vikram", "Deepa", "Sanjay", "Ananya", "Rohan", "Pooja", "Rajesh", "Meera", "Kabir"]
ID_CITIES = ["Pune", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Jaipur", "Kolkata", "Ahmedabad", "Chandigarh", "Nagpur", "Indore"]
ID_COMPANIES = ["TechNova", "Garuda Systems", "ChakrCorp", "IndusLogic", "ApexFlow", "VedicAI", "BharatCore", "KaveriNet"]
ID_STATUSES = ["ACTIVE", "PENDING", "INACTIVE", "ARCHIVED", "SUSPENDED", "QUEUED"]
ID_PORTS = ["8080", "3000", "5432", "6379", "8000", "443", "27017"]

# Completely disjoint OOD entity pool (Never in train or val!)
OOD_PERSONS = ["Aarav", "Ishaan", "Diya", "Sanya", "Tanya", "Aditya", "Rhea", "Manish", "Varun", "Alok", "Gaurav", "Nisha"]
OOD_CITIES = ["Bhopal", "Surat", "Patna", "Varanasi", "Dehradun", "Ranchi", "Guwahati", "Mysuru", "Kochi", "Coimbatore"]
OOD_COMPANIES = ["ZenithSoft", "OmniCloud", "QuantumGrid", "VortexLabs", "AuraScale", "Solaria"]
OOD_OBJECTS = ["Quasar", "Rhodium", "Platinum", "Osmium", "Titanium", "Cobalt", "Xylok", "Zentrix", "Pebble"]

DISTRACTORS = [
    "Weather in the region is sunny with mild humidity.",
    "The quarterly infrastructure meeting is set for Friday.",
    "System memory utilization is currently at 42 percent.",
    "Network latency across the backbone is 12 milliseconds.",
    "All security certificates have been verified for compliance.",
    "The automated backup sequence completed without errors.",
    "The regional database cluster operates in read-only replica mode."
]

CONVERSATIONAL_SEEDS = [
    {"p": "User: Hello!\nAssistant:", "r": "Good morning! How may I assist you with your tasks today?"},
    {"p": "User: Hi there, good morning.\nAssistant:", "r": "Good morning! I am online and ready to assist you."},
    {"p": "User: Good evening, how are you?\nAssistant:", "r": "Good evening! I am doing well and ready to assist."},
    {"p": "User: Who are you?\nAssistant:", "r": "I am ChakrView, an indigenous neural language model developed from first principles."},
    {"p": "User: What is your primary design philosophy?\nAssistant:", "r": "I am ChakrView, developed from first principles to explore compact, efficient neural intelligence on standard commodity hardware."},
    {"p": "User: Thank you for your assistance!\nAssistant:", "r": "You are very welcome! Please feel free to ask if you need anything else."},
    {"p": "User: Have a great day!\nAssistant:", "r": "Thank you! I wish you a productive and wonderful day ahead."},
    {"p": "User: Can you explain what a hash function does?\nAssistant:", "r": "A hash function maps input data of arbitrary size to a fixed-length string of bytes, ensuring deterministic verification of data integrity."},
    {"p": "User: What is the benefit of modular software design?\nAssistant:", "r": "Modular design breaks complex systems into independent, interchangeable components, improving maintainability, testing, and debugging."},
    {"p": "User: Bhai, kaise ho?\nAssistant:", "r": "Main theek hoon! Aap bataiye, aaj main aapki kya madad kar sakta hoon?"},
    {"p": "User: Kya aap technical tasks mein madad kar sakte hain?\nAssistant:", "r": "Haan, main technical tasks, code structure, aur general questions mein aapki madad kar sakta hoon."},
]

def make_sample(prompt, response, target_spans, category, split_type="train"):
    return {
        "prompt": prompt,
        "response": response,
        "target_spans": target_spans, # [[start, end], ...] in response
        "category": category,
        "split_type": split_type
    }

def find_spans(response, targets):
    spans = []
    for t in targets:
        start = response.find(t)
        if start != -1:
            spans.append([start, start + len(t)])
    return spans

def generate_dataset():
    rng = random.Random(SEED)
    
    # 1. Training Set Generation (~1,500 samples)
    train_samples = []
    
    # A. Single-hop retrieval: Person lives in City (300 samples)
    for _ in range(300):
        p = rng.choice(ID_PERSONS)
        c = rng.choice(ID_CITIES)
        prompt = f"Context: {p} lives in {c}.\nUser: Where does {p} live?\nAssistant:"
        response = f"{p} lives in {c}."
        spans = find_spans(response, [c])
        train_samples.append(make_sample(prompt, response, spans, "single_hop_person_city", "train"))

    # B. Reverse retrieval: Who lives in City? (200 samples)
    for _ in range(200):
        p = rng.choice(ID_PERSONS)
        c = rng.choice(ID_CITIES)
        prompt = f"Context: {p} lives in {c}.\nUser: Who lives in {c}?\nAssistant:"
        response = f"{p} lives in {c}."
        spans = find_spans(response, [p])
        train_samples.append(make_sample(prompt, response, spans, "reverse_retrieval_city", "train"))

    # C. Attribute association: Person owns Company (200 samples)
    for _ in range(200):
        p = rng.choice(ID_PERSONS)
        comp = rng.choice(ID_COMPANIES)
        prompt = f"Context: {p} founded {comp}.\nUser: Which company did {p} found?\nAssistant:"
        response = f"{p} founded {comp}."
        spans = find_spans(response, [comp])
        train_samples.append(make_sample(prompt, response, spans, "attribute_person_company", "train"))

    # D. Multiple entities disambiguation (200 samples)
    for _ in range(200):
        p1, p2 = rng.sample(ID_PERSONS, 2)
        c1, c2 = rng.sample(ID_CITIES, 2)
        target = rng.choice([p1, p2])
        ans_city = c1 if target == p1 else c2
        prompt = f"Context: {p1} lives in {c1}. {p2} lives in {c2}.\nUser: Where does {target} live?\nAssistant:"
        response = f"{target} lives in {ans_city}."
        spans = find_spans(response, [ans_city])
        train_samples.append(make_sample(prompt, response, spans, "multi_entity_disambiguation", "train"))

    # E. Distractor resistance (200 samples)
    for _ in range(200):
        p = rng.choice(ID_PERSONS)
        c = rng.choice(ID_CITIES)
        d1, d2 = rng.sample(DISTRACTORS, 2)
        prompt = f"Context: {d1} {p} lives in {c}. {d2}\nUser: Where does {p} live?\nAssistant:"
        response = f"{p} lives in {c}."
        spans = find_spans(response, [c])
        train_samples.append(make_sample(prompt, response, spans, "distractor_resistance", "train"))

    # F. Dynamic state updates (150 samples)
    for _ in range(150):
        st1, st2 = rng.sample(ID_STATUSES, 2)
        prompt = f"Context: Service status was {st1}. Later, service status changed to {st2}.\nUser: What is the current status of the service?\nAssistant:"
        response = f"The current service status is {st2}."
        spans = find_spans(response, [st2])
        train_samples.append(make_sample(prompt, response, spans, "dynamic_state_update", "train"))

    # G. Two-hop relationships (150 samples)
    for _ in range(150):
        p = rng.choice(ID_PERSONS)
        comp = rng.choice(ID_COMPANIES)
        city = rng.choice(ID_CITIES)
        prompt = f"Context: {p} owns {comp}. {comp} is located in {city}.\nUser: In which city is {p}'s company located?\nAssistant:"
        response = f"{p}'s company {comp} is located in {city}."
        spans = find_spans(response, [city])
        train_samples.append(make_sample(prompt, response, spans, "two_hop_relational", "train"))

    # H. Epistemic abstention / missing info (100 samples)
    for _ in range(100):
        p = rng.choice(ID_PERSONS)
        c = rng.choice(ID_CITIES)
        prompt = f"Context: {p} lives in {c}.\nUser: What car does {p} drive?\nAssistant:"
        response = "That information is not provided in the context."
        spans = find_spans(response, ["not provided in the context"])
        train_samples.append(make_sample(prompt, response, spans, "epistemic_abstention", "train"))

    # I. Conversational preservation (200 samples)
    for _ in range(200):
        conv = rng.choice(CONVERSATIONAL_SEEDS)
        train_samples.append(make_sample(conv["p"], conv["r"], [], "conversational_preservation", "train"))

    # Shuffle training set
    rng.shuffle(train_samples)
    print(f"Generated {len(train_samples)} training samples.")

    # 2. Validation Set Generation (200 samples)
    val_samples = []
    for _ in range(40):
        p = rng.choice(ID_PERSONS)
        c = rng.choice(ID_CITIES)
        prompt = f"Context: {p} lives in {c}.\nUser: Where does {p} live?\nAssistant:"
        response = f"{p} lives in {c}."
        val_samples.append(make_sample(prompt, response, find_spans(response, [c]), "single_hop_val", "validation"))
    for _ in range(30):
        p = rng.choice(ID_PERSONS)
        c = rng.choice(ID_CITIES)
        prompt = f"Context: {p} lives in {c}.\nUser: Who lives in {c}?\nAssistant:"
        response = f"{p} lives in {c}."
        val_samples.append(make_sample(prompt, response, find_spans(response, [p]), "reverse_val", "validation"))
    for _ in range(30):
        p1, p2 = rng.sample(ID_PERSONS, 2)
        c1, c2 = rng.sample(ID_CITIES, 2)
        prompt = f"Context: {p1} lives in {c1}. {p2} lives in {c2}.\nUser: Where does {p2} live?\nAssistant:"
        response = f"{p2} lives in {c2}."
        val_samples.append(make_sample(prompt, response, find_spans(response, [c2]), "multi_entity_val", "validation"))
    for _ in range(30):
        st1, st2 = rng.sample(ID_STATUSES, 2)
        prompt = f"Context: Status was {st1}. Later, status was updated to {st2}.\nUser: What is the current status?\nAssistant:"
        response = f"The current status is {st2}."
        val_samples.append(make_sample(prompt, response, find_spans(response, [st2]), "state_val", "validation"))
    for _ in range(30):
        p = rng.choice(ID_PERSONS)
        comp = rng.choice(ID_COMPANIES)
        city = rng.choice(ID_CITIES)
        prompt = f"Context: {p} leads {comp}. {comp} head office is in {city}.\nUser: Which city is {p}'s office in?\nAssistant:"
        response = f"{p}'s office is in {city}."
        val_samples.append(make_sample(prompt, response, find_spans(response, [city]), "two_hop_val", "validation"))
    for _ in range(40):
        conv = rng.choice(CONVERSATIONAL_SEEDS)
        val_samples.append(make_sample(conv["p"], conv["r"], [], "conversational_val", "validation"))
    rng.shuffle(val_samples)
    print(f"Generated {len(val_samples)} validation samples.")

    # 3. Test Splits (Completely Frozen & Isolated)
    # 3.1 Test In-Distribution (100 samples)
    test_id = []
    for _ in range(100):
        p = rng.choice(ID_PERSONS)
        c = rng.choice(ID_CITIES)
        prompt = f"Context: {p} lives in {c}.\nUser: Where does {p} live?\nAssistant:"
        response = f"{p} lives in {c}."
        test_id.append(make_sample(prompt, response, find_spans(response, [c]), "test_in_dist", "test"))

    # 3.2 Test OOD Entities (100 samples - ZERO overlap with ID persons/cities!)
    test_ood_entities = []
    for _ in range(100):
        p = rng.choice(OOD_PERSONS)
        c = rng.choice(OOD_CITIES)
        prompt = f"Context: {p} lives in {c}.\nUser: Where does {p} live?\nAssistant:"
        response = f"{p} lives in {c}."
        test_ood_entities.append(make_sample(prompt, response, find_spans(response, [c]), "test_ood_entities", "test"))

    # 3.3 Test OOD Templates (100 samples - novel relational syntax & synthetic objects)
    test_ood_templates = []
    for _ in range(100):
        obj1, obj2 = rng.sample(OOD_OBJECTS, 2)
        prompt = f"Record: Item {obj1} has been assigned code {obj2}.\nUser: What code is assigned to item {obj1}?\nAssistant:"
        response = f"Item {obj1} is assigned code {obj2}."
        test_ood_templates.append(make_sample(prompt, response, find_spans(response, [obj2]), "test_ood_templates", "test"))

    # 3.4 Test Multi-Hop (60 samples - strict 2-hop compositional queries)
    test_multihop = []
    for _ in range(60):
        p = rng.choice(OOD_PERSONS)
        comp = rng.choice(OOD_COMPANIES)
        city = rng.choice(OOD_CITIES)
        prompt = f"Context: {p} manages {comp}. {comp} operates out of {city}.\nUser: In which city does {p}'s company operate?\nAssistant:"
        response = f"{p}'s company operates in {city}."
        test_multihop.append(make_sample(prompt, response, find_spans(response, [city]), "test_multihop", "test"))

    # Write files
    def write_jsonl(path, samples):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    write_jsonl(DATA_DIR / "train.jsonl", train_samples)
    write_jsonl(DATA_DIR / "validation.jsonl", val_samples)
    write_jsonl(DATA_DIR / "test_in_distribution.jsonl", test_id)
    write_jsonl(DATA_DIR / "test_ood_entities.jsonl", test_ood_entities)
    write_jsonl(DATA_DIR / "test_ood_templates.jsonl", test_ood_templates)
    write_jsonl(DATA_DIR / "test_multihop.jsonl", test_multihop)

    # Manifest
    def hash_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "dataset_name": "CR-EXP-06-Relational-Curriculum-v1",
        "seed": SEED,
        "splits": {
            "train": {"file": "train.jsonl", "count": len(train_samples), "sha256": hash_file(DATA_DIR / "train.jsonl")},
            "validation": {"file": "validation.jsonl", "count": len(val_samples), "sha256": hash_file(DATA_DIR / "validation.jsonl")},
            "test_in_distribution": {"file": "test_in_distribution.jsonl", "count": len(test_id), "sha256": hash_file(DATA_DIR / "test_in_distribution.jsonl")},
            "test_ood_entities": {"file": "test_ood_entities.jsonl", "count": len(test_ood_entities), "sha256": hash_file(DATA_DIR / "test_ood_entities.jsonl")},
            "test_ood_templates": {"file": "test_ood_templates.jsonl", "count": len(test_ood_templates), "sha256": hash_file(DATA_DIR / "test_ood_templates.jsonl")},
            "test_multihop": {"file": "test_multihop.jsonl", "count": len(test_multihop), "sha256": hash_file(DATA_DIR / "test_multihop.jsonl")},
        },
        "entity_isolation": {
            "train_val_id_persons": ID_PERSONS,
            "train_val_id_cities": ID_CITIES,
            "strict_ood_persons": OOD_PERSONS,
            "strict_ood_cities": OOD_CITIES,
            "person_overlap_count": len(set(ID_PERSONS) & set(OOD_PERSONS)),
            "city_overlap_count": len(set(ID_CITIES) & set(OOD_CITIES))
        }
    }

    manifest_p = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/dataset_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Dataset generation complete. Manifest saved to {manifest_p}")

if __name__ == "__main__":
    generate_dataset()

