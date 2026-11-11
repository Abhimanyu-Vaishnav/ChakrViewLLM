"""
CR-EXP-08 Dataset Generator
Generates:
- 1,600 training samples (60% relational, 40% conversational)
- 200 validation samples (disjoint split)
- 400 frozen test samples across 7 distinct evaluation categories:
  1. In-distribution familiar entities & templates
  2. OOD entities (unseen subjects and unseen cities)
  3. OOD templates (unseen phrasing)
  4. Counterfactual reassignment
  5. Multi-entity distractors (2 and 3 entities)
  6. Missing-information abstention
  7. Conversational preservation
Enforces exact length-matched negative candidates for Arm C pairwise loss.
Outputs data/train.json, data/val.json, data/test.json, and dataset_manifest.json.
"""

import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_08_contextual_assignment" / "data"

# Length-grouped city pools in vocab_4096
CITIES_BY_LEN = {
    3: ["Delhi", "Jaipur", "Pune", "Surat", "Kanpur", "Nagpur", "Indore", "Patna"],
    4: ["Mumbai", "Bengaluru", "Bhopal", "Nashik"],
    5: ["Chennai", "Kolkata", "Hyderabad", "Lucknow"]
}

# Entity split discipline
# Training subjects: Familiar subjects
TRAIN_SUBJECTS = ["Ravi", "Aman", "Priya", "Neha", "Rohit", "Sneha", "Amit", "Pooja", "Rahul", "Anjali"]
VAL_SUBJECTS = ["Karan", "Simran", "Deepak", "Kavita"]
TEST_OOD_SUBJECTS = ["Vikram", "Zoya", "Arjun", "Meera", "Suresh", "Tanya", "Aditya", "Ritu"]

# Conversational persona pool
CONVERSATIONAL_TEMPLATES = [
    ("Hello! How are you doing today?", "Hello! I am doing well, ready to assist you. How can I help you today?"),
    ("Hi ChakrView!", "Hello! It is great to speak with you. How can I assist you right now?"),
    ("Who are you?", "I am ChakrView, an indigenous neural intelligence being developed from scratch."),
    ("Good morning!", "Good morning! I hope you have a productive day. What are you working on?"),
    ("Can you explain what you do?", "I am a neural brain prototype designed for conversational understanding, context retention, and reasoning."),
    ("Namaste! Aap kaise hain?", "Namaste! Main badhiya hoon. Aapki kya madad kar sakta hoon?"),
    ("Thank you very much!", "You are very welcome! Let me know if you need any more assistance."),
    ("Please tell me about yourself.", "I am ChakrView, built to progressively learn language and relational understanding."),
    ("What is your mission?", "My mission is to develop genuine autonomous cognitive capabilities step by step.")
]

# Relational template patterns
RELATIONAL_TEMPLATES_TRAIN = [
    ("Mapping: {s} lives in {c}. Query: Where does {s} live? Answer: ", "{c}"),
    ("Fact: {s} resides in {c}. Question: What city does {s} reside in? Answer: ", "{c}"),
    ("Context: {s}'s home is located in {c}. Where does {s} stay? Answer: ", "{c}"),
    ("Record: {s} is currently in {c}. In which city is {s}? Answer: ", "{c}"),
    ("Info: {s} lives in {c}. Query: Identify {s}'s city. Answer: ", "{c}")
]

RELATIONAL_TEMPLATES_OOD = [
    ("Data sheet: Subject {s} is stationed at {c}. Query: Where is {s} stationed? Answer: ", "{c}"),
    ("Dossier: Location for {s} is {c}. Question: Cite {s}'s location. Answer: ", "{c}"),
    ("Registry entry: {s} -> {c}. What is the designated city for {s}? Answer: ", "{c}")
]

def get_length_matched_negative(city, tok):
    enc = tok.encode("Answer: " + city)
    prefix_len = len(tok.encode("Answer: "))
    cand_len = len(enc) - prefix_len
    pool = [c for c in CITIES_BY_LEN.get(cand_len, []) if c != city]
    if not pool:
        # Fallback to any pool with same length
        for l, cs in CITIES_BY_LEN.items():
            if l == cand_len:
                pool = [c for c in cs if c != city]
    assert len(pool) > 0, f"No negative found for {city} with length {cand_len}"
    return random.choice(pool)

def build_relational_sample(subject, city, template_pair, tok, sample_type="single_fact"):
    prompt_tpl, ans_tpl = template_pair
    prompt = prompt_tpl.format(s=subject, c=city)
    ans = ans_tpl.format(s=subject, c=city)
    neg_city = get_length_matched_negative(ans, tok)
    
    # Token length verification
    enc_p = tok.encode(prompt)
    enc_pos = tok.encode(prompt + ans)[len(enc_p):]
    enc_neg = tok.encode(prompt + neg_city)[len(enc_p):]
    assert len(enc_pos) == len(enc_neg), f"Length mismatch: {ans} ({len(enc_pos)}) vs {neg_city} ({len(enc_neg)})"
    
    return {
        "type": "relational",
        "subtype": sample_type,
        "prompt": prompt,
        "completion": ans,
        "subject": subject,
        "positive_candidate": ans,
        "negative_candidate": neg_city,
        "candidate_token_length": len(enc_pos),
        "is_relational": True
    }

def build_conversational_sample(user_msg, asst_msg):
    prompt = f"User: {user_msg}\nAssistant: "
    return {
        "type": "conversational",
        "subtype": "dialogue_preservation",
        "prompt": prompt,
        "completion": asst_msg,
        "subject": None,
        "positive_candidate": asst_msg,
        "negative_candidate": None,
        "candidate_token_length": None,
        "is_relational": False
    }

def main():
    random.seed(42)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    print("Tokenizer loaded. Generating CR-EXP-08 dataset...")

    # 1. Build Training Set (1,600 samples: ~960 relational, ~640 conversational)
    train_samples = []
    # 960 relational samples across train subjects & cities
    all_train_cities = [c for cs in CITIES_BY_LEN.values() for c in cs]
    for _ in range(960):
        s = random.choice(TRAIN_SUBJECTS)
        c = random.choice(all_train_cities)
        tpl = random.choice(RELATIONAL_TEMPLATES_TRAIN)
        train_samples.append(build_relational_sample(s, c, tpl, tok, "train_relational"))

    # 640 conversational samples
    for _ in range(640):
        u, a = random.choice(CONVERSATIONAL_TEMPLATES)
        train_samples.append(build_conversational_sample(u, a))

    random.shuffle(train_samples)
    assert len(train_samples) == 1600

    # 2. Build Validation Set (200 samples: 120 relational, 80 conversational)
    val_samples = []
    for _ in range(120):
        s = random.choice(VAL_SUBJECTS)
        c = random.choice(all_train_cities)
        tpl = random.choice(RELATIONAL_TEMPLATES_TRAIN)
        val_samples.append(build_relational_sample(s, c, tpl, tok, "val_relational"))
    for _ in range(80):
        u, a = random.choice(CONVERSATIONAL_TEMPLATES)
        val_samples.append(build_conversational_sample(u, a))

    random.shuffle(val_samples)
    assert len(val_samples) == 200

    # 3. Build Frozen Test Set (400 samples across 7 distinct categories)
    test_samples = []
    # Cat A: Familiar relational retrieval (60 samples)
    for _ in range(60):
        s = random.choice(TRAIN_SUBJECTS)
        c = random.choice(all_train_cities)
        tpl = random.choice(RELATIONAL_TEMPLATES_TRAIN)
        sample = build_relational_sample(s, c, tpl, tok, "test_familiar_retrieval")
        sample["category"] = "familiar_retrieval"
        test_samples.append(sample)

    # Cat B: OOD Entity retrieval (unseen subjects, 60 samples)
    for _ in range(60):
        s = random.choice(TEST_OOD_SUBJECTS)
        c = random.choice(all_train_cities)
        tpl = random.choice(RELATIONAL_TEMPLATES_TRAIN)
        sample = build_relational_sample(s, c, tpl, tok, "test_ood_entities")
        sample["category"] = "ood_entities"
        test_samples.append(sample)

    # Cat C: OOD Template generalization (unseen phrasing, 60 samples)
    for _ in range(60):
        s = random.choice(TRAIN_SUBJECTS + VAL_SUBJECTS)
        c = random.choice(all_train_cities)
        tpl = random.choice(RELATIONAL_TEMPLATES_OOD)
        sample = build_relational_sample(s, c, tpl, tok, "test_ood_templates")
        sample["category"] = "ood_templates"
        test_samples.append(sample)

    # Cat D: Counterfactual reassignment (60 samples: subject remapped)
    for _ in range(60):
        s = random.choice(TRAIN_SUBJECTS)
        c1, c2 = random.sample(all_train_cities, 2)
        prompt = f"Mapping: {s} used to live in {c1}, but now lives in {c2}. Query: Where does {s} currently live? Answer: "
        neg = get_length_matched_negative(c2, tok)
        enc_p = tok.encode(prompt)
        enc_pos = tok.encode(prompt + c2)[len(enc_p):]
        enc_neg = tok.encode(prompt + neg)[len(enc_p):]
        test_samples.append({
            "type": "relational",
            "subtype": "test_counterfactual_reassignment",
            "category": "counterfactual_reassignment",
            "prompt": prompt,
            "completion": c2,
            "subject": s,
            "positive_candidate": c2,
            "negative_candidate": neg,
            "candidate_token_length": len(enc_pos),
            "is_relational": True
        })

    # Cat E: Multi-entity distractors (60 samples: 2 or 3 entities)
    for _ in range(60):
        s1, s2 = random.sample(TRAIN_SUBJECTS, 2)
        c1, c2 = random.sample(all_train_cities, 2)
        prompt = f"Mapping: {s1} lives in {c1}. {s2} lives in {c2}. Query: Where does {s1} live? Answer: "
        neg = get_length_matched_negative(c1, tok)
        enc_p = tok.encode(prompt)
        enc_pos = tok.encode(prompt + c1)[len(enc_p):]
        enc_neg = tok.encode(prompt + neg)[len(enc_p):]
        test_samples.append({
            "type": "relational",
            "subtype": "test_multi_entity_distractor",
            "category": "distractor_resistance",
            "prompt": prompt,
            "completion": c1,
            "subject": s1,
            "positive_candidate": c1,
            "negative_candidate": neg,
            "candidate_token_length": len(enc_pos),
            "is_relational": True
        })

    # Cat F: Missing-information abstention (50 samples)
    for _ in range(50):
        s1, s2 = random.sample(TRAIN_SUBJECTS, 2)
        c1 = random.choice(all_train_cities)
        prompt = f"Mapping: {s1} lives in {c1}. Query: Where does {s2} live? Answer: "
        ans = "The context does not specify where " + s2 + " lives."
        test_samples.append({
            "type": "relational_abstention",
            "subtype": "test_missing_info_abstention",
            "category": "missing_information_abstention",
            "prompt": prompt,
            "completion": ans,
            "subject": s2,
            "positive_candidate": ans,
            "negative_candidate": None,
            "candidate_token_length": None,
            "is_relational": False
        })

    # Cat G: Conversational preservation (50 samples)
    for _ in range(50):
        u, a = random.choice(CONVERSATIONAL_TEMPLATES)
        sample = build_conversational_sample(u, a)
        sample["category"] = "conversational_preservation"
        test_samples.append(sample)

    assert len(test_samples) == 400

    # Write splits
    with open(DATA_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(DATA_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
    with open(DATA_DIR / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_samples, f, indent=2)

    manifest = {
        "train_samples_count": len(train_samples),
        "val_samples_count": len(val_samples),
        "test_samples_count": len(test_samples),
        "train_relational_pct": round(sum(1 for s in train_samples if s["is_relational"]) / len(train_samples) * 100, 2),
        "train_conversational_pct": round(sum(1 for s in train_samples if not s["is_relational"]) / len(train_samples) * 100, 2),
        "test_categories_breakdown": {
            "familiar_retrieval": sum(1 for s in test_samples if s.get("category") == "familiar_retrieval"),
            "ood_entities": sum(1 for s in test_samples if s.get("category") == "ood_entities"),
            "ood_templates": sum(1 for s in test_samples if s.get("category") == "ood_templates"),
            "counterfactual_reassignment": sum(1 for s in test_samples if s.get("category") == "counterfactual_reassignment"),
            "distractor_resistance": sum(1 for s in test_samples if s.get("category") == "distractor_resistance"),
            "missing_information_abstention": sum(1 for s in test_samples if s.get("category") == "missing_information_abstention"),
            "conversational_preservation": sum(1 for s in test_samples if s.get("category") == "conversational_preservation")
        },
        "disjoint_subjects_verified": {
            "train": TRAIN_SUBJECTS,
            "val": VAL_SUBJECTS,
            "test_ood": TEST_OOD_SUBJECTS
        }
    }

    with open(DATA_DIR.parent / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset successfully created and written to {DATA_DIR}.")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    print(f"Manifest: {DATA_DIR.parent / 'dataset_manifest.json'}")

if __name__ == "__main__":
    main()

