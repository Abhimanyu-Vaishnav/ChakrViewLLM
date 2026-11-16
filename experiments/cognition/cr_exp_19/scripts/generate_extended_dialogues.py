import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_19/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Extended-horizon entities (separate from benchmarks)
TRAIN_NAMES = [
    "Aarav", "Ananya", "Rohan", "Meera", "Kabir", "Isha", "Aditya", "Riya",
    "Vikram", "Tara", "Arjun", "Pooja", "Siddharth", "Kavya", "Varun", "Diya"
]
TRAIN_CITIES = [
    "Jaipur", "Kochi", "Chandigarh", "Mysuru", "Bhopal", "Nagpur", "Surat", "Patna",
    "Varanasi", "Dehradun", "Indore", "Amritsar", "Ranchi", "Coimbatore", "Vadodara", "Shimla"
]
TRAIN_ITEMS = [
    "Laptop", "Tablet", "Monitor", "Drone", "Camera", "Sensor", "Router", "Microphone",
    "Projector", "Scanner", "Beacon", "Antenna", "Controller", "Analyzer"
]
TRAIN_TOPICS = [
    "quantum computing", "solar panels", "robotics", "urban farming", "machine learning",
    "electric vehicles", "data pipelines", "cloud security", "microservices", "compilers"
]

random.seed(42)

def generate_dialogue(n_user_turns):
    name = random.choice(TRAIN_NAMES)
    city = random.choice(TRAIN_CITIES)
    item = random.choice(TRAIN_ITEMS)
    topic = random.choice(TRAIN_TOPICS)
    
    # We construct a multi-turn dialogue where a fact is established in Turn 1,
    # intermediate conversation occurs in Turns 2 to (n-1),
    # and the final Turn asks for the fact or a correction.
    
    turns = []
    # Turn 1: Introduce entity and attribute
    turns.append((f"My name is {name} and I live in {city}.", f"Pleased to meet you, {name} from {city}. How can I assist you today?"))
    
    # Intermediary turns (distractors, topic discussions)
    if n_user_turns >= 3:
        turns.append((f"We are working on a new project involving {topic}.", f"That sounds like a promising initiative with {topic}."))
    if n_user_turns >= 4:
        turns.append((f"We recently procured a new {item} for the lab.", f"The {item} will be a valuable addition to your equipment."))
    if n_user_turns >= 5:
        turns.append(("Could you remind me what the weather was like yesterday?", "I do not have real-time weather sensors, but I hope it was pleasant."))
    if n_user_turns >= 6:
        turns.append(("Our sprint planning is scheduled for tomorrow morning.", "Best of luck with your team's sprint planning session."))
        
    # Query Turn (the final user prompt)
    query_type = random.choice(["name", "city", "correction"])
    if query_type == "name":
        final_user = "What was my name again?"
        final_assistant = f"Your name is {name}."
    elif query_type == "city":
        final_user = "Which city did I say I live in?"
        final_assistant = f"You live in {city}."
    else: # correction
        new_city = random.choice([c for c in TRAIN_CITIES if c != city])
        final_user = f"Correction: I recently relocated to {new_city}. Which city do I live in now?"
        final_assistant = f"Understood. You now live in {new_city}."
        
    # Format into prompt and response
    prompt_parts = []
    for u, a in turns:
        prompt_parts.append(f"User: {u}\nAssistant: {a}")
    prompt_parts.append(f"User: {final_user}\nAssistant:")
    
    return {
        "prompt": "\n".join(prompt_parts),
        "response": final_assistant,
        "turns_count": n_user_turns,
        "query_type": query_type
    }

# Generate 800 extended-horizon dialogues (3, 4, 5, 6 turns evenly distributed)
extended_dialogues = []
for _ in range(200):
    for n in [3, 4, 5, 6]:
        extended_dialogues.append(generate_dialogue(n))

random.shuffle(extended_dialogues)

# Split 90/10 into extended_train (720) and extended_val (80)
ext_val = extended_dialogues[:80]
ext_train = extended_dialogues[80:]

# Load 240 clean replay examples from CR-17 conversational train (25% replay proportion)
with open(REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_train_1080.jsonl", encoding="utf-8") as f:
    orig_train = [json.loads(line) for line in f if line.strip()]

random.seed(1337)
replay_pool = random.sample(orig_train, 240)

# Arm B Training Set: 720 extended-horizon + 240 replay = 960 total (25% replay)
arm_b_train = ext_train + replay_pool
random.shuffle(arm_b_train)

# Arm A Training Set (Short-Horizon Control):
# 720 short 1-2 turn dialogues with same fact types + 240 replay = 960 total
def generate_short_dialogue():
    name = random.choice(TRAIN_NAMES)
    city = random.choice(TRAIN_CITIES)
    u_turns = random.choice([1, 2])
    if u_turns == 1:
        u = f"Hello! My name is {name} from {city}."
        a = f"Hello {name}! How may I help you in {city} today?"
        return {"prompt": f"User: {u}\nAssistant:", "response": a, "turns_count": 1}
    else:
        u1 = f"My name is {name} and I live in {city}."
        a1 = f"Greetings {name} in {city}."
        u2 = "What was my name?"
        a2 = f"Your name is {name}."
        return {"prompt": f"User: {u1}\nAssistant: {a1}\nUser: {u2}\nAssistant:", "response": a2, "turns_count": 2}

arm_a_short = [generate_short_dialogue() for _ in range(720)]
arm_a_train = arm_a_short + replay_pool
random.shuffle(arm_a_train)

# Save files
with open(DATA_DIR / "extended_train.jsonl", "w", encoding="utf-8") as f:
    for ex in arm_b_train:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open(DATA_DIR / "extended_validation.jsonl", "w", encoding="utf-8") as f:
    for ex in ext_val:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open(DATA_DIR / "replay_train.jsonl", "w", encoding="utf-8") as f:
    for ex in replay_pool:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open(DATA_DIR / "short_control_train.jsonl", "w", encoding="utf-8") as f:
    for ex in arm_a_train:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

h_ext_tr = hashlib.sha256((DATA_DIR / "extended_train.jsonl").read_bytes()).hexdigest()
h_ext_va = hashlib.sha256((DATA_DIR / "extended_validation.jsonl").read_bytes()).hexdigest()
h_rep_tr = hashlib.sha256((DATA_DIR / "replay_train.jsonl").read_bytes()).hexdigest()
h_sho_tr = hashlib.sha256((DATA_DIR / "short_control_train.jsonl").read_bytes()).hexdigest()

manifest = {
    "arm_b_extended_train": {"file": "extended_train.jsonl", "sha256": h_ext_tr, "total": len(arm_b_train)},
    "extended_val": {"file": "extended_validation.jsonl", "sha256": h_ext_va, "total": len(ext_val)},
    "replay_pool": {"file": "replay_train.jsonl", "sha256": h_rep_tr, "total": len(replay_pool), "ratio": "25%"},
    "arm_a_short_control": {"file": "short_control_train.jsonl", "sha256": h_sho_tr, "total": len(arm_a_train)}
}

with open(DATA_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Generated training datasets successfully.")
print(f"Arm B Extended Train: {len(arm_b_train)} (720 extended + 240 replay)")
print(f"Arm A Short Control  : {len(arm_a_train)} (720 short + 240 replay)")
print(f"Extended Validation  : {len(ext_val)}")

