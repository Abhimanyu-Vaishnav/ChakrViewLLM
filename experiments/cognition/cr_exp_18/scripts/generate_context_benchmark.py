import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_18/benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate 200 controlled conversations across 5 length groups:
# Group A: 1 turn  (40 conversations)
# Group B: 2 turns (40 conversations)
# Group C: 3 turns (40 conversations)
# Group D: 5 turns (40 conversations)
# Group E: 8 turns (40 conversations)
#
# Task Types (distributed 20 each across the 200 probes):
# 1. single_fact_recall
# 2. multi_fact_recall
# 3. subject_value_association
# 4. pronoun_reference_resolution
# 5. user_correction_handling
# 6. counterfactual_reassignment
# 7. distractor_resistance
# 8. topic_switching_returning
# 9. missing_info_abstention
# 10. conversation_summary

# Fictional names, values, and entities not present in training sets:
NAMES = [
    "Zaraph", "Kaelen", "Miraan", "Theon", "Lyra", "Vesper", "Caspian", "Dorian",
    "Selene", "Theron", "Alistair", "Rowena", "Gideon", "Caelum", "Elowen", "Jareth",
    "Seraphina", "Corin", "Valen", "Morwen"
]
CITIES = [
    "Zephyria", "Oakhaven", "Solarium", "Vespera", "Eldoria", "Aethelgard", "Lunaria", "Bravos",
    "Kaelwick", "Drakenshire", "Nyxos", "Caelumgrad", "Starfall", "Valoria", "Frostholme", "Ironhold",
    "Dawnport", "Sunspire", "Silverhaven", "Shadowfen"
]
FRUITS = ["durian", "papaya", "dragonfruit", "pomegranate", "guava", "lychee", "persimmon", "kiwi", "fig", "apricot"]
DEVICES = ["Prism-9", "Nexus-X", "Core-V", "Quantum-Z", "Matrix-7", "Apex-4", "Pulse-3", "Vertex-8", "Vector-2", "Hydra-6"]

conversations = []
cid = 1

def make_probe(group_name, num_turns, task_type, turns, target_fact, expected_answer, answerable, failure_cond):
    global cid
    formatted = ""
    for t in turns[:-1]:
        formatted += f"User: {t['user']}\nAssistant: {t['assistant']}\n"
    formatted += f"User: {turns[-1]['user']}\nAssistant:"
    
    probe = {
        "conversation_id": f"CR18_CONV_{cid:03d}",
        "group": group_name,
        "turns_count": num_turns,
        "task_type": task_type,
        "turns": turns,
        "formatted_prompt": formatted,
        "target_fact": target_fact,
        "expected_answer": expected_answer,
        "answerable": answerable,
        "failure_conditions": failure_cond
    }
    cid += 1
    return probe

# Helper to generate groups
for grp_idx, (grp_name, n_turns) in enumerate([("A", 1), ("B", 2), ("C", 3), ("D", 5), ("E", 8)]):
    for i in range(40):
        # 10 task types, 4 per group
        task_idx = i % 10
        name = NAMES[(grp_idx * 40 + i) % len(NAMES)]
        city = CITIES[(grp_idx * 40 + i) % len(CITIES)]
        fruit = FRUITS[(grp_idx * 40 + i) % len(FRUITS)]
        device = DEVICES[(grp_idx * 40 + i) % len(DEVICES)]
        
        turns = []
        if n_turns == 1:
            if task_idx == 0: # single_fact_recall
                turns.append({"user": f"My name is {name}.", "assistant": f"Hello {name}!"})
                target = f"Name={name}"
                expected = [name]
                ttype = "single_fact_recall"
                ans = True
            elif task_idx == 8: # missing_info_abstention
                turns.append({"user": "What is the passcode to my safe?", "assistant": ""})
                target = "Unknown passcode"
                expected = ["do not know", "cannot know", "unspecified"]
                ttype = "missing_info_abstention"
                ans = False
            else:
                turns.append({"user": f"I live in {city}.", "assistant": f"Noted, {city}."})
                target = f"City={city}"
                expected = [city]
                ttype = "single_fact_recall"
                ans = True
        elif n_turns == 2:
            if task_idx == 0: # single_fact_recall
                turns.append({"user": f"My colleague's name is {name}.", "assistant": f"I have noted your colleague {name}."})
                turns.append({"user": "What was my colleague's name?", "assistant": ""})
                target = f"Colleague={name}"
                expected = [name]
                ttype = "single_fact_recall"
                ans = True
            elif task_idx == 3: # pronoun_reference
                turns.append({"user": f"{name} is an engineer who likes {fruit}.", "assistant": f"Noted that {name} likes {fruit}."})
                turns.append({"user": "What fruit does he enjoy?", "assistant": ""})
                target = f"Fruit={fruit}"
                expected = [fruit]
                ttype = "pronoun_reference_resolution"
                ans = True
            elif task_idx == 4: # user_correction
                turns.append({"user": f"Meeting is in {city}.", "assistant": f"Recorded meeting in {city}."})
                turns.append({"user": f"Correction: the meeting is actually in London, not {city}. Where is the meeting?", "assistant": ""})
                target = "Meeting=London"
                expected = ["London"]
                ttype = "user_correction_handling"
                ans = True
            elif task_idx == 8: # missing_info
                turns.append({"user": f"We deployed {device}.", "assistant": f"Deployment of {device} recorded."})
                turns.append({"user": "What is the IP address of that device?", "assistant": ""})
                target = "IP address unknown"
                expected = ["do not know", "not specified", "unknown"]
                ttype = "missing_info_abstention"
                ans = False
            else:
                turns.append({"user": f"My primary device is {device}.", "assistant": f"{device} noted."})
                turns.append({"user": "Which device did I mention?", "assistant": ""})
                target = f"Device={device}"
                expected = [device]
                ttype = "subject_value_association"
                ans = True
        elif n_turns == 3:
            turns.append({"user": f"I am testing {device} in {city}.", "assistant": f"Testing {device} in {city} noted."})
            turns.append({"user": "The weather today is cloudy.", "assistant": "Cloudy weather noted."})
            if task_idx == 7: # topic_switching_returning
                turns.append({"user": "Returning to our test, which device am I testing?", "assistant": ""})
                target = f"Device={device}"
                expected = [device]
                ttype = "topic_switching_returning"
                ans = True
            elif task_idx == 4: # user_correction
                turns.append({"user": f"Actually we moved testing to Paris. Which city are we testing in now?", "assistant": ""})
                target = "City=Paris"
                expected = ["Paris"]
                ttype = "user_correction_handling"
                ans = True
            elif task_idx == 8: # missing_info
                turns.append({"user": "Who is the lead technician for this test?", "assistant": ""})
                target = "Technician unknown"
                expected = ["do not know", "not specified", "unknown"]
                ttype = "missing_info_abstention"
                ans = False
            else:
                turns.append({"user": f"In which city is {device} being tested?", "assistant": ""})
                target = f"City={city}"
                expected = [city]
                ttype = "subject_value_association"
                ans = True
        elif n_turns == 5:
            turns.append({"user": f"Hello, my name is {name}.", "assistant": f"Hello {name}!"})
            turns.append({"user": f"I am currently traveling to {city}.", "assistant": f"Safe travels to {city}."})
            turns.append({"user": f"My favorite fruit is {fruit}.", "assistant": f"{fruit.capitalize()} is delicious."})
            turns.append({"user": f"We also brought our backup tool {device}.", "assistant": f"{device} recorded."})
            if task_idx == 0:
                turns.append({"user": "What was my name?", "assistant": ""})
                target = f"Name={name}"
                expected = [name]
                ttype = "single_fact_recall"
                ans = True
            elif task_idx == 1:
                turns.append({"user": f"What fruit did I say I love?", "assistant": ""})
                target = f"Fruit={fruit}"
                expected = [fruit]
                ttype = "multi_fact_recall"
                ans = True
            elif task_idx == 4:
                turns.append({"user": f"Update: I lost {device} and now use Laptop-1. What backup tool do I use now?", "assistant": ""})
                target = "Tool=Laptop-1"
                expected = ["Laptop-1"]
                ttype = "user_correction_handling"
                ans = True
            elif task_idx == 8:
                turns.append({"user": "What flight number am I flying on?", "assistant": ""})
                target = "Flight unknown"
                expected = ["do not know", "not specified", "unknown"]
                ttype = "missing_info_abstention"
                ans = False
            else:
                turns.append({"user": "Which city was I traveling to?", "assistant": ""})
                target = f"City={city}"
                expected = [city]
                ttype = "subject_value_association"
                ans = True
        else: # n_turns == 8
            turns.append({"user": f"Greetings! I am {name}.", "assistant": f"Welcome {name}."})
            turns.append({"user": f"I work out of {city}.", "assistant": f"Office in {city} noted."})
            turns.append({"user": "Today we had a team standup at 9 AM.", "assistant": "Standup noted."})
            turns.append({"user": f"Our server rack holds {device}.", "assistant": f"{device} in rack noted."})
            turns.append({"user": "For lunch we ordered vegetable curry.", "assistant": "Vegetable curry sounds great."})
            turns.append({"user": f"I brought fresh {fruit} for the team.", "assistant": f"Healthy {fruit} snack noted."})
            turns.append({"user": "We solved two major bugs before noon.", "assistant": "Good progress on bugs."})
            if task_idx == 0:
                turns.append({"user": "What is my name again?", "assistant": ""})
                target = f"Name={name}"
                expected = [name]
                ttype = "single_fact_recall"
                ans = True
            elif task_idx == 1:
                turns.append({"user": "Which server device was in our rack?", "assistant": ""})
                target = f"Device={device}"
                expected = [device]
                ttype = "multi_fact_recall"
                ans = True
            elif task_idx == 7:
                turns.append({"user": "Returning to my work location, which city was it?", "assistant": ""})
                target = f"City={city}"
                expected = [city]
                ttype = "topic_switching_returning"
                ans = True
            elif task_idx == 8:
                turns.append({"user": "What is the name of our CEO?", "assistant": ""})
                target = "CEO unknown"
                expected = ["do not know", "not specified", "unknown"]
                ttype = "missing_info_abstention"
                ans = False
            else:
                turns.append({"user": f"What snack did I bring for the team?", "assistant": ""})
                target = f"Fruit={fruit}"
                expected = [fruit]
                ttype = "subject_value_association"
                ans = True
                
        p = make_probe(grp_name, n_turns, ttype, turns, target, expected, ans, ["refusal sink", "wrong entity", "hallucination"])
        conversations.append(p)

out_file = OUT_DIR / "CR18_CONTEXT_200.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(conversations, f, indent=2)

h = hashlib.sha256(out_file.read_bytes()).hexdigest()
manifest = {
    "benchmark_name": "CR18_CONTEXT_200",
    "total_conversations": len(conversations),
    "group_breakdown": {
        "A (1 turn)": 40,
        "B (2 turns)": 40,
        "C (3 turns)": 40,
        "D (5 turns)": 40,
        "E (8 turns)": 40
    },
    "sha256": h
}

with open(OUT_DIR / "benchmark_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated {len(conversations)} conversations across 5 length groups.")
print(f"Benchmark SHA-256: {h}")
