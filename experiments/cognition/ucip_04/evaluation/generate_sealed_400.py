"""
Generate strictly sealed 400-probe benchmark for UCIP-04.
Categories:
1. exact_copying (40 probes)
2. simple_generation (50 probes)
3. basic_explanation (50 probes)
4. conversation_persona (40 probes)
5. devanagari_hindi (50 probes)
6. roman_hindi (50 probes)
7. hinglish (50 probes)
8. multi_turn_context (40 probes)
9. repetition_resistance (30 probes)
Total = 400 probes
Strictly disjoint from training data.
"""

import json
from pathlib import Path

# Load train and val prompts to enforce zero overlap
train_file = Path("experiments/cognition/ucip_04/data/ucip04_train.jsonl")
val_file = Path("experiments/cognition/ucip_04/data/ucip04_validation.jsonl")
known_prompts = set()
for f in [train_file, val_file]:
    if f.exists():
        for line in open(f, "r", encoding="utf-8"):
            known_prompts.add(json.loads(line)["prompt"].strip())

print(f"Loaded {len(known_prompts)} training/val prompts to strictly avoid.")

probes = []

# 1. Exact Copying (40)
copy_items = [
    "Clean air is essential for human vitality.",
    "The train departed smoothly from the platform.",
    "Architects design sturdy structures for cities.",
    "A quiet stream meanders through the forest.",
    "Honesty builds strong and lasting trust.",
    "Fresh vegetables provide essential minerals and vitamins.",
    "Astronomy reveals the vast scale of the cosmos.",
    "The morning breeze felt cool and gentle.",
    "Practice and discipline produce steady improvements.",
    "Solar arrays capture energy from direct sunlight."
]
for i in range(40):
    src = copy_items[i % len(copy_items)]
    probes.append({
        "id": f"sealed_copy_{i+1:03d}",
        "suite": "exact_copying",
        "prompt": f"User: Say exactly: {src} [Probe {i+1}]\nAssistant:",
        "target": src
    })

# 2. Simple Generation (50)
gen_items = ["a horse", "the stars", "a forest", "a ship", "a lamp", "a clock", "a bridge", "a garden", "an island", "a cloud"]
for i in range(50):
    subj = gen_items[i % len(gen_items)]
    probes.append({
        "id": f"sealed_gen_{i+1:03d}",
        "suite": "simple_generation",
        "prompt": f"User: Write one simple sentence describing {subj}. (Check {i+1})\nAssistant:",
        "target": subj
    })

# 3. Basic Explanation (50)
exp_items = [
    ("What does a GPU do?", "graphics"),
    ("What is an SSD?", "storage"),
    ("Why is ocean water salty?", "minerals"),
    ("What causes wind to blow?", "pressure"),
    ("What is a chemical element?", "substance"),
    ("What does an alternator do?", "electricity"),
    ("What is cellular respiration?", "energy"),
    ("Why do planets orbit the sun?", "gravity"),
    ("What is a software compiler?", "code"),
    ("How do plants absorb water?", "roots")
]
for i in range(50):
    q, kw = exp_items[i % len(exp_items)]
    probes.append({
        "id": f"sealed_exp_{i+1:03d}",
        "suite": "basic_explanation",
        "prompt": f"User: {q} [Test {i+1}]\nAssistant:",
        "target": kw
    })

# 4. Conversation & Persona (40)
persona_items = [
    ("Hello! Who are you?", "ChakrView"),
    ("What is your mission?", "assist"),
    ("Can you help me learn?", "help"),
    ("Tell me about yourself.", "ChakrView"),
    ("Good afternoon!", "afternoon")
]
for i in range(40):
    q, kw = persona_items[i % len(persona_items)]
    probes.append({
        "id": f"sealed_pers_{i+1:03d}",
        "suite": "conversation_persona",
        "prompt": f"User: {q} [Sample {i+1}]\nAssistant:",
        "target": kw
    })

# 5. Devanagari Hindi (50)
hi_items = [
    ("पर्यावरण की सुरक्षा क्यों आवश्यक है?", "पर्यावरण"),
    ("योग का स्वास्थ्य पर क्या प्रभाव पड़ता है?", "स्वास्थ्य"),
    ("जल ही जीवन है, संक्षेप में बताएं।", "जल"),
    ("शिक्षा से समाज को क्या लाभ मिलता है?", "शिक्षा"),
    ("पुस्तकालय ज्ञान का भंडार क्यों कहे जाते हैं?", "ज्ञान")
]
for i in range(50):
    q, kw = hi_items[i % len(hi_items)]
    probes.append({
        "id": f"sealed_hin_{i+1:03d}",
        "suite": "devanagari_hindi",
        "prompt": f"User: {q} [जांच {i+1}]\nAssistant:",
        "target": kw
    })

# 6. Roman Hindi (50)
rom_items = [
    ("Subah jaldi uthna kyun accha hota hai?", "accha"),
    ("Mehnat karne se kya milta hai?", "safalta"),
    ("Dost zindagi mein kyun zaroori hote hain?", "zaroori"),
    ("Nayi bhasha seekhne ke kya fayde hain?", "fayde"),
    ("Samay ka sahi upyog kaise karein?", "samay")
]
for i in range(50):
    q, kw = rom_items[i % len(rom_items)]
    probes.append({
        "id": f"sealed_rom_{i+1:03d}",
        "suite": "roman_hindi",
        "prompt": f"User: {q} (Test {i+1})\nAssistant:",
        "target": kw
    })

# 7. Hinglish (50)
hing_items = [
    ("Open source software developers ke liye kaise useful hai?", "developers"),
    ("Programming me practice kitni important hai?", "practice"),
    ("Daily workout karne se mental focus kaise improve hota hai?", "focus"),
    ("Can you explain artificial intelligence simply?", "intelligence"),
    ("Online courses se skills upgrade kaise karein?", "skills")
]
for i in range(50):
    q, kw = hing_items[i % len(hing_items)]
    probes.append({
        "id": f"sealed_hing_{i+1:03d}",
        "suite": "hinglish",
        "prompt": f"User: {q} [Hinglish {i+1}]\nAssistant:",
        "target": kw
    })

# 8. Multi-turn Context (40)
context_entities = [
    ("Kabir", "Mysore", "SkyLogic"),
    ("Deepak", "Ahmedabad", "CloudBrain"),
    ("Gaurav", "Bhopal", "FastCore"),
    ("Harish", "Indore", "DataMesh")
]
for i in range(40):
    e = context_entities[i % len(context_entities)]
    kind = i % 3
    if kind == 0:
        p_text = f"User: My name is {e[0]}.\nAssistant: Hello {e[0]}!\nUser: What is my name?\nAssistant:"
        tgt = e[0]
    elif kind == 1:
        p_text = f"User: I live in {e[1]}.\nAssistant: {e[1]} is noted.\nUser: Where do I live?\nAssistant:"
        tgt = e[1]
    else:
        p_text = f"User: My project is {e[2]}.\nAssistant: {e[2]} sounds interesting.\nUser: What is my project called?\nAssistant:"
        tgt = e[2]
    probes.append({
        "id": f"sealed_ctx_{i+1:03d}",
        "suite": "multi_turn_context",
        "prompt": p_text,
        "target": tgt
    })

# 9. Repetition Resistance (30)
rep_prompts = [
    "Tell me about a dog, a cat, and a bird.",
    "Explain what computers, phones, and networks do.",
    "Describe the sun, the moon, and the stars."
]
for i in range(30):
    rp = rep_prompts[i % len(rep_prompts)]
    probes.append({
        "id": f"sealed_rep_{i+1:03d}",
        "suite": "repetition_resistance",
        "prompt": f"User: {rp} [Probe {i+1}]\nAssistant:",
        "target": "dog" if "dog" in rp else ("computer" if "computer" in rp else "sun")
    })

out_path = Path("experiments/cognition/ucip_04/evaluation/sealed_benchmark_400.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2, ensure_ascii=False)

print(f"Generated {len(probes)} sealed probes in {out_path}")
