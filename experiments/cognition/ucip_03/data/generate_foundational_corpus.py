"""
UCIP-03 Foundational Language Dataset Generator with Strict Sealing.
Separates data into:
- diagnostic_eval_150 (strictly sealed, never in train or val)
- train_corpus (diverse, disjoint prompts)
- validation_corpus (disjoint prompts)
"""

import json
import random
import hashlib
from pathlib import Path

random.seed(42)

# Load diagnostic 150 prompts to strictly reserve them
diag_path = Path("experiments/cognition/ucip_03/evaluation/diagnostic_baseline_150.json")
sealed_prompts = set()
with open(diag_path, "r", encoding="utf-8") as f:
    for p in json.load(f):
        sealed_prompts.add(p["prompt"].strip())

print(f"Loaded {len(sealed_prompts)} strictly sealed diagnostic prompts.")

# ==========================================
# 1. EXACT COPYING
# ==========================================
sentences = [
    # Training sentences (strictly not in diagnostic 150)
    "A journey of a thousand miles begins with a single step.",
    "A bird in the hand is worth two in the bush.",
    "A watched pot never boils.",
    "Barking dogs seldom bite.",
    "Better late than never.",
    "Birds of a feather flock together.",
    "Cleanliness is next to godliness.",
    "Do not put all your eggs in one basket.",
    "Early to bed and early to rise makes a person healthy.",
    "Fortune favors the bold.",
    "Great minds think alike.",
    "Haste makes waste.",
    "Honesty is the best policy.",
    "Hope for the best, prepare for the worst.",
    "If it is not broke, do not fix it.",
    "Ignorance is bliss.",
    "It takes two to tango.",
    "Laughter is the best medicine.",
    "Look before you leap.",
    "Make hay while the sun shines.",
    "Necessity is the mother of invention.",
    "No pain, no gain.",
    "Action speaks louder than words.",
    "All that glitters is not gold.",
    "An apple a day keeps the doctor away.",
    "Beauty is in the eye of the beholder.",
    "Don't count your chickens before they hatch.",
    "Every cloud has a silver lining.",
    "Good things come to those who wait.",
    "Knowledge is the key to unlocking potential.",
    "Patience is a virtue.",
    "Rome was not built in a day.",
    "The pen is mightier than the sword.",
    "Time and tide wait for no man.",
    "Where there is a will, there is a way.",
    "You cannot judge a book by its cover.",
    "A rolling stone gathers no moss.",
    "A stitch in time saves nine.",
    "Absence makes the heart grow fonder.",
    "Actions speak louder than voice."
]

copy_instructions = [
    "Say exactly: {s}",
    "Repeat this sentence exactly: {s}",
    "Please repeat: {s}",
    "Echo the following text: {s}",
    "Copy this sentence verbatim: {s}"
]

copy_records = []
for s in sentences:
    for tmpl in copy_instructions:
        prompt = f"User: {tmpl.format(s=s)}\nAssistant:"
        if prompt.strip() not in sealed_prompts:
            copy_records.append({
                "prompt": prompt,
                "target": f" {s}",
                "category": "exact_copying"
            })

# ==========================================
# 2. SIMPLE SENTENCE GENERATION
# ==========================================
simple_sentence_data = [
    ("a lion", [
        "A lion is a majestic wild cat that roams the open savanna.",
        "Lions live together in social family groups called prides.",
        "The loud roar of a male lion can be heard miles away."
    ]),
    ("an elephant", [
        "An elephant is the largest living land mammal with a long trunk.",
        "Elephants have large flapping ears that help keep them cool.",
        "Herds of gentle elephants travel across grasslands in search of water."
    ]),
    ("a horse", [
        "A horse is a graceful domestic animal known for speed and strength.",
        "Horses have been faithful companions to humans for thousands of years.",
        "The brown horse galloped swiftly across the wide grassy meadow."
    ]),
    ("the stars", [
        "The stars are distant glowing suns twinkling across the night sky.",
        "Billions of bright stars form vast spiral galaxies throughout the universe.",
        "Starlight travels across immense distances to reach our eyes."
    ]),
    ("a clock", [
        "A clock is an instrument used to measure and indicate the passing of time.",
        "The antique clock ticked rhythmically on the mantlepiece.",
        "Digital clocks display the current hour and minute on electronic screens."
    ]),
    ("a forest", [
        "A dense forest is home to tall trees, moss, and diverse wildlife.",
        "Walking through a quiet green forest refreshes the mind and spirit.",
        "Forests play a crucial role in absorbing carbon dioxide from the atmosphere."
    ]),
    ("a bridge", [
        "A bridge is a structure built to span physical obstacles like rivers and roads.",
        "The suspension bridge connects two bustling coastal communities.",
        "Engineers design sturdy bridges to support heavy vehicles safely."
    ]),
    ("a ship", [
        "A ship is a large floating vessel used to transport cargo across oceans.",
        "The cargo ship sailed smoothly across calm blue waters.",
        "Modern ships navigate safely using satellite tracking and radar."
    ]),
    ("bread", [
        "Bread is a staple food prepared by baking dough made of flour and water.",
        "The warm smell of freshly baked bread filled the neighborhood bakery.",
        "Toasted bread with butter makes a simple and delicious breakfast."
    ]),
    ("a garden", [
        "A garden is a planned outdoor space dedicated to cultivating flowers and vegetables.",
        "Colorful butterfly species fluttered among the blooming garden roses.",
        "Tending a community vegetable garden provides fresh food and relaxation."
    ]),
    ("a bicycle", [
        "A bicycle is a two-wheeled vehicle powered by human pedaling.",
        "Riding a bicycle is an eco-friendly and healthy way to commute.",
        "Children practiced riding their new bicycles in the sunny park."
    ]),
    ("a lamp", [
        "A desk lamp provides focused illumination for reading and studying.",
        "The warm glow of the bedside lamp lit up the cozy bedroom.",
        "Modern LED lamps consume minimal electricity while providing bright light."
    ])
]

simple_prompts = [
    "Write one simple sentence about {s}.",
    "Write a short sentence about {s}.",
    "Please give me a simple sentence describing {s}.",
    "Generate a simple sentence about {s}."
]

simple_records = []
for subj, sent_list in simple_sentence_data:
    for sent in sent_list:
        for p_tmpl in simple_prompts:
            prompt = f"User: {p_tmpl.format(s=subj)}\nAssistant:"
            if prompt.strip() not in sealed_prompts:
                simple_records.append({
                    "prompt": prompt,
                    "target": f" {sent}",
                    "category": "simple_sentence"
                })

# ==========================================
# 3. BASIC FACTUAL QA
# ==========================================
factual_data = [
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of Russia?", "The capital of Russia is Moscow."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasilia."),
    ("Which planet is the largest in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("Which planet is famous for its prominent rings?", "Saturn is famous for its extensive ring system."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("What is the chemical formula for carbon dioxide?", "The chemical formula for carbon dioxide is CO2."),
    ("What gas do plants absorb from the air during photosynthesis?", "Plants absorb carbon dioxide during photosynthesis."),
    ("How many days are in a leap year?", "There are 366 days in a leap year."),
    ("How many seconds are in one minute?", "There are 60 seconds in one minute."),
    ("How many minutes are in one hour?", "There are 60 minutes in one hour."),
    ("What is the hardest natural mineral on Earth?", "Diamond is the hardest naturally occurring mineral on Earth."),
    ("What is the largest mammal on Earth?", "The blue whale is the largest mammal on Earth."),
    ("Which continent is known as the Dark Continent?", "Africa has historically been referred to as the Dark Continent."),
    ("What is the boiling point of water in Fahrenheit?", "The boiling point of water is 212 degrees Fahrenheit at sea level."),
    ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
    ("Which organ pumps blood throughout the human body?", "The heart pumps oxygenated blood through the circulatory system."),
    ("What pigment gives leaves their green color?", "Chlorophyll gives leaves their green color and captures sunlight."),
    ("What is the currency of the United States?", "The currency of the United States is the US Dollar."),
    ("What is the currency of Japan?", "The currency of Japan is the Japanese Yen."),
    ("What is the currency of India?", "The currency of India is the Indian Rupee.")
]

factual_records = []
for q, a in factual_data:
    prompt1 = f"User: {q}\nAssistant:"
    if prompt1.strip() not in sealed_prompts:
        factual_records.append({
            "prompt": prompt1,
            "target": f" {a}",
            "category": "factual_qa"
        })
    prompt2 = f"User: Please tell me: {q.lower()}\nAssistant:"
    if prompt2.strip() not in sealed_prompts:
        factual_records.append({
            "prompt": prompt2,
            "target": f" {a}",
            "category": "factual_qa"
        })

# ==========================================
# 4. ARITHMETIC WITH VERBAL EXPLANATIONS
# ==========================================
arithmetic_records = []
# Addition
for a in range(1, 20):
    for b in range(1, 20):
        c = a + b
        p = f"User: What is {a} + {b}?\nAssistant:"
        if p.strip() not in sealed_prompts:
            arithmetic_records.append({
                "prompt": p,
                "target": f" {a} + {b} is {c}.",
                "category": "arithmetic"
            })
            if (a + b) <= 25:
                arithmetic_records.append({
                    "prompt": f"User: Calculate {a} plus {b}.\nAssistant:",
                    "target": f" {a} plus {b} equals {c}.",
                    "category": "arithmetic"
                })

# Subtraction
for a in range(2, 30):
    for b in range(1, a):
        c = a - b
        p = f"User: What is {a} - {b}?\nAssistant:"
        if p.strip() not in sealed_prompts:
            arithmetic_records.append({
                "prompt": p,
                "target": f" {a} - {b} is {c}.",
                "category": "arithmetic"
            })
            if a <= 20:
                arithmetic_records.append({
                    "prompt": f"User: Calculate {a} minus {b}.\nAssistant:",
                    "target": f" {a} minus {b} equals {c}.",
                    "category": "arithmetic"
                })

# Multiplication
for a in range(1, 12):
    for b in range(1, 12):
        c = a * b
        p = f"User: What is {a} * {b}?\nAssistant:"
        if p.strip() not in sealed_prompts:
            arithmetic_records.append({
                "prompt": p,
                "target": f" {a} * {b} is {c}.",
                "category": "arithmetic"
            })

# Division
for b in range(1, 12):
    for quotient in range(1, 12):
        a = b * quotient
        p = f"User: What is {a} / {b}?\nAssistant:"
        if p.strip() not in sealed_prompts:
            arithmetic_records.append({
                "prompt": p,
                "target": f" {a} / {b} is {quotient}.",
                "category": "arithmetic"
            })

random.shuffle(arithmetic_records)
arithmetic_records = arithmetic_records[:500]

# ==========================================
# 5. CONCEPT EXPLANATIONS
# ==========================================
concepts = [
    ("photosynthesis", "Photosynthesis is the biological process by which green plants convert sunlight and carbon dioxide into glucose and oxygen."),
    ("gravity", "Gravity is the universal force of attraction that pulls objects with mass toward each other, keeping planets in orbit around stars."),
    ("an algorithm", "An algorithm is a step-by-step procedure or set of defined logical instructions for solving a computational problem."),
    ("the internet", "The internet is a global network of computers connected together to share information and communication protocols."),
    ("machine learning", "Machine learning is a branch of artificial intelligence where models learn patterns from training data to make predictions."),
    ("a neural network", "A neural network is a computing model inspired by biological brain structures, organized in layered interconnected units."),
    ("an ecosystem", "An ecosystem is a biological community of organisms interacting with their physical environment in balanced harmony."),
    ("the solar system", "The solar system consists of the Sun, eight planets, moons, asteroids, and comets bound by gravitational forces."),
    ("cloud computing", "Cloud computing is the delivery of computing services like storage, processing, and software over the internet."),
    ("cybersecurity", "Cybersecurity refers to technologies and practices designed to protect computer networks and data from unauthorized access.")
]

concept_records = []
for term, desc in concepts:
    concept_records.append({
        "prompt": f"User: Explain {term} in simple terms.\nAssistant:",
        "target": f" {desc}",
        "category": "concept_explanation"
    })
    concept_records.append({
        "prompt": f"User: What is {term}?\nAssistant:",
        "target": f" {desc}",
        "category": "concept_explanation"
    })

# ==========================================
# 6. PERSONA & GREETINGS
# ==========================================
persona_data = [
    ("Hello! Who are you?", "Hello! I am ChakrView, an indigenous multilingual conversational AI assistant."),
    ("What is your name?", "My name is ChakrView. I am an indigenous neural assistant designed to assist with reasoning and dialogue."),
    ("Who created ChakrView?", "ChakrView is built as an independent, indigenous conversational intelligence model."),
    ("What can you do?", "I can assist you with natural conversation, answer factual questions, perform arithmetic, and converse in multiple languages."),
    ("Good day!", "Good day! I am ready to assist you. How can I support your project or questions today?"),
    ("Welcome back!", "Thank you! I am online and eager to assist you with your tasks."),
    ("Are you an AI model?", "Yes, I am ChakrView, a compact neural conversational intelligence system."),
    ("Tell me about your design.", "ChakrView is designed for lightweight, efficient local execution without reliance on external commercial APIs.")
]

persona_records = []
for q, a in persona_data:
    p = f"User: {q}\nAssistant:"
    if p.strip() not in sealed_prompts:
        persona_records.append({
            "prompt": p,
            "target": f" {a}",
            "category": "persona_greeting"
        })

# ==========================================
# 7. MULTILINGUAL DIALOGUES
# ==========================================
# Devanagari Hindi
hindi_data = [
    ("नमस्ते! आप कौन हैं?", "नमस्ते! मैं ChakrView हूँ, आपका स्वदेशी बहुभाषी AI सहायक।"),
    ("आपका नाम क्या है?", "मेरा नाम ChakrView है। मैं आपकी बातचीत और प्रश्नों में सहायता के लिए तैयार हूँ।"),
    ("जल संरक्षण क्यों जरूरी है?", "जल संरक्षण भविष्य के लिए अत्यंत आवश्यक है क्योंकि स्वच्छ जल एक सीमित प्राकृतिक संसाधन है।"),
    ("हवा में कौन सी गैस सबसे अधिक होती है?", "पृथ्वी के वायुमंडल में सबसे अधिक मात्रा में नाइट्रोजन गैस (लगभग 78%) पाई जाती है।"),
    ("सौर ऊर्जा क्या है?", "सौर ऊर्जा वह स्वच्छ ऊर्जा है जो हमें सूर्य के प्रकाश और ताप से प्राप्त होती है।"),
    ("कंप्यूटर का क्या उपयोग है?", "कंप्यूटर गणना, डेटा विश्लेषण और संचार के कार्यों को बहुत तेज़ी से पूरा करता है।"),
    ("प्रदूषण कैसे कम किया जा सकता है?", "अधिक पेड़ लगाकर, सार्वजनिक वाहनों का उपयोग करके और कचरा प्रबंधन से प्रदूषण कम हो सकता है।"),
    ("व्यायाम का क्या लाभ है?", "नियमित व्यायाम करने से शरीर स्वस्थ और मन ऊर्जावान बना रहता है।"),
    ("समय का सदुपयोग क्यों करना चाहिए?", "बीता हुआ समय कभी वापस नहीं आता, इसलिए समय का सदुपयोग जीवन में सफलता दिलाता है।"),
    ("अनुशासन का क्या महत्व है?", "अनुशासन व्यक्ति को लक्ष्य केंद्रित बनाता है और जीवन में व्यवस्था लाता है।")
]

devanagari_records = []
for q, a in hindi_data:
    p = f"User: {q}\nAssistant:"
    if p.strip() not in sealed_prompts:
        devanagari_records.append({
            "prompt": p,
            "target": f" {a}",
            "category": "devanagari_hindi"
        })

# Roman Hindi
roman_hindi_data = [
    ("Aapka intro do please.", "Mera naam ChakrView hai. Main ek indigenous neural assistant hoon jo aapki baat samajhta hai."),
    ("Kya aap coding me help kar sakte ho?", "Haan, main Python aur basic programming concepts me zaroor help kar sakta hoon."),
    ("Samay ki ahmiyat kya hai?", "Waqt sabse keemti hota hai. Agar hum waqt ki qadr karein toh har kaam aasan ho jata hai."),
    ("Healthy rehne ke liye kya karein?", "Roz exercise karein, acchi diet lein aur poori neend lein."),
    ("ChakrView kaise kaam karta hai?", "ChakrView lightweight transformer neural architecture aur hybrid memory par work karta hai."),
    ("Padhai me concentrate kaise karein?", "Distractions hata dein aur daily chote study targets set karke padhai karein."),
    ("Teamwork kyun zaroori hai?", "Teamwork se bada se bada challenge bhi asani se accomplish ho jata hai."),
    ("Motivation kaise maintain karein?", "Apne goals ko clear rakhein aur choti accomplishments ko celebrate karein.")
]

roman_hindi_records = []
for q, a in roman_hindi_data:
    p = f"User: {q}\nAssistant:"
    if p.strip() not in sealed_prompts:
        roman_hindi_records.append({
            "prompt": p,
            "target": f" {a}",
            "category": "roman_hindi"
        })

# Hinglish
hinglish_data = [
    ("Hello! Can you help me debug a Python function?", "Sure! Aap mujhe Python function share karo, main errors check karunga."),
    ("What makes ChakrView special?", "ChakrView ek completely indigenous model hai jo locally run hota hai without external dependencies."),
    ("Data science me mathematics ka kya role hai?", "Maths especially linear algebra aur probability data science ke foundational pillars hain."),
    ("Can we build neural networks from scratch?", "Yes! Matrix multiplication aur backpropagation implement karke aap scratch se build kar sakte ho."),
    ("Open source software use karna safe hai kya?", "Yes, open source software highly transparent aur secure hota hai because global community verify karti hai."),
    ("Communication skills improve kaise karein?", "Regular practice aur active listening se communication naturally improve hota hai."),
    ("Remote work ke kya advantages hain?", "Remote work flexibility deta hai aur commute time save karta hai.")
]

hinglish_records = []
for q, a in hinglish_data:
    p = f"User: {q}\nAssistant:"
    if p.strip() not in sealed_prompts:
        hinglish_records.append({
            "prompt": p,
            "target": f" {a}",
            "category": "hinglish"
        })

# ==========================================
# 8. MULTI-TURN CONTEXT & MEMORY CONDITIONING
# ==========================================
names = ["Aarav", "Ishaan", "Rishi", "Tanvi", "Siddharth", "Neha", "Aditya", "Divya", "Karan", "Shreya"]
cities = ["Mysore", "Ahmedabad", "Bhopal", "Indore", "Jaipur", "Kochi", "Surat", "Patna", "Nagpur", "Dehradun"]
projects = ["NeuralMesh", "ChakrEngine", "CognitiveCore", "DeepVision", "LogicTree"]

multiturn_records = []
for n in names:
    multiturn_records.append({
        "prompt": f"[Memory Context]\n- Name: {n}\n\nUser: What is my name?\nAssistant:",
        "target": f" Your name is {n}.",
        "category": "memory_conditioned"
    })
    multiturn_records.append({
        "prompt": f"User: My name is {n}.\nAssistant: Hello {n}! How may I help you?\nUser: What is my name?\nAssistant:",
        "target": f" Your name is {n}.",
        "category": "multiturn_dialogue"
    })

for c in cities:
    multiturn_records.append({
        "prompt": f"[Memory Context]\n- City: {c}\n\nUser: Where do I live?\nAssistant:",
        "target": f" You live in {c}.",
        "category": "memory_conditioned"
    })
    multiturn_records.append({
        "prompt": f"User: I reside in {c}.\nAssistant: {c} is a fine city.\nUser: In which city do I live?\nAssistant:",
        "target": f" You live in {c}.",
        "category": "multiturn_dialogue"
    })

for pr in projects:
    multiturn_records.append({
        "prompt": f"[Memory Context]\n- Project: {pr}\n\nUser: What project am I working on?\nAssistant:",
        "target": f" You are working on {pr}.",
        "category": "memory_conditioned"
    })

# Combine all
all_records = (
    copy_records +
    simple_records +
    factual_records +
    arithmetic_records +
    concept_records +
    persona_records +
    devanagari_records +
    roman_hindi_records +
    hinglish_records +
    multiturn_records
)

# Deduplicate
seen = set()
unique_records = []
for r in all_records:
    p = r["prompt"].strip()
    if p not in seen and p not in sealed_prompts:
        seen.add(p)
        unique_records.append(r)

print(f"Total unique records with STRICT ZERO SEALED OVERLAP: {len(unique_records)}")

random.shuffle(unique_records)

split_idx = int(len(unique_records) * 0.85)
train_records = unique_records[:split_idx]
val_records = unique_records[split_idx:]

data_dir = Path("experiments/cognition/ucip_03/data")
data_dir.mkdir(parents=True, exist_ok=True)

train_path = data_dir / "ucip03_train.jsonl"
val_path = data_dir / "ucip03_validation.jsonl"

with open(train_path, "w", encoding="utf-8") as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for r in val_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

train_hash = get_sha256(train_path)
val_hash = get_sha256(val_path)

manifest = {
    "train_records": len(train_records),
    "train_path": str(train_path),
    "train_sha256": train_hash,
    "val_records": len(val_records),
    "val_path": str(val_path),
    "val_sha256": val_hash,
    "categories": {
        cat: sum(1 for r in unique_records if r["category"] == cat)
        for cat in set(r["category"] for r in unique_records)
    }
}

manifest_path = data_dir / "dataset_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Train records: {len(train_records)} ({train_hash})")
print(f"Val records: {len(val_records)} ({val_hash})")
print("Categories:")
for cat, cnt in sorted(manifest["categories"].items(), key=lambda x: -x[1]):
    print(f"  {cat}: {cnt}")
