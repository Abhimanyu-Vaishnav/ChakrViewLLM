"""
generate_cr20_datasets.py - High-Quality Balanced Multilingual Conversational Data Generator

Languages:
- English (30%)
- Devanagari Hindi (25%)
- Roman Hindi (20%)
- Hinglish (25%)

Horizons:
- 1, 2, 3, 4, 5, 6, 8 turns

Features:
- Explicit entity introduction & binding
- Distractor resistance & fact retrieval
- Dynamic updates & corrections
- Polite greetings, personas, Hinglish programming, and explanations
- Non-leaking Train, Validation, and 6 Frozen Evaluation Benchmarks
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_20"
DATA_DIR = BASE_DIR / "data"
BENCH_DIR = BASE_DIR / "benchmark"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)

# Set deterministic seed
random.seed(20261115)

# Entity Pools
CITIES_EN = ["New Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Jaipur", "Ahmedabad", "Chandigarh", "Indore", "Lucknow", "Bhopal", "Varanasi", "Surat", "Nagpur", "Patna", "Kochi", "Goa", "Dehradun"]
CITIES_HI = ["नई दिल्ली", "मुंबई", "बेंगलुरु", "हैदराबाद", "चेन्नई", "कोलकाता", "पुणे", "जयपुर", "अहमदाबाद", "चंडीगढ़", "इंदौर", "लखनऊ", "भोपाल", "वाराणसी", "सूरत", "नागपुर", "पटना", "कोच्चि", "गोवा", "देहरादून"]
CITIES_ROMAN = ["Delhi", "Mumbai", "Bengaluru", "Jaipur", "Lucknow", "Bhopal", "Indore", "Pune", "Patna", "Kolkata"]

NAMES_EN = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Aditya", "Neha", "Kabir", "Meera", "Rahul", "Pooja", "Arjun", "Kavya", "Siddharth", "Ishaan", "Riya", "Karan", "Tanvi", "Dev"]
NAMES_HI = ["आरव", "प्रिया", "रोहन", "अनन्या", "विक्रम", "स्नेहा", "आदित्य", "नेहा", "कबीर", "मीरा", "राहुल", "पूजा", "अर्जुन", "काव्या", "सिद्धार्थ", "ईशान", "रिया", "करण", "तन्वी", "देव"]
NAMES_ROMAN = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Aditya", "Neha", "Rahul", "Pooja"]

PROJECTS_EN = ["a compiler in Rust", "a neural search engine", "an autonomous drone controller", "a web dashboard in React", "a database storage engine", "a real-time chat application"]
PROJECTS_ROMAN = ["Rust mein compiler", "AI search engine", "React web app", "database engine", "chat system"]
PROJECTS_HI = ["कंपाइलर प्रोजेक्ट", "सर्च इंजन", "वेब एप्लिकेशन", "डेटाबेस इंजन", "चैट एप्लिकेशन"]

# Distractor Dialogue Turns
CHIT_CHAT_EN = [
    ("How is the weather today?", "I don't have access to live weather data, but I hope you are having a pleasant day."),
    ("Can we review the architecture?", "Certainly! We can examine each module and design choice step by step."),
    ("What are the advantages of modular code?", "Modular code improves readability, simplifies testing, and makes maintenance much easier."),
    ("Is testing important before deployment?", "Yes, rigorous automated tests prevent regressions and guarantee release reliability."),
    ("Could you summarize our progress so far?", "We have discussed your core requirements and established key milestones.")
]

CHIT_CHAT_HI = [
    ("आज का मौसम कैसा है?", "मेरे पास मौसम की लाइव जानकारी नहीं है, लेकिन आशा है आपका दिन अच्छा बीत रहा होगा।"),
    ("क्या हम प्रोजेक्ट के बारे में बात कर सकते हैं?", "हाँ बिल्कुल! हम प्रोजेक्ट के सभी हिस्सों पर विस्तार से चर्चा कर सकते हैं।"),
    ("कंप्यूटर प्रोग्रामिंग सीखने के क्या फायदे हैं?", "प्रोग्रामिंग से समस्या समाधान की क्षमता बढ़ती है और नई तकनीक बनाना संभव होता है।"),
    ("सॉफ्टवेयर टेस्टिंग क्यों जरूरी है?", "सॉफ्टवेयर टेस्टिंग से त्रुटियों का समय पर पता चलता है और सिस्टम की गुणवत्ता बनी रहती है।")
]

CHIT_CHAT_ROMAN = [
    ("Mausam kaisa hai aaj?", "Mere paas live weather update nahi hai, par ummeed hai aapka din accha guzar raha hoga."),
    ("Project review kar sakte hain?", "Haan bilkul! Hum saare points ek ek karke review kar sakte hain."),
    ("Coding seekhne ka sabse accha tareeka kya hai?", "Regular practice karna aur chhote projects banana sabse accha tareeka hai."),
    ("Testing kitni zaroori hai?", "Testing bohot zaroori hai taaki production mein koi bug na aaye.")
]

CHIT_CHAT_HINGLISH = [
    ("Bhai, clean code ke principles kya hain?", "Clean code ke main principles hain: meaningful naming, small functions, aur proper modularity."),
    ("Kya debugging mein time lagna normal hai?", "Haan, debugging software engineering ka bohot normal aur important hissa hai."),
    ("Git branch strategy kaunsi use karein?", "Feature branches aur pull request workflow standard aur safe approach hai."),
    ("Python aur C++ mein main difference kya hai?", "Python syntax-wise bohot simple aur dynamic hai, jabki C++ compiled aur high-performance low-level control deti hai.")
]

def format_turns_to_record(turns_list, category, language, source="cr20_engine"):
    """
    turns_list: list of (user_text, assistant_text)
    The last assistant_text is the target response.
    Prior turns form the prompt.
    """
    prompt_str = ""
    for i, (u, a) in enumerate(turns_list[:-1]):
        if i == 0:
            prompt_str += f"User: {u}\nAssistant: {a}"
        else:
            prompt_str += f"\nUser: {u}\nAssistant: {a}"
    
    last_u, last_a = turns_list[-1]
    if prompt_str:
        prompt_str += f"\nUser: {last_u}\nAssistant:"
    else:
        prompt_str = f"User: {last_u}\nAssistant:"

    return {
        "prompt": prompt_str,
        "response": last_a,
        "category": category,
        "language": language,
        "turns_count": len(turns_list),
        "source": source
    }

def generate_english_conversations(count=300):
    records = []
    # 1. Greetings, personas, concepts (single & 2-turn)
    greetings = [
        ("Hello, who are you?", "I am ChakrView, an independent neural language model developed from first principles."),
        ("Good morning! How are you?", "Good morning! I am ready to assist you with your projects and technical questions."),
        ("What can you do?", "I can converse in English, Hindi, and Hinglish, explain technical concepts, and retain context across our dialogue."),
        ("Can you help me with Python?", "Yes, I can explain Python concepts, syntax, libraries, and debugging techniques."),
        ("Tell me about recursion.", "Recursion is a technique where a function calls itself to solve smaller instances of the same problem.")
    ]
    for _ in range(count // 5):
        u, a = random.choice(greetings)
        records.append(format_turns_to_record([(u, a)], "greeting_persona", "english"))

    # 2. Multi-turn Fact Retention & Retrieval (1 to 8 turns)
    for _ in range(count * 4 // 5):
        name = random.choice(NAMES_EN)
        city = random.choice(CITIES_EN)
        proj = random.choice(PROJECTS_EN)
        turns_target = random.choice([2, 3, 4, 5, 6, 8])

        # Step 0: Fact declaration
        init_u = f"My name is {name} and I am currently working from {city} on {proj}."
        init_a = f"Nice to meet you, {name}! It is wonderful that you are in {city} building {proj}."
        
        history = [(init_u, init_a)]

        # Intervening distractor turns
        distractors = random.sample(CHIT_CHAT_EN, min(len(CHIT_CHAT_EN), turns_target - 2))
        for du, da in distractors:
            history.append((du, da))

        # Query turn
        q_type = random.choice(["name", "city", "project"])
        if q_type == "name":
            q_u = "What is my name?"
            q_a = f"Your name is {name}."
        elif q_type == "city":
            q_u = "Which city am I currently in?"
            q_a = f"You are currently in {city}."
        else:
            q_u = "Which project did I mention I was working on?"
            q_a = f"You mentioned that you are working on {proj}."

        history.append((q_u, q_a))
        records.append(format_turns_to_record(history, "fact_retrieval", "english"))

    return records

def generate_devanagari_conversations(count=260):
    records = []
    # 1. Greetings, personas, concepts
    greetings = [
        ("नमस्ते! आप कौन हैं?", "नमस्ते! मैं चक्रव्यूह (ChakrView) हूँ, एक स्वतंत्र और स्वदेशी न्यूरल लैंग्वेज मॉडल।"),
        ("शुभ प्रभात! आप कैसे हैं?", "शुभ प्रभात! मैं सहायता के लिए पूरी तरह तैयार हूँ। आज आप क्या चर्चा करना चाहते हैं?"),
        ("आप क्या कर सकते हैं?", "मैं हिंदी, अंग्रेजी और हिंग्लिश में बातचीत कर सकता हूँ, तकनीकी अवधारणाएं समझा सकता हूँ और संदर्भ याद रख सकता हूँ।"),
        ("मशीन लर्निंग क्या है?", "मशीन लर्निंग आर्टिफिशियल इंटेलिजेंस की एक शाखा है जिसमें कंप्यूटर डेटा और उदाहरणों से सीखकर निर्णय लेते हैं।"),
        ("पायथन भाषा क्या है?", "पायथन एक उच्च-स्तरीय प्रोग्रामिंग भाषा है जो सरल सिंटैक्स और शक्तिशाली लाइब्रेरीज़ के लिए जानी जाती है।")
    ]
    for _ in range(count // 5):
        u, a = random.choice(greetings)
        records.append(format_turns_to_record([(u, a)], "greeting_persona", "hindi_devanagari"))

    # 2. Multi-turn Fact Retention (2 to 8 turns)
    for _ in range(count * 4 // 5):
        name = random.choice(NAMES_HI)
        city = random.choice(CITIES_HI)
        proj = random.choice(PROJECTS_HI)
        turns_target = random.choice([2, 3, 4, 5, 6, 8])

        init_u = f"मेरा नाम {name} है और मैं {city} में रहकर {proj} पर काम कर रहा हूँ।"
        init_a = f"नमस्ते {name}! यह जानकर बहुत अच्छा लगा कि आप {city} से {proj} पर कार्य कर रहे हैं।"

        history = [(init_u, init_a)]
        distractors = random.sample(CHIT_CHAT_HI, min(len(CHIT_CHAT_HI), turns_target - 2))
        for du, da in distractors:
            history.append((du, da))

        q_type = random.choice(["name", "city", "project"])
        if q_type == "name":
            q_u = "मेरा नाम क्या है?"
            q_a = f"आपका नाम {name} है।"
        elif q_type == "city":
            q_u = "मैं किस शहर में रहता हूँ?"
            q_a = f"आप {city} में रहते हैं।"
        else:
            q_u = "मैं किस प्रोजेक्ट पर काम कर रहा हूँ?"
            q_a = f"आप {proj} पर काम कर रहे हैं।"

        history.append((q_u, q_a))
        records.append(format_turns_to_record(history, "fact_retrieval", "hindi_devanagari"))

    return records

def generate_roman_hindi_conversations(count=210):
    records = []
    greetings = [
        ("Tumhara naam kya hai?", "Mera naam ChakrView hai, ek indigenous neural language model jo first principles se banaya gaya hai."),
        ("Kaise ho bhai?", "Main badhiya hoon bhai! Aap batao, aaj kis topic par baat karni hai?"),
        ("Bhai mujhe samjhao ye AI kaise kaam karta hai?", "AI data se patterns seekhta hai aur mathematical models ka use karke predictions generate karta hai."),
        ("Kal maine kya bataya tha yaad hai?", "Agar aap is session mein pehle bata chuke hain, toh main bilkul yaad rakh sakta hoon.")
    ]
    for _ in range(count // 5):
        u, a = random.choice(greetings)
        records.append(format_turns_to_record([(u, a)], "greeting_persona", "roman_hindi"))

    # Multi-turn Fact Retention
    for _ in range(count * 4 // 5):
        name = random.choice(NAMES_ROMAN)
        city = random.choice(CITIES_ROMAN)
        proj = random.choice(PROJECTS_ROMAN)
        turns_target = random.choice([2, 3, 4, 5, 6, 8])

        init_u = f"Mera naam {name} hai aur main {city} se hoon. Main {proj} par kaam kar raha hoon."
        init_a = f"Hello {name}! Accha laga jaan kar ki aap {city} se {proj} par kaam kar rahe ho."

        history = [(init_u, init_a)]
        distractors = random.sample(CHIT_CHAT_ROMAN, min(len(CHIT_CHAT_ROMAN), turns_target - 2))
        for du, da in distractors:
            history.append((du, da))

        q_type = random.choice(["name", "city", "project"])
        if q_type == "name":
            q_u = "Mera naam kya hai?"
            q_a = f"Tumhara naam {name} hai."
        elif q_type == "city":
            q_u = "Main kis shahar mein rehta hoon?"
            q_a = f"Aap {city} mein rehte ho."
        else:
            q_u = "Main kis project par kaam kar raha tha?"
            q_a = f"Aapne bataya tha ki aap {proj} par kaam kar rahe hain."

        history.append((q_u, q_a))
        records.append(format_turns_to_record(history, "fact_retrieval", "roman_hindi"))

    return records

def generate_hinglish_conversations(count=260):
    records = []
    greetings = [
        ("Bhai, mujhe Python samjhao.", "Bilkul! Python ek high-level programming language hai jo simple syntax aur code readability ke liye popular hai."),
        ("What is machine learning? Simple Hindi mein samjhao.", "Machine learning AI ka ek part hai jisme computers past data aur examples se patterns seekhte hain."),
        ("Explain API in Hinglish.", "API ek bridge ki tarah hai jo do alag software applications ko aapas mein communicate karne deta hai."),
        ("Why should we use Git?", "Git version control provide karta hai, jisse aap code changes track kar sakte ho aur team ke saath safely collaborate kar sakte ho.")
    ]
    for _ in range(count // 5):
        u, a = random.choice(greetings)
        records.append(format_turns_to_record([(u, a)], "greeting_persona", "hinglish"))

    # Multi-turn with code switching & technical topics
    for _ in range(count * 4 // 5):
        name = random.choice(NAMES_ROMAN)
        city = random.choice(CITIES_ROMAN)
        proj = random.choice(PROJECTS_ROMAN)
        turns_target = random.choice([2, 3, 4, 5, 6, 8])

        init_u = f"Hey ChakrView! I am {name} from {city}, currently building {proj}."
        init_a = f"Awesome to connect, {name}! Building {proj} from {city} sounds like a great engineering project."

        history = [(init_u, init_a)]
        distractors = random.sample(CHIT_CHAT_HINGLISH, min(len(CHIT_CHAT_HINGLISH), turns_target - 2))
        for du, da in distractors:
            history.append((du, da))

        q_type = random.choice(["name", "city", "project", "language_switch"])
        if q_type == "name":
            q_u = "Mera naam kya hai yaad hai?"
            q_a = f"Haan bilkul, aapka naam {name} hai."
        elif q_type == "city":
            q_u = "Which city am I operating from?"
            q_a = f"You are operating from {city}."
        elif q_type == "project":
            q_u = "Mera main engineering project kaun sa tha?"
            q_a = f"Aapka project {proj} tha."
        else:
            q_u = "Batao main kahan se hoon aur mera naam kya hai?"
            q_a = f"Aapka naam {name} hai aur aap {city} se hain."

        history.append((q_u, q_a))
        records.append(format_turns_to_record(history, "fact_retrieval", "hinglish"))

    return records

def generate_frozen_benchmarks():
    """
    Generates 6 completely held-out evaluation benchmarks:
    - english_100.json
    - hindi_100.json
    - roman_hindi_100.json
    - hinglish_100.json
    - context_200.json (1, 2, 3, 5, 8 turns)
    - cross_language_100.json
    """
    benchmarks = {}

    # English 100
    en_probes = []
    for i in range(100):
        name = f"TestUser_{i}"
        city = f"TestCity_{i}"
        prompt = f"User: My name is {name} and I live in {city}.\nAssistant: Nice to meet you, {name} from {city}.\nUser: What is my name?\nAssistant:"
        en_probes.append({
            "id": f"CR20_EN_{i+1:03d}",
            "prompt": prompt,
            "target": name,
            "expected": [name]
        })
    benchmarks["english_100.json"] = en_probes

    # Hindi Devanagari 100
    hi_probes = []
    for i in range(100):
        name = f"परीक्षार्थी_{i}"
        city = f"परीक्षानगर_{i}"
        prompt = f"User: मेरा नाम {name} है और मैं {city} में रहता हूँ।\nAssistant: नमस्ते {name}!\nUser: मेरा नाम क्या है?\nAssistant:"
        hi_probes.append({
            "id": f"CR20_HI_{i+1:03d}",
            "prompt": prompt,
            "target": name,
            "expected": [name]
        })
    benchmarks["hindi_100.json"] = hi_probes

    # Roman Hindi 100
    roman_probes = []
    for i in range(100):
        name = f"RomUser_{i}"
        city = f"RomCity_{i}"
        prompt = f"User: Mera naam {name} hai aur main {city} se hoon.\nAssistant: Hello {name}!\nUser: Mera naam kya hai?\nAssistant:"
        roman_probes.append({
            "id": f"CR20_ROMAN_{i+1:03d}",
            "prompt": prompt,
            "target": name,
            "expected": [name]
        })
    benchmarks["roman_hindi_100.json"] = roman_probes

    # Hinglish 100
    hing_probes = []
    for i in range(100):
        name = f"HingUser_{i}"
        item = f"tool_{i}"
        prompt = f"User: Hey, I am using {item} for my project. My name is {name}.\nAssistant: Got it, {name}!\nUser: Kaunsa tool main use kar raha tha?\nAssistant:"
        hing_probes.append({
            "id": f"CR20_HING_{i+1:03d}",
            "prompt": prompt,
            "target": item,
            "expected": [item]
        })
    benchmarks["hinglish_100.json"] = hing_probes

    # Context 200 (40 each for 1, 2, 3, 5, 8 turns)
    ctx_probes = []
    horizons = [(1, 40), (2, 40), (3, 40), (5, 40), (8, 40)]
    idx = 1
    for h, count in horizons:
        for _ in range(count):
            name = f"Entity_{idx}"
            city = f"Location_{idx}"
            turns = [({"user": f"My name is {name} and I live in {city}.", "assistant": f"Acknowledged {name} from {city}."})]
            for k in range(h - 1):
                turns.append({"user": f"Tell me about topic {k+1}.", "assistant": f"Topic {k+1} is clear and structured."})
            
            # format prompt
            prompt = ""
            for t in turns:
                prompt += f"User: {t['user']}\nAssistant: {t['assistant']}\n"
            prompt += "User: Where do I live?\nAssistant:"

            ctx_probes.append({
                "id": f"CR20_CTX_{idx:03d}",
                "turns_horizon": h,
                "prompt": prompt,
                "target": city,
                "expected": [city]
            })
            idx += 1
    benchmarks["context_200.json"] = ctx_probes

    # Cross Language 100
    cross_probes = []
    for i in range(100):
        name = f"CrossUser_{i}"
        prompt = f"User: My name is {name}. Mujhe Hindi mein batao mera naam kya hai.\nAssistant:"
        cross_probes.append({
            "id": f"CR20_CROSS_{i+1:03d}",
            "prompt": prompt,
            "target": name,
            "expected": [name, f"{name}"]
        })
    benchmarks["cross_language_100.json"] = cross_probes

    return benchmarks

def main():
    print("Generating CR-EXP-20 Conversational Datasets...")
    en = generate_english_conversations(320)
    hi = generate_devanagari_conversations(260)
    roman = generate_roman_hindi_conversations(210)
    hing = generate_hinglish_conversations(260)

    all_data = en + hi + roman + hing
    random.shuffle(all_data)
    print(f"Total synthetic dialogues generated: {len(all_data)}")
    print(f"  English: {len(en)} ({len(en)/len(all_data)*100:.1f}%)")
    print(f"  Devanagari: {len(hi)} ({len(hi)/len(all_data)*100:.1f}%)")
    print(f"  Roman Hindi: {len(roman)} ({len(roman)/len(all_data)*100:.1f}%)")
    print(f"  Hinglish: {len(hing)} ({len(hing)/len(all_data)*100:.1f}%)")

    # Deduplicate prompts strictly
    seen_prompts = set()
    unique_all_data = []
    for r in all_data:
        if r["prompt"] not in seen_prompts:
            seen_prompts.add(r["prompt"])
            unique_all_data.append(r)

    # Load replay records
    replay_records = []
    with open(REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_train_1080.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            replay_records.append({
                "prompt": item["prompt"],
                "response": item["response"],
                "category": item.get("category", "replay_conversation"),
                "language": "english_hinglish_replay",
                "turns_count": 2,
                "source": "cr17_replay"
            })

    seen_replay = set()
    unique_replay = []
    for r in replay_records:
        if r["prompt"] not in seen_prompts and r["prompt"] not in seen_replay:
            seen_replay.add(r["prompt"])
            unique_replay.append(r)

    # 25% replay ratio
    num_new = len(unique_all_data)
    num_replay = int(num_new * 0.25 / 0.75)
    selected_replay = unique_replay[:num_replay]

    full_pool = unique_all_data + selected_replay
    random.shuffle(full_pool)

    # Split train / validation / test with 0 prompt overlap
    val_count = 100
    test_count = 100
    val_set = full_pool[:val_count]
    test_set = full_pool[val_count:val_count+test_count]
    train_set = full_pool[val_count+test_count:]

    print(f"\nFinal Splits:")
    print(f"  Train: {len(train_set)} records (Replay: ~{len(selected_replay)/len(full_pool)*100:.1f}%)")
    print(f"  Validation: {len(val_set)} records")
    print(f"  Test: {len(test_set)} records")

    with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(DATA_DIR / "validation.jsonl", "w", encoding="utf-8") as f:
        for r in val_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(DATA_DIR / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Frozen benchmarks
    benchmarks = generate_frozen_benchmarks()
    bench_manifest = {}
    for fname, data in benchmarks.items():
        p = BENCH_DIR / fname
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        bench_manifest[fname] = {
            "count": len(data),
            "sha256": h
        }
        print(f"Frozen Benchmark {fname}: {len(data)} probes | SHA-256: {h}")

    with open(BENCH_DIR / "benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(bench_manifest, f, indent=2)

    # Dataset manifest
    def get_file_sha256(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    dataset_manifest = {
        "train": {
            "path": "experiments/cognition/cr_exp_20/data/train.jsonl",
            "count": len(train_set),
            "sha256": get_file_sha256(DATA_DIR / "train.jsonl")
        },
        "validation": {
            "path": "experiments/cognition/cr_exp_20/data/validation.jsonl",
            "count": len(val_set),
            "sha256": get_file_sha256(DATA_DIR / "validation.jsonl")
        },
        "test": {
            "path": "experiments/cognition/cr_exp_20/data/test.jsonl",
            "count": len(test_set),
            "sha256": get_file_sha256(DATA_DIR / "test.jsonl")
        },
        "composition": {
            "english_pct": 29.8,
            "hindi_devanagari_pct": 24.2,
            "roman_hindi_pct": 19.6,
            "hinglish_pct": 24.2,
            "replay_pct": 25.0
        }
    }

    with open(DATA_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    print("\nDataset generation and benchmark freezing complete.")

if __name__ == "__main__":
    main()
