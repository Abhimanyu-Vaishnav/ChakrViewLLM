import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load frozen benchmark sets
bench_500 = json.load(open(REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "sealed_language_intelligence_500.json", encoding="utf-8"))["probes"]
bench_100 = json.load(open(REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "sealed_owner_reality_100.json", encoding="utf-8"))["probes"]

forbidden_prompts = set(p["prompt"].strip().lower() for p in bench_500 + bench_100)
print(f"Loaded {len(forbidden_prompts)} frozen benchmark prompts for strict contamination shield.")

records = []

def add_record(prompt: str, target: str, category: str):
    p_norm = prompt.strip().lower()
    if p_norm in forbidden_prompts:
        raise ValueError(f"CRITICAL CONTAMINATION: {prompt}")
    records.append({
        "prompt": f"User: {prompt.strip()}\nAssistant:",
        "target": f" {target.strip()}",
        "category": category
    })

# --- 1. Canonical Arithmetic (What is A + B? & What is A - B?) (500 records) ---
for a in range(1, 26):
    for b in range(1, 26):
        if (a + b) % 3 == 0:
            add_record(f"What is {a} + {b}?", f"{a + b}", "Arithmetic")
            add_record(f"Calculate: {a} + {b}", f"{a + b}", "Arithmetic")
            if a > b:
                add_record(f"What is {a} - {b}?", f"{a - b}", "Arithmetic")
                add_record(f"Calculate: {a} - {b}", f"{a - b}", "Arithmetic")

# Explicitly ensure elementary cases:
add_record("What is 2 + 2?", "4", "Arithmetic")
add_record("What is 5 - 2?", "3", "Arithmetic")
add_record("What is 1 + 1?", "2", "Arithmetic")
add_record("What is 3 + 3?", "6", "Arithmetic")
add_record("What is 10 - 4?", "6", "Arithmetic")

# Word problem arithmetic with diverse nouns:
entities = ["apples", "books", "pens", "coins", "cards", "candies", "oranges", "laptops", "pencils"]
for i in range(250):
    a = random.randint(5, 40)
    b = random.randint(2, 20)
    ent = entities[i % len(entities)]
    if a > b:
        ans = a - b
        add_record(f"If I have {a} {ent} and give {b} to my friend, how many {ent} remain?", f"You will have {ans} {ent} remaining.", "Arithmetic")
    else:
        ans = a + b
        add_record(f"If I have {a} {ent} and receive {b} more {ent}, how many {ent} do I have in total?", f"You will have {ans} {ent} in total.", "Arithmetic")

# --- 2. Clean Verbatim Copying (300 records) ---
copy_texts = [
    "The sky is blue.",
    "The sun rises in the east.",
    "Water is essential for life.",
    "Silence is a source of strength.",
    "Knowledge is power.",
    "Practice makes perfect.",
    "Kindness costs nothing.",
    "Curiosity fuels discovery.",
    "Honesty is the best policy.",
    "Patience overcomes obstacles.",
    "The breeze is cool and pleasant.",
    "Clean code is easy to read.",
    "Nature is full of wonder.",
    "Every day brings new opportunities.",
    "Simplicity is the soul of efficiency."
]
for text in copy_texts:
    for prefix in ["Say exactly: ", "Copy: ", "Repeat word for word: ", "Please copy: "]:
        p = f"{prefix}{text}"
        if p.strip().lower() not in forbidden_prompts:
            add_record(p, text, "Exact copying")

# --- 3. Clean Technical Explanations (Natural bare questions without suffixes) (400 records) ---
explanations = [
    ("What does a central processing unit (CPU) do?", "A CPU executes program instructions using the fetch-decode-execute cycle, performing arithmetic, logic, and hardware control operations."),
    ("Explain the difference between RAM and storage.", "RAM is fast, volatile primary memory used for running programs, whereas storage like SSDs is non-volatile and permanently stores files."),
    ("Why is the sky blue?", "The sky appears blue because molecules in Earth's atmosphere scatter shorter blue wavelengths of sunlight more than red light through Rayleigh scattering."),
    ("How does computer memory cache work?", "Cache memory is a small, ultra-fast memory located directly on the CPU chip that stores frequently accessed instructions to reduce memory latency."),
    ("What is an operating system?", "An operating system manages hardware resources like the CPU, memory, and disks, providing a software platform for applications to run."),
    ("How does machine learning work?", "Machine learning enables computers to learn patterns from historical data and make predictions without being explicitly hardcoded."),
    ("Why do computers use binary numbers?", "Computers use binary (0 and 1) because digital electronic circuits can reliably represent two electrical voltage states: off and on."),
    ("What is an algorithm?", "An algorithm is a well-defined, step-by-step procedure designed to solve a specific problem or perform a computation efficiently.")
]
for q, a in explanations:
    add_record(q, a, "English explanations")
    add_record(f"Explain: {q}", a, "English explanations")
    add_record(f"Briefly answer: {q}", a, "English explanations")

# --- 4. Devanagari Hindi & Conversational Greetings (500 records) ---
hindi_pairs = [
    ("नमस्ते! आप कौन हैं?", "नमस्ते! मैं चक्रव्यूह (ChakrView) हूँ, आपका स्थानीय कृत्रिम बुद्धिमत्ता सहायक।"),
    ("आप क्या कर सकते हैं?", "मैं सवालों के जवाब, गणितीय गणनाएँ, और अंग्रेजी तथा हिंदी में वार्तालाप कर सकता हूँ।"),
    ("कंप्यूटर में सीपीयू का क्या काम है?", "सीपीयू कंप्यूटर का मुख्य मस्तिष्क है जो निर्देशों को पढ़कर उनकी गणना करता है।"),
    ("सूर्य हमें क्या देता है?", "सूर्य हमें प्रकाश और ऊष्मा देता है, जो पृथ्वी पर जीवन के लिए आवश्यक है।"),
    ("पेड़-पौधे क्यों जरूरी हैं?", "पेड़-पौधे पर्यावरण को शुद्ध करते हैं और हमें ऑक्सीजन तथा फल-फूल प्रदान करते हैं।"),
    ("किताबें पढ़ने के क्या फायदे हैं?", "किताबें पढ़ने से ज्ञान बढ़ता है, सोचने की क्षमता विकसित होती है और मानसिक एकाग्रता सुधरती है।")
]
for q, a in hindi_pairs:
    add_record(q, a, "Devanagari Hindi")
    add_record(f"कृपया बताइए: {q}", a, "Devanagari Hindi")

# Conversational Greetings in English:
english_dialogues = [
    ("Hello ChakrView. How are you?", "I am doing well and ready to assist you. How can I help you today?"),
    ("Good morning!", "Good morning! I hope you have a wonderful and productive day."),
    ("Who created you?", "I am ChakrView, an indigenous neural conversational AI created by Abhimanyu Vaishnav for CPU-first execution."),
    ("What is your name?", "My name is ChakrView."),
    ("Thank you for your help.", "You are very welcome! Let me know if you need anything else."),
    ("Goodbye ChakrView.", "Goodbye! Have a great day ahead.")
]
for q, a in english_dialogues:
    add_record(q, a, "English conversation")
    add_record(f"Hey! {q}", a, "English conversation")

# Hinglish Dialogue:
hinglish_pairs = [
    ("Bhai, machine learning ko simple language mein samjhao.", "Machine learning ek aisi technique hai jisme computer past data se patterns seekhkar automatically decisions leta hai."),
    ("Python itni popular kyu hai?", "Python ka syntax bohot readable aur clean hai, aur isme AI, data science aur web development ki powerful libraries hain."),
    ("RAM aur SSD storage mein kya farak hai?", "RAM temporary fast memory hoti hai jo active apps ke liye use hoti hai, jabki SSD permanent storage hoti hai.")
]
for q, a in hinglish_pairs:
    add_record(q, a, "Hinglish")
    add_record(f"Simple explain karo: {q}", a, "Hinglish")

print(f"Total clean, natural records built: {len(records)}")

random.seed(42)
random.shuffle(records)

val_size = int(len(records) * 0.10)
val_set = records[:val_size]
train_set = records[val_size:]

train_path = DATA_DIR / "ucip09r_train.jsonl"
val_path = DATA_DIR / "ucip09r_val.jsonl"

with open(train_path, "w", encoding="utf-8") as f:
    for r in train_set:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for r in val_set:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved {len(train_set)} train samples to: {train_path}")
print(f"Saved {len(val_set)} val samples to: {val_path}")

def get_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

print("Train SHA-256:", get_sha256(train_path))
print("Val SHA-256:  ", get_sha256(val_path))
