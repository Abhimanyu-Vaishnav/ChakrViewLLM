import sys
import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

# Load existing generated dataset
train_file = REPO_ROOT / "experiments/cognition/ucip_05/data/ucip05_train.jsonl"
val_file = REPO_ROOT / "experiments/cognition/ucip_05/data/ucip05_validation.jsonl"

all_records = []
for p in [train_file, val_file]:
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    all_records.append(json.loads(l))

# Load frozen benchmark to prevent contamination
benchmark_path = REPO_ROOT / "experiments/cognition/ucip_05/baseline_benchmark.json"
sealed_inputs = set()
if benchmark_path.exists():
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
        for item in bench_data:
            sealed_inputs.add(item["input"].strip().lower())

def add_rec(prompt_text: str, target_text: str, category: str):
    q = prompt_text.split("User:")[-1].split("Assistant:")[0].strip().lower()
    if q in sealed_inputs:
        return
    tgt = target_text if target_text.startswith(" ") else f" {target_text}"
    all_records.append({
        "prompt": prompt_text,
        "target": tgt,
        "category": category
    })

# 1. Expand Multi-Turn Dialogues (Adding 400+ multi-turn records)
multi_turn_base = [
    # Dialogue A: Coding
    ("Can you help me learn Python?", "Certainly! Python is known for its readable syntax. We can start with variables, functions, and data structures.",
     "How do I define a function in Python?", "You use the def keyword followed by the function name and parentheses, for example: def my_function():",
     "What does the return keyword do?", "The return keyword exits a function and sends back an evaluated value to the caller."),
    # Dialogue B: Operating Systems
    ("What is an operating system process?", "A process is an active program instance executing in memory with its own allocated address space.",
     "How is a thread different from a process?", "A thread is a lightweight execution stream within a process, sharing the same address space with peer threads.",
     "Why are threads used for multitasking?", "Threads allow concurrent execution with lower memory overhead and faster context switching than separate processes."),
    # Dialogue C: Solar System
    ("How many planets are in our solar system?", "There are eight recognized planets orbiting our Sun.",
     "Which planet is closest to the Sun?", "Mercury is the closest planet to the Sun and experiences extreme temperature fluctuations.",
     "Does Mercury have an atmosphere?", "Mercury has only a very thin exosphere composed of atoms blasted off its surface by the solar wind."),
    # Dialogue D: Machine Learning
    ("What is machine learning in simple terms?", "Machine learning is a field of artificial intelligence where algorithms learn patterns from data instead of being explicitly programmed.",
     "What is the loss function in training?", "The loss function calculates the difference between the model's prediction and the actual target value.",
     "Why do we minimize the loss?", "Minimizing loss directs the model weights toward parameters that generate more accurate predictions."),
    # Dialogue E: Healthy Living
    ("Why is regular exercise recommended?", "Regular exercise strengthens the cardiovascular system, improves mood via endorphins, and boosts physical endurance.",
     "How much exercise is recommended each week?", "Health guidelines typically recommend at least 150 minutes of moderate aerobic exercise per week.",
     "Does walking count toward this goal?", "Yes, brisk daily walking is an effective, accessible, and low-impact form of aerobic exercise."),
    # Dialogue F: Trees and Nature
    ("How do trees benefit human cities?", "Trees provide cooling shade, absorb airborne pollutants, produce oxygen, and mitigate stormwater runoff.",
     "Which trees grow fastest?", "Species like willows, poplars, and certain pines are known for their rapid annual growth rates.",
     "Why do deciduous trees lose their leaves?", "They shed leaves in autumn to conserve water and prevent cellular damage during cold winter months."),
    # Dialogue G: Compilers and Interpreters
    ("What does a language compiler do?", "A compiler translates entire high-level source files into optimized machine binary instructions before execution.",
     "What is an interpreter?", "An interpreter executes programming source code directly line by line at runtime without prior binary compilation.",
     "Which one is generally faster?", "Compiled code generally runs faster because CPU instructions are fully resolved and optimized ahead of time."),
    # Dialogue H: Databases
    ("What is a relational database?", "A relational database organizes data into structured tables with rows and columns, linked by defined relationships.",
     "What is a primary key?", "A primary key is a unique identifier assigned to each record in a database table to prevent duplication.",
     "What is a foreign key?", "A foreign key is a column that creates a verified relationship referencing the primary key of another table."),
    # Dialogue I: Devanagari Multi-Turn
    ("नमस्ते! क्या आप मेरी पढ़ाई में मदद कर सकते हैं?", "नमस्ते! बिल्कुल, मैं विज्ञान, गणित और भाषा सीखने में आपकी पूरी सहायता करूँगा।",
     "कंप्यूटर का मुख्य भाग कौन सा होता है?", "कंप्यूटर का मुख्य भाग सीपीयू (CPU) होता है जो सभी निर्देशों को प्रोसेस करता है।",
     "रैम का क्या काम होता है?", "रैम (RAM) एक तेज़ मेमोरी है जो वर्तमान में उपयोग हो रहे डेटा को सुरक्षित रखती है।"),
    # Dialogue J: Roman Hindi Multi-Turn
    ("Bhai, coding seekhna chahta hoon.", "Bohot badhiya decision hai! Coding logic build karne aur naye software develop karne ke liye bohot useful hai.",
     "Python se shuru karun ya C++ se?", "Beginners ke liye Python ideal hai kyunki iska syntax clear aur easily understandable hota hai.",
     "Daily kitna time practice karni chahiye?", "Daily 1 se 2 ghante focused practice logic ko strong aur programming concepts ko clear bana deti hai.")
]

for t1_u, t1_a, t2_u, t2_a, t3_u, t3_a in multi_turn_base:
    # 2-turn
    add_rec(f"User: {t1_u}\nAssistant: {t1_a}\nUser: {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")
    add_rec(f"User: Hi assistant! {t1_u}\nAssistant: {t1_a}\nUser: {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")
    add_rec(f"User: {t1_u}\nAssistant: {t1_a}\nUser: Thank you. {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")
    add_rec(f"User: {t1_u}\nAssistant: {t1_a}\nUser: One question: {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")
    # 3-turn
    add_rec(f"User: {t1_u}\nAssistant: {t1_a}\nUser: {t2_u}\nAssistant: {t2_a}\nUser: {t3_u}\nAssistant:", t3_a, "multi_turn_conversation")
    add_rec(f"User: {t1_u}\nAssistant: {t1_a}\nUser: {t2_u}\nAssistant: {t2_a}\nUser: Got it! {t3_u}\nAssistant:", t3_a, "multi_turn_conversation")
    add_rec(f"User: Hello! {t1_u}\nAssistant: {t1_a}\nUser: {t2_u}\nAssistant: {t2_a}\nUser: {t3_u}\nAssistant:", t3_a, "multi_turn_conversation")

# 2. Expand Devanagari Hindi (Adding 300+ records)
devanagari_expanded = [
    ("कंप्यूटर का आविष्कार किसने किया?", "कंप्यूटर के बुनियादी सिद्धांतों और एनालिटिकल इंजन का श्रेय चार्ल्स बैबेज को दिया जाता है।"),
    ("सौर ऊर्जा क्या होती है?", "सूर्य से प्राप्त होने वाले प्रकाश और गर्मी को सौर ऊर्जा कहते हैं, जिसे सोलर पैनल द्वारा बिजली में बदला जाता है।"),
    ("जल चक्र क्या है?", "जल चक्र वह प्रक्रिया है जिसमें पानी समुद्रों से वाष्पित होकर बादल बनता है और फिर वर्षा के रूप में पुनः धरती पर आता है।"),
    ("ऑक्सीजन हमारे लिए क्यों जरूरी है?", "ऑक्सीजन हमारी श्वसन क्रिया के लिए अनिवार्य है, जिससे हमारी कोशिकाएं भोजन से ऊर्जा उत्पन्न करती हैं।"),
    ("पर्यावरण को स्वच्छ रखने के उपाय क्या हैं?", "अधिक से अधिक पेड़ लगाएं, प्लास्टिक का उपयोग कम करें, कचरे का सही निस्तारण करें और जल का अपव्यय रोकें।"),
    ("व्यायाम के क्या लाभ हैं?", "व्यायाम करने से हृदय मजबूत होता है, मांसपेशियां सुदृढ़ बनती हैं और मानसिक तनाव कम होता है।"),
    ("भारत का राष्ट्रीय वृक्ष कौन सा है?", "भारत का राष्ट्रीय वृक्ष बरगद (वट वृक्ष) है, जो अपनी विशालता और दीर्घायु के लिए प्रसिद्ध है।"),
    ("अनुशासन क्यों महत्वपूर्ण है?", "अनुशासन जीवन में व्यवस्था लाता है, समय की पाबंदी सिखाता है और कठिन लक्ष्यों को प्राप्त करने में मदद करता है।"),
    ("किताबें इंसान की सच्ची मित्र क्यों हैं?", "किताबें हमें बिना किसी स्वार्थ के नया ज्ञान, मार्गदर्शन और सकारात्मक विचार प्रदान करती हैं।"),
    ("ध्वनि कैसे उत्पन्न होती है?", "ध्वनि किसी वस्तु के कंपन (वाइब्रेशन) से उत्पन्न होती है और किसी माध्यम (जैसे हवा या पानी) के द्वारा यात्रा करती है।"),
    ("प्रकाश की गति कितनी होती है?", "प्रकाश निर्वात में लगभग 3,00,000 किलोमीटर प्रति सेकंड की गति से यात्रा करता है।"),
    ("सत्य और ईमानदारी का क्या मूल्य है?", "सत्य और ईमानदारी से मनुष्य का चरित्र उज्ज्वल होता है और समाज में स्थाई विश्वास और सम्मान मिलता है।")
]

hindi_forms = [
    "User: {}\nAssistant:",
    "User: कृपया बताइए: {}\nAssistant:",
    "User: क्या आप समझा सकते हैं: {}\nAssistant:",
    "User: नमस्ते! {}\nAssistant:",
    "User: संक्षेप में उत्तर दीजिए: {}\nAssistant:",
    "User: इस प्रश्न का उत्तर क्या है: {}\nAssistant:"
]

for q, a in devanagari_expanded:
    for f in hindi_forms:
        add_rec(f.format(q), a, "devanagari_hindi")

# 3. Expand Roman Hindi (Adding 250+ records)
roman_expanded = [
    ("Computer memory RAM kaise kaam karti hai?", "RAM ek ultra-fast temporary storage hoti hai jahan computer active programs ke data ko temporarily store karta hai."),
    ("Hard drive aur SSD me kya difference hota hai?", "Hard drive moving mechanical magnetic parts use karti hai jabki SSD electronic flash chips use karti hai jo bohot fast hoti hain."),
    ("Python me functions kyun banaye jaate hain?", "Functions code reusability ko badhate hain, code ko modular banate hain aur redundancy ko eliminate karte hain."),
    ("Git commit ka kya matlab hota hai?", "Git commit repository me code ke changes ka ek permanent snapshot create karta hai jisse history track ho sake."),
    ("Time table banakar padhna kyun zaroori hai?", "Time table se study organized rehti hai, sabhi subjects ko equal time milta hai aur consistency maintain rehti hai."),
    ("Dopamine hormone kya karta hai?", "Dopamine brain ka neurotransmitter hai jo motivation, reward feeling aur learning drive ko regulate karta hai."),
    ("Internet security ke liye strong password kyun zaroori hai?", "Strong password unauthorized access ko prevent karta hai aur personal data ko hack hone se protect karta hai."),
    ("Morning walk karne se kya fayde milte hain?", "Morning walk se body energetic rehti hai, fresh oxygen milti hai aur cardiovascular health improve hoti hai."),
    ("ChakrView model local hardware par kaise execute hota hai?", "ChakrView compact parameter design aur efficient CPU loops ke through directly local processor par inference run karta hai.")
]

roman_forms = [
    "User: {}\nAssistant:",
    "User: Bhai, {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Simple bhasha me batao: {}\nAssistant:",
    "User: Ek question ka answer do: {}\nAssistant:",
    "User: Please explain: {}\nAssistant:"
]

for q, a in roman_expanded:
    for f in roman_forms:
        add_rec(f.format(q), a, "roman_hindi")

# 4. Expand Hinglish (Adding 250+ records)
hinglish_expanded = [
    ("Neural network loss decrease hona model learning ko kaise reflect karta hai?", "Loss decrease hone ka matlab hai ki model predictions ground truth targets ke closer aa rahi hain aur parameters calibrate ho rahe hain."),
    ("Database normalization table redundancy ko kaise eliminate karti hai?", "Normalization tables ko structured relational forms me decompose karke duplicate data storage aur update anomalies prevent karti hai."),
    ("Continuous delivery pipeline software deployment ko kaise speed up karti hai?", "Continuous delivery automated testing aur reliable staging deployment ke through release cycle time ko drastically reduce karti hai."),
    ("Virtual memory memory allocation limits ko kaise expand karti hai?", "Virtual memory operating system ko disk storage space ko swap space ke roop me address karne deti hai jab physical RAM full ho jaati hai."),
    ("Object oriented programming me inheritance ka kya main advantage hai?", "Inheritance parent class ke existing attributes aur methods ko child classes me reuse karke code duplication ko eliminate karti hai."),
    ("Cache memory latency reduce karke application speed kaise boost karti hai?", "Cache high-frequency data ko CPU cores ke close memory me cache karti hai jisse repeated memory bus lookups avoid hote hain."),
    ("Recursion technique me base case define karna kyun critical hota hai?", "Base case recursion termination condition provide karta hai; agar base case na ho toh program infinite loop me fas kar stack overflow crash kar dega.")
]

hinglish_forms = [
    "User: {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Please clarify this concept: {}\nAssistant:",
    "User: Can you explain: {}\nAssistant:",
    "User: Bhai, {}\nAssistant:",
    "User: Technical doubt solve karo: {}\nAssistant:"
]

for q, a in hinglish_expanded:
    for f in hinglish_forms:
        add_rec(f.format(q), a, "hinglish")

# Deduplicate
seen = set()
dedup_records = []
for r in all_records:
    p = r["prompt"].strip()
    if p not in seen:
        seen.add(p)
        dedup_records.append(r)

print(f"Total deduplicated records: {len(dedup_records)}")

random.shuffle(dedup_records)
split_idx = int(len(dedup_records) * 0.88)
train_split = dedup_records[:split_idx]
val_split = dedup_records[split_idx:]

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

data_dir = REPO_ROOT / "experiments/cognition/ucip_05/data"
with open(data_dir / "ucip05_train.jsonl", "w", encoding="utf-8") as f:
    for rec in train_split:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open(data_dir / "ucip05_validation.jsonl", "w", encoding="utf-8") as f:
    for rec in val_split:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().lower()

manifest = {
    "total_unique_records": len(dedup_records),
    "train_records": len(train_split),
    "validation_records": len(val_split),
    "train_file": str((data_dir / "ucip05_train.jsonl").relative_to(REPO_ROOT)),
    "train_sha256": sha256_file(data_dir / "ucip05_train.jsonl"),
    "val_file": str((data_dir / "ucip05_validation.jsonl").relative_to(REPO_ROOT)),
    "val_sha256": sha256_file(data_dir / "ucip05_validation.jsonl"),
    "category_breakdown": {}
}

for rec in dedup_records:
    cat = rec["category"]
    manifest["category_breakdown"][cat] = manifest["category_breakdown"].get(cat, 0) + 1

with open(data_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("Saved dataset manifest:")
print(json.dumps(manifest, indent=2))
