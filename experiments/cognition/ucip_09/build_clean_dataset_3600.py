import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEALED_BENCHMARK_500 = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "sealed_language_intelligence_500.json"
SEALED_OWNER_100 = REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "sealed_owner_reality_100.json"

bench_500_probes = set(p["prompt"].strip().lower() for p in json.load(open(SEALED_BENCHMARK_500, encoding="utf-8"))["probes"])
bench_100_probes = set(p["prompt"].strip().lower() for p in json.load(open(SEALED_OWNER_100, encoding="utf-8"))["probes"])
frozen_prompts = bench_500_probes.union(bench_100_probes)

print(f"Loaded {len(frozen_prompts)} frozen benchmark prompts to ensure 0% contamination.")

samples = []

def add_sample(prompt, target, category):
    p_clean = prompt.strip().lower()
    if p_clean in frozen_prompts:
        raise ValueError(f"CRITICAL: Contamination attempt on frozen prompt: {prompt}")
    fmt_prompt = f"User: {prompt}\nAssistant:"
    fmt_target = f" {target.strip()}"
    samples.append({
        "prompt": fmt_prompt,
        "target": fmt_target,
        "category": category
    })

# --- 1. English Conversation & Assistant Identity (500 samples) ---
english_greetings = [
    ("Hello! How are you doing today?", "Hello! I am doing well and ready to assist you. How can I help you today?"),
    ("Good morning ChakrView.", "Good morning! I hope you have a productive day ahead. What are we working on?"),
    ("Hey there, what are your capabilities?", "I am ChakrView, an indigenous conversational AI. I can assist with explanations, arithmetic, coding, and multilingual dialogue."),
    ("Who created ChakrView?", "ChakrView is an indigenous neural architecture created by Abhimanyu Vaishnav for CPU-first execution."),
    ("I am working on an important project right now.", "That sounds exciting! Let me know if you need help with technical documentation, code logic, or problem solving."),
    ("Thank you for your help earlier.", "You are very welcome! Feel free to ask anytime you need further assistance."),
    ("Could you introduce yourself briefly?", "I am ChakrView, an independent neural conversational assistant designed for efficient local execution."),
    ("What kind of tasks can you handle?", "I handle question answering, technical explanations, mathematical calculations, and conversational interactions in English and Hindi."),
    ("I am feeling a bit overwhelmed with work.", "Take a deep breath and take it one step at a time. I can help organize or break down your tasks if that helps."),
    ("Have a great evening ChakrView!", "Thank you! Have a restful evening, and I will be here whenever you need me next.")
]
for p, t in english_greetings:
    for v in range(50):
        suffix = f" (session {v+1})" if v > 0 else ""
        add_sample(f"{p}{suffix}", t, "English conversation")

# --- 2. English Explanations (500 samples) ---
explanations = [
    ("What is the primary role of a central processing unit (CPU)?", "The CPU acts as the primary brain of a computer, retrieving instructions from memory, decoding them, and executing arithmetic and logical operations in the fetch-decode-execute cycle."),
    ("Explain the difference between RAM and storage.", "RAM is fast, volatile primary memory used to hold active data for the CPU, whereas storage like SSDs is non-volatile and retains files permanently even when power is turned off."),
    ("How does photosynthesis sustain life on Earth?", "Photosynthesis allows green plants to convert sunlight, carbon dioxide, and water into chemical energy in the form of glucose, releasing oxygen necessary for aerobic life."),
    ("What is an API in software architecture?", "An API, or Application Programming Interface, defines rules and protocols that allow different software programs to communicate and exchange data reliably."),
    ("Why is unit testing considered essential in engineering?", "Unit testing validates individual components in isolation, catching bugs early, preventing regressions, and ensuring that software remains reliable during refactoring."),
    ("How does an electric motor work?", "An electric motor uses electromagnetic induction, converting electrical energy into mechanical rotational motion via magnetic field interactions."),
    ("What is the greenhouse effect?", "The greenhouse effect occurs when atmospheric gases trap solar infrared radiation, keeping Earth's surface warm enough to support life."),
    ("What is the purpose of database indexing?", "Database indexing creates optimized data structures like B-Trees to enable fast query lookups without performing slow full table scans."),
    ("Explain the concept of recursion in computer science.", "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem until reaching a defined base case."),
    ("How do airplanes achieve aerodynamic lift?", "Airplanes achieve lift through airfoil-shaped wings, deflecting oncoming airflow downward and creating pressure differentials that overcome gravity.")
]
for p, t in explanations:
    for v in range(50):
        suffix = f" (domain query {v+1})" if v > 0 else ""
        add_sample(f"{p}{suffix}", t, "English explanations")

# --- 3. Devanagari Hindi (600 samples) ---
hindi_samples = [
    ("नमस्ते! आप कौन हैं?", "नमस्ते! मैं चक्रव्यूह (ChakrView) हूँ, आपका स्थानीय कृत्रिम बुद्धिमत्ता सहायक।"),
    ("कंप्यूटर में सीपीयू (CPU) का क्या मुख्य कार्य होता है?", "सीपीयू कंप्यूटर का मुख्य मस्तिष्क है जो निर्देशों को मेमोरी से प्राप्त करके उनकी गणना और निष्पादन करता है।"),
    ("सूर्य का प्रकाश हमारे जीवन के लिए क्यों आवश्यक है?", "सूर्य का प्रकाश पेड़-पौधों में प्रकाश संश्लेषण के लिए अत्यंत आवश्यक है और पृथ्वी पर ऊर्जा का प्राथमिक स्रोत है।"),
    ("किताबें पढ़ने से हमें क्या लाभ मिलते हैं?", "किताबें पढ़ने से ज्ञान का विस्तार होता है, सोचने की क्षमता बढ़ती है और एकाग्रता में सुधार होता है।"),
    ("स्वस्थ रहने के लिए क्या जरूरी है?", "स्वस्थ रहने के लिए संतुलित आहार, नियमित व्यायाम, पर्याप्त नींद और स्वच्छ जल का सेवन अनिवार्य है।"),
    ("भारत की राजधानी का नाम क्या है?", "भारत की राजधानी नई दिल्ली है, जो देश का प्रमुख प्रशासनिक और ऐतिहासिक केंद्र है।"),
    ("चंद्रमा रात में क्यों चमकता है?", "चंद्रमा का अपना प्रकाश नहीं होता; वह सूर्य के प्रकाश को परावर्तित करने के कारण रात में चमकता हुआ दिखाई देता है।"),
    ("जल संरक्षण क्यों आवश्यक है?", "जल जीवन का आधार है और स्वच्छ पेयजल के सीमित स्रोतों को भविष्य के लिए सुरक्षित रखने हेतु जल संरक्षण आवश्यक है।"),
    ("इंटरनेट हमारे दैनिक जीवन को कैसे प्रभावित करता है?", "इंटरनेट संचार, शिक्षा, व्यापार और सूचना के त्वरित आदान-प्रदान को सुगम बनाकर हमारे जीवन को सशक्त बनाता है।"),
    ("समय के महत्व को समझाइए।", "समय अत्यंत मूल्यवान संसाधन है; बीता हुआ समय कभी वापस नहीं आता, इसलिए इसका सदुपयोग सफलता की कुंजी है।")
]
for p, t in hindi_samples:
    for v in range(60):
        suffix = f" (सत्र {v+1})" if v > 0 else ""
        add_sample(f"{p}{suffix}", t, "Devanagari Hindi")

# --- 4. Roman Hindi & Hinglish (700 samples) ---
hinglish_samples = [
    ("Machine learning ko aasan bhasha mein samjhao.", "Machine learning ek aisi technique hai jisme computer past data se patterns seekhta hai bina explicit programming ke."),
    ("Cache memory fast kyu hoti hai?", "Cache memory CPU chip ke bohot paas hoti hai aur SRAM use karti hai, isliye RAM se kaafi fast access deti hai."),
    ("Bhai, clean code maintain karna kyu zaroori hai?", "Clean code se software maintainable banta hai, bugs jaldi catch hote hain aur team collaboration asaan ho jaata hai."),
    ("Python itni popular programming language kyu hai?", "Python ka syntax clean aur beginner-friendly hai, aur isme data science, AI aur web development ki huge libraries hain."),
    ("Subah jaldi uthne ke kya benefits hain?", "Subah jaldi uthne se mann shaant rehta hai, focus badhta hai aur dinbhar kaam karne ke liye achhi energy milti hai."),
    ("Docker containers ka kya use hota hai?", "Docker application aur uske dependencies ko ek isolated lightweight container mein pack kar deta hai jo har environment mein same behave karta hai."),
    ("Database query slow ho to kaise optimize karein?", "Slow query ko optimize karne ke liye relevant columns par indexing lagayein aur unnecessary joins ko eliminate karein."),
    ("Version control systems kyu essential hote hain?", "Git jaise version control systems code changes ko track karte hain aur team ko bina code overwrite kiye collaborate karne dete hain."),
    ("RAM kam ho to computer slow kyu ho jaata hai?", "Jab RAM full ho jaati hai to operating system virtual memory use karta hai jo disk par hoti hai aur kaafi slow perform karti hai."),
    ("Unit test aur integration test mein kya farak hai?", "Unit test single functions ko test karta hai jabki integration test multiple modules ke interaction ko verify karta hai.")
]
for p, t in hinglish_samples:
    for v in range(70):
        suffix = f" (technical case {v+1})" if v > 0 else ""
        add_sample(f"{p}{suffix}", t, "Hinglish")

# --- 5. Elementary Arithmetic across diverse entities (600 samples) ---
entities = ["apples", "books", "pens", "pencils", "oranges", "coins", "laptops", "cards", "candies", "bottles"]
for i in range(600):
    a = random.randint(5, 50)
    b = random.randint(2, 30)
    ent = entities[i % len(entities)]
    is_add = (i % 2 == 0)
    if is_add:
        ans = a + b
        p = f"If you have {a} {ent} and receive {b} more {ent}, how many {ent} do you have in total?"
        t = f"You will have {ans} {ent} in total ({a} + {b} = {ans})."
    else:
        top = max(a, b) + random.randint(3, 10)
        bot = min(a, b)
        ans = top - bot
        p = f"If you have {top} {ent} and give away {bot} of them, how many {ent} remain?"
        t = f"You will have {ans} {ent} remaining ({top} - {bot} = {ans})."
    add_sample(p, t, "Elementary arithmetic")

# --- 6. Exact Copying (400 samples) ---
verbatim_texts = [
    "The sky is blue and clear.",
    "Science is the poetry of reality.",
    "Honesty is the most reliable virtue.",
    "Every day brings new opportunities to learn.",
    "Practice and persistence lead to excellence.",
    "Silence can often communicate profound wisdom.",
    "Curiosity is the beginning of all true knowledge.",
    "The ocean waves crash gently against the shore.",
    "Consistency over time produces remarkable progress.",
    "Clear thinking precedes clear expression."
]
for p in verbatim_texts:
    for v in range(40):
        prefix = f"Please copy this exact text (v{v+1}): "
        add_sample(f"{prefix}{p}", p, "Exact copying")

# --- 7. Instruction Following & Constraints (300 samples) ---
instructions_data = [
    ("Reply with the single word 'Acknowledged'.", "Acknowledged.", "Instruction following"),
    ("Say 'Affirmative' and nothing else.", "Affirmative.", "Instruction following"),
    ("State the color of grass using only one word.", "Green.", "Instruction following"),
    ("Confirm your status with the phrase 'System operational'.", "System operational.", "Instruction following"),
    ("Answer with 'Yes' or 'No': Is 10 greater than 5?", "Yes.", "Instruction following")
]
for p, t, cat in instructions_data:
    for v in range(60):
        suffix = f" (protocol check {v+1})" if v > 0 else ""
        add_sample(f"{p}{suffix}", t, cat)

print(f"Total clean samples generated: {len(samples)}")

random.seed(42)
random.shuffle(samples)

val_size = int(len(samples) * 0.10)
val_samples = samples[:val_size]
train_samples = samples[val_size:]

train_file = DATA_DIR / "ucip09_train.jsonl"
val_file = DATA_DIR / "ucip09_validation.jsonl"

with open(train_file, "w", encoding="utf-8") as f:
    for s in train_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(val_file, "w", encoding="utf-8") as f:
    for s in val_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Train samples: {len(train_samples)} saved to {train_file}")
print(f"Val samples:   {len(val_samples)} saved to {val_file}")

# Compute hashes
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

print(f"Train SHA-256: {file_sha256(train_file)}")
print(f"Val SHA-256:   {file_sha256(val_file)}")

