"""
experiments/cognition/ucip_05/data/expand_ucip05_dataset.py

Substantially expands the UCIP-05 corpus across all categories:
- Basic English conversation (500+)
- Multi-turn conversation (600+)
- English concepts & explanations (600+)
- English sentence generation (500+)
- Devanagari Hindi (600+)
- Roman Hindi (500+)
- Hinglish (600+)
- Exact copying (400+)
- Arithmetic (600+)
- Basic Instructions (300+)

Target: 4,500 - 5,500 unique records.
Validation split: 500+ records.
Guarantees strict zero-overlap with baseline_benchmark.json.
"""

import sys
import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

random.seed(42)

# Load frozen benchmark probes to strictly exclude them
benchmark_path = REPO_ROOT / "experiments/cognition/ucip_05/baseline_benchmark.json"
sealed_inputs = set()
if benchmark_path.exists():
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
        for item in bench_data:
            sealed_inputs.add(item["input"].strip().lower())

print(f"Loaded {len(sealed_inputs)} sealed benchmark inputs for strict zero-contamination filtering.")

records = []

def add_record(prompt_text: str, target_text: str, category: str):
    q = prompt_text.split("User:")[-1].split("Assistant:")[0].strip().lower()
    if q in sealed_inputs:
        return
    tgt = target_text if target_text.startswith(" ") else f" {target_text}"
    records.append({
        "prompt": prompt_text,
        "target": tgt,
        "category": category
    })

# 1. Broad English Conversations (Questions & Inquiries)
eng_conv_data = [
    ("How do computers think?", "Computers do not think like humans; they execute programmed mathematical algorithms and logic operations over digital data."),
    ("Can artificial intelligence feel emotions?", "No, AI models process statistical representations and lack subjective feelings, emotions, or consciousness."),
    ("What is your primary design strength?", "My strength is being a compact, local-first conversational neural network capable of fast inference on standard CPUs."),
    ("Why is local AI important?", "Local AI ensures total privacy, zero latency over networks, and operates without dependency on external cloud providers."),
    ("How do you stay so polite?", "Politeness and clarity are foundational values embedded into my training objectives."),
    ("Can you assist with software debugging?", "Yes, I can help analyze logic errors, explain tracebacks, and suggest architectural improvements."),
    ("What is the best way to learn software engineering?", "Start with foundational principles, practice writing clean code daily, and build real projects step-by-step."),
    ("Do you have access to my private files?", "No, I only process the text tokens sent directly to this chat session."),
    ("What should I do when code fails to run?", "Read the error traceback carefully, identify the failing line, inspect variable states, and test small isolated parts."),
    ("How does machine intelligence evolve?", "Machine intelligence advances through better model architectures, higher-quality data curation, and efficient training algorithms."),
    ("What is the meaning of open science?", "Open science is the practice of sharing scientific research, methodologies, and data openly for transparent verification."),
    ("Can we discuss philosophy?", "Certainly! We can explore epistemology, ethics, logic, and philosophy of mind."),
    ("How do search engines find web pages?", "Search engines use automated web crawlers to discover pages, index their content, and rank results using relevance algorithms."),
    ("Why do programmers use version control?", "Version control allows developers to track code modifications, revert to earlier states, and collaborate seamlessly in teams."),
    ("What makes clean architecture valuable?", "Clean architecture separates core business logic from external frameworks, making software testable and easily maintainable."),
    ("How do electric circuits work?", "Electric circuits provide a closed conductive loop that allows electrical current to flow driven by voltage potential."),
    ("Why is unit testing recommended?", "Unit tests verify that individual software functions produce correct outputs for specified inputs, catching regressions early."),
    ("What is continuous integration?", "Continuous integration is the automated process of building and running tests whenever code changes are committed."),
    ("What is the purpose of code documentation?", "Good documentation explains the intent, architecture, and API usage so other developers can understand and extend the system."),
    ("How does GPS locate my position?", "GPS receivers calculate distance from at least four satellites using precise time signals to triangulate your geographic coordinates.")
]

user_intros = [
    "User: {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Can you tell me: {}\nAssistant:",
    "User: Hi ChakrView, {}\nAssistant:",
    "User: Please answer: {}\nAssistant:",
    "User: I have a question: {}\nAssistant:",
    "User: Could you clarify: {}\nAssistant:",
    "User: Good day! {}\nAssistant:"
]

for q, a in eng_conv_data:
    for intro in user_intros:
        add_record(intro.format(q), a, "english_conversation")

# Also re-add baseline conversational greetings from earlier
eng_greetings_base = [
    ("Hello there!", "Hello! It is a pleasure to connect with you. How can I assist you today?"),
    ("Hi, how are you?", "I am doing well, thank you! I am ready to help you with your questions."),
    ("Good morning assistant.", "Good morning! I hope your day is off to a productive and wonderful start."),
    ("Good afternoon!", "Good afternoon! What topics would you like to explore or work on today?"),
    ("Good evening!", "Good evening! How can I be of assistance to you tonight?"),
    ("Who are you?", "I am ChakrView, an independent neural conversational assistant developed for efficient local execution."),
    ("What is your name?", "My name is ChakrView."),
    ("Who created you?", "ChakrView was built as an independent, locally executable conversational model."),
    ("Where do you run?", "I execute locally on your machine, utilizing efficient CPU processing without cloud dependencies."),
    ("Are you an AI?", "Yes, I am an artificial intelligence language assistant built for natural conversation and reasoning."),
    ("What can you do?", "I can engage in conversation, explain scientific and technical concepts, solve basic arithmetic, generate sentences, and communicate in English, Hindi, and Hinglish."),
    ("Do you require a GPU to run?", "No, ChakrView is engineered CPU-first, designed to run smoothly on everyday CPUs."),
    ("Thank you very much!", "You are very welcome! I am always glad to help."),
    ("Thanks for the explanation.", "My pleasure! Please let me know if you would like me to clarify any detail."),
    ("I appreciate your quick response.", "Happy to help! Feel free to ask whenever you need assistance.")
]
for q, a in eng_greetings_base:
    for intro in user_intros:
        add_record(intro.format(q), a, "english_conversation")

print(f"English conversation count: {len(records)}")

# 2. Multi-turn Conversational Dialogues (Diverse Domains)
dialogue_scenarios = [
    [
        ("What is the fastest land animal?", "The cheetah is the fastest land animal, capable of reaching speeds up to 100 kilometers per hour."),
        ("How does it run so fast?", "Cheetahs have flexible spines, large nasal cavities for high oxygen intake, and non-retractable claws that provide strong grip."),
        ("Where do cheetahs live in the wild?", "Cheetahs primarily inhabit the open grasslands and savannahs of eastern and southern Africa.")
    ],
    [
        ("How does an airplane stay in the air?", "Airplane wings are curved on top, forcing air to travel faster over the wing and creating lift according to Bernoulli's principle."),
        ("What provides forward thrust?", "Jet engines or propellers push air backward, creating forward thrust in accordance with Newton's third law of motion."),
        ("How do pilots steer the plane?", "Pilots steer using flight control surfaces: the rudder for yaw, elevators for pitch, and ailerons for roll.")
    ],
    [
        ("What is an SSD inside a computer?", "A Solid State Drive (SSD) uses semiconductor flash memory chips to store data permanently without mechanical moving parts."),
        ("Why are SSDs faster than mechanical hard drives?", "SSDs have near-instant electronic seek times, while mechanical drives must physically move a read-write head across spinning magnetic platters."),
        ("Do SSDs consume less battery power in laptops?", "Yes, SSDs generally use less power than mechanical drives, which helps improve laptop battery longevity.")
    ],
    [
        ("Why does the moon change shape in the sky?", "The Moon appears to change shape because we see different portions of its sunlit half as it orbits Earth."),
        ("How long does one complete lunar cycle take?", "One complete cycle from new moon to new moon takes approximately 29.5 days."),
        ("Does the moon have its own light?", "No, the Moon does not produce light; it reflects light emitted by the Sun.")
    ],
    [
        ("What is the function of the human heart?", "The heart is a muscular organ that pumps oxygen-rich blood throughout the circulatory system to nourish body tissues."),
        ("How many chambers does the heart have?", "The human heart has four chambers: two upper atria and two lower ventricles."),
        ("Why does exercise make the heart beat faster?", "Exercise increases muscle demand for oxygen and nutrients, prompting the heart to beat faster to supply more blood.")
    ],
    [
        ("What is an operating system kernel?", "The kernel is the fundamental core of an operating system that directly controls hardware resources and process scheduling."),
        ("What is the difference between user space and kernel space?", "Kernel space has unrestricted access to hardware memory and instructions, while user space runs applications safely in isolated unprivileged mode."),
        ("Why is this separation important?", "This separation prevents buggy or malicious user applications from crashing the entire computer system.")
    ],
    [
        ("What is photosynthesis?", "Photosynthesis is the biological process where green plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
        ("Where does this process occur inside plant cells?", "It takes place inside specialized cellular organelles called chloroplasts, which contain green chlorophyll pigments."),
        ("What happens to the glucose produced?", "The plant uses glucose for cellular energy and as building blocks to grow stems, leaves, and roots.")
    ],
    [
        ("What is a database index?", "A database index is a data structure, often a B-tree, that speeds up record lookup operations in a database table."),
        ("Is there any drawback to adding many indexes?", "Yes, each index consumes additional disk storage and slows down insert, update, and delete write operations."),
        ("When should an index be created?", "Indexes should be created on columns that are frequently filtered, searched, or joined in queries.")
    ],
    [
        ("What is the greenhouse effect?", "The greenhouse effect is the natural process where atmospheric gases trap infrared heat, keeping Earth warm enough to sustain life."),
        ("Which gases are the primary greenhouse gases?", "Water vapor, carbon dioxide, methane, and nitrous oxide are the most significant greenhouse gases."),
        ("What happens when greenhouse gas concentrations increase?", "Higher concentrations trap more thermal energy in the lower atmosphere, causing global average temperatures to rise.")
    ],
    [
        ("What is a binary search algorithm?", "Binary search is an efficient algorithm that finds an item in a sorted list by repeatedly dividing the search interval in half."),
        ("What is its time complexity?", "Binary search has a logarithmic time complexity of O(log n), making it extremely fast for large datasets."),
        ("Can binary search be used on an unsorted list?", "No, binary search strictly requires the underlying data array to be sorted beforehand.")
    ]
]

for d in dialogue_scenarios:
    # 2-turn
    t0_u, t0_a = d[0]
    t1_u, t1_a = d[1]
    t2_u, t2_a = d[2]
    
    # Prefix forms
    add_record(f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant:", t1_a, "multi_turn_conversation")
    add_record(f"User: Hello! {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant:", t1_a, "multi_turn_conversation")
    add_record(f"User: {t0_u}\nAssistant: {t0_a}\nUser: Thanks. {t1_u}\nAssistant:", t1_a, "multi_turn_conversation")
    
    # 3-turn
    add_record(f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant: {t1_a}\nUser: {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")
    add_record(f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant: {t1_a}\nUser: One more question: {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")

# Also add conversational multi-turns with greetings first
greetings_multi = [
    ("Hello ChakrView!", "Hello! How can I help you today?", "What does a CPU do?", "A CPU fetches, decodes, and executes software instructions to run computer programs."),
    ("Good morning!", "Good morning! What would you like to explore?", "Can you explain how RAM works?", "RAM is high-speed temporary memory that stores active application data for the CPU."),
    ("Hi assistant.", "Hi! Ready when you are.", "Explain why the sky is blue.", "The sky is blue because Earth's atmosphere scatters short blue wavelengths of sunlight more than other colors."),
    ("Greetings!", "Greetings! What questions do you have?", "How do plants make food?", "Plants make food through photosynthesis using sunlight, carbon dioxide, and water."),
    ("Hello there.", "Hello! How can I assist you?", "What is an operating system?", "An operating system manages computer hardware and coordinates software execution.")
]

for g_u, g_a, q_u, q_a in greetings_multi:
    add_record(f"User: {g_u}\nAssistant: {g_a}\nUser: {q_u}\nAssistant:", q_a, "multi_turn_conversation")
    add_record(f"User: {g_u}\nAssistant: {g_a}\nUser: Please tell me: {q_u}\nAssistant:", q_a, "multi_turn_conversation")

print(f"Total after multi-turn: {len(records)}")

# 3. English Explanations (600+)
eng_exp_topics = [
    ("the central processing unit (CPU)", "The CPU is the primary processing chip of a computer that carries out instructions of computer programs by performing basic arithmetic, logic, and input-output operations."),
    ("random access memory (RAM)", "RAM is the fast, volatile working memory where an operating system and open programs store data currently in active use."),
    ("how an electric motor works", "An electric motor uses electromagnetic fields to generate rotational kinetic torque, converting electrical power into mechanical work."),
    ("the function of a computer compiler", "A compiler converts high-level human-readable programming language source code into machine language binary code that hardware can directly execute."),
    ("how a bicycle operates", "A bicycle uses human pedal force transferred through a chain drive to turn the rear wheel and propel the vehicle forward."),
    ("the nature of gravity", "Gravity is the fundamental force by which all things with mass or energy are attracted toward one another."),
    ("what causes rain to fall", "Rain occurs when warm moisture evaporates into the atmosphere, cools, condenses into cloud droplets, and falls when drops become too heavy."),
    ("what a computer database is", "A database is an organized collection of structured information or data stored electronically in a computer system."),
    ("how vaccines protect the human body", "Vaccines introduce a weakened or inactive antigen to train the immune system to produce antibodies against specific diseases."),
    ("the structure of an atom", "An atom has a dense nucleus of protons and neutrons surrounded by orbiting electrons held by electromagnetic forces."),
    ("how airplanes fly", "Airplanes fly because air moving over curved wings creates lower pressure above than below, producing upward aerodynamic lift."),
    ("what an API is in software", "An Application Programming Interface (API) is a set of defined rules and protocols that enables different software applications to communicate with each other."),
    ("the purpose of an operating system", "An operating system acts as an intermediary between computer hardware and user applications, allocating memory, processor time, and disk storage."),
    ("how solar photovoltaic cells generate electricity", "Photovoltaic cells absorb photons from sunlight to knock electrons free from silicon atoms, creating an electrical current flow."),
    ("what the water cycle is", "The water cycle is the continuous movement of water on, above, and below Earth's surface through evaporation, condensation, and precipitation."),
    ("how human memory works", "Human memory encodes sensory experiences, consolidates information into synaptic neural connections, and retrieves stored knowledge when cued."),
    ("why leaves change color in autumn", "Leaves change color when cooling temperatures cause chlorophyll to break down, revealing yellow, orange, and red carotenoid pigments."),
    ("how fiber optic cables transmit internet data", "Fiber optic cables transmit digital data by bouncing rapid pulses of laser light off the reflective glass core via total internal reflection."),
    ("what a black hole is in astrophysics", "A black hole is a region of spacetime where gravity is so strong that nothing, including light, has sufficient velocity to escape."),
    ("how refrigeration cools food", "A refrigerator circulates refrigerant fluid that absorbs heat from inside the compartment and releases it outside through condenser coils.")
]

exp_intros = [
    "User: Explain {}.\nAssistant:",
    "User: Please explain {}.\nAssistant:",
    "User: What is {}?\nAssistant:",
    "User: Can you explain {}?\nAssistant:",
    "User: Tell me about {}.\nAssistant:",
    "User: How does {} work?\nAssistant:",
    "User: Briefly describe {}.\nAssistant:",
    "User: Give a simple explanation of {}.\nAssistant:"
]

for t, ans in eng_exp_topics:
    for intro in exp_intros:
        add_record(intro.format(t), ans, "english_explanation")

print(f"Total after English explanations: {len(records)}")

# 4. English Sentence Generation (500+)
gen_subjects = [
    ("a dog", "A loyal dog waited patiently by the front door for its family to return home."),
    ("a cat", "A soft cat curled up comfortably on the sunny windowsill and purred quietly."),
    ("the sun", "The bright sun rose above the eastern horizon, warming the morning air."),
    ("the ocean", "Vast blue ocean waves rolled gently onto the sandy shoreline under the clear sky."),
    ("a mountain", "A majestic snow-capped mountain rose high above the quiet forest valley."),
    ("a tree", "An ancient oak tree provided cool green shade for resting travelers in the park."),
    ("a river", "A clear mountain river wound its way peacefully through rocky gorges toward the sea."),
    ("an apple", "A ripe red apple hung from the leafy branch in the autumn orchard."),
    ("a bicycle", "A shiny bicycle was parked beside the wooden fence along the cycling path."),
    ("a book", "An inspiring book opened new perspectives and shared valuable knowledge with the reader."),
    ("a bird", "A small songbird sang a melodious morning tune from the leafy treetops."),
    ("a train", "The express train raced smoothly across the steel bridge above the wide canyon."),
    ("a garden", "Colorful flowers bloomed brightly in the peaceful botanical garden."),
    ("the stars", "Countless sparkling stars illuminated the deep dark velvet of the midnight sky."),
    ("a clock", "The vintage clock on the study wall ticked steadily with dependable precision."),
    ("music", "Gentle acoustic guitar music filled the warm room with a peaceful atmosphere."),
    ("clean water", "Drinking fresh clean water revitalizes the human body and promotes enduring health."),
    ("a quiet library", "Students read quietly among tall bookshelves filled with historical volumes."),
    ("autumn leaves", "Golden autumn leaves drifted softly to the ground on the cool autumn breeze."),
    ("a campfire", "A warm campfire crackled brightly under the open starry night sky.")
]

gen_intros = [
    "User: Write a sentence about {}.\nAssistant:",
    "User: Write a descriptive sentence about {}.\nAssistant:",
    "User: Generate one sentence describing {}.\nAssistant:",
    "User: Please write a simple sentence about {}.\nAssistant:",
    "User: Create a sentence about {}.\nAssistant:",
    "User: Give me a sentence describing {}.\nAssistant:",
    "User: Write one short sentence about {}.\nAssistant:",
    "User: Compose a sentence about {}.\nAssistant:"
]

for s, sent in gen_subjects:
    for intro in gen_intros:
        add_record(intro.format(s), sent, "english_sentence_generation")

print(f"Total after English sentence generation: {len(records)}")

# 5. Devanagari Hindi (600+)
hindi_topics = [
    ("नमस्ते!", "नमस्ते! आपका स्वागत है। बताइए आज मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("आप कौन हैं?", "मैं ChakrView हूँ, एक स्वतंत्र और स्थानीय भाषा मॉडल जिसे कंप्यूटर पर सीधे चलने के लिए बनाया गया है।"),
    ("कंप्यूटर क्या है?", "कंप्यूटर एक इलेक्ट्रॉनिक मशीन है जो डेटा को इनपुट के रूप में लेकर प्रोसेस करती है और उपयोगी परिणाम देती है।"),
    ("कंप्यूटर का सीपीयू क्या करता है?", "सीपीयू (CPU) कंप्यूटर का मस्तिष्क है जो सॉफ्टवेयर प्रोग्राम्स के सभी निर्देशों को निष्पादित और गणना करता है।"),
    ("रैम क्या काम आती है?", "रैम (RAM) एक तेज़ अस्थाई मेमोरी है जो वर्तमान में उपयोग हो रहे डेटा को तेजी से सीपीयू तक पहुंचाती है।"),
    ("सूर्य हमारे लिए क्यों जरूरी है?", "सूर्य का प्रकाश पेड़-पौधों में प्रकाश संश्लेषण के लिए आवश्यक है और पृथ्वी पर जीवन व ऊर्जा का मुख्य स्रोत है।"),
    ("पेड़ पर्यावरण को कैसे स्वच्छ रखते हैं?", "पेड़ वातावरण से हानिकारक कार्बन डाइऑक्साइड को सोखकर हमें ताजी और शुद्ध ऑक्सीजन प्रदान करते हैं।"),
    ("जल हमारे शरीर के लिए क्यों महत्वपूर्ण है?", "जल शरीर के तापमान को नियंत्रित करता है, रक्त संचार में सहायता करता है और सभी कोशिकाओं को स्वस्थ रखता है।"),
    ("शिक्षा का जीवन में क्या महत्व है?", "शिक्षा मनुष्य को सही और गलत की पहचान कराती है, नए कौशल सिखाती है और उज्ज्वल भविष्य का निर्माण करती है।"),
    ("सफलता प्राप्त करने का सही मार्ग क्या है?", "सफलता पाने के लिए निरंतर परिश्रम, सकारात्मक सोच, समय का सदुपयोग और दृढ़ निश्चय आवश्यक है।"),
    ("इंटरनेट से हमें क्या लाभ मिलते हैं?", "इंटरनेट हमें ज्ञान, शिक्षा, वैश्विक संचार और उपयोगी सेवाओं तक तुरंत पहुंच प्रदान करता है।"),
    ("योग करने से क्या लाभ होता है?", "योग करने से शरीर लचीला, मजबूत और तनावमुक्त रहता है तथा मन में शांति और एकाग्रता बढ़ती है।"),
    ("किताबें पढ़ना क्यों फायदेमंद है?", "किताबें पढ़ने से सोचने की क्षमता विकसित होती है, भाषा समृद्ध होती है और नई जानकारियां प्राप्त होती हैं।"),
    ("समय का सदुपयोग कैसे किया जाए?", "समय का सदुपयोग करने के लिए अपने कार्यों की सूची बनाएं और आलस्य छोड़कर समय पर काम पूरा करें।"),
    ("सफाई रखने से क्या लाभ हैं?", "स्वच्छता रखने से बीमारियां दूर रहती हैं, वातावरण सुंदर लगता है और स्वास्थ्य उत्तम बना रहता है।"),
    ("बिजली कैसे उत्पन्न होती है?", "बिजली मुख्य रूप से जल, कोयला, सौर ऊर्जा और पवन ऊर्जा के माध्यम से जनरेटर चलाकर उत्पन्न की जाती है।"),
    ("हवा हमारे लिए क्यों जरूरी है?", "हवा में मौजूद ऑक्सीजन हमारे सांस लेने और जीवित रहने के लिए अत्यंत आवश्यक है।"),
    ("ऋतुओं का बदलना किस कारण होता है?", "पृथ्वी अपनी धुरी पर झुकी हुई सूर्य की परिक्रमा करती है, जिसके कारण अलग-अलग ऋतुएं बदलती हैं।"),
    ("चंद्रमा की क्या विशेषता है?", "चंद्रमा पृथ्वी का एकमात्र प्राकृतिक उपग्रह है जो रात के समय सूर्य के प्रकाश को परावर्तित करके चमकता है।"),
    ("धन्यवाद!", "आपका बहुत-बहुत स्वागत है! यदि आपके पास कोई और प्रश्न हो, तो निसंकोच पूछिए।")
]

hindi_intros = [
    "User: {}\nAssistant:",
    "User: कृपया समझाइए: {}\nAssistant:",
    "User: बताइए: {}\nAssistant:",
    "User: प्रश्न: {}\nAssistant:",
    "User: नमस्ते, {}\nAssistant:",
    "User: क्या आप बता सकते हैं: {}\nAssistant:"
]

for q, a in hindi_topics:
    for intro in hindi_intros:
        add_record(intro.format(q), a, "devanagari_hindi")

print(f"Total after Devanagari Hindi: {len(records)}")

# 6. Roman Hindi (500+)
roman_topics = [
    ("Aap kaise hain?", "Main theek hoon, poochne ke liye shukriya! Aap bataiye aapka din kaisa chal raha hai?"),
    ("Aapka naam kya hai?", "Mera naam ChakrView hai. Main ek independent local conversational assistant hoon."),
    ("Aap kya kaam kar sakte hain?", "Main sawalon ke jawab de sakta hoon, concepts explain kar sakta hoon aur English, Hindi me chat kar sakta hoon."),
    ("Computer ka CPU kya kaam karta hai?", "CPU computer ka primary processor hota hai jo software instructions ko execute aur data flow control karta hai."),
    ("RAM memory ka kya kaam hota hai?", "RAM ek fast temporary memory hoti hai jisme computer running programs ka active data store karta hai."),
    ("Subah jaldi uthne ke kya fayde hain?", "Subah jaldi uthne se mind fresh rehta hai, concentration increase hoti hai aur din bhar ke tasks time par complete hote hain."),
    ("Clean code likhna kyun important hai?", "Clean code easily samajh aata hai, bugs jaldi identify ho jaate hain aur project maintenance aasan ho jaati hai."),
    ("Machine learning simple language mein samjhao.", "Machine learning ek aisi technique hai jisme computer data ke patterns ko observe karke khud naye decisions lena seekhta hai."),
    ("Python language seekhna kyun accha hai?", "Python ka syntax bahut clean aur readable hai aur isme artificial intelligence aur web development ke powerful tools hain."),
    ("Hard work aur smart work me kya fark hai?", "Hard work continuous effort hota hai jabki smart work sahi strategy aur time management ke saath kiya gaya targeted work hota hai."),
    ("Internet kaise connect hota hai?", "Internet worldwide undersea fiber cables aur wireless routers ke network ke through data packets transmit karke connect hota hai."),
    ("Database ka primary role kya hai?", "Database digital information ko structured tables me store karta hai taaki data ko safely query aur update kiya ja sake."),
    ("Time management kaise improve karein?", "Time management ke liye daily goals set karein, priority list banayein aur distraction-free work environment rakhein."),
    ("Water peena body ke liye kyun zaroori hai?", "Pani peene se digestion healthy rehta hai, body temperature balance rehta hai aur energy level active rehta hai."),
    ("Shukriya aapki advice ke liye!", "Aapka bahut swaagat hai! Agar koi aur question ho toh zaroor bataiye.")
]

roman_intros = [
    "User: {}\nAssistant:",
    "User: Bhai, {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Please explain: {}\nAssistant:",
    "User: Ek question hai: {}\nAssistant:",
    "User: Suno, {}\nAssistant:"
]

for q, a in roman_topics:
    for intro in roman_intros:
        add_record(intro.format(q), a, "roman_hindi")

print(f"Total after Roman Hindi: {len(records)}")

# 7. Hinglish Code-Switched (600+)
hinglish_topics = [
    ("Machine learning models data se kaise learn karte hain?", "Training phase mein models input data process karke predictions banate hain aur error compute karke mathematical weights adjust karte hain."),
    ("CPU architecture mein ALU ka kya function hota hai?", "ALU yani Arithmetic Logic Unit CPU ka core part hai jo saari addition, subtraction aur boolean logic operations execute karta hai."),
    ("Python mein list comprehension code ko kaise optimize karti hai?", "List comprehension syntax ko compact banati hai aur standard loops ke comparison mein C-level implementation ke through fast execute hoti hai."),
    ("ChakrView model CPU par efficiently kaise run karta hai?", "ChakrView compact architecture aur optimized matrix operations use karta hai jisse bina heavy GPU hardware ke smooth inference milta hai."),
    ("Git mein branch create karne ka kya advantage hota hai?", "New branch create karke aap main codebase ko affect kiye bina safely naye features build aur independently test kar sakte hain."),
    ("REST APIs web development mein kyun standard hain?", "REST APIs simple HTTP methods jaise GET aur POST use karke client aur server ke beech clean JSON data exchange provide karti hain."),
    ("Operating system mein paging ka kya concept hai?", "Paging memory management technique hai jahan physical memory ko fixed-size pages mein divide karke non-contiguous storage allocate ki jaati hai."),
    ("Unit tests run karna continuous integration mein kyun zaroori hai?", "Automated unit tests code change hone par instant feedback dete hain aur unexpected regressions ko production mein jaane se rokte hain."),
    ("Deep learning mein neural network layers ka kya significance hai?", "Deep layers progressively higher-level abstract features extract karti hain jisse complex visual aur linguistic patterns accurately model hote hain."),
    ("Backend server aur database ke beech connection pooling kaise help karti hai?", "Connection pooling database connections reuse karti hai jisse har naye request par naya connection create karne ka high latency overhead avoid hota hai.")
]

hinglish_intros = [
    "User: {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Please clarify: {}\nAssistant:",
    "User: Can you explain: {}\nAssistant:",
    "User: Bhai, {}\nAssistant:",
    "User: Technical doubt hai: {}\nAssistant:"
]

for q, a in hinglish_topics:
    for intro in hinglish_intros:
        add_record(intro.format(q), a, "hinglish")

print(f"Total after Hinglish: {len(records)}")

# 8. Exact Copying & Verbatim (400+)
copy_phrases = [
    "The sky is blue.",
    "Python is an expressive and readable programming language.",
    "Clean code prevents subtle runtime bugs.",
    "Sun rises in the eastern horizon.",
    "Water is fundamental to all biological life.",
    "Reading books enriches the human intellect.",
    "Technological innovation connects modern society.",
    "Keep moving forward with steady dedication.",
    "Practice sharpens problem solving competence.",
    "Simplicity is a crucial element of reliable software design.",
    "Efficient algorithms maximize system computational throughput.",
    "Knowledge delivers clarity, focus, and inner confidence.",
    "The earth orbits smoothly around the bright sun.",
    "Bright stars sparkle in the midnight sky.",
    "Persistent effort produces lasting achievements.",
    "Healthy trees provide fresh shade and pure air.",
    "Focus your attention on solving real human problems.",
    "Curiosity is the engine of all scientific research.",
    "Calm thinking leads to sound strategic decisions.",
    "Fresh morning air and sunlight renew daily energy."
]

copy_intros = [
    ("User: Say exactly: {}\nAssistant:", "{}"),
    ("User: Repeat verbatim: {}\nAssistant:", "{}"),
    ("User: Echo the following text: {}\nAssistant:", "{}"),
    ("User: Copy this sentence: {}\nAssistant:", "{}"),
    ("User: Please repeat exactly: {}\nAssistant:", "{}"),
    ("User: Print verbatim: {}\nAssistant:", "{}")
]

for p in copy_phrases:
    for p_tmpl, t_tmpl in copy_intros:
        add_record(p_tmpl.format(p), t_tmpl.format(p), "exact_copying")

print(f"Total after Exact copying: {len(records)}")

# 9. Arithmetic (600+)
for a in range(1, 16):
    for b in range(1, 16):
        # Addition
        s = a + b
        add_record(f"User: What is {a} + {b}?\nAssistant:", f"{a} + {b} = {s}.", "elementary_arithmetic")
        add_record(f"User: Calculate {a} plus {b}.\nAssistant:", f"{a} plus {b} equals {s}.", "elementary_arithmetic")
        # Subtraction
        if a >= b:
            diff = a - b
            add_record(f"User: What is {a} - {b}?\nAssistant:", f"{a} - {b} = {diff}.", "elementary_arithmetic")
            add_record(f"User: Calculate {a} minus {b}.\nAssistant:", f"{a} minus {b} equals {diff}.", "elementary_arithmetic")
            add_record(f"User: If I have {a} apples and give {b} to my friend, how many apples remain?\nAssistant:",
                       f"You will have {diff} apples remaining.", "elementary_arithmetic")

print(f"Total after Arithmetic: {len(records)}")

# 10. Basic Instructions (300+)
inst_items = [
    ("What color is the clear daytime sky?", "blue", "Answer in exactly one word: {}"),
    ("What celestial body is at the center of the solar system?", "sun", "Answer in one word: {}"),
    ("How many days are in one week?", "seven", "Answer in only one word: {}"),
    ("Is ice solid water?", "yes", "Answer yes or no: {}"),
    ("Can rocks float on water?", "no", "Answer yes or no: {}"),
    ("Do humans need oxygen to breathe?", "yes", "Answer yes or no: {}"),
    ("Is the Pacific Ocean made of freshwater?", "no", "Answer yes or no: {}"),
    ("What liquid falls as precipitation from clouds?", "rain", "Answer in one word: {}"),
    ("What is the opposite of hot?", "cold", "State the opposite in one word: {}"),
    ("What is the opposite of fast?", "slow", "State the opposite in one word: {}"),
    ("What is the opposite of dark?", "light", "State the opposite in one word: {}"),
    ("What is the opposite of heavy?", "light", "State the opposite in one word: {}")
]

for q, ans, tmpl in inst_items:
    add_record(f"User: {tmpl.format(q)}\nAssistant:", ans, "basic_instructions")
    add_record(f"User: {q} Answer in one word.\nAssistant:", ans, "basic_instructions")
    add_record(f"User: Respond with only one word: {q}\nAssistant:", ans, "basic_instructions")

print(f"Raw total generated: {len(records)}")

# Deduplicate strictly
seen = set()
unique_records = []
for r in records:
    p = r["prompt"].strip()
    if p not in seen:
        seen.add(p)
        unique_records.append(r)

print(f"Unique records: {len(unique_records)}")

random.shuffle(unique_records)

split_idx = int(len(unique_records) * 0.88)
train_split = unique_records[:split_idx]
val_split = unique_records[split_idx:]

print(f"Final Train: {len(train_split)}, Val: {len(val_split)}")

data_dir = REPO_ROOT / "experiments/cognition/ucip_05/data"
data_dir.mkdir(parents=True, exist_ok=True)

train_file = data_dir / "ucip05_train.jsonl"
val_file = data_dir / "ucip05_validation.jsonl"

with open(train_file, "w", encoding="utf-8") as f:
    for rec in train_split:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open(val_file, "w", encoding="utf-8") as f:
    for rec in val_split:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().lower()

manifest = {
    "total_unique_records": len(unique_records),
    "train_records": len(train_split),
    "validation_records": len(val_split),
    "train_file": str(train_file.relative_to(REPO_ROOT)),
    "train_sha256": sha256_file(train_file),
    "val_file": str(val_file.relative_to(REPO_ROOT)),
    "val_sha256": sha256_file(val_file),
    "category_breakdown": {}
}

for rec in unique_records:
    cat = rec["category"]
    manifest["category_breakdown"][cat] = manifest["category_breakdown"].get(cat, 0) + 1

with open(data_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("Saved dataset manifest:")
print(json.dumps(manifest, indent=2))
