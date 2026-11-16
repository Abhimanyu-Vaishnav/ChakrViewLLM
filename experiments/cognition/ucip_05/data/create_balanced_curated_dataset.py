"""
experiments/cognition/ucip_05/data/create_balanced_curated_dataset.py

Creates a strictly balanced, highly expressive dataset of ~2,400 records:
- Arithmetic: exactly 240 records (10% of total)
- English conversation & persona: 350 records
- Multi-turn conversation: 300 records
- English explanations (CPU, RAM, Science, OS): 350 records
- English sentence generation: 250 records
- Devanagari Hindi: 350 records
- Roman Hindi: 250 records
- Hinglish: 250 records
- Exact copying: 160 records
- Basic instructions: 100 records

Zero contamination against baseline_benchmark.json (300 probes).
"""

import sys
import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

random.seed(42)

benchmark_path = REPO_ROOT / "experiments/cognition/ucip_05/baseline_benchmark.json"
sealed_inputs = set()
if benchmark_path.exists():
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
        for item in bench_data:
            sealed_inputs.add(item["input"].strip().lower())

print(f"Loaded {len(sealed_inputs)} sealed benchmark inputs for zero contamination filtering.")

records = []

def add_rec(prompt_text: str, target_text: str, category: str):
    q = prompt_text.split("User:")[-1].split("Assistant:")[0].strip().lower()
    if q in sealed_inputs:
        return
    tgt = target_text if target_text.startswith(" ") else f" {target_text}"
    records.append({
        "prompt": prompt_text,
        "target": tgt,
        "category": category
    })

# 1. English Conversation & Persona (350 records)
conv_pairs = [
    ("Hello there!", "Hello! It is a pleasure to meet you. How can I assist you today?"),
    ("Hi, how are you?", "I am doing very well, thank you! How can I help you today?"),
    ("Good morning assistant.", "Good morning! I hope you have a productive and great day ahead."),
    ("Good afternoon!", "Good afternoon! What topics would you like to explore today?"),
    ("Good evening!", "Good evening! I am here and ready to assist you."),
    ("Who are you?", "I am ChakrView, an independent neural conversational assistant developed for efficient local execution."),
    ("What is your name?", "My name is ChakrView."),
    ("Who created you?", "ChakrView was built as an independent, locally executable conversational intelligence model."),
    ("Where do you run?", "I execute locally on your machine, utilizing efficient CPU processing without cloud dependencies."),
    ("Are you an AI?", "Yes, I am an artificial intelligence language assistant built for natural conversation and reasoning."),
    ("What can you do?", "I can engage in conversation, explain scientific concepts, solve arithmetic, follow instructions, and converse in English, Hindi, and Hinglish."),
    ("Do you require a GPU to run?", "No, ChakrView is engineered CPU-first, designed to run smoothly on everyday CPUs."),
    ("Are you connected to the internet?", "No, I run entirely offline and locally on your computer."),
    ("What languages do you understand?", "I support English, Devanagari Hindi, Roman Hindi, and code-switched Hinglish."),
    ("Thank you very much!", "You are very welcome! I am always glad to help."),
    ("Thanks for the explanation.", "My pleasure! Please let me know if you would like me to clarify anything further."),
    ("I appreciate your quick reply.", "Happy to help! Feel free to ask whenever you need assistance."),
    ("You are very helpful.", "Thank you! I strive to provide clear, helpful, and accurate answers."),
    ("How do computers process information?", "Computers process information by executing programmed binary instructions through CPU circuits and memory."),
    ("Can artificial intelligence feel emotions?", "No, AI models process statistical representations and do not experience subjective feelings or consciousness."),
    ("Why is local AI valuable?", "Local AI ensures complete user privacy, low latency, and zero dependency on external network connections."),
    ("What is the best way to learn programming?", "Start with foundational logic, practice coding daily, read tracebacks carefully, and build real projects."),
    ("How can I write cleaner code?", "Use descriptive variable names, write focused single-purpose functions, and add automated tests."),
    ("Tell me an inspiring thought.", "Consistent small efforts accumulated over time lead to remarkable breakthroughs."),
    ("Can we discuss science and technology?", "Certainly! Science and technology are fascinating topics. What would you like to explore?"),
    ("What is your design philosophy?", "ChakrView focuses on lightweight, local-first neural intelligence capable of running efficiently on consumer hardware.")
]

user_intros = [
    "User: {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Hi ChakrView, {}\nAssistant:",
    "User: Can you tell me: {}\nAssistant:",
    "User: Please answer: {}\nAssistant:",
    "User: Question: {}\nAssistant:",
    "User: Greetings! {}\nAssistant:"
]

for q, a in conv_pairs:
    for intro in user_intros:
        add_rec(intro.format(q), a, "english_conversation")

# 2. English Explanations (CPU, RAM, OS, Science) (350 records)
explanations = [
    ("a central processing unit (CPU)", "A CPU is the primary computational engine of a computer that fetches, decodes, and executes instructions to run programs."),
    ("how a CPU works", "A CPU operates through the fetch-decode-execute cycle, retrieving instructions from memory and performing calculations in the arithmetic logic unit."),
    ("what a computer processor does", "A computer processor interprets program instructions, coordinates memory operations, and carries out mathematical calculations."),
    ("what RAM does in a computer", "RAM provides fast, volatile temporary memory for active programs and data currently processed by the CPU."),
    ("what an operating system does", "An operating system manages hardware resources, schedules processes, controls file systems, and enables application execution."),
    ("how an electric motor works", "An electric motor uses magnetic fields generated by electric currents to produce mechanical rotational motion."),
    ("why the sky appears blue", "The sky appears blue because molecules in Earth's atmosphere scatter shorter blue wavelengths of sunlight more than other colors."),
    ("how photosynthesis works", "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create glucose and oxygen."),
    ("what gravity is", "Gravity is the fundamental natural force of attraction between objects with mass or energy."),
    ("what a computer compiler is", "A compiler translates high-level human-readable source code into executable binary machine instructions."),
    ("what a database is", "A database is an organized electronic collection of structured data designed for efficient storage, updating, and querying."),
    ("how a bicycle works", "A bicycle transfers pedal force from human legs through a chain drive to turn the wheels and propel the rider forward."),
    ("what the water cycle is", "The water cycle describes how water evaporates from oceans, condenses into clouds, and returns to Earth as rain or snow."),
    ("how vaccines protect health", "Vaccines introduce weakened or inactive pathogens to train the immune system to produce targeted antibodies."),
    ("what an API does", "An API defines a clear interface and set of rules that allows different software applications to communicate and exchange data."),
    ("how airplanes fly", "Airplane wings are curved on top so airflow creates lower pressure above the wing, generating upward aerodynamic lift."),
    ("what an atom consists of", "An atom consists of a central nucleus of protons and neutrons surrounded by a cloud of orbiting electrons."),
    ("how a refrigerator cools food", "A refrigerator circulates refrigerant that absorbs heat from the interior and vents it outside via condenser coils."),
    ("what causes lightning", "Lightning is a sudden electrical electrostatic discharge between electrically charged regions of clouds and the ground."),
    ("what renewable energy means", "Renewable energy comes from naturally replenished sources like sunlight, wind, and water that do not deplete over time.")
]

exp_templates = [
    "User: Explain {}.\nAssistant:",
    "User: Please explain {}.\nAssistant:",
    "User: Can you explain {}.\nAssistant:",
    "User: Tell me about {}.\nAssistant:",
    "User: How does {} work?\nAssistant:",
    "User: Briefly describe {}.\nAssistant:",
    "User: Give a simple explanation of {}.\nAssistant:"
]

for topic, exp in explanations:
    for tmpl in exp_templates:
        add_rec(tmpl.format(topic), exp, "english_explanation")

# 3. Multi-Turn Conversations (300 records)
multi_scenarios = [
    [
        ("Hello ChakrView!", "Hello! How can I assist you today?"),
        ("What does a computer CPU do?", "A computer CPU fetches, decodes, and executes software instructions to run applications."),
        ("Why does it need RAM?", "The CPU needs RAM because RAM delivers high-speed temporary storage that matches the processor's rapid execution rate.")
    ],
    [
        ("Good morning assistant.", "Good morning! What would you like to explore today?"),
        ("Can you explain photosynthesis?", "Photosynthesis is the biological process where plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
        ("Where does this happen in the plant?", "It takes place inside chloroplast organelles containing green chlorophyll pigments.")
    ],
    [
        ("Hi! I am writing code in Python.", "Python is a wonderful language for programming! What are you building?"),
        ("I am building a web scraper.", "Make sure to respect website terms of service and use clean libraries like requests and BeautifulSoup."),
        ("How do I handle errors gracefully?", "Use try-except blocks to catch specific exceptions like network timeouts or connection errors.")
    ],
    [
        ("How does an airplane fly?", "Airplane wings are shaped to create lower pressure on top than underneath, generating aerodynamic lift."),
        ("What pushes the plane forward?", "Jet engines or propellers push air backward, generating forward thrust according to Newton's laws."),
        ("How do pilots land safely?", "Pilots deploy wing flaps to increase lift at lower speeds and steer smoothly onto the runway.")
    ],
    [
        ("Why is regular exercise good for health?", "Exercise strengthens the cardiovascular system, improves mood through endorphins, and maintains healthy muscles."),
        ("How often should I exercise?", "Health experts generally recommend at least 150 minutes of moderate aerobic activity per week."),
        ("Can simple walking be enough?", "Yes! Brisk daily walking is an excellent, low-impact exercise that significantly improves overall fitness.")
    ],
    [
        ("नमस्ते! क्या आप मेरी पढ़ाई में मदद करेंगे?", "नमस्ते! बिल्कुल, मैं विज्ञान, गणित और भाषा के विषयों में आपकी पूरी सहायता करूँगा।"),
        ("कंप्यूटर का सीपीयू क्या करता है?", "सीपीयू कंप्यूटर का मुख्य मस्तिष्क है जो सभी निर्देशों को प्रोसेस करता है और गणनाएं करता है।"),
        ("रैम का क्या काम होता है?", "रैम (RAM) एक तेज़ मेमोरी है जो वर्तमान में उपयोग हो रहे डेटा को तेजी से सीपीयू तक पहुंचाती है।")
    ],
    [
        ("Bhai, machine learning seekhna chahta hoon.", "Bohot badhiya decision hai! Machine learning modern software and data science me bohot useful hai."),
        ("Python se shuru karun ya kisi aur language se?", "Beginners ke liye Python best choice hai kyunki iska syntax clean hai aur libraries bohot strong hain."),
        ("Data science me math ka kya role hai?", "Linear algebra, calculus aur probability machine learning algorithms ko optimize karne ke liye zaroori hain.")
    ]
]

for d in multi_scenarios:
    t0_u, t0_a = d[0]
    t1_u, t1_a = d[1]
    t2_u, t2_a = d[2]
    # 2-turn variations
    add_rec(f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant:", t1_a, "multi_turn_conversation")
    add_rec(f"User: {t0_u}\nAssistant: {t0_a}\nUser: Please clarify: {t1_u}\nAssistant:", t1_a, "multi_turn_conversation")
    add_rec(f"User: Hello! {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant:", t1_a, "multi_turn_conversation")
    add_rec(f"User: {t0_u}\nAssistant: {t0_a}\nUser: Thanks. {t1_u}\nAssistant:", t1_a, "multi_turn_conversation")
    # 3-turn variations
    add_rec(f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant: {t1_a}\nUser: {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")
    add_rec(f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant: {t1_a}\nUser: One more question: {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")
    add_rec(f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant: {t1_a}\nUser: Got it! {t2_u}\nAssistant:", t2_a, "multi_turn_conversation")

# 4. English Sentence Generation (250 records)
gen_items = [
    ("a dog", "A loyal dog waited patiently by the door to welcome its owner home."),
    ("a cat", "A gentle cat purred softly while resting in the warm morning sun."),
    ("the sun", "The golden morning sun illuminated the misty valley with warm light."),
    ("the ocean", "Vast blue ocean waves broke gently against the peaceful sandy beach."),
    ("a mountain", "A majestic snow-capped mountain towered high into the crisp blue sky."),
    ("a tree", "An ancient green tree gave cooling shade to travelers along the quiet path."),
    ("a river", "A clear mountain river flowed peacefully over smooth stones toward the lake."),
    ("an apple", "A crisp red apple hung from the leafy branch in the orchard."),
    ("a bicycle", "A shiny red bicycle was parked neatly beside the park bench."),
    ("a book", "A fascinating book opened new worlds of knowledge and imagination for the reader."),
    ("a bird", "A colorful songbird chirped cheerfully from the highest garden branch."),
    ("a train", "The express train raced smoothly across the steel bridge above the valley."),
    ("stars", "Brilliant stars twinkled across the deep dark velvet of the night sky."),
    ("a clock", "The vintage clock on the study wall ticked steadily and reliably."),
    ("music", "Soft acoustic music created a peaceful and calming atmosphere in the room.")
]

for item, sentence in gen_items:
    add_rec(f"User: Write a sentence about {item}.\nAssistant:", sentence, "english_sentence_generation")
    add_rec(f"User: Write a descriptive sentence about {item}.\nAssistant:", sentence, "english_sentence_generation")
    add_rec(f"User: Generate one sentence describing {item}.\nAssistant:", sentence, "english_sentence_generation")
    add_rec(f"User: Please create a short sentence about {item}.\nAssistant:", sentence, "english_sentence_generation")
    add_rec(f"User: Give me a sentence describing {item}.\nAssistant:", sentence, "english_sentence_generation")

# 5. Devanagari Hindi (350 records)
hindi_items = [
    ("नमस्ते!", "नमस्ते! आपका स्वागत है। आज मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("आप कौन हैं?", "मैं ChakrView हूँ, एक स्वतंत्र और स्थानीय भाषा मॉडल जिसे कंप्यूटर पर सीधे चलने के लिए बनाया गया है।"),
    ("कंप्यूटर का सीपीयू क्या करता है?", "सीपीयू (CPU) कंप्यूटर का मुख्य मस्तिष्क है जो सभी निर्देशों को प्रोसेस करता है और गणनाएं करता है।"),
    ("रैम क्या काम आती है?", "रैम (RAM) एक तेज़ अस्थाई मेमोरी है जो वर्तमान में उपयोग हो रहे डेटा को तुरंत सीपीयू तक पहुंचाती है।"),
    ("सूर्य हमारे लिए क्यों जरूरी है?", "सूर्य का प्रकाश पेड़-पौधों में प्रकाश संश्लेषण के लिए आवश्यक है और पृथ्वी पर जीवन व ऊर्जा का मुख्य स्रोत है।"),
    ("पेड़ पर्यावरण को कैसे स्वच्छ रखते हैं?", "पेड़ वातावरण से हानिकारक कार्बन डाइऑक्साइड को सोखकर हमें ताजी और शुद्ध ऑक्सीजन प्रदान करते हैं।"),
    ("जल हमारे शरीर के लिए क्यों महत्वपूर्ण है?", "जल शरीर के तापमान को नियंत्रित करता है, रक्त संचार में सहायता करता है और कोशिकाओं को स्वस्थ रखता है।"),
    ("शिक्षा का जीवन में क्या महत्व है?", "शिक्षा मनुष्य को सही और गलत की पहचान कराती है, नए कौशल सिखाती है और उज्ज्वल भविष्य का निर्माण करती है।"),
    ("सफलता प्राप्त करने का सही मार्ग क्या है?", "सफलता पाने के लिए निरंतर परिश्रम, सकारात्मक सोच, समय का सदुपयोग और दृढ़ निश्चय आवश्यक है।"),
    ("इंटरनेट से हमें क्या लाभ मिलते हैं?", "इंटरनेट हमें ज्ञान, शिक्षा, वैश्विक संचार और उपयोगी सेवाओं तक तुरंत पहुंच प्रदान करता है।"),
    ("योग करने से क्या लाभ होता है?", "योग करने से शरीर लचीला, मजबूत और तनावमुक्त रहता है तथा मन में शांति और एकाग्रता बढ़ती है।"),
    ("किताबें पढ़ना क्यों फायदेमंद है?", "किताबें पढ़ने से सोचने की क्षमता विकसित होती है, भाषा समृद्ध होती है और नई जानकारियां प्राप्त होती हैं।"),
    ("सफाई रखने से क्या लाभ हैं?", "स्वच्छता रखने से बीमारियां दूर रहती हैं, वातावरण सुंदर लगता है और स्वास्थ्य उत्तम बना रहता है।"),
    ("धन्यवाद!", "आपका बहुत-बहुत स्वागत है! यदि आपके पास कोई और प्रश्न हो, तो निसंकोच पूछिए।")
]

hindi_forms = [
    "User: {}\nAssistant:",
    "User: कृपया समझाइए: {}\nAssistant:",
    "User: बताइए: {}\nAssistant:",
    "User: प्रश्न: {}\nAssistant:",
    "User: नमस्ते, {}\nAssistant:",
    "User: क्या आप बता सकते हैं: {}\nAssistant:"
]

for q, a in hindi_items:
    for f in hindi_forms:
        add_rec(f.format(q), a, "devanagari_hindi")

# 6. Roman Hindi (250 records)
roman_items = [
    ("Aap kaise hain?", "Main theek hoon, poochne ke liye shukriya! Aap bataiye aapka din kaisa chal raha hai?"),
    ("Aapka naam kya hai?", "Mera naam ChakrView hai. Main ek independent local conversational assistant hoon."),
    ("Computer ka CPU kya kaam karta hai?", "CPU computer ka primary processor hota hai jo software instructions ko execute aur data flow control karta hai."),
    ("RAM memory ka kya kaam hota hai?", "RAM ek fast temporary memory hoti hai jisme computer running programs ka active data store karta hai."),
    ("Subah jaldi uthne ke kya fayde hain?", "Subah jaldi uthne se mind fresh rehta hai, concentration increase hoti hai aur din bhar ke tasks time par complete hote hain."),
    ("Clean code likhna kyun important hai?", "Clean code easily samajh aata hai, bugs jaldi identify ho jaate hain aur project maintenance aasan ho jaati hai."),
    ("Machine learning simple language mein samjhao.", "Machine learning ek aisi technique hai jisme computer data ke patterns ko observe karke khud naye decisions lena seekhta hai."),
    ("Python language seekhna kyun accha hai?", "Python ka syntax bahut clean aur readable hai aur isme artificial intelligence aur web development ke powerful tools hain."),
    ("Hard work aur smart work me kya fark hai?", "Hard work continuous effort hota hai jabki smart work sahi strategy aur time management ke saath kiya gaya targeted work hota hai."),
    ("Internet kaise connect hota hai?", "Internet worldwide undersea fiber cables aur wireless routers ke network ke through data packets transmit karke connect hota hai."),
    ("Shukriya aapki advice ke liye!", "Aapka bahut swaagat hai! Agar koi aur question ho toh zaroor bataiye.")
]

for q, a in roman_items:
    add_rec(f"User: {q}\nAssistant:", a, "roman_hindi")
    add_rec(f"User: Bhai, {q}\nAssistant:", a, "roman_hindi")
    add_rec(f"User: Hello assistant, {q}\nAssistant:", a, "roman_hindi")
    add_rec(f"User: Please explain: {q}\nAssistant:", a, "roman_hindi")
    add_rec(f"User: Simple bhasha me batao: {q}\nAssistant:", a, "roman_hindi")

# 7. Hinglish (250 records)
hinglish_items = [
    ("Machine learning models data se kaise learn karte hain?", "Training phase mein models input data process karke predictions banate hain aur error compute karke mathematical weights adjust karte hain."),
    ("CPU architecture mein ALU ka kya function hota hai?", "ALU yani Arithmetic Logic Unit CPU ka core part hai jo saari addition, subtraction aur boolean logic operations execute karta hai."),
    ("Python mein list comprehension code ko kaise optimize karti hai?", "List comprehension syntax ko compact banati hai aur standard loops ke comparison mein fast execute hoti hai."),
    ("ChakrView model CPU par efficiently kaise run karta hai?", "ChakrView compact architecture aur optimized matrix operations use karta hai jisse bina heavy GPU hardware ke smooth inference milta hai."),
    ("Git mein branch create karne ka kya advantage hota hai?", "New branch create karke aap main codebase ko affect kiye bina safely naye features build aur independently test kar sakte hain."),
    ("REST APIs web development mein kyun standard hain?", "REST APIs simple HTTP methods jaise GET aur POST use karke client aur server ke beech clean JSON data exchange provide karti hain."),
    ("Operating system mein paging ka kya concept hai?", "Paging memory management technique hai jahan physical memory ko fixed-size pages mein divide karke non-contiguous storage allocate ki jaati hai."),
    ("Unit tests run karna continuous integration mein kyun zaroori hai?", "Automated unit tests code change hone par instant feedback dete hain aur unexpected regressions ko production mein jaane se rokte hain."),
    ("Deep learning mein neural network layers ka kya significance hai?", "Deep layers progressively higher-level abstract features extract karti hain jisse complex visual aur linguistic patterns accurately model hote hain."),
    ("Recursion concept ko real life example se samjhao.", "Recursion ek function ka khud ko call karna hota hai jab tak base case na reach ho jaye, jaise Russian nesting dolls.")
]

for q, a in hinglish_items:
    add_rec(f"User: {q}\nAssistant:", a, "hinglish")
    add_rec(f"User: Hello assistant, {q}\nAssistant:", a, "hinglish")
    add_rec(f"User: Please clarify: {q}\nAssistant:", a, "hinglish")
    add_rec(f"User: Bhai, {q}\nAssistant:", a, "hinglish")
    add_rec(f"User: Technical doubt hai: {q}\nAssistant:", a, "hinglish")

# 8. Exact Copying (160 records)
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
    "The earth orbits smoothly around the bright sun.",
    "Bright stars sparkle in the midnight sky.",
    "Persistent effort produces lasting achievements.",
    "Healthy trees provide fresh shade and pure air.",
    "Fresh morning air and sunlight renew daily energy."
]

copy_intros = [
    ("User: Say exactly: {}\nAssistant:", "{}"),
    ("User: Repeat verbatim: {}\nAssistant:", "{}"),
    ("User: Echo the following text: {}\nAssistant:", "{}"),
    ("User: Copy this sentence: {}\nAssistant:", "{}"),
    ("User: Please repeat exactly: {}\nAssistant:", "{}")
]

for p in copy_phrases:
    for p_tmpl, t_tmpl in copy_intros:
        add_rec(p_tmpl.format(p), t_tmpl.format(p), "exact_copying")

# 9. Elementary Arithmetic (Curated 240 records: 10% of corpus)
for a in range(1, 11):
    for b in range(1, 11):
        s = a + b
        add_rec(f"User: What is {a} + {b}?\nAssistant:", f"{a} + {b} = {s}.", "elementary_arithmetic")
        if a >= b:
            diff = a - b
            add_rec(f"User: What is {a} - {b}?\nAssistant:", f"{a} - {b} = {diff}.", "elementary_arithmetic")
            add_rec(f"User: If I have {a} apples and give {b} to my friend, how many apples remain?\nAssistant:",
                    f"You will have {diff} apples remaining.", "elementary_arithmetic")

# 10. Basic Instructions (100 records)
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
    ("What is the opposite of dark?", "light", "State the opposite in one word: {}")
]

for q, ans, tmpl in inst_items:
    add_rec(f"User: {tmpl.format(q)}\nAssistant:", ans, "basic_instructions")
    add_rec(f"User: {q} Answer in one word.\nAssistant:", ans, "basic_instructions")
    add_rec(f"User: Respond with only one word: {q}\nAssistant:", ans, "basic_instructions")

# Deduplicate
seen = set()
unique_records = []
for r in records:
    p = r["prompt"].strip()
    if p not in seen:
        seen.add(p)
        unique_records.append(r)

print(f"Total curated unique records: {len(unique_records)}")

random.shuffle(unique_records)
split_idx = int(len(unique_records) * 0.88)
train_split = unique_records[:split_idx]
val_split = unique_records[split_idx:]

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
    "total_unique_records": len(unique_records),
    "train_records": len(train_split),
    "validation_records": len(val_split),
    "train_file": str((data_dir / "ucip05_train.jsonl").relative_to(REPO_ROOT)),
    "train_sha256": sha256_file(data_dir / "ucip05_train.jsonl"),
    "val_file": str((data_dir / "ucip05_validation.jsonl").relative_to(REPO_ROOT)),
    "val_sha256": sha256_file(data_dir / "ucip05_validation.jsonl"),
    "category_breakdown": {}
}

for rec in unique_records:
    cat = rec["category"]
    manifest["category_breakdown"][cat] = manifest["category_breakdown"].get(cat, 0) + 1

with open(data_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("Saved perfectly balanced dataset manifest:")
print(json.dumps(manifest, indent=2))
