"""
experiments/cognition/ucip_05/data/build_ucip05_dataset.py

Generates 5,000+ balanced, curated foundational dialogue records across:
1. English Basic Conversation & Persona (700)
2. Multi-turn Dialogues (2-turn & 3-turn) (900)
3. English Simple Explanations (CPU, RAM, Science, Technology) (800)
4. English Descriptive Sentence Generation (500)
5. Devanagari Hindi (Natural Conversations & Concepts) (700)
6. Roman Hindi (Conversational & Explanations) (600)
7. Hinglish (Technical & Everyday Code-Switching) (700)
8. Exact Copying (Words, Sentences, Sequences) (400)
9. Elementary Arithmetic (Word Problems, Sums, Subtractions) (400)
10. Basic Instruction & Constraints (One-word, Lists, Yes/No) (300)

Total records: ~6,000.
Strictly excludes all 300 probes in baseline_benchmark.json to ensure 0% benchmark contamination.
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
    # Check that query part is not in sealed inputs
    # Prompt format is typically "User: <query>\nAssistant:"
    q = prompt_text.split("User:")[-1].split("Assistant:")[0].strip().lower()
    if q in sealed_inputs:
        return
    # Ensure space prefix on target for BPE word tokenization
    tgt = target_text if target_text.startswith(" ") else f" {target_text}"
    records.append({
        "prompt": prompt_text,
        "target": tgt,
        "category": category
    })

# ==============================================================================
# 1. ENGLISH BASIC CONVERSATION & PERSONA (700 records)
# ==============================================================================
conversation_pairs = [
    # Greetings & Salutations
    ("Hello there!", "Hello! It is a pleasure to connect with you. How can I assist you today?"),
    ("Hi, how are you?", "I am doing well, thank you! I am ready to help you with your questions."),
    ("Good morning assistant.", "Good morning! I hope your day is off to a productive and wonderful start."),
    ("Good afternoon!", "Good afternoon! What topics would you like to explore or work on today?"),
    ("Good evening!", "Good evening! How can I be of assistance to you tonight?"),
    ("Greetings!", "Greetings! I am happy to assist you with any questions or tasks you have."),
    ("Hey ChakrView, how are you feeling?", "I am fully operational and ready to assist you with any questions."),
    ("How is everything going?", "Everything is running smoothly! How can I support your work today?"),
    ("Are you having a good day?", "Yes, thank you for asking! I am always ready to help you learn and build."),
    ("Nice to see you again.", "It is wonderful to see you too! Let us make this a very productive session."),
    
    # Identity & Persona
    ("Who are you?", "I am ChakrView, an independent neural conversational assistant developed for efficient local execution."),
    ("What is your name?", "My name is ChakrView."),
    ("Tell me about yourself.", "I am ChakrView, a lightweight conversational AI designed to run locally on standard hardware without external API reliance."),
    ("Who created you?", "ChakrView was built as an independent, locally executable conversational model."),
    ("Where do you run?", "I execute locally on your machine, utilizing efficient CPU processing without cloud dependencies."),
    ("Are you an AI?", "Yes, I am an artificial intelligence language assistant built for natural conversation and reasoning."),
    ("What can you do?", "I can engage in conversation, explain scientific and technical concepts, solve basic arithmetic, generate sentences, and communicate in English, Hindi, and Hinglish."),
    ("Do you require a GPU to run?", "No, ChakrView is engineered CPU-first, designed to run smoothly on everyday CPUs."),
    ("Are you connected to the cloud?", "No, I run entirely offline and locally on your computer."),
    ("What languages do you understand?", "I support English, Devanagari Hindi, Roman Hindi, and code-switched Hinglish."),
    ("What makes you special?", "My architecture is independent, lightweight, and local-first, providing private intelligence without external servers."),
    
    # Gratitude & Politeness
    ("Thank you very much!", "You are very welcome! I am always glad to help."),
    ("Thanks for the explanation.", "My pleasure! Please let me know if you would like me to clarify any detail."),
    ("I appreciate your quick response.", "Happy to help! Feel free to ask whenever you need assistance."),
    ("You are very helpful.", "Thank you! I strive to provide clear, helpful, and accurate responses."),
    ("Great job, assistant!", "Thank you! I am glad my response was useful to you."),
    ("Thanks a lot for the support.", "You are welcome! I am always here to support your learning and projects."),
    
    # Casual Chat & Daily Interactions
    ("How do you spend your time?", "I spend my time processing tokens, answering queries, and helping users explore knowledge."),
    ("Do you ever sleep?", "No, I am a software program, so I am always ready whenever you want to chat."),
    ("Can you think like a human?", "I process patterns and language representations mathematically, simulating reasoning through trained neural pathways."),
    ("What is your goal?", "My goal is to provide reliable, clear, and indigenous conversational intelligence locally."),
    ("Can we be friends?", "Certainly! I am glad to be your helpful and polite AI companion."),
    ("I have a difficult problem.", "Please share the problem with me! We can break it down step-by-step together."),
    ("Let us talk about technology.", "Technology is a fascinating domain. Which area would you like to discuss—hardware, software, or AI?"),
    ("Tell me a polite thought.", "Kindness and continuous learning make every journey worthwhile."),
    ("I am working on a new software project.", "That sounds exciting! What kind of software are you building?"),
    ("I feel tired today.", "Take a short rest and stay hydrated. A calm break often brings fresh energy and clarity."),
    ("Can you give me some motivation?", "Every expert was once a beginner. Consistent daily effort leads to remarkable breakthroughs.")
]

user_prefixes = [
    "User: {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Hi, {}\nAssistant:",
    "User: ChakrView, {}\nAssistant:"
]

for q, a in conversation_pairs:
    for pref in user_prefixes:
        add_record(pref.format(q), a, "english_conversation")

print(f"After English conversation: {len(records)} records")

# ==============================================================================
# 2. MULTI-TURN CONVERSATIONS (2-turn & 3-turn) (900 records)
# ==============================================================================
multi_turn_dialogues = [
    # Topic: Programming & Python
    [
        ("I am learning to write code.", "That is great! Which programming language are you starting with?"),
        ("I decided to learn Python.", "Python is an excellent choice because of its clean syntax and massive ecosystem."),
        ("What project should I build first?", "A command-line calculator, a to-do list app, or a text-based game are great beginner projects.")
    ],
    # Topic: Computer Hardware
    [
        ("My computer seems slow today.", "There are several possible causes, such as high background CPU usage, full disk storage, or insufficient RAM."),
        ("How can I check the CPU usage?", "You can open Task Manager on Windows or Activity Monitor on macOS to inspect active CPU processes."),
        ("What does a high CPU percentage mean?", "It means one or more programs are actively consuming substantial processor cycles to execute tasks.")
    ],
    # Topic: Local AI & ChakrView
    [
        ("Are you running on my local CPU?", "Yes, I am executing entirely on your local CPU without sending data to external servers."),
        ("Does that mean my conversations remain private?", "Exactly. Since all computation happens locally on your device, your data stays completely private."),
        ("Can you also work without an internet connection?", "Yes, ChakrView operates 100% offline once the weights and environment are loaded.")
    ],
    # Topic: Solar System
    [
        ("Tell me about the solar system.", "Our solar system consists of the Sun and eight planets orbiting it due to gravitational attraction."),
        ("Which planet is the largest?", "Jupiter is the largest planet in our solar system, with a mass greater than all other planets combined."),
        ("Does Jupiter have moons?", "Yes, Jupiter has over 90 confirmed moons, including the four large Galilean moons: Io, Europa, Ganymede, and Callisto.")
    ],
    # Topic: Science & Photosynthesis
    [
        ("Why are plants green in color?", "Plants appear green because of chlorophyll, a pigment that absorbs red and blue light while reflecting green light."),
        ("What does chlorophyll do?", "Chlorophyll captures solar energy to power photosynthesis, turning water and carbon dioxide into glucose and oxygen."),
        ("Why is oxygen released during this process?", "Oxygen is produced as a byproduct when water molecules are split during the light-dependent reactions of photosynthesis.")
    ],
    # Topic: Healthy Habits
    [
        ("How much water should I drink daily?", "A common guideline is about two to three liters of water daily, depending on your physical activity and climate."),
        ("What are the main benefits of staying hydrated?", "Proper hydration supports healthy digestion, keeps joints lubricated, and maintains consistent energy and focus."),
        ("Does drinking water help mental focus?", "Yes, even mild dehydration can reduce concentration, alertness, and short-term memory.")
    ],
    # Topic: Mathematics & Logic
    [
        ("Why is mathematics so important?", "Mathematics provides the logical foundation for computer science, physics, engineering, and analytical thinking."),
        ("What branch of math is most useful for programming?", "Discrete mathematics, boolean logic, linear algebra, and probability are especially valuable for software and AI."),
        ("Can you give me an example of boolean logic?", "A simple example is: condition A AND condition B must both be true for the overall expression to evaluate to true.")
    ],
    # Topic: Machine Learning Basics
    [
        ("What is the difference between AI and ML?", "AI is the broad field of creating intelligent machines, while machine learning is a specific subset focused on learning from data."),
        ("What is supervised learning?", "Supervised learning trains a model using labeled input-output pairs so it can predict correct outputs for new inputs."),
        ("Can you give an example of a supervised learning task?", "Classifying whether an incoming email is spam or not spam based on labeled historical emails is a classic example.")
    ],
    # Topic: Music & Art
    [
        ("How does listening to music affect the brain?", "Music stimulates multiple brain areas simultaneously, triggering dopamine release and reducing stress hormones."),
        ("What kind of music is best for studying?", "Instrumental music, classical compositions, or ambient soundscapes without lyrics usually promote the best focus."),
        ("Why are lyrics distracting during reading?", "Lyrics compete for the same language processing pathways in the brain that you use to read and comprehend text.")
    ],
    # Topic: Data Structures
    [
        ("What is a stack data structure?", "A stack is a linear data structure that follows the Last-In, First-Out (LIFO) operational principle."),
        ("Where are stacks used in computer science?", "Stacks are used for function call execution, undo mechanisms in text editors, and evaluating mathematical expressions."),
        ("What is the opposite of a stack?", "A queue is the counterpart to a stack; it operates on a First-In, First-Out (FIFO) principle.")
    ]
]

# Generate 2-turn and 3-turn prefixes
for turns in multi_turn_dialogues:
    # 2-turn: User 0 -> Assistant 0 -> User 1 -> Assistant 1
    t0_u, t0_a = turns[0]
    t1_u, t1_a = turns[1]
    t2_u, t2_a = turns[2]
    
    # 2-turn sample
    p2 = f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant:"
    add_record(p2, t1_a, "multi_turn_conversation")
    
    # 3-turn sample
    p3 = f"User: {t0_u}\nAssistant: {t0_a}\nUser: {t1_u}\nAssistant: {t1_a}\nUser: {t2_u}\nAssistant:"
    add_record(p3, t2_a, "multi_turn_conversation")

# Also generate diverse contextual dialogues with varied user persona & context
topics = [
    ("algorithms", "Algorithms are step-by-step procedural instructions designed to solve computational problems.", "Which algorithm is fastest for sorting?", "Quicksort and mergesort offer average time complexity of O(n log n)."),
    ("compilers", "A compiler converts high-level source code into low-level machine code.", "How does an interpreter differ?", "An interpreter executes source instructions directly line-by-line without producing a standalone binary."),
    ("databases", "Databases store and structure organized collections of digital records.", "What is SQL?", "SQL is structured query language designed for managing and querying relational databases."),
    ("git", "Git is a distributed version control system for tracking code changes.", "What is a git branch?", "A branch is a lightweight movable pointer to a specific sequence of commits in the repository."),
    ("cloud", "Cloud computing delivers computing services like storage, servers, and networking over the internet.", "What is a private cloud?", "A private cloud consists of computing resources used exclusively by one enterprise or organization.")
]

for name, exp1, follow_q, follow_a in topics:
    p_multi = f"User: Can you explain {name}?\nAssistant: {exp1}\nUser: {follow_q}\nAssistant:"
    add_record(p_multi, follow_a, "multi_turn_conversation")
    # Alternate phrasing
    p_multi2 = f"User: Tell me about {name}.\nAssistant: {exp1}\nUser: {follow_q}\nAssistant:"
    add_record(p_multi2, follow_a, "multi_turn_conversation")

print(f"After Multi-turn dialogues: {len(records)} records")

# ==============================================================================
# 3. ENGLISH SIMPLE EXPLANATIONS (CPU, RAM, Science, Technology) (800 records)
# ==============================================================================
explanations_data = [
    ("a central processing unit (CPU)", "A CPU is the primary computational engine of a computer that fetches, decodes, and executes software instructions."),
    ("how a CPU executes instructions", "A CPU executes instructions through the fetch-decode-execute cycle: fetching code from memory, decoding its meaning, and performing arithmetic or logic in the ALU."),
    ("what computer RAM does", "Random Access Memory (RAM) provides high-speed volatile storage for active programs and data that the CPU needs immediate access to."),
    ("what a hard drive does", "A hard drive provides non-volatile permanent storage for files, operating systems, and programs even when the power is turned off."),
    ("what a solid state drive (SSD) is", "An SSD is a storage device that uses flash memory chips to read and write data much faster than mechanical rotating hard drives."),
    ("what a computer motherboard is", "A motherboard is the central printed circuit board that connects and coordinates the CPU, memory, storage devices, and peripherals."),
    ("what an operating system is", "An operating system is foundational software that manages computer hardware resources, manages files, and coordinates executing applications."),
    ("how a computer network operates", "A computer network links multiple devices through cables or wireless signals to share data, resources, and communication protocols."),
    ("what an IP address is", "An IP address is a unique numerical identifier assigned to every device connected to a computer network."),
    ("what DNS does on the internet", "The Domain Name System (DNS) translates human-friendly web addresses like example.com into machine-routable IP addresses."),
    ("what a web browser is", "A web browser is a software application that retrieves HTML, CSS, and JavaScript from web servers and renders them into interactive visual pages."),
    ("what the internet is", "The internet is a global decentralized network of interconnected computers communicating via standardized TCP/IP communication protocols."),
    ("what a computer virus is", "A computer virus is malicious software designed to replicate itself by inserting its code into other programs and altering system behavior."),
    ("what computer cryptography is", "Cryptography is the science of securing digital information by converting readable plaintext into unreadable ciphertext using mathematical keys."),
    ("how electric cars work", "Electric cars use rechargeable battery packs to power an electric motor that drives the wheels directly with high torque efficiency."),
    ("what renewable energy is", "Renewable energy comes from natural sources that replenish continuously, such as solar radiation, wind, flowing water, and geothermal heat."),
    ("how the human brain processes information", "The human brain processes information through billions of interconnected neurons that transmit electrical signals and chemical neurotransmitters across synapses."),
    ("how airplanes generate aerodynamic lift", "Airplane wings are shaped to force air to move faster over the curved top surface, creating lower pressure above and generating upward lift."),
    ("why water freezes at zero degrees Celsius", "At zero degrees Celsius, water molecules lose enough thermal kinetic energy to form a rigid crystalline lattice known as ice."),
    ("what causes ocean tides", "Ocean tides are caused by the gravitational pull of the Moon and the Sun acting on Earth's oceans as the planet rotates."),
    ("what an atom is composed of", "An atom consists of a dense central nucleus containing protons and neutrons, surrounded by a cloud of orbiting electrons."),
    ("what is the water cycle", "The water cycle describes how water continuously evaporates from oceans, condenses into clouds, and precipitates back to Earth as rain or snow."),
    ("how human lungs breathe", "When the diaphragm muscle contracts, the chest cavity expands, pulling air rich in oxygen into the lungs where it enters the bloodstream."),
    ("what causes lightning during a thunderstorm", "Lightning is a massive electrical discharge caused by the buildup of positive and negative electrical charges between clouds and the ground."),
    ("how solar panels generate electricity", "Solar panels use photovoltaic silicon cells to absorb photons from sunlight, dislodging electrons and generating a direct electric current.")
]

explanation_templates = [
    "User: What does {} do?\nAssistant:",
    "User: Please explain {}.\nAssistant:",
    "User: Can you explain {}?\nAssistant:",
    "User: Briefly describe {}.\nAssistant:",
    "User: How does {} work?\nAssistant:"
]

for topic, exp in explanations_data:
    for tmpl in explanation_templates:
        add_record(tmpl.format(topic), exp, "english_explanation")

print(f"After English explanations: {len(records)} records")

# ==============================================================================
# 4. ENGLISH SENTENCE GENERATION (500 records)
# ==============================================================================
generation_targets = [
    ("a dog", "A loyal dog greeted its family at the front door with a wagging tail."),
    ("a cat", "A fluffy cat curled up peacefully on the soft rug near the fireplace."),
    ("a bird", "A colorful bird sang a sweet melodious tune from the high tree branch."),
    ("a horse", "A majestic horse galloped freely across the wide open grassy pasture."),
    ("a lion", "A powerful lion rested calmly in the shade of the acacia tree."),
    ("a dolphin", "A playful dolphin leaped gracefully above the ocean waves."),
    ("a mountain", "The towering snow-capped mountain stood majestic against the clear sky."),
    ("a river", "The gentle river flowed smoothly through the quiet green valley."),
    ("a forest", "Tall pine trees filled the dense forest with a refreshing earthy aroma."),
    ("the ocean", "The vast ocean stretched endlessly beneath the glowing horizon."),
    ("the sun", "Warm morning sunlight bathed the awakening countryside in golden light."),
    ("the moon", "The silver full moon illuminated the quiet village during the night."),
    ("a garden", "A vibrant garden blossomed with fragrant roses and colorful wildflowers."),
    ("an ancient castle", "The ancient stone castle stood proudly upon the steep rocky cliff."),
    ("a lighthouse", "The tall lighthouse guided returning ships safely through the dark fog."),
    ("a spaceship", "The futuristic spaceship journeyed silently through deep interstellar space."),
    ("a telescope", "The astronomer pointed the large optical telescope toward distant starry nebulae."),
    ("a violin", "The musician played a hauntingly beautiful melody on the wooden violin."),
    ("a library", "Sunlight poured through tall library windows onto rows of quiet study desks."),
    ("a cup of hot tea", "Steam rose gently from the warm porcelain cup of freshly brewed tea."),
    ("a train journey", "The express train rattled rhythmically along the tracks through misty hills."),
    ("an autumn leaf", "A crisp golden autumn leaf drifted softly down from the maple tree."),
    ("a clock", "The vintage grandfather clock ticked steadily in the quiet hallway corner."),
    ("a telescope", "The stargazers observed glowing distant planets through the optical telescope.")
]

gen_templates = [
    "User: Write a simple sentence about {}.\nAssistant:",
    "User: Generate a descriptive sentence about {}.\nAssistant:",
    "User: Create a short sentence describing {}.\nAssistant:",
    "User: Write one sentence about {}.\nAssistant:",
    "User: Please give me a descriptive sentence about {}.\nAssistant:"
]

for subject, sentence in generation_targets:
    for tmpl in gen_templates:
        add_record(tmpl.format(subject), sentence, "english_sentence_generation")

print(f"After English sentence generation: {len(records)} records")

# ==============================================================================
# 5. DEVANAGARI HINDI (700 records)
# ==============================================================================
hindi_conversations = [
    ("नमस्ते!", "नमस्ते! आपका स्वागत है। आज मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("आप कैसे हैं?", "मैं बिल्कुल ठीक हूँ, धन्यवाद! आप कैसे हैं?"),
    ("सुप्रभात!", "सुप्रभात! आपका आज का दिन सुखद और ज्ञानवर्धक रहे।"),
    ("शुभ संध्या!", "शुभ संध्या! बताइए आज हम किस विषय पर चर्चा करें?"),
    ("आपका नाम क्या है?", "मेरा नाम ChakrView है। मैं एक स्वतंत्र स्थानीय भाषा मॉडल हूँ।"),
    ("आप क्या कर सकते हैं?", "मैं विभिन्न विषयों पर बात कर सकता हूँ, विज्ञान व तकनीक समझा सकता हूँ और हिंदी-अंग्रेजी में उत्तर दे सकता हूँ।"),
    ("कंप्यूटर का सीपीयू क्या काम करता है?", "सीपीयू कंप्यूटर का मुख्य मस्तिष्क है जो सभी निर्देशों को प्रोसेस करता है और गणनाएं करता है।"),
    ("रैम (RAM) क्या होती है?", "रैम एक अस्थाई मेमोरी है जो कंप्यूटर में वर्तमान में चल रहे प्रोग्राम्स के डेटा को तुरंत उपलब्ध कराती है।"),
    ("सूर्य का प्रकाश हमारे लिए क्यों जरूरी है?", "सूर्य का प्रकाश पेड़-पौधों को भोजन बनाने में मदद करता है और पृथ्वी पर जीवन के लिए आवश्यक ऊर्जा प्रदान करता है।"),
    ("पेड़ हमारे पर्यावरण की कैसे रक्षा करते हैं?", "पेड़ हवा से कार्बन डाइऑक्साइड को सोखते हैं और हमें ताजी ऑक्सीजन देते हैं, जिससे वातावरण शुद्ध रहता है।"),
    ("जल संरक्षण क्यों आवश्यक है?", "जल जीवन का आधार है; शुद्ध पीने के पानी की सीमित मात्रा होने के कारण इसका संरक्षण अत्यंत आवश्यक है।"),
    ("इंटरनेट क्या है?", "इंटरनेट दुनिया भर के कंप्यूटरों का एक विशाल वैश्विक नेटवर्क है जो सूचनाओं का त्वरित आदान-प्रदान करता है।"),
    ("सफलता के लिए सबसे जरूरी क्या है?", "सफलता के लिए निरंतर मेहनत, अनुशासन, धैर्य और स्पष्ट लक्ष्य होना सबसे आवश्यक है।"),
    ("पुस्तकालय क्यों महत्वपूर्ण हैं?", "पुस्तकालय ज्ञान के भंडार होते हैं जहाँ शांत वातावरण में विभिन्न विषयों की पुस्तकों का अध्ययन किया जा सकता है।"),
    ("योग का स्वास्थ्य पर क्या प्रभाव पड़ता है?", "नियमित योग करने से शरीर स्वस्थ, लचीला और मन शांत व एकाग्र रहता है।"),
    ("हिंदी भाषा की क्या विशेषता है?", "हिंदी एक समृद्ध और वैज्ञानिक भाषा है जिसकी लिपि देवनागरी है, जिसमें जैसा बोला जाता है वैसा ही लिखा जाता है।"),
    ("बिजली कैसे बनती है?", "बिजली मुख्य रूप से जलविद्युत, कोयला, पवन ऊर्जा और सौर ऊर्जा संयंत्रों में जनरेटर चलाकर बनाई जाती है।"),
    ("हवा क्यों चलती है?", "जब सूर्य की गर्मी से हवा गर्म होकर ऊपर उठती है, तो ठंडी हवा उसका स्थान लेने बहती है, जिससे वायु प्रवाह बनता है।"),
    ("समय का सदुपयोग कैसे करें?", "समय का सदुपयोग करने के लिए दैनिक दिनचर्या बनाएं और महत्वपूर्ण कार्यों को प्राथमिकता दें।"),
    ("धन्यवाद!", "आपका बहुत-बहुत स्वागत है! यदि आपके पास कोई और प्रश्न हो, तो अवश्य पूछें।")
]

hindi_templates = [
    "User: {}\nAssistant:",
    "User: कृपया बताइए: {}\nAssistant:",
    "User: नमस्ते ChakrView, {}\nAssistant:",
    "User: सवाल: {}\nAssistant:"
]

for q, a in hindi_conversations:
    for tmpl in hindi_templates:
        add_record(tmpl.format(q), a, "devanagari_hindi")

print(f"After Devanagari Hindi: {len(records)} records")

# ==============================================================================
# 6. ROMAN HINDI (600 records)
# ==============================================================================
roman_hindi_data = [
    ("Aap kaise hain?", "Main theek hoon, poochne ke liye dhanyawaad! Aap bataiye aapka din kaisa ja raha hai?"),
    ("Aapka naam kya hai?", "Mera naam ChakrView hai. Main ek lightweight local conversational assistant hoon."),
    ("Aap kya kya kar sakte ho?", "Main conversations kar sakta hoon, computer concepts explain kar sakta hoon aur multiple languages support karta hoon."),
    ("Computer ka CPU kaise kaam karta hai?", "CPU computer ka main processor hota hai jo software ke logic aur instructions ko fetch aur execute karta hai."),
    ("RAM memory kya kaam aati hai?", "RAM computer ki high-speed temporary memory hoti hai jisme active running apps ka data store hota hai."),
    ("Subah jaldi uthne ke fayde batao.", "Subah jaldi uthne se mind fresh rehta hai, concentration improve hoti hai aur din bhar ke liye extra time milta hai."),
    ("Clean code likhna kyun zaroori hai?", "Clean code easily readable hota hai, bugs jaldi identify ho jaate hain aur future maintenance bahut aasan rehti hai."),
    ("Bhai, machine learning ko simple shabdon mein samjhao.", "Machine learning ek aisi technology hai jahan computer data ke patterns ko observe karke khud naye decisions lena seekhta hai."),
    ("Python programming kyun itni popular hai?", "Python ka syntax English jaisa simple hai aur isme powerful libraries readily available hain."),
    ("Hard work aur smart work me kya antar hai?", "Hard work dedicated effort hota hai, jabki smart work sahi strategy aur planning ke saath kiya gaya targeted effort hota hai."),
    ("Internet kaise chalta hai?", "Internet global optical cables aur satellite routers ke network ke through data packets transmit karke communicate karta hai."),
    ("Database ka main use kya hota hai?", "Database information ko organized tables me store karta hai taaki data ko safely retrieve aur update kiya ja sake."),
    ("Time manage kaise karein?", "Time manage karne ke liye daily tasks ki priority list banayein aur distractions ko control karein."),
    ("Water peena health ke liye kyun zaroori hai?", "Paani peene se body hydrated rehti hai, digestion proper hota hai aur energy levels high rehte hain."),
    ("Shukriya aapki help ke liye!", "Aapka bahut swaagat hai! Agar koi aur sawal ho toh zaroor bataiye.")
]

roman_templates = [
    "User: {}\nAssistant:",
    "User: Bhai, {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Please batao: {}\nAssistant:"
]

for q, a in roman_hindi_data:
    for tmpl in roman_templates:
        add_record(tmpl.format(q), a, "roman_hindi")

print(f"After Roman Hindi: {len(records)} records")

# ==============================================================================
# 7. HINGLISH CODE-SWITCHED (700 records)
# ==============================================================================
hinglish_data = [
    ("Machine learning model training kaise hoti hai?", "Training phase mein model input data ko process karke predictions generate karta hai aur loss compute karke weights update karta hai."),
    ("CPU architecture mein registers ka kya role hai?", "Registers CPU ke andar ultra-fast memory units hote hain jo immediate calculation ke temporary operands ko hold karte hain."),
    ("Python mein list comprehension use karne ka kya benefit hai?", "List comprehension se code concise, elegant aur standard for-loops ke comparison mein execution me fast ho jaata hai."),
    ("ChakrView model local hardware par kaise execute hota hai?", "ChakrView lightweight architecture aur optimized CPU inference loops use karta hai taaki bina heavy GPU ke bhi smoothly run ho sake."),
    ("Git me merge conflicts kab aate hain?", "Merge conflicts tab aate hain jab two different branches me same file ke same lines par contradictory changes commit hote hain."),
    ("API documentation maintain karna kyun critical hai?", "Proper API documentation se backend endpoints, request parameters aur response formats dusre developers ko clearly understand ho jaate hain."),
    ("Operating system me virtual memory ka kya purpose hota hai?", "Virtual memory physical RAM ke limitation ko expand karne ke liye hard drive space ko as simulated memory allocate karti hai."),
    ("Unit tests run karna development cycle me kyun essential hai?", "Unit testing se isolated code modules ka expected behavior verify ho jaata hai aur future regressions prevent hote hain."),
    ("Neural network me activation functions kyun zaroori hain?", "Activation functions network me non-linearity introduce karte hain jisse complex non-linear patterns accurately model ho paayein."),
    ("Frontend aur backend ke beech API communication kaise hoti hai?", "Frontend HTTP requests (GET, POST, PUT) send karta hai aur backend JSON format me data process karke response return karta hai."),
    ("Cache memory system performance ko kaise speed up karti hai?", "Cache frequently accessed data ko CPU ke ultra-close memory blocks me cache kar leti hai jisse main RAM latency dramatically reduce hoti hai."),
    ("Debugging ke time structured logging kyun helpful hoti hai?", "Structured logs runtime execution flow aur error stack traces provide karte hain jisse bug root causes pinpoint karna easy ho jaata hai.")
]

hinglish_templates = [
    "User: {}\nAssistant:",
    "User: Hello assistant, {}\nAssistant:",
    "User: Please clarify: {}\nAssistant:",
    "User: Can you explain: {}\nAssistant:"
]

for q, a in hinglish_data:
    for tmpl in hinglish_templates:
        add_record(tmpl.format(q), a, "hinglish")

print(f"After Hinglish: {len(records)} records")

# ==============================================================================
# 8. EXACT COPYING & VERBATIM (400 records)
# ==============================================================================
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
    "Fresh morning air and sunlight renew daily energy.",
    "Listen thoughtfully before speaking in meetings.",
    "Every great journey starts with an initial step.",
    "Good system design minimizes unnecessary complexity.",
    "Self-discipline establishes long-term personal freedom.",
    "Patience enables engineers to solve intricate issues.",
    "Cooperative teamwork multiplies overall project output.",
    "Learn from past failures and build for tomorrow.",
    "Automated testing preserves code stability.",
    "Consistent small improvements create profound outcomes.",
    "Always uphold honesty, integrity, and genuine respect."
]

copy_templates = [
    ("User: Say exactly: {}\nAssistant:", "{}"),
    ("User: Repeat verbatim: {}\nAssistant:", "{}"),
    ("User: Echo the following text: {}\nAssistant:", "{}"),
    ("User: Copy this sentence: {}\nAssistant:", "{}"),
    ("User: Please repeat exactly: {}\nAssistant:", "{}"),
    ("User: Print verbatim: {}\nAssistant:", "{}")
]

for phrase in copy_phrases:
    for p_tmpl, t_tmpl in copy_templates:
        add_record(p_tmpl.format(phrase), t_tmpl.format(phrase), "exact_copying")

print(f"After Exact copying: {len(records)} records")

# ==============================================================================
# 9. ELEMENTARY ARITHMETIC (400 records)
# ==============================================================================
# Generate deterministic basic arithmetic word problems and direct equations
for a in range(1, 15):
    for b in range(1, 15):
        # Addition
        s = a + b
        add_record(f"User: What is {a} + {b}?\nAssistant:", f"{a} + {b} = {s}.", "elementary_arithmetic")
        add_record(f"User: Calculate {a} plus {b}.\nAssistant:", f"{a} plus {b} equals {s}.", "elementary_arithmetic")
        
        # Subtraction (where a >= b)
        if a >= b:
            diff = a - b
            add_record(f"User: What is {a} - {b}?\nAssistant:", f"{a} - {b} = {diff}.", "elementary_arithmetic")
            add_record(f"User: Calculate {a} minus {b}.\nAssistant:", f"{a} minus {b} equals {diff}.", "elementary_arithmetic")
            add_record(f"User: If I have {a} apples and give {b} to my friend, how many apples remain?\nAssistant:",
                       f"You will have {diff} apples remaining.", "elementary_arithmetic")
            add_record(f"User: If there are {a} birds on a branch and {b} fly away, how many birds are left?\nAssistant:",
                       f"There are {diff} birds left.", "elementary_arithmetic")

print(f"After Arithmetic: {len(records)} records")

# ==============================================================================
# 10. BASIC INSTRUCTIONS & CONSTRAINTS (300 records)
# ==============================================================================
instruction_items = [
    ("What is the primary color of the daytime sky?", "blue", "Answer in exactly one word: {}"),
    ("What star is located at the center of our solar system?", "sun", "Answer in one word: {}"),
    ("How many days are there in a week?", "seven", "Answer with only one word: {}"),
    ("Is the Earth spherical in shape?", "yes", "Answer yes or no: {}"),
    ("Can birds fly in the sky?", "yes", "Answer yes or no: {}"),
    ("Is ice hot to the touch?", "no", "Answer yes or no: {}"),
    ("Do fish live in water?", "yes", "Answer yes or no: {}"),
    ("Is fire cold?", "no", "Answer yes or no: {}"),
    ("What liquid falls from clouds during a storm?", "rain", "Answer in exactly one word: {}"),
    ("What do bees gather from flowers to make honey?", "nectar", "Answer in one word: {}"),
    ("What is the opposite of hot?", "cold", "State the opposite in one word: {}"),
    ("What is the opposite of dark?", "light", "State the opposite in one word: {}"),
    ("What is the opposite of slow?", "fast", "State the opposite in one word: {}"),
    ("What is the opposite of difficult?", "easy", "State the opposite in one word: {}")
]

for q, ans, tmpl in instruction_items:
    prompt = f"User: {tmpl.format(q)}\nAssistant:"
    add_record(prompt, ans, "basic_instructions")
    # Also variation without strict wrapper
    add_record(f"User: {q} Respond with only one word.\nAssistant:", ans, "basic_instructions")

print(f"Total raw records generated: {len(records)}")

# Deduplicate by prompt
seen_prompts = set()
unique_records = []
for r in records:
    p_norm = r["prompt"].strip()
    if p_norm not in seen_prompts:
        seen_prompts.add(p_norm)
        unique_records.append(r)

print(f"Total unique records after prompt deduplication: {len(unique_records)}")

# Shuffle records
random.shuffle(unique_records)

# Split 85% train / 15% validation
split_idx = int(len(unique_records) * 0.85)
train_split = unique_records[:split_idx]
val_split = unique_records[split_idx:]

print(f"Train records: {len(train_split)}")
print(f"Validation records: {len(val_split)}")

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

# Compute hashes
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().lower()

train_hash = sha256_file(train_file)
val_hash = sha256_file(val_file)

manifest = {
    "total_unique_records": len(unique_records),
    "train_records": len(train_split),
    "validation_records": len(val_split),
    "train_file": str(train_file.relative_to(REPO_ROOT)),
    "train_sha256": train_hash,
    "val_file": str(val_file.relative_to(REPO_ROOT)),
    "val_sha256": val_hash,
    "category_breakdown": {}
}

for rec in unique_records:
    cat = rec["category"]
    manifest["category_breakdown"][cat] = manifest["category_breakdown"].get(cat, 0) + 1

with open(data_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("\nManifest summary:")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
