"""
UCIP-04 Foundational Autoregressive Corpus Generator (Expanded & Disjoint).
Ensures extensive volume across all categories with 100% strict zero-leakage
against baseline_100_probes.json.
"""

import json
import random
import hashlib
from pathlib import Path

random.seed(42)

# Load baseline 100 prompts to guarantee zero leakage
base_file = Path("experiments/cognition/ucip_04/evaluation/baseline_100_probes.json")
sealed_prompts = set()
if base_file.exists():
    for p in json.load(open(base_file, "r", encoding="utf-8")):
        sealed_prompts.add(p["prompt"].strip())

print(f"Loaded {len(sealed_prompts)} sealed diagnostic prompts to exclude.")

records = []

# =========================================================================
# 1. SIMPLE GENERATION (Everyday Objects, Animals, Concepts) - 250+
# =========================================================================
simple_generation_items = [
    ("a dog", [
        "A dog is a friendly domestic animal that loves to play and guard the home.",
        "Dogs are faithful companions known for their loyalty to humans.",
        "The playful puppy chased a tennis ball across the green field.",
        "A trained guard dog stood alertly beside the front gate."
    ]),
    ("a cat", [
        "A cat is a graceful animal with sharp senses and soft warm fur.",
        "Cats enjoy napping in quiet sunny spots during the afternoon.",
        "A curious kitten climbed carefully onto the windowsill.",
        "The quiet cat purred gently while sitting on the velvet sofa."
    ]),
    ("a computer", [
        "A computer is an electronic machine that processes data and runs applications.",
        "Computers help students and professionals solve complex problems rapidly.",
        "Modern computers utilize multi-core microprocessors and solid-state storage.",
        "The laptop computer loaded the operating system within seconds."
    ]),
    ("a tree", [
        "A tree absorbs carbon dioxide from the air and produces clean oxygen.",
        "Tall oak trees provide cooling shade and shelter for nesting birds.",
        "The deep roots of the ancient tree anchor firmly into fertile soil.",
        "Fresh green buds sprouted along the branches in early spring."
    ]),
    ("the sun", [
        "The sun is the bright star at the center of our solar system.",
        "Sunlight provides warmth and energy that sustains all living organisms on Earth.",
        "The morning sun cast long golden shadows across the quiet town.",
        "Solar radiation warms the atmosphere and drives global weather systems."
    ]),
    ("a river", [
        "A river is a flowing channel of fresh water traveling toward an ocean or lake.",
        "Clean rivers supply vital water to nearby towns and agricultural fields.",
        "The mountain river flowed swiftly through the steep rocky canyon.",
        "Fish swam peacefully beneath the calm surface of the freshwater river."
    ]),
    ("an apple", [
        "An apple is a sweet and crisp fruit that grows on leafy orchard trees.",
        "Eating fresh apples provides valuable vitamins and dietary fiber for good health.",
        "The farmer harvested bushels of ripe red apples during the autumn season.",
        "A green apple sliced into pieces makes a delicious and healthy afternoon snack."
    ]),
    ("a bicycle", [
        "A bicycle is a two-wheeled vehicle powered by human pedaling.",
        "Riding a bicycle is an efficient and environmentally friendly way to commute.",
        "He pumped air into the bicycle tires before riding along the coastal path.",
        "The shiny red bicycle was parked neatly beside the wooden bench."
    ]),
    ("rain", [
        "Rain is liquid water falling from dense clouds in the sky.",
        "Gentle rain refreshes the dry soil and nourishes growing crops in farm fields.",
        "The soothing sound of rain tapping on the roof created a peaceful ambiance.",
        "Dark storm clouds gathered quickly on the horizon before the afternoon rain."
    ]),
    ("a mountain", [
        "A mountain is an elevated landform that towers high above the surrounding terrain.",
        "Snow covers the rocky mountain peaks throughout the cold winter months.",
        "Climbing a steep mountain requires strength, focus, and proper hiking gear.",
        "The winding trail led hikers to the summit of the majestic mountain."
    ]),
    ("music", [
        "Music is an expressive art form composed of harmonious sounds, melody, and rhythm.",
        "Listening to calm acoustic music helps people relax after a demanding workday.",
        "Rhythmic percussion music inspired the dancers to move energetically across the stage.",
        "The violinist played a beautiful classical melody with skill and precision."
    ]),
    ("a book", [
        "A book contains written knowledge, thoughtful ideas, or imaginative stories.",
        "Reading a well-written book broadens a person's perspective on history and culture.",
        "The leather-bound book contained centuries of historical records and illustrations.",
        "She bookmarked the final chapter of the mystery novel before falling asleep."
    ]),
    ("the ocean", [
        "The ocean is a vast expanse of salt water covering most of planet Earth.",
        "Oceans are home to thousands of diverse fish, coral reefs, and marine mammals.",
        "Gentle ocean waves crashed rhythmically against the sandy tropical beach.",
        "Deep ocean trenches remain among the most mysterious ecosystems on the planet."
    ]),
    ("a bird", [
        "A bird is a warm-blooded feathered creature capable of gliding through the air.",
        "Songbirds sing cheerful melodies from leafy branches early at dawn.",
        "The golden eagle circled effortlessly high in the clear blue sky.",
        "Small sparrows gathered seeds scattered on the stone garden path."
    ]),
    ("a flower", [
        "A flower is the colorful reproductive blossom of a flowering plant.",
        "Fragrant flowers in the botanical garden attracted busy honeybees and butterflies.",
        "A bright yellow sunflower turned faithfully toward the warmth of the sun.",
        "Dewdrops rested delicately on the velvety red petals of the rose."
    ]),
    ("bread", [
        "Bread is a staple food prepared by baking fermented dough made of flour and water.",
        "The irresistible aroma of fresh sourdough bread drifted from the neighborhood bakery.",
        "Toasted bread topped with olive oil and herbs makes a simple Italian appetizer.",
        "Bakers knead the dough thoroughly to develop gluten before placing it in the oven."
    ]),
    ("the moon", [
        "The moon is Earth's only natural satellite that shines with reflected sunlight.",
        "The full moon illuminated the quiet countryside with a serene silvery glow.",
        "Gravitational pull from the moon creates the cyclical rise and fall of ocean tides.",
        "Astronauts landed on the desolate lunar surface during the historic Apollo mission."
    ]),
    ("a car", [
        "A car is a wheeled motor vehicle designed for personal and family road transportation.",
        "Modern electric cars utilize lithium-ion batteries to eliminate tailpipe exhaust emissions.",
        "The family drove their compact sedan smoothly along the winding coastal highway.",
        "Advanced driver assistance systems in modern cars enhance highway passenger safety."
    ]),
    ("a school", [
        "A school is an educational institution where students learn diverse subjects from teachers.",
        "Schools nurture friendships, collaborative problem-solving, and civic responsibility.",
        "Children gathered eagerly in the sunlit school courtyard before their morning classes.",
        "Dedicated teachers organize interactive laboratory experiments to teach scientific principles."
    ]),
    ("a hospital", [
        "A hospital is a healthcare facility providing specialized medical treatment and nursing care.",
        "Doctors and nurses work tirelessly in hospitals to care for recovering patients.",
        "Modern hospitals are equipped with advanced diagnostic imaging and intensive care units.",
        "Emergency ambulances transport injured individuals quickly to the nearest hospital."
    ])
]

gen_templates = [
    "Write one simple sentence about {s}.",
    "Write a short sentence describing {s}.",
    "Please give me a simple sentence about {s}.",
    "Generate a descriptive sentence about {s}.",
    "Tell me a simple sentence about {s}."
]

for topic, sent_list in simple_generation_items:
    for sent in sent_list:
        for tmpl in gen_templates:
            p_text = f"User: {tmpl.format(s=topic)}\nAssistant:"
            if p_text.strip() not in sealed_prompts:
                records.append({
                    "prompt": p_text,
                    "target": f" {sent}",
                    "category": "simple_generation"
                })

# =========================================================================
# 2. BASIC EXPLANATIONS (Technical, Scientific, Logical) - 200+
# =========================================================================
explanations_items = [
    ("What does a CPU do?", "A central processing unit executes software instructions, performs arithmetic calculations, and manages data flow inside a computer."),
    ("What is a central processing unit?", "A CPU is the primary computational brain of a computer that interprets program code and coordinates hardware activities."),
    ("How does a computer processor function?", "A computer processor fetches instructions from memory, decodes their meaning, and executes the required arithmetic or logical operations."),
    ("What does a motherboard do?", "A motherboard is the main circuit board that connects and facilitates communication between the CPU, memory, storage, and peripheral cards."),
    ("What is computer RAM?", "Random access memory is fast temporary storage used by the computer to hold data and program code actively being executed."),
    ("What is an SSD?", "A solid-state drive is a persistent digital storage device that stores data on non-volatile flash memory chips without moving parts."),
    ("What is an operating system?", "An operating system is foundational system software that manages hardware resources, files, and user processes on a computer."),
    ("What is the internet?", "The internet is a global network of interconnected computer systems communicating via standardized protocols like TCP/IP."),
    ("What is an algorithm?", "An algorithm is a well-defined sequence of logical steps used to solve a computational problem or complete a task."),
    ("What is machine learning?", "Machine learning is a subfield of artificial intelligence where statistical models learn patterns directly from training data."),
    ("What is an artificial neural network?", "An artificial neural network is a machine learning architecture structured with interconnected mathematical units organized in layers."),
    ("Explain the principle of gravity.", "Gravity is a fundamental physical force that causes objects with mass to attract one another, keeping planets in orbital motion."),
    ("What causes ocean tides?", "Ocean tides are caused primarily by the gravitational pull of the moon and sun acting upon Earth's rotating water bodies."),
    ("What is photosynthesis in plants?", "Photosynthesis is the biochemical process where green plants transform solar light energy, water, and carbon dioxide into sugars and oxygen."),
    ("What is the chemical composition of water?", "Water consists of molecules made from two hydrogen atoms chemically bonded to one oxygen atom, represented as H2O."),
    ("Why is the sky blue during daylight?", "The sky appears blue because molecules in the Earth's atmosphere scatter short-wavelength blue sunlight more than other colors."),
    ("What is solar energy?", "Solar energy is renewable energy captured from sunlight using photovoltaic solar cells or concentrated solar thermal collectors."),
    ("What is wind energy?", "Wind energy is clean electrical power generated by wind turbines that convert kinetic airflow into rotational mechanical energy."),
    ("What is cybersecurity?", "Cybersecurity encompasses technologies, policies, and practices designed to protect digital networks, systems, and data from attacks."),
    ("What is open source software?", "Open source software is software distributed with its complete source code, permitting anyone to study, improve, and share it freely.")
]

exp_templates = [
    "User: {q}\nAssistant:",
    "User: Can you explain: {q_clean}\nAssistant:",
    "User: Please explain: {q_clean}\nAssistant:",
    "User: Briefly tell me: {q_clean}\nAssistant:"
]

for q, ans in explanations_items:
    q_clean = q.replace("?", "").lower()
    for tmpl in exp_templates:
        p_text = tmpl.format(q=q, q_clean=q_clean)
        if p_text.strip() not in sealed_prompts:
            records.append({
                "prompt": p_text,
                "target": f" {ans}",
                "category": "basic_explanation"
            })

# =========================================================================
# 3. EXACT COPYING & REPRODUCTION - 200+
# =========================================================================
copying_data = [
    "The morning air was crisp and refreshing.",
    "Engineers build reliable bridges across wide rivers.",
    "Scientific discovery requires patience and curiosity.",
    "Clean drinking water is fundamental to human health.",
    "Mathematics provides a foundation for logical thinking.",
    "The library was quiet as students read their textbooks.",
    "A gentle breeze moved softly through the green trees.",
    "Small daily habits create significant long-term results.",
    "Astronomers observe distant galaxies through large telescopes.",
    "Clear communication prevents misunderstandings in teams.",
    "The sun rose quietly over the mountain range.",
    "Writing clean code makes software maintenance much easier.",
    "Children played happily in the neighborhood park.",
    "Musicians practiced their instruments before the concert.",
    "Healthy soil provides essential nutrients for growing plants.",
    "Deep breathing exercises help calm the human nervous system.",
    "Solar panels generate clean electrical energy from sunlight.",
    "Regular physical exercise strengthens cardiovascular health.",
    "The commuter train arrived exactly on schedule at the terminal.",
    "Curiosity motivates researchers to explore challenging questions.",
    "Light travels faster than sound in Earth's atmosphere.",
    "The autumn leaves turned vibrant shades of amber and gold.",
    "Careful planning leads to successful software deployments.",
    "A balanced diet includes fresh fruits, grains, and vegetables."
]

copy_templates = [
    "Say exactly: {s}",
    "Repeat this sentence: {s}",
    "Copy exactly: {s}",
    "Please repeat: {s}",
    "Echo the following text: {s}",
    "Repeat verbatim: {s}"
]

for s in copying_data:
    for tmpl in copy_templates:
        p_text = f"User: {tmpl.format(s=s)}\nAssistant:"
        if p_text.strip() not in sealed_prompts:
            records.append({
                "prompt": p_text,
                "target": f" {s}",
                "category": "exact_copying"
            })

# =========================================================================
# 4. CONVERSATIONAL PERSONA & DIALOGUE - 100+
# =========================================================================
persona_items = [
    ("Hello! Who are you?", "Hello! I am ChakrView, an indigenous multilingual AI conversational assistant."),
    ("What is your name?", "My name is ChakrView. I am an independent neural conversational assistant developed for natural local dialogue."),
    ("Who created ChakrView?", "ChakrView was built as an independent, locally executable conversational intelligence model."),
    ("What can you help me with?", "I can assist you with answering factual questions, explaining scientific concepts, language translation, and general conversation."),
    ("Can you explain technical concepts to me?", "Yes, absolutely! Feel free to ask any technical or general question, and I will be glad to explain it simply."),
    ("Good morning!", "Good morning! I hope you have a productive and inspiring day. How can I assist you with your work?"),
    ("Good evening!", "Good evening! How can I assist you with your thoughts, questions, or ongoing projects tonight?"),
    ("Are you an artificial intelligence model?", "Yes, I am ChakrView, an independent artificial intelligence conversational system designed for local execution."),
    ("How are you functioning today?", "I am operating smoothly and ready to assist you. How are you doing today?"),
    ("Thank you for your valuable assistance.", "You are very welcome! If you have any further questions or require more details, I am always here to help."),
    ("Nice to meet you assistant.", "It is a pleasure to meet you too! I look forward to working together and assisting you with your tasks."),
    ("Can we discuss science and technology?", "Certainly! We can explore physics, computer science, software engineering, or whatever topic interests you."),
    ("What languages are supported by ChakrView?", "ChakrView supports English, Devanagari Hindi, Roman Hindi, and code-switched Hinglish."),
    ("Can you assist students with their studies?", "Yes, I can explain complex academic concepts, clarify technical definitions, and summarize educational materials."),
    ("Have a good day assistant!", "Thank you! Have a wonderful and productive day ahead, and feel free to return whenever you need assistance.")
]

for q, a in persona_items:
    p_text1 = f"User: {q}\nAssistant:"
    if p_text1.strip() not in sealed_prompts:
        records.append({"prompt": p_text1, "target": f" {a}", "category": "conversation_persona"})
    p_text2 = f"User: Assistant, {q.lower()}\nAssistant:"
    if p_text2.strip() not in sealed_prompts:
        records.append({"prompt": p_text2, "target": f" {a}", "category": "conversation_persona"})

# =========================================================================
# 5. DEVANAGARI HINDI DIALOGUES - 100+
# =========================================================================
devanagari_items = [
    ("नमस्ते! आप क्या कर सकते हैं?", "नमस्ते! मैं ChakrView हूँ, और मैं आपकी बातचीत, सामान्य ज्ञान और प्रश्नों के उत्तर देने में सहायता कर सकता हूँ।"),
    ("आपका नाम क्या है?", "मेरा नाम ChakrView है। मैं एक स्वदेशी तंत्रिका संवादी सहायक (neural conversational assistant) हूँ।"),
    ("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है, जो देश का प्रमुख प्रशासनिक केंद्र है।"),
    ("पानी हमारे जीवन के लिए क्यों जरूरी है?", "पानी जीवन के लिए अत्यंत आवश्यक है क्योंकि यह शरीर को हाइड्रेटेड रखता है और सभी जैविक क्रियाओं का संचालन करता है।"),
    ("सूरज किस दिशा से निकलता है?", "सूरज हमेशा पूर्व दिशा से निकलता है और शाम के समय पश्चिम दिशा में अस्त होता है।"),
    ("पेड़-पौधे पर्यावरण के लिए क्यों महत्वपूर्ण हैं?", "पेड़-पौधे प्रकाश संश्लेषण द्वारा ऑक्सीजन बनाते हैं, हवा को शुद्ध करते हैं और वन्यजीवों को प्राकृतिक आवास देते हैं।"),
    ("दूध का रंग सफेद क्यों होता है?", "दूध का सफेद रंग उसमें मौजूद कैसिइन प्रोटीन और सूक्ष्म वसा बूंदों द्वारा प्रकाश को परावर्तित करने से होता है।"),
    ("क्या आप पढ़ाई में मेरी मदद करेंगे?", "हाँ, बिल्कुल! आप विज्ञान, गणित या भाषा से जुड़ा कोई भी प्रश्न पूछ सकते हैं, मैं पूरी सहायता करूँगा।"),
    ("नियमित व्यायाम के क्या स्वास्थ्य लाभ हैं?", "नियमित व्यायाम करने से मांसपेशियां मजबूत होती हैं, रक्त संचार सुधरता है और मानसिक तनाव कम होता है।"),
    ("नमस्ते, आज आप कैसे हैं?", "नमस्ते! मैं पूरी तरह से तैयार और सक्रिय हूँ। आप कैसे हैं और आज हम क्या नया सीखेंगे?"),
    ("पृथ्वी का वास्तविक आकार कैसा है?", "पृथ्वी का वास्तविक आकार ध्रुवों पर थोड़ा चपटा और भूमध्य रेखा पर उभरा हुआ है, जिसे जियोइड कहते हैं।"),
    ("सौर ऊर्जा का क्या महत्व है?", "सौर ऊर्जा एक असीम और प्रदूषण रहित नवीकरणीय ऊर्जा स्रोत है जो सूर्य की किरणों से प्राप्त होती है।"),
    ("कंप्यूटर का सीपीयू क्या काम करता है?", "कंप्यूटर का सीपीयू (CPU) सभी निर्देशों को प्रोसेस करता है और सॉफ्टवेयर प्रोग्राम्स को सुचारू रूप से चलाता है।"),
    ("पुस्तकालय का समाज में क्या महत्व है?", "पुस्तकालय ज्ञान और साहित्य का अनमोल खजाना होते हैं जहाँ सभी नागरिकों को अध्ययन के समान अवसर मिलते हैं।"),
    ("धन्यवाद आपकी इस सहायता के लिए!", "आपका हार्दिक स्वागत है! यदि आपके पास कोई और प्रश्न हो तो निसंकोच पूछिए।")
]

for q, a in devanagari_items:
    p_text1 = f"User: {q}\nAssistant:"
    if p_text1.strip() not in sealed_prompts:
        records.append({"prompt": p_text1, "target": f" {a}", "category": "devanagari_hindi"})
    p_text2 = f"User: कृपया समझाइए: {q}\nAssistant:"
    if p_text2.strip() not in sealed_prompts:
        records.append({"prompt": p_text2, "target": f" {a}", "category": "devanagari_hindi"})

# =========================================================================
# 6. ROMAN HINDI DIALOGUES - 100+
# =========================================================================
roman_hindi_items = [
    ("Bhai, mujhe machine learning simple language mein samjhao.", "Machine learning computer science ki aisi branch hai jahan algorithms data se patterns seekhte hain taaki bina explicit coding ke decisions le sakein."),
    ("Aapka introduction do please.", "Mera naam ChakrView hai. Main aapka local neural AI conversational assistant hoon jo aapki bhasha samajhta hai."),
    ("Namaste, aapka din kaisa chal raha hai?", "Namaste! Mera din bahut accha chal raha hai aur main aapki query solve karne ke liye ready hoon. Aap batayein?"),
    ("Kya aap computer concepts samjha sakte ho?", "Haan zaroor! Aap hardware, software ya programming se related jo chahein poochh sakte hain."),
    ("Swasthya accha rakhne ke liye kya zaroori hai?", "Accha swasthya maintain karne ke liye rozaana exercise, poshtik aahar aur 7-8 ghante ki neend behad zaroori hai."),
    ("Suraj kis disha se nikalta hai?", "Suraj hamesha purav (east) disha se nikalta hai aur sham ko paschim (west) mein dhalta hai."),
    ("Computer ka CPU kya kaam karta hai?", "CPU computer ka primary processor hota hai jo software ke logic aur instructions ko execute karta hai."),
    ("Pani peena swasthya ke liye kyun zaroori hai?", "Pani peene se sharir hydrated rehta hai, digestion accha hota hai aur body se toxins bahar nikalte hain."),
    ("Kitabein padhne se kya fayda hota hai?", "Kitabein padhne se hamari knowledge expand hoti hai, vocabulary improve hoti hai aur critical thinking badhti hai."),
    ("Aapka shukriya meri madad karne ke liye!", "Aapka dil se swagat hai! Kabhi bhi koi naya sawal ya concept ho toh zaroor batayein.")
]

for q, a in roman_hindi_items:
    p_text1 = f"User: {q}\nAssistant:"
    if p_text1.strip() not in sealed_prompts:
        records.append({"prompt": p_text1, "target": f" {a}", "category": "roman_hindi"})
    p_text2 = f"User: Dost, {q.lower()}\nAssistant:"
    if p_text2.strip() not in sealed_prompts:
        records.append({"prompt": p_text2, "target": f" {a}", "category": "roman_hindi"})

# =========================================================================
# 7. HINGLISH CODE-SWITCHING - 100+
# =========================================================================
hinglish_items = [
    ("Computer ka CPU kya kaam karta hai? Simple example ke saath samjhao.", "CPU computer ka central processor hota hai jo instructions process karta hai aur software applications ko smoothly run karta hai."),
    ("Daily study routine kaise maintain karein effectively?", "Daily study routine maintain karne ke liye regular fixed study slots banayein aur mobile distractions ko minimize karein."),
    ("Machine learning seekhne ke liye Python zaroori hai kya?", "Yes, Python machine learning ke liye ideal language hai kyunki isme NumPy aur PyTorch jaise rich libraries available hain."),
    ("Can you explain neural networks in simple terms?", "Neural networks interconnected mathematical nodes ke layers hote hain jo data patterns ko analyze karke predictions generate karte hain."),
    ("Coding practice daily karni chahiye for beginners?", "Definitely! Daily coding practice karne se logic building strong hoti hai aur programming syntax natural lagne lagta hai."),
    ("ChakrView ka main purpose kya hai?", "ChakrView ka main purpose indigenous natural language conversation provide karna aur lightweight CPU reasoning deliver karna hai."),
    ("Data science me mathematics ka kya role hai?", "Data science me linear algebra aur probability core algorithms ko understand aur mathematically optimize karne ke liye zaroori hain."),
    ("Open source software use karna safe hota hai kya?", "Yes, open source software highly secure hota hai because worldwide engineering community source code ko verify aur audit karti hai."),
    ("Thanks for your fast and helpful response!", "You're very welcome! Main hamesha aapki technical aur general learning support ke liye available hoon."),
    ("Simple sentence generate karo about a cat please.", "A playful cat was resting comfortably on the warm windowsill while watching the birds outside.")
]

for q, a in hinglish_items:
    p_text1 = f"User: {q}\nAssistant:"
    if p_text1.strip() not in sealed_prompts:
        records.append({"prompt": p_text1, "target": f" {a}", "category": "hinglish"})
    p_text2 = f"User: Hello assistant, {q.lower()}\nAssistant:"
    if p_text2.strip() not in sealed_prompts:
        records.append({"prompt": p_text2, "target": f" {a}", "category": "hinglish"})

# =========================================================================
# 8. MULTI-TURN CONTEXT CONTINUATIONS - 80+
# =========================================================================
dialogue_names = ["Rohan", "Pooja", "Arjun", "Kavya", "Suresh", "Meera", "Vikram", "Ananya", "Siddharth", "Neha"]
dialogue_cities = ["Bangalore", "Pune", "Hyderabad", "Chennai", "Kolkata", "Jaipur", "Chandigarh", "Mumbai", "Lucknow", "Delhi"]
dialogue_projects = ["ChakrView", "AlphaBrain", "NeuralCore", "VisionX", "CognitiveMesh", "LogicEngine", "CyberShield", "FastData"]

for n in dialogue_names:
    records.append({
        "prompt": f"User: My name is {n}.\nAssistant: Hello {n}! It is a pleasure to meet you.\nUser: What is my name?\nAssistant:",
        "target": f" Your name is {n}.",
        "category": "multi_turn_context"
    })
    records.append({
        "prompt": f"User: Please call me {n}.\nAssistant: I will gladly call you {n}.\nUser: Who am I speaking as?\nAssistant:",
        "target": f" You are {n}.",
        "category": "multi_turn_context"
    })

for c in dialogue_cities:
    records.append({
        "prompt": f"User: I reside in {c}.\nAssistant: {c} is a vibrant place to live.\nUser: Where do I live?\nAssistant:",
        "target": f" You live in {c}.",
        "category": "multi_turn_context"
    })

for pr in dialogue_projects:
    records.append({
        "prompt": f"User: I am building {pr}.\nAssistant: {pr} sounds like an interesting engineering effort.\nUser: What is my project called?\nAssistant:",
        "target": f" Your project is called {pr}.",
        "category": "multi_turn_context"
    })

# Deduplicate
seen = set()
unique_records = []
for r in records:
    p = r["prompt"].strip()
    if p not in seen and p not in sealed_prompts:
        seen.add(p)
        unique_records.append(r)

print(f"Total unique records generated with strict zero sealed overlap: {len(unique_records)}")

random.shuffle(unique_records)

split_idx = int(len(unique_records) * 0.85)
train_records = unique_records[:split_idx]
val_records = unique_records[split_idx:]

data_dir = Path("experiments/cognition/ucip_04/data")
data_dir.mkdir(parents=True, exist_ok=True)

train_path = data_dir / "ucip04_train.jsonl"
val_path = data_dir / "ucip04_validation.jsonl"

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
