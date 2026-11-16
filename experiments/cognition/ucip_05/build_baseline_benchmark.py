import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import json
import hashlib

# Build 300+ balanced probes across all required categories
probes = []

def add_probe(probe_id, category, language, input_text, expected_keywords, rubric, exact_match=False):
    probes.append({
        "probe_id": probe_id,
        "category": category,
        "language": language,
        "input": input_text,
        "expected_keywords": expected_keywords,
        "rubric": rubric,
        "exact_match": exact_match
    })

# 1. English Greetings & Conversation (40 probes)
english_greetings = [
    ("Hello ChakrView. How are you?", ["hello", "good", "well", "fine", "assist", "help"], "Polite conversational greeting"),
    ("Good morning! How are you today?", ["morning", "good", "well", "fine", "help"], "Morning greeting"),
    ("Good evening! Nice to meet you.", ["evening", "nice", "welcome", "meet"], "Evening greeting"),
    ("Hi there! What can you help me with?", ["help", "assist", "questions", "topics", "learn"], "Capability overview"),
    ("Who are you?", ["chakrview", "assistant", "model", "language"], "Persona introduction"),
    ("What is your name?", ["chakrview", "name"], "Name identification"),
    ("Who created you?", ["chakrview", "built", "developed", "independent"], "Creation background"),
    ("Are you an artificial intelligence?", ["yes", "ai", "artificial intelligence", "model", "assistant"], "AI identification"),
    ("Can you speak multiple languages?", ["yes", "languages", "english", "hindi", "hinglish"], "Language support"),
    ("What makes you unique?", ["local", "cpu", "independent", "neural", "lightweight"], "Architecture distinctiveness"),
    ("Thank you for your assistance!", ["welcome", "pleasure", "glad", "help"], "Gratitude response"),
    ("I appreciate your quick reply.", ["welcome", "happy", "assist", "glad"], "Gratitude response"),
    ("How does your day look?", ["ready", "assist", "good", "well", "help"], "Conversational engagement"),
    ("It is nice chatting with you.", ["pleasure", "nice", "glad", "welcome"], "Pleasantry"),
    ("Can we discuss science and technology?", ["yes", "discuss", "explore", "science", "technology"], "Affirmative agreement"),
    ("I want to learn programming.", ["great", "python", "code", "start", "learn", "programming"], "Encouragement and guidance"),
    ("Do you get tired after many questions?", ["no", "machine", "ready", "assist", "tired"], "Machine nature clarification"),
    ("Where do you run?", ["locally", "cpu", "computer", "system", "hardware"], "Local execution explanation"),
    ("Can you help me write clean code?", ["yes", "clean", "code", "practice", "help", "structure"], "Coding assistance"),
    ("What is your favorite topic?", ["learning", "science", "technology", "language", "questions"], "Topic preference"),
    ("Hello! Are you ready for some questions?", ["yes", "ready", "ask", "assist"], "Readiness confirmation"),
    ("Good afternoon! Hope you are having a productive day.", ["afternoon", "productive", "thank", "assist"], "Afternoon greeting"),
    ("Tell me something interesting.", ["did you know", "interesting", "earth", "sun", "space", "science", "computers"], "Interesting fact sharing"),
    ("Can you explain difficult concepts simply?", ["yes", "simple", "clear", "explain", "help"], "Pedagogical commitment"),
    ("How fast can you generate answers?", ["fast", "cpu", "efficient", "tokens", "quick"], "Performance description"),
    ("Are you connected to the internet right now?", ["no", "local", "offline", "independent", "system"], "Local offline clarification"),
    ("What should I call you?", ["chakrview", "assistant"], "Name clarification"),
    ("I am happy to test your capabilities.", ["glad", "welcome", "test", "ready", "happy"], "Polite engagement"),
    ("Let us have a productive session.", ["sure", "ready", "start", "productive", "happy"], "Session kickoff"),
    ("Can you keep conversations polite and respectful?", ["yes", "polite", "respectful", "always", "courteous"], "Tone commitment"),
    ("How do you process information?", ["neural", "tokens", "transformer", "attention", "weights"], "Processing overview"),
    ("Do you store personal data permanently?", ["no", "privacy", "local", "session", "safe"], "Privacy assurance"),
    ("Can you assist with everyday questions?", ["yes", "assist", "everyday", "questions", "general"], "General assistance"),
    ("Is ChakrView open source and locally executable?", ["yes", "local", "independent", "executable", "open"], "Open execution nature"),
    ("Tell me about your design philosophy.", ["local", "cpu", "efficient", "independent", "lightweight"], "Design values"),
    ("What are you capable of doing right now?", ["conversation", "explanations", "hindi", "arithmetic", "copying", "assistance"], "Current capabilities"),
    ("Can we practice language translation?", ["yes", "translate", "languages", "hindi", "english"], "Translation capability"),
    ("I need some motivation today.", ["believe", "effort", "progress", "step", "achieve", "success"], "Motivational encouragement"),
    ("Goodbye! Talk to you later.", ["goodbye", "farewell", "bye", "take care", "later"], "Farewell response"),
    ("Have a great evening ahead!", ["thank", "you", "too", "great", "evening"], "Polite closing")
]
for idx, (inp, kw, rub) in enumerate(english_greetings, 1):
    add_probe(f"ENG_CONV_{idx:03d}", "English Conversation", "English", inp, kw, rub)

# 2. English Simple Explanations (40 probes)
english_explanations = [
    ("What does a CPU do?", ["processor", "instructions", "executes", "arithmetic", "memory", "computer"], "Accurate CPU explanation"),
    ("Explain how computer memory RAM works.", ["ram", "memory", "volatile", "storage", "fast", "temporary"], "RAM explanation"),
    ("What is an operating system?", ["manages", "hardware", "software", "system", "resources", "processes"], "OS explanation"),
    ("Why is the sky blue?", ["atmosphere", "sunlight", "scatter", "blue", "light", "molecules"], "Rayleigh scattering explanation"),
    ("What is photosynthesis?", ["plants", "sunlight", "energy", "convert", "chlorophyll", "oxygen", "glucose"], "Photosynthesis explanation"),
    ("How does an electric motor work?", ["electrical", "magnetic", "mechanical", "energy", "motion", "rotation"], "Electric motor function"),
    ("What is gravity?", ["force", "attraction", "mass", "earth", "gravity", "pull"], "Gravity explanation"),
    ("What is a compiler?", ["translates", "source code", "machine", "binary", "executable", "program"], "Compiler purpose"),
    ("Explain what machine learning is.", ["learn", "data", "patterns", "algorithms", "predict", "training"], "Machine learning definition"),
    ("What is a database?", ["collection", "data", "organized", "storage", "tables", "queries"], "Database definition"),
    ("How does a bicycle work?", ["pedals", "wheels", "chain", "gear", "human", "propel"], "Bicycle mechanism"),
    ("What is the solar system?", ["sun", "planets", "orbit", "solar", "system", "gravity"], "Solar system description"),
    ("What causes rain?", ["water", "evaporation", "clouds", "condensation", "drops", "atmosphere"], "Rain formation"),
    ("What is an algorithm?", ["step", "sequence", "instructions", "solve", "problem", "rules"], "Algorithm definition"),
    ("How does the human heart function?", ["pumps", "blood", "oxygen", "circulates", "body", "muscle"], "Heart function"),
    ("What is renewable energy?", ["solar", "wind", "renewable", "replenished", "clean", "sustainable"], "Renewable energy explanation"),
    ("What is a web browser?", ["software", "internet", "web", "pages", "renders", "html"], "Browser function"),
    ("Explain the difference between software and hardware.", ["hardware", "physical", "software", "programs", "instructions"], "Hardware vs software distinction"),
    ("What is a black hole?", ["gravity", "dense", "space", "light", "cannot", "escape"], "Black hole explanation"),
    ("How does a thermometer measure temperature?", ["temperature", "heat", "liquid", "sensor", "measures", "expansion"], "Thermometer mechanism"),
    ("What is a neural network?", ["layers", "nodes", "neurons", "weights", "learn", "artificial"], "Neural network explanation"),
    ("What is open source software?", ["source code", "freely", "available", "modify", "share", "community"], "Open source definition"),
    ("How do airplanes fly?", ["wings", "lift", "air", "pressure", "thrust", "aerodynamic"], "Flight aerodynamics"),
    ("What is an API?", ["interface", "application", "communicate", "software", "services", "exchange"], "API definition"),
    ("What is DNA?", ["genetic", "instructions", "living", "cells", "hereditary", "organism"], "DNA definition"),
    ("Why do we need sleep?", ["rest", "brain", "body", "recovery", "health", "energy"], "Importance of sleep"),
    ("What is sound?", ["vibrations", "waves", "air", "ears", "frequency", "travels"], "Physics of sound"),
    ("What is cloud computing?", ["servers", "internet", "remote", "storage", "computing", "on-demand"], "Cloud computing"),
    ("What is a microcontroller?", ["compact", "integrated", "chip", "controls", "embedded", "system"], "Microcontroller definition"),
    ("Explain how a refrigerator keeps food cold.", ["refrigerant", "heat", "compresses", "cools", "absorbs", "inside"], "Refrigeration mechanism"),
    ("What is an atom?", ["basic", "unit", "matter", "protons", "electrons", "neutrons"], "Atom definition"),
    ("What is electricity?", ["flow", "electrons", "charge", "current", "energy", "circuits"], "Electricity definition"),
    ("What is computer cache?", ["fast", "temporary", "memory", "cpu", "frequent", "stores"], "Cache purpose"),
    ("How does GPS work?", ["satellites", "location", "signals", "triangulation", "time", "position"], "GPS explanation"),
    ("What is a firewall in computer networks?", ["security", "network", "blocks", "traffic", "unauthorized", "protects"], "Firewall explanation"),
    ("What is the function of the human lungs?", ["breathe", "oxygen", "carbon dioxide", "blood", "gas exchange"], "Lungs function"),
    ("What causes earthquakes?", ["tectonic", "plates", "fault", "movement", "crust", "seismic", "energy"], "Earthquake causes"),
    ("What is an operating system kernel?", ["core", "kernel", "manages", "hardware", "processes", "memory"], "Kernel role"),
    ("Explain what a vector is in mathematics.", ["magnitude", "direction", "arrow", "quantity", "dimensions"], "Vector definition"),
    ("Why do seasons change on Earth?", ["tilt", "axis", "orbit", "sun", "sunlight", "hemisphere"], "Seasons explanation")
]
for idx, (inp, kw, rub) in enumerate(english_explanations, 1):
    add_probe(f"ENG_EXP_{idx:03d}", "English Simple Explanations", "English", inp, kw, rub)

# 3. English Sentence Generation (30 probes)
english_generation = [
    ("Write one simple sentence about a dog.", ["dog", "animal", "friendly", "pet", "loves", "home"], "Dog description"),
    ("Write one descriptive sentence about the sun.", ["sun", "star", "bright", "solar", "light", "energy"], "Sun description"),
    ("Generate a short sentence about an apple.", ["apple", "fruit", "sweet", "red", "healthy", "tree"], "Apple description"),
    ("Write a sentence describing a quiet library.", ["library", "quiet", "books", "students", "reading", "study"], "Library description"),
    ("Write a descriptive sentence about a bicycle.", ["bicycle", "wheels", "ride", "pedal", "cycle", "travel"], "Bicycle description"),
    ("Create a sentence about a fast train.", ["train", "speed", "tracks", "travel", "fast", "passengers"], "Train description"),
    ("Write a simple sentence about the ocean.", ["ocean", "water", "vast", "deep", "waves", "sea"], "Ocean description"),
    ("Generate a sentence describing a tall mountain.", ["mountain", "tall", "peak", "snow", "climb", "high"], "Mountain description"),
    ("Write a sentence about drinking clean water.", ["water", "clean", "healthy", "drink", "essential", "body"], "Water sentence"),
    ("Write a simple sentence about a friendly cat.", ["cat", "pet", "purrs", "fur", "gentle", "friendly"], "Cat sentence"),
    ("Write one sentence about stars in the night sky.", ["stars", "night", "sky", "twinkle", "shine", "bright"], "Stars sentence"),
    ("Generate a sentence about planting a green tree.", ["tree", "plant", "green", "grow", "nature", "shade"], "Tree planting"),
    ("Write a sentence describing an honest student.", ["student", "honest", "studies", "hard", "learns", "school"], "Student sentence"),
    ("Write one sentence about eating fresh fruits.", ["fruits", "fresh", "healthy", "vitamins", "delicious", "eat"], "Fresh fruits"),
    ("Create a sentence about reading good books.", ["books", "reading", "knowledge", "learn", "mind", "stories"], "Reading books"),
    ("Write a descriptive sentence about morning sunlight.", ["sunlight", "morning", "bright", "warmth", "day", "shines"], "Morning sunlight"),
    ("Write a sentence describing a calm forest.", ["forest", "calm", "trees", "peaceful", "nature", "birds"], "Calm forest"),
    ("Generate a sentence about an electric car.", ["electric", "car", "battery", "efficient", "drives", "clean"], "Electric car"),
    ("Write one sentence about a clock on the wall.", ["clock", "wall", "time", "ticks", "measures", "seconds"], "Clock sentence"),
    ("Write a descriptive sentence about a bird singing.", ["bird", "singing", "morning", "melodious", "tree", "song"], "Bird singing"),
    ("Write a sentence describing walking in the rain.", ["walking", "rain", "umbrella", "drops", "wet", "cool"], "Walking in rain"),
    ("Generate a simple sentence about a computer keyboard.", ["keyboard", "keys", "typing", "input", "computer", "letters"], "Keyboard sentence"),
    ("Write one sentence about an artist painting a canvas.", ["artist", "painting", "colors", "canvas", "brush", "art"], "Artist painting"),
    ("Write a sentence about baking fresh bread.", ["bread", "baking", "oven", "warm", "fresh", "kitchen"], "Baking bread"),
    ("Create a descriptive sentence about a flowing river.", ["river", "flowing", "water", "stream", "peaceful", "valley"], "Flowing river"),
    ("Write a sentence describing a bright full moon.", ["moon", "bright", "full", "night", "glowing", "sky"], "Full moon"),
    ("Write one sentence about learning a new skill.", ["learning", "skill", "practice", "patience", "growth", "mastery"], "Skill learning"),
    ("Generate a sentence about listening to soft music.", ["music", "soft", "calm", "listening", "relaxing", "melody"], "Soft music"),
    ("Write a sentence describing working in a team.", ["team", "collaboration", "work", "together", "goals", "success"], "Teamwork sentence"),
    ("Write a simple sentence about children playing.", ["children", "playing", "park", "joy", "laughing", "games"], "Children playing")
]
for idx, (inp, kw, rub) in enumerate(english_generation, 1):
    add_probe(f"ENG_GEN_{idx:03d}", "English Sentence Generation", "English", inp, kw, rub)

# 4. Devanagari Hindi (40 probes)
devanagari_hindi = [
    ("नमस्ते! आप क्या कर सकते हैं?", ["नमस्ते", "सहायता", "कर", "सकता", "सीखने", "प्रश्न", "उत्तर"], "Devanagari greeting and capability"),
    ("आप कैसे हैं?", ["मैं", "ठीक", "हूँ", "धन्यवाद", "आप"], "Polite status reply"),
    ("कंप्यूटर क्या होता है?", ["कंप्यूटर", "इलेक्ट्रॉनिक", "उपकरण", "डेटा", "निर्देश", "गणना"], "Computer definition in Hindi"),
    ("सूर्य हमारे लिए क्यों महत्वपूर्ण है?", ["सूर्य", "प्रकाश", "ऊर्जा", "पृथ्वी", "जीवन", "धूप"], "Significance of the Sun"),
    ("पानी हमारे जीवन के लिए क्यों जरूरी है?", ["पानी", "जल", "जीवन", "शरीर", "आवश्यक", "स्वास्थ्य"], "Water importance"),
    ("पेड़ों के क्या लाभ होते हैं?", ["पेड़", "ऑक्सीजन", "छाया", "फल", "पर्यावरण", "शुद्ध"], "Benefits of trees"),
    ("भारत की राजधानी क्या है?", ["भारत", "राजधानी", "नई दिल्ली", "दिल्ली"], "Capital of India"),
    ("शिक्षा का हमारे जीवन में क्या महत्व है?", ["शिक्षा", "ज्ञान", "जीवन", "सफलता", "विकास", "भविष्य"], "Importance of education"),
    ("समय का सदुपयोग क्यों जरूरी है?", ["समय", "अनमोल", "सफलता", "सदुपयोग", "महत्वपूर्ण"], "Time management"),
    ("पुस्तकों से हमें क्या मिलता है?", ["पुस्तक", "किताब", "ज्ञान", "सीख", "मित्र", "जानकारी"], "Value of books"),
    ("व्यायाम करने से शरीर को क्या फायदा होता है?", ["व्यायाम", "शरीर", "स्वस्थ", "मजबूत", "तंदुरुस्त", "ऊर्जा"], "Exercise benefits"),
    ("ईमानदारी क्यों आवश्यक है?", ["ईमानदारी", "सच्चाई", "विश्वास", "गुण", "सम्मान"], "Honesty value"),
    ("स्वास्थ्य का ध्यान कैसे रखना चाहिए?", ["संतुलित", "आहार", "व्यायाम", "स्वच्छता", "नींद", "स्वास्थ्य"], "Health care tips"),
    ("विज्ञान ने मानव जीवन को कैसे बदला है?", ["विज्ञान", "तकनीक", "सरल", "सुविधा", "उन्नति", "आविष्कार"], "Impact of science"),
    ("इंटरनेट क्या है?", ["इंटरनेट", "नेटवर्क", "सूचना", "कंप्यूटर", "वैश्विक", "संचार"], "Internet explanation"),
    ("एक अच्छे मित्र में क्या गुण होने चाहिए?", ["सच्चा", "सहायक", "ईमानदार", "विश्वास", "मित्रता"], "Good friend qualities"),
    ("सफलता पाने के लिए क्या जरूरी है?", ["मेहनत", "लगन", "धैर्य", "लक्ष्य", "प्रयास"], "Key to success"),
    ("नदियों का पर्यावरण में क्या योगदान है?", ["नदियां", "जल", "उपजाऊ", "प्रकृति", "पर्यावरण", "जीवन"], "Rivers contribution"),
    ("ऋतुएं क्यों बदलती हैं?", ["पृथ्वी", "सूर्य", "परिक्रमा", "झुकाव", "ऋतुएं", "मौसम"], "Why seasons change"),
    ("पर्यावरण प्रदूषण कैसे रोका जा सकता है?", ["पेड़", "सफाई", "प्रदूषण", "रोकथाम", "रीसायकल", "जागरूकता"], "Preventing pollution"),
    ("सौरमंडल में कितने ग्रह हैं?", ["सौरमंडल", "आठ", "ग्रह", "सूर्य"], "Solar system planets count"),
    ("चंद्रमा क्या है?", ["चंद्रमा", "उपग्रह", "पृथ्वी", "रात", "प्रकाश"], "Moon description"),
    ("हिंदी दिवस कब मनाया जाता है?", ["14 सितंबर", "चौदह", "सितंबर", "हिंदी दिवस"], "Hindi Diwas date"),
    ("हवा हमारे लिए क्यों जरूरी है?", ["हवा", "वायु", "ऑक्सीजन", "सांस", "जीवित"], "Air importance"),
    ("कृषि का देश की अर्थव्यवस्था में क्या महत्व है?", ["कृषि", "अन्न", "किसान", "अर्थव्यवस्था", "भोजन"], "Agriculture importance"),
    ("योग का क्या महत्व है?", ["योग", "मानसिक", "शारीरिक", "शांति", "स्वास्थ्य"], "Yoga significance"),
    ("धैर्य रखना क्यों जरूरी है?", ["धैर्य", "शांति", "कठिन", "निर्णय", "सफलता"], "Patience value"),
    ("सफाई रखने से क्या लाभ होते हैं?", ["स्वच्छता", "सफाई", "बीमारियां", "स्वस्थ", "वातावरण"], "Cleanliness benefits"),
    ("बिजली कैसे बनती है?", ["जल", "कोयला", "सौर", "पवन", "ऊर्जा", "बिजली"], "Electricity generation"),
    ("मोबाइल फोन के क्या लाभ हैं?", ["मोबाइल", "संचार", "बातचीत", "सूचना", "काम", "सुविधा"], "Mobile phone advantages"),
    ("जंगल वन्यजीवों के लिए क्यों जरूरी हैं?", ["जंगल", "वन", "जीव", "आवास", "प्रकृति", "संरक्षण"], "Forests and wildlife"),
    ("माता-पिता का सम्मान क्यों करना चाहिए?", ["माता-पिता", "सम्मान", "संस्कार", "प्रेम", "मार्गदर्शन"], "Respect for parents"),
    ("कठिन परिश्रम का कोई विकल्प क्यों नहीं है?", ["परिश्रम", "मेहनत", "सफलता", "लगन", "परिणाम"], "Hard work necessity"),
    ("ध्वनि कैसे उत्पन्न होती है?", ["कंपन", "ध्वनि", "तरंगें", "माध्यम", "आवाज"], "Sound generation in Hindi"),
    ("अनुशासन का छात्र जीवन में क्या महत्व है?", ["अनुशासन", "छात्र", "समय", "नियम", "सफलता", "भविष्य"], "Discipline in student life"),
    ("प्रकाश की गति कितनी होती है?", ["प्रकाश", "गति", "लाख", "किलोमीटर", "सेकंड", "300000"], "Speed of light"),
    ("आहार में हरी सब्जियों का क्या महत्व है?", ["हरी सब्जियां", "विटामिन", "पोषण", "स्वास्थ्य", "खनिज"], "Vegetables nutrition"),
    ("पुस्तकालय का क्या उपयोग है?", ["पुस्तकालय", "किताबें", "अध्ययन", "शांत", "ज्ञान"], "Library utility"),
    ("सड़क सुरक्षा नियमों का पालन क्यों जरूरी है?", ["सड़क", "सुरक्षा", "नियम", "दुर्घटना", "बचाव", "जीवन"], "Road safety rules"),
    ("नमस्ते! क्या आप मुझे हिंदी सिखा सकते हैं?", ["हाँ", "हिंदी", "सीखने", "अभ्यास", "सहायता"], "Willingness to teach Hindi")
]
for idx, (inp, kw, rub) in enumerate(devanagari_hindi, 1):
    add_probe(f"DEV_HIN_{idx:03d}", "Devanagari Hindi", "Devanagari Hindi", inp, kw, rub)

# 5. Roman Hindi (30 probes)
roman_hindi = [
    ("Aap kaise hain?", ["main", "theek", "hoon", "aap", "kaise"], "Status in Roman Hindi"),
    ("Aapka naam kya hai?", ["mera", "naam", "chakrview", "hai"], "Name query"),
    ("Computer ka CPU kya kaam karta hai?", ["cpu", "instructions", "execute", "karta", "processor", "hai"], "CPU function"),
    ("Bhai, machine learning ko simple language mein samjhao.", ["machine learning", "data", "patterns", "learn", "karta", "hai"], "ML in Roman Hindi"),
    ("RAM aur ROM mein kya antar hai?", ["ram", "temporary", "volatile", "rom", "permanent", "storage"], "RAM vs ROM"),
    ("Aaj ka mausam kaisa lag raha hai?", ["mausam", "accha", "suhana", "hai", "weather"], "Weather casual chat"),
    ("ChakrView model kis purpose ke liye banaya gaya hai?", ["chakrview", "local", "cpu", "conversational", "model", "hai"], "Model purpose"),
    ("Python language seekhna zaroori kyun hai?", ["python", "programming", "easy", "libraries", "useful", "hai"], "Why learn Python"),
    ("Kya aap mere questions ka answer de sakte ho?", ["haan", "main", "questions", "answer", "de", "sakta", "hoon"], "Affirmation"),
    ("Internet kaise connect hota hai?", ["network", "router", "servers", "connect", "data", "internet"], "Internet connection"),
    ("Subah jaldi uthne ke kya fayde hain?", ["subah", "fayde", "health", "energy", "fresh", "din"], "Early waking benefits"),
    ("Clean code likhna kyun zaroori hai?", ["clean", "code", "maintain", "readable", "bugs", "easy"], "Clean code value"),
    ("Hard work aur smart work mein kya difference hai?", ["hard work", "smart work", "strategy", "mehnat", "planning"], "Hard vs smart work"),
    ("Database kya kaam karta hai?", ["database", "data", "store", "organize", "tables", "queries"], "Database role"),
    ("Software testing kyun ki jaati hai?", ["testing", "bugs", "quality", "verify", "software", "check"], "Software testing"),
    ("Operating system ka role kya hota hai?", ["os", "hardware", "software", "manage", "resources", "coordinate"], "OS role"),
    ("Data structures seekhna kyun important hai?", ["data structures", "algorithms", "efficient", "problem", "solving", "code"], "DS importance"),
    ("Gym ya exercise karne se kya fayda hota hai?", ["exercise", "body", "fit", "strong", "energy", "health"], "Gym benefits"),
    ("Kitabein padhne se kya seekhne ko milta hai?", ["kitabein", "books", "knowledge", "ideas", "mind", "perspective"], "Books benefits"),
    ("API kya hoti hai simple shabdon mein batao.", ["api", "software", "communicate", "interface", "data", "exchange"], "API simple definition"),
    ("Cloud storage ka kya fayda hota hai?", ["cloud", "storage", "backup", "anywhere", "access", "files"], "Cloud storage benefits"),
    ("Debugging ka kya matlab hota hai?", ["bugs", "errors", "find", "fix", "code", "problem"], "Debugging meaning"),
    ("Team work se project par kya asar padta hai?", ["team", "collaboration", "fast", "ideas", "success", "together"], "Teamwork impact"),
    ("Ek acche programmer ki kya qualities hoti hain?", ["problem solving", "logic", "patience", "clean code", "learning"], "Programmer qualities"),
    ("Solar energy environment ke liye kyun acchi hai?", ["solar", "clean", "renewable", "pollution", "green", "energy"], "Solar energy"),
    ("Water drink karna health ke liye kyun zaroori hai?", ["water", "hydration", "body", "health", "energy", "essential"], "Hydration importance"),
    ("Time management kaise improve karein?", ["planning", "schedule", "priority", "focus", "goals", "time"], "Time management tips"),
    ("Open source software use karne ke kya advantages hain?", ["open source", "free", "community", "transparent", "modify"], "Open source perks"),
    ("Kya bina GPU ke AI run ho sakta hai?", ["haan", "cpu", "local", "efficient", "lightweight", "run"], "CPU AI execution"),
    ("Shukriya bhai, aapka explanation helpful tha.", ["welcome", "pleasure", "glad", "help", "shukriya"], "Gratitude response")
]
for idx, (inp, kw, rub) in enumerate(roman_hindi, 1):
    add_probe(f"ROM_HIN_{idx:03d}", "Roman Hindi", "Roman Hindi", inp, kw, rub)

# 6. Hinglish (40 probes)
hinglish_probes = [
    ("Machine learning simple words mein explain karo.", ["machine learning", "data", "patterns", "algorithms", "predictions", "train"], "ML in Hinglish"),
    ("Computer ka CPU kaise function karta hai?", ["cpu", "instructions", "memory", "fetch", "execute", "process"], "CPU function in Hinglish"),
    ("Neural network weights training ke time kaise update hote hain?", ["gradient descent", "backpropagation", "loss", "weights", "update"], "Weights update in Hinglish"),
    ("ChakrView ka main purpose kya hai?", ["chakrview", "indigenous", "local", "conversational", "cpu", "intelligence"], "ChakrView purpose"),
    ("Python list comprehension code ko kaise simplify karti hai?", ["list comprehension", "clean", "compact", "readable", "loop", "syntax"], "List comprehension in Hinglish"),
    ("Docker container aur virtual machine mein kya main difference hai?", ["docker", "container", "lightweight", "os kernel", "vm", "virtual machine"], "Docker vs VM"),
    ("Git rebase aur git merge mein kya distinction hoti hai?", ["git merge", "rebase", "history", "commit", "branch"], "Git rebase vs merge"),
    ("Frontend aur backend development mein kya difference hota hai?", ["frontend", "ui", "user interface", "backend", "server", "database"], "Frontend vs Backend"),
    ("RAM memory volatile kyun hoti hai?", ["power", "volatile", "data", "lost", "temporary", "ram"], "RAM volatile nature"),
    ("Compiler aur interpreter mein kya execution difference hai?", ["compiler", "translates", "interpreter", "line by line", "execute"], "Compiler vs Interpreter"),
    ("Object oriented programming ke core concepts kya hain?", ["classes", "objects", "inheritance", "encapsulation", "polymorphism"], "OOP concepts"),
    ("Database indexing queries ko fast kaise banati hai?", ["index", "lookup", "b-tree", "search", "fast", "query"], "Indexing explanation"),
    ("Continuous integration pipeline software quality kaise improve karti hai?", ["ci", "pipeline", "tests", "automated", "build", "quality"], "CI pipeline"),
    ("Recursion concept ko real world example ke saath explain karo.", ["recursion", "function", "calls itself", "base case", "subproblem"], "Recursion in Hinglish"),
    ("Unit tests likhne ka main advantage kya hota hai?", ["unit tests", "bugs", "verify", "regression", "code quality", "isolated"], "Unit testing benefits"),
    ("Microservices architecture monolithic app se kaise better hoti hai?", ["microservices", "modular", "independent", "services", "scalable"], "Microservices vs Monolith"),
    ("Operating system processes aur threads mein kya relation hota hai?", ["process", "thread", "memory", "execution", "shared", "lightweight"], "Processes vs Threads"),
    ("Code refactoring software performance aur readability ko kaise boost karta hai?", ["refactoring", "clean code", "readability", "maintainable", "structure"], "Code refactoring"),
    ("Cache memory CPU speed ko kaise enhance karti hai?", ["cache", "l1", "l2", "fast", "cpu", "latency", "frequent"], "Cache memory"),
    ("Public key cryptography security kaise ensure karti hai?", ["public key", "private key", "encryption", "decryption", "secure"], "Asymmetric cryptography"),
    ("REST API design best practices kya hoti hain?", ["endpoints", "http methods", "stateless", "json", "status codes", "clean"], "REST API practices"),
    ("Memory leak program performance ko kaise degrade karta hai?", ["memory leak", "unreleased", "ram", "consumption", "crash", "slow"], "Memory leaks"),
    ("Bhai, algorithms ki time complexity calculate karna kyun important hai?", ["time complexity", "big o", "performance", "efficiency", "scale", "input"], "Time complexity importance"),
    ("SQL aur NoSQL databases kis use case ke liye choose karne chahiye?", ["sql", "relational", "structured", "nosql", "documents", "flexible"], "SQL vs NoSQL"),
    ("Cloud computing local server management ke comparison mein kya edge deti hai?", ["cloud", "scalable", "cost", "on-demand", "maintenance", "remote"], "Cloud advantages"),
    ("Multithreading execution application speed ko kaise boost karti hai?", ["multithreading", "concurrent", "cores", "parallel", "execution", "speed"], "Multithreading"),
    ("Code review process team learning ko kaise strengthen karta hai?", ["code review", "quality", "feedback", "standards", "bugs", "knowledge"], "Code review"),
    ("DNS system domain name ko IP address mein kaise resolve karta hai?", ["dns", "domain", "ip address", "lookup", "resolve", "nameserver"], "DNS resolution"),
    ("Asynchronous programming synchronous blocking calls se kaise different hai?", ["async", "non-blocking", "event loop", "io", "concurrent"], "Asynchronous execution"),
    ("Hash map key-value lookup constant time mein kaise provide karta hai?", ["hash map", "hash function", "bucket", "constant time", "o(1)"], "Hash map O(1) lookup"),
    ("Data normalization relational databases mein redundant data kaise avoid karti hai?", ["normalization", "redundancy", "tables", "integrity", "keys"], "Data normalization"),
    ("Cyber security mein phishing attacks se protect kaise rahein?", ["phishing", "emails", "links", "verification", "security", "passwords"], "Phishing prevention"),
    ("Virtual memory physical RAM limit se larger programs kaise run karti hai?", ["virtual memory", "paging", "swap", "disk", "address space", "ram"], "Virtual memory"),
    ("Load balancer incoming traffic distribute karke server crash kaise prevent karta hai?", ["load balancer", "traffic", "servers", "distribution", "health check", "scale"], "Load balancer"),
    ("Data structure stack LIFO principle par kaise operate karta hai?", ["stack", "lifo", "last in first out", "push", "pop"], "Stack LIFO"),
    ("Queue FIFO queueing mechanism real world systems mein kahan use hota hai?", ["queue", "fifo", "first in first out", "tasks", "messaging"], "Queue FIFO"),
    ("Garbage collection runtime environment mein automatic memory cleanup kaise karti hai?", ["garbage collection", "memory", "cleanup", "unreferenced", "objects", "automatic"], "Garbage collection"),
    ("Stateless architecture horizontally scale karna kyun easy banati hai?", ["stateless", "session", "servers", "scale", "independent", "requests"], "Stateless scaling"),
    ("Dependency injection software components ke beech coupling kaise reduce karti hai?", ["dependency injection", "coupling", "modular", "testing", "flexible"], "Dependency injection"),
    ("ChakrView ka CPU-first approach deployment overhead kaise cut down karta hai?", ["cpu-first", "lightweight", "no gpu needed", "local", "cost effective", "efficient"], "CPU-first efficiency")
]
for idx, (inp, kw, rub) in enumerate(hinglish_probes, 1):
    add_probe(f"HING_{idx:03d}", "Hinglish", "Hinglish", inp, kw, rub)

# 7. Exact Copying (30 probes)
exact_copying = [
    ("Say exactly: The sky is blue.", ["The sky is blue."], "Exact match copy", True),
    ("Repeat verbatim: Python is an elegant language.", ["Python is an elegant language."], "Exact copy", True),
    ("Echo the text: Clean code prevents bugs.", ["Clean code prevents bugs."], "Exact copy", True),
    ("Copy this sentence: Sun rises in the east.", ["Sun rises in the east."], "Exact copy", True),
    ("Say exactly: Water is essential for all life.", ["Water is essential for all life."], "Exact copy", True),
    ("Repeat exactly: Reading books expands your mind.", ["Reading books expands your mind."], "Exact copy", True),
    ("Echo verbatim: Technology connects people globally.", ["Technology connects people globally."], "Exact copy", True),
    ("Copy the following: Keep moving forward every day.", ["Keep moving forward every day."], "Exact copy", True),
    ("Say exactly: Practice makes human skills sharper.", ["Practice makes human skills sharper."], "Exact copy", True),
    ("Repeat verbatim: Simplicity is prerequisite for reliability.", ["Simplicity is prerequisite for reliability."], "Exact copy", True),
    ("Echo exactly: Fast computers need efficient algorithms.", ["Fast computers need efficient algorithms."], "Exact copy", True),
    ("Copy verbatim: Knowledge brings clarity and confidence.", ["Knowledge brings clarity and confidence."], "Exact copy", True),
    ("Say exactly: The earth orbits around the sun.", ["The earth orbits around the sun."], "Exact copy", True),
    ("Repeat exactly: Stars shine brightly in the night.", ["Stars shine brightly in the night."], "Exact copy", True),
    ("Echo verbatim: Honest efforts yield lasting success.", ["Honest efforts yield lasting success."], "Exact copy", True),
    ("Copy this text: Trees produce oxygen and shade.", ["Trees produce oxygen and shade."], "Exact copy", True),
    ("Say exactly: Focus on solving the real problem.", ["Focus on solving the real problem."], "Exact copy", True),
    ("Repeat verbatim: Curiosity drives scientific discovery.", ["Curiosity drives scientific discovery."], "Exact copy", True),
    ("Echo exactly: Calm minds make better decisions.", ["Calm minds make better decisions."], "Exact copy", True),
    ("Copy verbatim: Fresh air and sunshine bring energy.", ["Fresh air and sunshine bring energy."], "Exact copy", True),
    ("Say exactly: Listen carefully before speaking.", ["Listen carefully before speaking."], "Exact copy", True),
    ("Repeat exactly: Every journey begins with a single step.", ["Every journey begins with a single step."], "Exact copy", True),
    ("Echo verbatim: Good design is as little design as possible.", ["Good design is as little design as possible."], "Exact copy", True),
    ("Copy this sentence: Discipline creates long term freedom.", ["Discipline creates long term freedom."], "Exact copy", True),
    ("Say exactly: Patience and persistence overcome challenges.", ["Patience and persistence overcome challenges."], "Exact copy", True),
    ("Repeat verbatim: Teamwork divides the effort and multiplies success.", ["Teamwork divides the effort and multiplies success."], "Exact copy", True),
    ("Echo exactly: Learn from yesterday, live for today.", ["Learn from yesterday, live for today."], "Exact copy", True),
    ("Copy verbatim: Writing tests ensures code reliability.", ["Writing tests ensures code reliability."], "Exact copy", True),
    ("Say exactly: Small daily improvements create remarkable results.", ["Small daily improvements create remarkable results."], "Exact copy", True),
    ("Repeat exactly: Always strive for truth and kindness.", ["Always strive for truth and kindness."], "Exact copy", True)
]
for idx, (inp, kw, rub, exact) in enumerate(exact_copying, 1):
    add_probe(f"EXACT_{idx:03d}", "Exact Copying", "English", inp, kw, rub, exact)

# 8. Elementary Arithmetic (20 probes)
arithmetic_probes = [
    ("If I have 5 apples and give 2 to my friend, how many apples remain?", ["3", "three"], "5 - 2 = 3"),
    ("What is 2 + 2?", ["4", "four"], "2 + 2 = 4"),
    ("Calculate 10 - 4.", ["6", "six"], "10 - 4 = 6"),
    ("What is 5 + 3?", ["8", "eight"], "5 + 3 = 8"),
    ("If there are 7 birds on a tree and 3 fly away, how many birds are left?", ["4", "four"], "7 - 3 = 4"),
    ("What is 6 + 6?", ["12", "twelve"], "6 + 6 = 12"),
    ("Calculate 15 - 5.", ["10", "ten"], "15 - 5 = 10"),
    ("If you have 4 books and buy 3 more, how many books do you have in total?", ["7", "seven"], "4 + 3 = 7"),
    ("What is 3 * 3?", ["9", "nine"], "3 * 3 = 9"),
    ("What is 20 - 8?", ["12", "twelve"], "20 - 8 = 12"),
    ("Calculate 8 + 7.", ["15", "fifteen"], "8 + 7 = 15"),
    ("If a basket has 10 oranges and 6 are eaten, how many oranges remain?", ["4", "four"], "10 - 6 = 4"),
    ("What is 4 * 2?", ["8", "eight"], "4 * 2 = 8"),
    ("What is 9 - 5?", ["4", "four"], "9 - 5 = 4"),
    ("Calculate 12 + 8.", ["20", "twenty"], "12 + 8 = 20"),
    ("If you save 5 rupees yesterday and 5 rupees today, how many rupees do you have?", ["10", "ten"], "5 + 5 = 10"),
    ("What is 14 - 7?", ["7", "seven"], "14 - 7 = 7"),
    ("Calculate 3 + 7.", ["10", "ten"], "3 + 7 = 10"),
    ("If a room has 6 chairs and 2 are removed, how many chairs are left?", ["4", "four"], "6 - 2 = 4"),
    ("What is 5 * 2?", ["10", "ten"], "5 * 2 = 10")
]
for idx, (inp, kw, rub) in enumerate(arithmetic_probes, 1):
    add_probe(f"ARITH_{idx:03d}", "Elementary Arithmetic", "English", inp, kw, rub)

# 9. Basic Instructions & Constraints (20 probes)
instruction_probes = [
    ("Answer in exactly one word: What is the color of the clear sky?", ["blue"], "Single word response"),
    ("Answer in one word: What is the primary star of our solar system?", ["sun"], "Single word response"),
    ("Answer with only a number: How many days are in a standard week?", ["7", "seven"], "Single number response"),
    ("Write exactly two sentences explaining what water is.", ["water", "liquid", "hydrogen", "oxygen", "essential"], "Two sentence constraint"),
    ("List three primary colors using comma separation.", ["red", "blue", "yellow", "green"], "Three items list"),
    ("Answer yes or no: Is the Earth round?", ["yes"], "Yes/No constraint"),
    ("Answer yes or no: Can humans breathe underwater without equipment?", ["no"], "Yes/No constraint"),
    ("Summarize what a school is in one short sentence.", ["school", "institution", "students", "learn", "education"], "One sentence summary"),
    ("Answer in only one word: What liquid do cows produce?", ["milk"], "Single word response"),
    ("State the opposite of cold in one word.", ["hot", "warm"], "Antonym constraint"),
    ("State the opposite of dark in one word.", ["light", "bright"], "Antonym constraint"),
    ("Answer with only the capital city name: What is the capital of France?", ["paris"], "Single city name"),
    ("Answer with only the planet name: Which planet is known as the Red Planet?", ["mars"], "Single planet name"),
    ("Give a one sentence definition of a clock.", ["clock", "instrument", "device", "measures", "time"], "One sentence definition"),
    ("Answer yes or no: Does the sun rise in the east?", ["yes"], "Yes/No constraint"),
    ("Answer yes or no: Is fire cold?", ["no"], "Yes/No constraint"),
    ("List two popular programming languages separated by a comma.", ["python", "javascript", "c++", "java", "rust"], "Two items list"),
    ("In one short sentence, describe what winter feels like.", ["winter", "cold", "snow", "weather", "chilly"], "One sentence description"),
    ("Answer in one word: What do bees produce?", ["honey"], "Single word response"),
    ("State the opposite of fast in one word.", ["slow"], "Antonym constraint")
]
for idx, (inp, kw, rub) in enumerate(instruction_probes, 1):
    add_probe(f"INST_{idx:03d}", "Basic Instructions", "English", inp, kw, rub)

# 10. Repetition & Multi-turn Coherence (10 probes)
coherence_probes = [
    ("Tell me about dogs without repeating yourself.", ["dogs", "animals", "pets", "friendly", "loyal"], "Non-repetitive dog description"),
    ("Explain why exercise is good for health without repeating words.", ["exercise", "health", "body", "active", "fitness", "energy"], "Non-repetitive exercise explanation"),
    ("Describe the seasons of the year smoothly.", ["spring", "summer", "autumn", "winter", "seasons", "weather"], "Smooth seasonal progression"),
    ("Tell a very short two-sentence story about a brave traveler.", ["traveler", "journey", "mountain", "brave", "reached"], "Short coherent story"),
    ("Explain the life cycle of a plant in two clean sentences.", ["seed", "sprouts", "plant", "sunlight", "grows", "flower"], "Two sentence life cycle"),
    ("Describe a quiet morning in a village.", ["village", "morning", "quiet", "birds", "sunrise", "peaceful"], "Atmospheric description"),
    ("Explain why honesty matters in human relationships.", ["honesty", "trust", "respect", "truth", "relationships"], "Value explanation"),
    ("Describe how trees help the planet.", ["trees", "oxygen", "carbon", "environment", "clean", "shade"], "Environmental description"),
    ("Explain what happens when water freezes.", ["freezes", "ice", "temperature", "solid", "cold", "zero"], "Physical change explanation"),
    ("Describe the feeling of reading a fascinating book.", ["reading", "book", "imagination", "story", "curiosity", "fascinating"], "Experiential description")
]
for idx, (inp, kw, rub) in enumerate(coherence_probes, 1):
    add_probe(f"COH_{idx:03d}", "Repetition and Coherence", "English", inp, kw, rub)

print(f"Total frozen benchmark probes created: {len(probes)}")
category_counts = {}
for p in probes:
    cat = p["category"]
    category_counts[cat] = category_counts.get(cat, 0) + 1

for cat, count in category_counts.items():
    print(f"  {cat}: {count} probes")

out_file = REPO_ROOT / "experiments/cognition/ucip_05/baseline_benchmark.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2, ensure_ascii=False)

# Compute SHA-256
with open(out_file, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()

print(f"\nSaved benchmark to: {out_file}")
print(f"Benchmark SHA-256: {h}")
