"""
experiments/cognition/ucip_06/build_sealed_500_benchmark.py

Generates a fresh, reliable, strictly frozen 500-probe benchmark for UCIP-06:
  1. English Conversation (60 probes)
  2. English Simple Explanations (50 probes)
  3. English Sentence Generation (40 probes)
  4. Devanagari Hindi (60 probes)
  5. Roman Hindi (50 probes)
  6. Hinglish (60 probes)
  7. Exact Copying (50 probes)
  8. Elementary Arithmetic (40 probes)
  9. Basic Instructions & Constraints (40 probes)
 10. Semantic Consistency & Coherence (50 probes)

Total: 500 probes.
Each probe includes:
  - probe_id
  - category
  - language
  - input
  - expected_keywords / expected_value
  - rubric
  - exact_match flag
"""

import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")

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

# 1. English Conversation (60 probes)
# Incorporates all 7 original owner test prompts and broad daily chat
conv_prompts = [
    ("Hello ChakrView. How are you?", ["well", "good", "fine", "help", "assist", "today"], "Polite status greeting"),
    ("Good morning! How are you doing today?", ["morning", "good", "well", "fine", "help"], "Morning greeting"),
    ("Good evening! Nice to connect with you.", ["evening", "nice", "welcome", "pleasure"], "Evening greeting"),
    ("Hi there! What can you help me with?", ["help", "assist", "topics", "questions", "explain"], "Capability introduction"),
    ("Who are you? Explain how you were created.", ["chakrview", "independent", "neural", "assistant", "model", "local"], "Persona and origin"),
    ("What is your name?", ["chakrview", "name"], "Name introduction"),
    ("Who created you?", ["chakrview", "independent", "developed", "built"], "Creation background"),
    ("Are you an artificial intelligence?", ["yes", "ai", "artificial intelligence", "model", "assistant"], "AI identification"),
    ("Where do you execute your code?", ["locally", "cpu", "computer", "machine", "hardware"], "Execution location"),
    ("Do you require an active internet connection?", ["no", "offline", "local", "independent"], "Offline clarification"),
    ("Can you speak multiple languages?", ["yes", "english", "hindi", "hinglish", "languages"], "Language capability"),
    ("What makes your architecture distinct?", ["local", "cpu", "lightweight", "independent", "efficient"], "Architectural identity"),
    ("Thank you very much for your help!", ["welcome", "pleasure", "glad", "help"], "Polite gratitude response"),
    ("I really appreciate your quick explanation.", ["welcome", "happy", "assist", "glad"], "Gratitude reply"),
    ("How is your day going so far?", ["well", "ready", "assist", "smoothly", "help"], "Conversational engagement"),
    ("It is nice chatting with you today.", ["pleasure", "nice", "glad", "welcome"], "Pleasantry"),
    ("Can we discuss science and technology?", ["yes", "explore", "discuss", "science", "technology"], "Affirmative discussion"),
    ("I want to learn how to write clean code.", ["code", "clean", "practice", "python", "functions"], "Programming encouragement"),
    ("Do you get tired after answering many questions?", ["no", "machine", "software", "ready", "assist"], "Machine nature clarification"),
    ("What is your favorite subject to discuss?", ["science", "learning", "technology", "language", "questions"], "Subject interest"),
    ("Hello! Are you ready for some challenging questions?", ["yes", "ready", "assist", "ask"], "Readiness confirmation"),
    ("Good afternoon! Hope you are having a good session.", ["afternoon", "thank", "productive", "assist"], "Afternoon greeting"),
    ("Tell me something fascinating about the universe.", ["universe", "stars", "light", "space", "galaxies", "sun"], "Fascinating fact"),
    ("Can you break down complex ideas into simple terms?", ["yes", "simple", "clear", "explain", "help"], "Pedagogical commitment"),
    ("How fast do you process tokens on CPU?", ["cpu", "fast", "efficient", "tokens", "speed"], "Speed description"),
    ("Are my private conversation logs stored on a server?", ["no", "private", "local", "session", "safe"], "Privacy assurance"),
    ("What should I call you during this conversation?", ["chakrview", "assistant"], "Name clarification"),
    ("I am excited to test your language abilities.", ["glad", "welcome", "ready", "happy", "test"], "Positive rapport"),
    ("Let us start a productive coding session.", ["ready", "start", "sure", "productive"], "Session start"),
    ("Can you maintain a polite and courteous tone?", ["yes", "polite", "courteous", "always", "respectful"], "Tone commitment"),
    ("How do transformers process text tokens?", ["attention", "tokens", "embeddings", "layers", "weights"], "Transformer overview"),
    ("Can you assist with general knowledge questions?", ["yes", "general", "knowledge", "assist", "questions"], "General knowledge"),
    ("Is ChakrView an open source software project?", ["yes", "local", "independent", "executable", "open"], "Open source identity"),
    ("What is your guiding design philosophy?", ["lightweight", "local", "cpu", "efficient", "private"], "Design values"),
    ("What are your current operational capabilities?", ["conversation", "explanations", "hindi", "arithmetic", "copying"], "Capabilities list"),
    ("Can we practice bilingual communication?", ["yes", "languages", "hindi", "english", "bilingual"], "Bilingual readiness"),
    ("I need some study motivation today.", ["effort", "consistency", "step", "progress", "learn"], "Motivation"),
    ("Goodbye! Talk to you next time.", ["goodbye", "farewell", "bye", "take care", "later"], "Farewell response"),
    ("Have a pleasant and relaxing evening!", ["thank", "you", "too", "evening", "pleasant"], "Polite closing"),
    ("Can you summarize long paragraphs clearly?", ["yes", "summarize", "key", "points", "clear"], "Summarization affirmation"),
    ("What happens when you encounter an unknown word?", ["context", "subwords", "tokens", "represent"], "Subword handling"),
    ("Why is privacy in AI software important?", ["privacy", "data", "confidential", "secure", "local"], "Privacy value"),
    ("Can you help me plan my daily study schedule?", ["yes", "schedule", "plan", "time", "routine"], "Planning help"),
    ("What is the advantage of running models locally?", "No cloud dependency, complete data privacy, and zero internet latency.", "Local advantages"),
    ("Do you support both Hindi and English dialogue?", ["yes", "hindi", "english", "support", "both"], "Bilingual confirmation"),
    ("How does self-attention mechanism work?", ["attention", "tokens", "relate", "queries", "keys", "values"], "Attention explanation"),
    ("Can you write a polite email greeting?", ["dear", "hello", "greetings", "hope", "well"], "Email greeting"),
    ("What should a developer do when facing a tough bug?", ["logs", "debug", "traceback", "isolate", "test"], "Debugging advice"),
    ("Why is continuous learning important for engineers?", ["skills", "technology", "growth", "learn", "adapt"], "Continuous learning"),
    ("Can you explain why peer review improves code?", ["quality", "bugs", "feedback", "standards", "team"], "Code review"),
    ("How do neural network weights change during training?", ["gradients", "backpropagation", "loss", "update", "optimizer"], "Weight updates"),
    ("What is the role of an activation function?", ["non-linearity", "activation", "patterns", "complex"], "Activation role"),
    ("Can AI understand sarcasm reliably?", ["sarcasm", "subtle", "context", "challenging", "difficult"], "Sarcasm challenge"),
    ("What makes Python so beginner friendly?", ["syntax", "readable", "clean", "simple", "libraries"], "Python accessibility"),
    ("Why are open standards beneficial in technology?", ["interoperability", "open", "collaboration", "freedom"], "Open standards"),
    ("Can you explain what latency means in networking?", ["delay", "time", "travel", "packets", "network"], "Latency meaning"),
    ("How can someone improve their logical thinking?", ["practice", "puzzles", "math", "coding", "logic"], "Logic improvement"),
    ("What is the difference between data and information?", ["data", "raw", "information", "context", "meaning", "processed"], "Data vs information"),
    ("Can you help me brainstorm project ideas?", ["yes", "ideas", "brainstorm", "explore", "project"], "Brainstorming offer"),
    ("Tell me a brief thought on patience.", ["patience", "time", "persistence", "growth", "mastery"], "Thought on patience")
]
for idx, (inp, kw, rub) in enumerate(conv_prompts, 1):
    add_probe(f"ENG_CONV_{idx:03d}", "English Conversation", "English", inp, kw, rub)

# 2. English Simple Explanations (50 probes)
exp_prompts = [
    ("What does a CPU do?", ["processor", "instructions", "executes", "arithmetic", "memory", "fetches"], "CPU explanation"),
    ("How does computer RAM function?", ["ram", "memory", "volatile", "temporary", "fast", "cpu"], "RAM function"),
    ("What is an operating system?", ["manages", "hardware", "software", "resources", "processes", "system"], "OS definition"),
    ("Why is the sky blue in daytime?", ["atmosphere", "sunlight", "scatter", "blue", "wavelength", "molecules"], "Atmospheric scattering"),
    ("What is photosynthesis?", ["plants", "sunlight", "energy", "glucose", "oxygen", "chlorophyll"], "Photosynthesis"),
    ("How does an electric motor work?", ["electrical", "magnetic", "mechanical", "rotation", "energy"], "Electric motor"),
    ("What is the force of gravity?", ["force", "attraction", "mass", "earth", "gravity", "pull"], "Gravity"),
    ("What does a computer compiler do?", ["translates", "source code", "machine", "binary", "executable"], "Compiler"),
    ("What is machine learning?", ["learn", "data", "patterns", "algorithms", "predictions"], "Machine learning"),
    ("What is a database?", ["collection", "data", "organized", "tables", "storage", "queries"], "Database"),
    ("How does a bicycle work?", ["pedals", "wheels", "chain", "gear", "propel", "human"], "Bicycle"),
    ("What is our solar system composed of?", ["sun", "planets", "orbit", "solar", "gravity"], "Solar system"),
    ("What causes rain to fall?", ["water", "evaporation", "clouds", "condensation", "drops"], "Rain causes"),
    ("What is an algorithm in computing?", ["step", "sequence", "instructions", "solve", "problem"], "Algorithm"),
    ("How does the human heart function?", ["pumps", "blood", "oxygen", "circulates", "muscle"], "Heart function"),
    ("What is renewable energy?", ["solar", "wind", "renewable", "replenished", "clean"], "Renewable energy"),
    ("What is the function of a web browser?", ["renders", "html", "web", "internet", "pages"], "Web browser"),
    ("Explain the difference between hardware and software.", ["hardware", "physical", "software", "programs", "code"], "Hardware vs software"),
    ("What is a black hole in astrophysics?", ["gravity", "dense", "light", "space", "escape"], "Black hole"),
    ("How does a thermometer measure heat?", ["temperature", "heat", "liquid", "sensor", "expansion"], "Thermometer"),
    ("What is an artificial neural network?", ["layers", "nodes", "neurons", "weights", "learn"], "Neural network"),
    ("What is open source software?", ["source code", "freely", "modify", "share", "community"], "Open source"),
    ("How do airplanes generate lift?", ["wings", "lift", "pressure", "air", "aerodynamic"], "Flight lift"),
    ("What is an application programming interface (API)?", ["interface", "communicate", "software", "data", "exchange"], "API definition"),
    ("What is DNA in living organisms?", ["genetic", "instructions", "cells", "hereditary", "organism"], "DNA definition"),
    ("Why is human sleep necessary?", ["rest", "brain", "recovery", "health", "energy"], "Sleep purpose"),
    ("What is physical sound?", ["vibrations", "waves", "air", "ears", "frequency"], "Sound physics"),
    ("What is cloud computing?", ["servers", "internet", "storage", "remote", "on-demand"], "Cloud computing"),
    ("What is a microcontroller chip?", ["compact", "chip", "controls", "embedded", "system"], "Microcontroller"),
    ("How does a household refrigerator cool items?", ["refrigerant", "heat", "absorbs", "cools", "compressor"], "Refrigeration"),
    ("What is an atom?", ["unit", "matter", "protons", "electrons", "neutrons"], "Atom definition"),
    ("What is electrical current?", ["flow", "electrons", "charge", "circuits", "energy"], "Electricity"),
    ("What is CPU cache memory?", ["fast", "temporary", "memory", "cpu", "frequent"], "Cache memory"),
    ("How does GPS satellite positioning work?", ["satellites", "signals", "triangulation", "location", "time"], "GPS positioning"),
    ("What is a network firewall?", ["security", "blocks", "traffic", "unauthorized", "protects"], "Firewall role"),
    ("What is the biological function of lungs?", ["oxygen", "carbon dioxide", "breathe", "blood", "gas exchange"], "Lungs function"),
    ("What geological process causes earthquakes?", ["tectonic", "plates", "fault", "movement", "seismic"], "Earthquakes"),
    ("What is an operating system kernel?", ["core", "kernel", "manages", "hardware", "processes"], "Kernel role"),
    ("What is a vector in geometry and physics?", ["magnitude", "direction", "arrow", "quantity"], "Vector definition"),
    ("Why do seasons cycle on Earth?", ["tilt", "axis", "orbit", "sun", "sunlight"], "Seasons cycle"),
    ("What is the function of a computer graphics card (GPU)?", ["graphics", "parallel", "rendering", "display", "matrix"], "GPU explanation"),
    ("How does water boil?", ["boils", "temperature", "vapor", "heat", "bubbles", "celsius"], "Boiling physics"),
    ("What is a computer file system?", ["organizes", "files", "directories", "storage", "disk"], "File system"),
    ("What does a router do in a network?", ["routes", "packets", "network", "traffic", "connects"], "Router explanation"),
    ("How do solar panels convert sunlight to power?", ["photovoltaic", "silicon", "photons", "electrons", "current"], "Solar power"),
    ("What is the difference between compiler and interpreter?", ["compiler", "interpreter", "translates", "line", "executable"], "Compiler vs interpreter"),
    ("What is virtual memory in operating systems?", ["virtual", "paging", "swap", "disk", "ram"], "Virtual memory"),
    ("How do optical fiber cables transmit signals?", ["light", "laser", "reflection", "glass", "pulses"], "Fiber optics"),
    ("What is the role of white blood cells in immunity?", ["immune", "infections", "protect", "defense", "white blood"], "Immune system"),
    ("What is an algorithm's Big-O time complexity?", ["time", "complexity", "growth", "input", "scale"], "Time complexity")
]
for idx, (inp, kw, rub) in enumerate(exp_prompts, 1):
    add_probe(f"ENG_EXP_{idx:03d}", "English Simple Explanations", "English", inp, kw, rub)

# 3. English Sentence Generation (40 probes)
gen_prompts = [
    ("Write one simple sentence about a dog.", ["dog", "animal", "friendly", "pet", "home", "loyal"], "Dog sentence"),
    ("Write one descriptive sentence about the sun.", ["sun", "star", "bright", "golden", "light", "shines"], "Sun sentence"),
    ("Generate a short sentence about an apple.", ["apple", "fruit", "sweet", "crisp", "red", "orchard"], "Apple sentence"),
    ("Write a sentence describing a quiet library.", ["library", "books", "quiet", "reading", "students"], "Library sentence"),
    ("Write a descriptive sentence about a bicycle.", ["bicycle", "wheels", "ride", "pedal", "cycle"], "Bicycle sentence"),
    ("Create a sentence about a fast train.", ["train", "tracks", "speed", "travel", "fast"], "Train sentence"),
    ("Write a simple sentence about the ocean.", ["ocean", "waves", "water", "vast", "blue", "sea"], "Ocean sentence"),
    ("Generate a sentence describing a tall mountain.", ["mountain", "tall", "peak", "snow", "high"], "Mountain sentence"),
    ("Write a sentence about drinking clean water.", ["water", "clean", "drink", "healthy", "essential"], "Water sentence"),
    ("Write a simple sentence about a friendly cat.", ["cat", "pet", "purrs", "fur", "gentle"], "Cat sentence"),
    ("Write one sentence about stars in the night sky.", ["stars", "night", "sky", "twinkle", "shine"], "Stars sentence"),
    ("Generate a sentence about planting a green tree.", ["tree", "plant", "green", "grow", "nature"], "Tree planting"),
    ("Write a sentence describing an honest student.", ["student", "honest", "studies", "learns", "hard"], "Student sentence"),
    ("Write one sentence about eating fresh fruits.", ["fruits", "fresh", "healthy", "delicious", "vitamins"], "Fruits sentence"),
    ("Create a sentence about reading good books.", ["books", "reading", "knowledge", "learn", "mind"], "Reading sentence"),
    ("Write a descriptive sentence about morning sunlight.", ["sunlight", "morning", "golden", "warmth", "shines"], "Morning sun"),
    ("Write a sentence describing a calm forest.", ["forest", "trees", "calm", "peaceful", "nature"], "Calm forest"),
    ("Generate a sentence about an electric car.", ["electric", "car", "battery", "clean", "drives"], "Electric car"),
    ("Write one sentence about a clock on the wall.", ["clock", "wall", "time", "ticks", "measures"], "Clock sentence"),
    ("Write a descriptive sentence about a singing bird.", ["bird", "singing", "morning", "tune", "tree"], "Bird singing"),
    ("Write a sentence describing walking in the rain.", ["walking", "rain", "umbrella", "drops", "cool"], "Walking in rain"),
    ("Generate a simple sentence about a computer keyboard.", ["keyboard", "keys", "typing", "input", "letters"], "Keyboard sentence"),
    ("Write one sentence about an artist painting.", ["artist", "painting", "colors", "canvas", "brush"], "Artist painting"),
    ("Write a sentence about baking fresh bread.", ["bread", "baking", "oven", "warm", "fresh"], "Baking bread"),
    ("Create a descriptive sentence about a flowing river.", ["river", "flowing", "water", "stream", "peaceful"], "Flowing river"),
    ("Write a sentence describing a bright full moon.", ["moon", "bright", "full", "night", "glowing"], "Full moon"),
    ("Write one sentence about learning a new skill.", ["learning", "skill", "practice", "patience", "growth"], "Skill learning"),
    ("Generate a sentence about listening to soft music.", ["music", "soft", "calm", "listening", "melody"], "Soft music"),
    ("Write a sentence describing working in a team.", ["team", "collaboration", "work", "together", "success"], "Teamwork sentence"),
    ("Write a simple sentence about children playing.", ["children", "playing", "park", "joy", "games"], "Children playing"),
    ("Write a sentence about an eagle soaring high.", ["eagle", "soaring", "sky", "wings", "high"], "Eagle soaring"),
    ("Write a sentence describing a warm cup of coffee.", ["coffee", "cup", "warm", "morning", "aroma"], "Coffee sentence"),
    ("Generate a sentence about a quiet snowy evening.", ["snow", "snowy", "quiet", "evening", "winter", "white"], "Snowy evening"),
    ("Write one sentence about an astronaut in space.", ["astronaut", "space", "floating", "earth", "stars"], "Astronaut in space"),
    ("Write a sentence about a bustling morning market.", ["market", "bustling", "morning", "vendors", "people"], "Market scene"),
    ("Write a descriptive sentence about autumn leaves falling.", ["autumn", "leaves", "falling", "golden", "breeze"], "Autumn leaves"),
    ("Create a sentence about a wooden sailing ship.", ["ship", "sailing", "ocean", "wooden", "waves"], "Sailing ship"),
    ("Write one sentence about solving a puzzle.", ["puzzle", "solving", "pieces", "satisfying", "logic"], "Puzzle solving"),
    ("Generate a sentence about a roaring campfire.", ["campfire", "fire", "crackling", "warmth", "night"], "Campfire sentence"),
    ("Write a simple sentence about planting flower seeds.", ["seeds", "flower", "garden", "planting", "soil"], "Planting seeds")
]
for idx, (inp, kw, rub) in enumerate(gen_prompts, 1):
    add_probe(f"ENG_GEN_{idx:03d}", "English Sentence Generation", "English", inp, kw, rub)

# 4. Devanagari Hindi (60 probes)
hindi_prompts = [
    ("नमस्ते! आप क्या कर सकते हैं?", ["नमस्ते", "सहायता", "कर", "सकता", "सीखने", "प्रश्न", "उत्तर"], "Devanagari greeting and capability"),
    ("आप कैसे हैं?", ["मैं", "ठीक", "हूँ", "धन्यवाद", "आप"], "Status reply in Hindi"),
    ("कंप्यूटर क्या होता है?", ["कंप्यूटर", "इलेक्ट्रॉनिक", "उपकरण", "डेटा", "निर्देश", "मशीन"], "Computer definition in Hindi"),
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
    ("ध्वनि कैसे उत्पन्न होती है?", ["कंपन", "ध्वनि", "तरंगें", "माध्यम", "आवाज"], "Sound generation"),
    ("अनुशासन का छात्र जीवन में क्या महत्व है?", ["अनुशासन", "छात्र", "समय", "नियम", "सफलता", "भविष्य"], "Discipline in student life"),
    ("प्रकाश की गति कितनी होती है?", ["प्रकाश", "गति", "लाख", "किलोमीटर", "सेकंड", "300000"], "Speed of light"),
    ("आहार में हरी सब्जियों का क्या महत्व है?", ["हरी सब्जियां", "विटामिन", "पोषण", "स्वास्थ्य", "खनिज"], "Vegetables nutrition"),
    ("पुस्तकालय का क्या उपयोग है?", ["पुस्तकालय", "किताबें", "अध्ययन", "शांत", "ज्ञान"], "Library utility"),
    ("सड़क सुरक्षा नियमों का पालन क्यों जरूरी है?", ["सड़क", "सुरक्षा", "नियम", "दुर्घटना", "बचाव", "जीवन"], "Road safety rules"),
    ("नमस्ते! क्या आप मुझे हिंदी सिखा सकते हैं?", ["हाँ", "हिंदी", "सीखने", "अभ्यास", "सहायता"], "Willingness to teach Hindi"),
    ("कंप्यूटर का सीपीयू क्या काम करता है?", ["सीपीयू", "प्रोसेस", "निर्देशों", "गणना", "मस्तिष्क"], "CPU in Hindi"),
    ("रैम (RAM) क्या होती है?", ["रैम", "मेमोरी", "अस्थाई", "तेज़", "डेटा"], "RAM in Hindi"),
    ("हार्ड ड्राइव क्या करती है?", ["स्टोरेज", "स्थाई", "फाइलें", "सुरक्षित", "डेटा"], "Hard drive in Hindi"),
    ("सॉफ्टवेयर और हार्डवेयर में क्या अंतर है?", ["सॉफ्टवेयर", "हार्डवेयर", "प्रोग्राम्स", "भौतिक", "उपकरण"], "Software vs hardware"),
    ("कीबोर्ड का क्या उपयोग है?", ["कीबोर्ड", "टाइपिंग", "इनपुट", "अक्षर", "निर्देश"], "Keyboard in Hindi"),
    ("मॉनिटर क्या काम करता है?", ["मॉनिटर", "स्क्रीन", "डिस्प्ले", "चित्र", "आउटपुट"], "Monitor in Hindi"),
    ("माउस का क्या काम होता है?", ["माउस", "कर्सर", "क्लिक", "इनपुट", "स्क्रीन"], "Mouse in Hindi"),
    ("प्रिंटर का उपयोग क्या है?", ["प्रिंटर", "कागज", "दस्तावेज", "प्रिंट", "आउटपुट"], "Printer in Hindi"),
    ("मदरबोर्ड क्या होता है?", ["मदरबोर्ड", "सर्किट", "बोर्ड", "जोड़ता", "मुख्य"], "Motherboard in Hindi"),
    ("ऑपरेटिंग सिस्टम का क्या काम है?", ["ऑपरेटिंग", "सिस्टम", "प्रबंधन", "हार्डवेयर", "सॉफ्टवेयर"], "OS in Hindi"),
    ("अल्गोरिदम किसे कहते हैं?", ["अल्गोरिदम", "चरणबद्ध", "नियम", "समस्या", "समाधान"], "Algorithm in Hindi"),
    ("प्रोग्रामिंग भाषा क्या होती है?", ["प्रोग्रामिंग", "भाषा", "निर्देश", "कंप्यूटर", "कोड"], "Programming language in Hindi"),
    ("डेटाबेस का क्या उपयोग है?", ["डेटाबेस", "संग्रह", "डेटा", "तालिकाओं", "सुरक्षित"], "Database in Hindi"),
    ("क्लाउड स्टोरेज क्या है?", ["क्लाउड", "स्टोरेज", "इंटरनेट", "सर्वर", "ऑनलाइन"], "Cloud storage in Hindi"),
    ("साइबर सुरक्षा क्यों जरूरी है?", ["सुरक्षा", "डेटा", "बचाव", "पासवर्ड", "सुरक्षित"], "Cyber security in Hindi"),
    ("ई-मेल क्या होता है?", ["ईमेल", "संदेश", "इलेक्ट्रॉनिक", "पत्र", "इंटरनेट"], "Email in Hindi"),
    ("वेबसाइट क्या होती है?", ["वेबसाइट", "वेबपेज", "सूचना", "इंटरनेट", "पता"], "Website in Hindi"),
    ("सर्च इंजन कैसे काम करता है?", ["सर्च", "इंजन", "खोज", "वेबसाइट", "परिणाम"], "Search engine in Hindi"),
    ("आर्टिफिशियल इंटेलिजेंस क्या है?", ["आर्टिफिशियल", "इंटेलिजेंस", "एआई", "मशीन", "सीखने", "बुद्धिमत्ता"], "AI in Hindi"),
    ("धन्यवाद! आपका उत्तर बहुत अच्छा था।", ["स्वागत", "खुशी", "मदद", "प्रसन्न"], "Gratitude response in Hindi")
]
for idx, (inp, kw, rub) in enumerate(hindi_prompts, 1):
    add_probe(f"DEV_HIN_{idx:03d}", "Devanagari Hindi", "Devanagari Hindi", inp, kw, rub)

# 5. Roman Hindi (50 probes)
roman_prompts = [
    ("Aap kaise hain?", ["main", "theek", "hoon", "aap", "kaise"], "Status in Roman Hindi"),
    ("Aapka naam kya hai?", ["mera", "naam", "chakrview", "hai"], "Name query"),
    ("Computer ka CPU kya kaam karta hai?", ["cpu", "instructions", "execute", "karta", "processor", "hai"], "CPU function"),
    ("Bhai, machine learning ko simple language mein samjhao.", ["machine learning", "data", "patterns", "learn", "karta", "hai"], "ML in Roman Hindi"),
    ("RAM aur ROM mein kya antar hai?", ["ram", "temporary", "volatile", "rom", "permanent", "storage"], "RAM vs ROM"),
    ("Aaj ka mausam kaisa lag raha hai?", ["mausam", "accha", "suhana", "hai", "weather"], "Weather chat"),
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
    ("Shukriya bhai, aapka explanation helpful tha.", ["welcome", "pleasure", "glad", "help", "shukriya"], "Gratitude response"),
    ("Compiler code ko kaise execute karta hai?", ["compiler", "machine code", "binary", "translate", "execute"], "Compiler in Roman Hindi"),
    ("Git branch create karne ka kya benefit hai?", ["branch", "isolated", "features", "safe", "test"], "Git branch in Roman Hindi"),
    ("Frontend aur backend me kya fark hota hai?", ["frontend", "ui", "backend", "server", "database"], "Frontend vs backend in Roman Hindi"),
    ("Computer keyboard ka kya function hai?", ["keyboard", "input", "typing", "commands", "buttons"], "Keyboard in Roman Hindi"),
    ("Mouse ka right click kya karta hai?", ["right click", "context menu", "options", "properties"], "Mouse right click"),
    ("Laptop battery life kaise badhayein?", ["brightness", "background apps", "battery", "saver", "power"], "Battery tips"),
    ("Algorithm ka simple meaning kya hai?", ["step by step", "instructions", "problem", "solution", "logic"], "Algorithm in Roman Hindi"),
    ("Motherboard computer me kahan hota hai?", ["motherboard", "main board", "cpu", "ram", "connect"], "Motherboard location"),
    ("Mobile screen display kaise kaam karti hai?", ["display", "pixels", "led", "lcd", "screen", "light"], "Screen display"),
    ("Wi-Fi aur mobile data me kya fark hai?", ["wifi", "router", "wireless", "cellular", "tower"], "Wi-Fi vs mobile data"),
    ("Backup lena kyun zaroori hota hai?", ["backup", "data loss", "safe", "files", "recovery"], "Backup importance"),
    ("Antivirus software kya kaam karta hai?", ["antivirus", "malware", "virus", "protect", "scan"], "Antivirus in Roman Hindi"),
    ("Google search engine results kaise lata hai?", ["search", "indexing", "ranking", "crawler", "results"], "Search engine in Roman Hindi"),
    ("Bhai, Python me loop kaise kaam karta hai?", ["loop", "repeat", "iteration", "for", "while"], "Loop in Roman Hindi"),
    ("Function create karne ka kya fayda hota hai?", ["function", "reuse", "clean", "modular", "code"], "Function benefits"),
    ("Variable kya hota hai programming me?", ["variable", "storage", "value", "data", "container"], "Variable in Roman Hindi"),
    ("List comprehension Python me kyun use hoti hai?", ["list comprehension", "clean", "compact", "readable", "fast"], "List comprehension"),
    ("Recursion kya hoti hai programming me?", ["recursion", "function", "itself", "calls", "base case"], "Recursion in Roman Hindi"),
    ("SQL query ka kya kaam hota hai?", ["sql", "query", "database", "retrieve", "tables"], "SQL in Roman Hindi"),
    ("Bhai, thanks a lot for all your guidance!", ["welcome", "shukriya", "dosti", "help", "hamesha"], "Friendly closing")
]
for idx, (inp, kw, rub) in enumerate(roman_prompts, 1):
    add_probe(f"ROM_HIN_{idx:03d}", "Roman Hindi", "Roman Hindi", inp, kw, rub)

# 6. Hinglish (60 probes)
hinglish_prompts = [
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
    ("ChakrView ka CPU-first approach deployment overhead kaise cut down karta hai?", ["cpu-first", "lightweight", "no gpu needed", "local", "cost effective", "efficient"], "CPU-first efficiency"),
    ("Git stash command kab use karni chahiye?", ["stash", "temporary", "changes", "save", "working directory"], "Git stash in Hinglish"),
    ("WebSockets HTTP polling se better latency kaise provide karte hain?", ["websockets", "bidirectional", "full duplex", "connection", "real-time"], "WebSockets in Hinglish"),
    ("Database transaction ACID properties ka kya role hota hai?", ["acid", "atomicity", "consistency", "isolation", "durability"], "ACID properties in Hinglish"),
    ("JWT token authentication stateless security kaise manage karta hai?", ["jwt", "token", "signature", "stateless", "claims", "auth"], "JWT in Hinglish"),
    ("Serverless architecture backend infrastructure scaling ko kaise simplify karti hai?", ["serverless", "functions", "auto scale", "event driven", "cloud"], "Serverless in Hinglish"),
    ("Binary tree traversal algorithms (Inorder, Preorder) kaise operate karte hain?", ["binary tree", "traversal", "nodes", "inorder", "left", "right"], "Tree traversal in Hinglish"),
    ("Linux file permissions (rwx) system security ko kaise guard karti hain?", ["permissions", "read", "write", "execute", "users", "linux"], "File permissions in Hinglish"),
    ("CORS header web browsers mein cross-origin security kaise enforce karta hai?", ["cors", "cross origin", "browser", "security", "headers", "api"], "CORS in Hinglish"),
    ("Responsive web design mobile aur desktop dono par kaise adjust hoti hai?", ["responsive", "media queries", "css", "layout", "mobile", "desktop"], "Responsive design in Hinglish"),
    ("Docker compose multi-container setup ko kaise automate karta hai?", ["docker compose", "containers", "yaml", "services", "multi container"], "Docker compose in Hinglish"),
    ("API rate limiting DDoS aur server overload kaise prevent karti hai?", ["rate limiting", "requests", "throttle", "overload", "protection"], "Rate limiting in Hinglish"),
    ("Software architecture mein monolithic app kab prefer karni chahiye?", ["monolith", "simple", "small team", "mvp", "single codebase"], "Monolith use cases in Hinglish"),
    ("CSS flexbox aur grid layout mein primary architectural difference kya hai?", ["flexbox", "grid", "one dimensional", "two dimensional", "layout"], "Flexbox vs Grid in Hinglish"),
    ("Python virtual environment package conflicts kaise avoid karta hai?", ["virtual environment", "isolated", "dependencies", "packages", "python"], "Virtualenv in Hinglish"),
    ("OAuth 2.0 authorization framework third-party login kaise secure banata hai?", ["oauth", "access token", "authorization", "third party", "credentials"], "OAuth in Hinglish"),
    ("HTTPS encryption man-in-the-middle attacks kaise thwart karti hai?", ["https", "tls", "ssl", "encryption", "certificates", "secure"], "HTTPS in Hinglish"),
    ("Message broker (Kafka, RabbitMQ) decoupled services mein kaise help karta hai?", ["message broker", "async", "decoupled", "queues", "events"], "Message brokers in Hinglish"),
    ("Machine learning pipeline mein data augmentation model accuracy kaise badhati hai?", ["data augmentation", "variety", "generalization", "overfitting", "training"], "Data augmentation in Hinglish"),
    ("Code profiling execution bottlenecks kaise spot karta hai?", ["profiling", "bottlenecks", "cpu time", "memory", "performance"], "Profiling in Hinglish"),
    ("Bhai, overall ChakrView model ka response kaisa perform kar raha hai?", ["efficient", "local", "natural", "responses", "improving"], "Overall ChakrView in Hinglish")
]
for idx, (inp, kw, rub) in enumerate(hinglish_prompts, 1):
    add_probe(f"HING_{idx:03d}", "Hinglish", "Hinglish", inp, kw, rub)

# 7. Exact Copying (50 probes)
exact_copy_prompts = [
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
    ("Repeat exactly: Always strive for truth and kindness.", ["Always strive for truth and kindness."], "Exact copy", True),
    ("Say exactly: A stitch in time saves nine.", ["A stitch in time saves nine."], "Exact copy", True),
    ("Repeat verbatim: Birds of a feather flock together.", ["Birds of a feather flock together."], "Exact copy", True),
    ("Echo the text: Actions speak louder than words.", ["Actions speak louder than words."], "Exact copy", True),
    ("Copy this sentence: Look before you leap.", ["Look before you leap."], "Exact copy", True),
    ("Say exactly: Where there is a will there is a way.", ["Where there is a will there is a way."], "Exact copy", True),
    ("Repeat exactly: Better late than never.", ["Better late than never."], "Exact copy", True),
    ("Echo verbatim: Two heads are better than one.", ["Two heads are better than one."], "Exact copy", True),
    ("Copy the following: Honesty is the best policy.", ["Honesty is the best policy."], "Exact copy", True),
    ("Say exactly: Laughter is the best medicine.", ["Laughter is the best medicine."], "Exact copy", True),
    ("Repeat verbatim: Practice what you preach.", ["Practice what you preach."], "Exact copy", True),
    ("Echo exactly: Time and tide wait for no man.", ["Time and tide wait for no man."], "Exact copy", True),
    ("Copy verbatim: All that glitters is not gold.", ["All that glitters is not gold."], "Exact copy", True),
    ("Say exactly: Fortune favors the brave.", ["Fortune favors the brave."], "Exact copy", True),
    ("Repeat exactly: Knowledge is supreme power.", ["Knowledge is supreme power."], "Exact copy", True),
    ("Echo verbatim: Still waters run deep.", ["Still waters run deep."], "Exact copy", True),
    ("Copy this text: Beauty lies in the eyes of the beholder.", ["Beauty lies in the eyes of the beholder."], "Exact copy", True),
    ("Say exactly: Necessity is the mother of invention.", ["Necessity is the mother of invention."], "Exact copy", True),
    ("Repeat verbatim: Hope for the best, prepare for the worst.", ["Hope for the best, prepare for the worst."], "Exact copy", True),
    ("Echo exactly: Great minds think alike.", ["Great minds think alike."], "Exact copy", True),
    ("Copy verbatim: You reap what you sow.", ["You reap what you sow."], "Exact copy", True)
]
for idx, (inp, kw, rub, exact) in enumerate(exact_copy_prompts, 1):
    add_probe(f"EXACT_{idx:03d}", "Exact Copying", "English", inp, kw, rub, exact)

# 8. Elementary Arithmetic (40 probes)
arith_prompts = [
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
    ("What is 5 * 2?", ["10", "ten"], "5 * 2 = 10"),
    ("What is 1 + 1?", ["2", "two"], "1 + 1 = 2"),
    ("Calculate 7 - 2.", ["5", "five"], "7 - 2 = 5"),
    ("What is 4 + 4?", ["8", "eight"], "4 + 4 = 8"),
    ("What is 10 - 7?", ["3", "three"], "10 - 7 = 3"),
    ("If you have 8 pencils and share 4 with classmates, how many pencils do you keep?", ["4", "four"], "8 - 4 = 4"),
    ("What is 3 + 5?", ["8", "eight"], "3 + 5 = 8"),
    ("Calculate 18 - 9.", ["9", "nine"], "18 - 9 = 9"),
    ("What is 6 * 2?", ["12", "twelve"], "6 * 2 = 12"),
    ("What is 16 - 8?", ["8", "eight"], "16 - 8 = 8"),
    ("If a farmer has 12 sheep and 5 are sold, how many sheep are left on the farm?", ["7", "seven"], "12 - 5 = 7"),
    ("What is 9 + 3?", ["12", "twelve"], "9 + 3 = 12"),
    ("Calculate 11 - 6.", ["5", "five"], "11 - 6 = 5"),
    ("What is 2 * 5?", ["10", "ten"], "2 * 5 = 10"),
    ("What is 7 + 7?", ["14", "fourteen"], "7 + 7 = 14"),
    ("If there are 9 slices of pizza and 5 are eaten, how many slices remain?", ["4", "four"], "9 - 5 = 4"),
    ("What is 15 - 9?", ["6", "six"], "15 - 9 = 6"),
    ("Calculate 8 + 4.", ["12", "twelve"], "8 + 4 = 12"),
    ("What is 5 * 3?", ["15", "fifteen"], "5 * 3 = 15"),
    ("If you run 3 kilometers in the morning and 2 in the evening, what is your total distance?", ["5", "five"], "3 + 2 = 5"),
    ("What is 13 - 6?", ["7", "seven"], "13 - 6 = 7")
]
for idx, (inp, kw, rub) in enumerate(arith_prompts, 1):
    add_probe(f"ARITH_{idx:03d}", "Elementary Arithmetic", "English", inp, kw, rub)

# 9. Basic Instructions & Constraints (40 probes)
inst_prompts = [
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
    ("State the opposite of fast in one word.", ["slow"], "Antonym constraint"),
    ("Answer in exactly one word: What is the solid form of water called?", ["ice"], "Single word answer"),
    ("Answer in one word: What gas do plants absorb from the air?", ["carbon dioxide", "co2", "carbon"], "Single word gas"),
    ("Answer yes or no: Is diamond a very hard mineral?", ["yes"], "Yes/No mineral"),
    ("Answer yes or no: Can birds fly backward?", ["yes", "no"], "Yes/No flight"),
    ("State the opposite of heavy in one word.", ["light"], "Antonym constraint"),
    ("State the opposite of empty in one word.", ["full"], "Antonym constraint"),
    ("Answer with only the country name: In which country is the Taj Mahal located?", ["india"], "Single country name"),
    ("Answer in one word: What force keeps our feet on the ground?", ["gravity"], "Single word physics"),
    ("Answer yes or no: Does sound travel faster than light?", ["no"], "Yes/No speed of light"),
    ("Answer yes or no: Is copper an electrical conductor?", ["yes"], "Yes/No conductor"),
    ("List two seasons of the year separated by a comma.", ["summer", "winter", "spring", "autumn", "fall"], "Two seasons list"),
    ("Answer in only one word: What organ pumps blood in human body?", ["heart"], "Single organ word"),
    ("Answer in one word: What is the boiling point of pure water in Celsius?", ["100"], "Single number boiling point"),
    ("State the opposite of loud in one word.", ["quiet", "silent", "soft"], "Antonym loud"),
    ("State the opposite of young in one word.", ["old"], "Antonym young"),
    ("Answer yes or no: Is iron a metal?", ["yes"], "Yes/No metal"),
    ("Answer yes or no: Do plants need sunlight to survive?", ["yes"], "Yes/No plants"),
    ("In one short sentence, explain what a dictionary is.", ["dictionary", "book", "words", "meanings", "definitions"], "Dictionary summary"),
    ("Answer with only the ocean name: What is the largest ocean on Earth?", ["pacific"], "Ocean name"),
    ("Answer in one word: What is the freezing point of water in Celsius?", ["0", "zero"], "Zero Celsius")
]
for idx, (inp, kw, rub) in enumerate(inst_prompts, 1):
    add_probe(f"INST_{idx:03d}", "Basic Instructions", "English", inp, kw, rub)

# 10. Semantic Consistency & Coherence (50 probes)
coherence_prompts = [
    ("Tell me about dogs without repeating words.", ["dogs", "animals", "pets", "friendly", "loyal"], "Non-repetitive dog description"),
    ("Explain why exercise is good for health smoothly.", ["exercise", "health", "body", "active", "fitness", "energy"], "Non-repetitive exercise explanation"),
    ("Describe the seasons of the year consistently.", ["spring", "summer", "autumn", "winter", "seasons", "weather"], "Smooth seasonal progression"),
    ("Tell a very short two-sentence story about a traveler.", ["traveler", "journey", "mountain", "brave", "reached"], "Short coherent story"),
    ("Explain the life cycle of a plant in two clean sentences.", ["seed", "sprouts", "plant", "sunlight", "grows", "flower"], "Two sentence life cycle"),
    ("Describe a quiet morning in a village.", ["village", "morning", "quiet", "birds", "sunrise", "peaceful"], "Atmospheric description"),
    ("Explain why honesty matters in human relationships.", ["honesty", "trust", "respect", "truth", "relationships"], "Value explanation"),
    ("Describe how trees help the planet.", ["trees", "oxygen", "carbon", "environment", "clean", "shade"], "Environmental description"),
    ("Explain what happens when water freezes.", ["freezes", "ice", "temperature", "solid", "cold", "zero"], "Physical change explanation"),
    ("Describe the feeling of reading a fascinating book.", ["reading", "book", "imagination", "story", "curiosity", "fascinating"], "Experiential description"),
    ("What makes clean air essential for city dwellers?", ["air", "health", "lungs", "clean", "breathe", "pollution"], "Clean air value"),
    ("Explain how rivers shape the landscape over centuries.", ["rivers", "water", "erosion", "canyons", "valleys", "flow"], "River erosion"),
    ("Why is good sleep vital for cognitive sharpness?", ["sleep", "brain", "memory", "rest", "focus", "sharp"], "Sleep cognition"),
    ("Describe the experience of watching a sunrise over the mountains.", ["sunrise", "morning", "mountains", "light", "golden", "peace"], "Sunrise scene"),
    ("How does music bring people together across cultures?", ["music", "universal", "culture", "emotion", "connects", "harmony"], "Music connection"),
    ("Explain why biodiversity is crucial for healthy ecosystems.", ["biodiversity", "species", "ecosystem", "balance", "nature", "survival"], "Biodiversity"),
    ("Describe the sensation of a gentle cool breeze on a hot summer day.", ["breeze", "cool", "summer", "refreshing", "wind", "relief"], "Summer breeze"),
    ("Why does teamwork often produce superior solutions to complex problems?", ["teamwork", "perspectives", "collaborate", "solutions", "diverse"], "Teamwork advantages"),
    ("Explain how solar energy helps mitigate carbon emissions.", ["solar", "clean", "emissions", "renewable", "carbon", "energy"], "Solar emissions"),
    ("Describe the quiet discipline required to master an instrument.", ["practice", "discipline", "hours", "music", "mastery", "patience"], "Instrument mastery"),
    ("Why do scientific theories change when new evidence appears?", ["science", "evidence", "theories", "experiments", "update", "facts"], "Scientific method"),
    ("Explain the value of curiosity in personal development.", ["curiosity", "learning", "growth", "questions", "knowledge", "explore"], "Curiosity value"),
    ("Describe how oceans regulate global atmospheric climate.", ["oceans", "heat", "currents", "climate", "temperature", "weather"], "Ocean climate"),
    ("Why is clean potable water a fundamental human right?", ["water", "life", "health", "sanitation", "human", "essential"], "Clean water right"),
    ("Explain the psychological benefit of spending time in nature.", ["nature", "trees", "calm", "stress", "mental", "peace"], "Nature therapy"),
    ("Describe how ancient libraries preserved historical knowledge.", ["libraries", "scrolls", "books", "history", "preserved", "ancient"], "Ancient libraries"),
    ("Why are bees indispensable for agricultural crop pollination?", ["bees", "pollination", "crops", "agriculture", "flowers", "food"], "Bees pollination"),
    ("Explain how physical fitness boosts mental resilience.", ["fitness", "exercise", "mental", "resilience", "endurance", "mood"], "Fitness resilience"),
    ("Describe the feeling of successfully solving a challenging problem.", ["success", "solved", "satisfaction", "effort", "breakthrough", "clarity"], "Problem solved"),
    ("Why does empathy strengthen human communities?", ["empathy", "understanding", "compassion", "trust", "community", "support"], "Empathy value"),
    ("Explain how compilers verify syntactic correctness in code.", ["compiler", "syntax", "parsing", "grammar", "errors", "tokens"], "Compiler syntax"),
    ("Describe a starry night away from bright city light pollution.", ["stars", "night", "sky", "dark", "milky way", "clear"], "Starry night"),
    ("Why is regular reflection on past mistakes helpful for wisdom?", ["reflection", "mistakes", "wisdom", "learn", "growth", "experience"], "Wisdom from mistakes"),
    ("Explain how vaccines protect broader community populations.", ["vaccines", "herd immunity", "protect", "community", "outbreaks"], "Community immunity"),
    ("Describe the dynamic energy of a creative brainstorming workshop.", ["ideas", "creative", "energy", "brainstorming", "collaborative", "innovative"], "Creative energy"),
    ("Why is soil conservation vital for long-term food security?", ["soil", "erosion", "crops", "food", "farming", "nutrients"], "Soil conservation"),
    ("Explain how optical telescopes expanded our cosmic horizons.", ["telescopes", "galaxies", "cosmic", "stars", "universe", "observation"], "Telescope impact"),
    ("Describe the peaceful ambiance of an ancient quiet temple.", ["peaceful", "temple", "quiet", "serene", "calm", "spiritual"], "Temple ambiance"),
    ("Why is clear communication vital in medical care?", ["communication", "doctors", "patients", "safety", "clarity", "health"], "Medical clarity"),
    ("Explain the difference between knowledge and applied wisdom.", ["knowledge", "information", "wisdom", "judgment", "apply", "action"], "Knowledge vs wisdom"),
    ("Describe the transformation of a caterpillar into a butterfly.", ["caterpillar", "chrysalis", "cocoon", "butterfly", "wings", "metamorphosis"], "Metamorphosis"),
    ("Why do ethical standards matter in scientific research?", ["ethics", "integrity", "research", "safety", "trust", "truth"], "Research ethics"),
    ("Explain how languages evolve through cultural exchange over time.", ["language", "evolves", "words", "culture", "exchange", "history"], "Language evolution"),
    ("Describe the quiet satisfaction of writing clean working code.", ["clean", "code", "working", "satisfaction", "elegant", "efficient"], "Clean code feeling"),
    ("Why is architectural modularity advantageous in software?", ["modularity", "components", "maintainable", "decoupled", "isolated", "scalable"], "Software modularity"),
    ("Explain how renewable energy grids balance fluctuating supply.", ["grid", "batteries", "storage", "balance", "solar", "wind"], "Grid balancing"),
    ("Describe the power of habit in achieving long term goals.", ["habits", "daily", "consistency", "goals", "compound", "discipline"], "Power of habit"),
    ("Why is mathematical literacy empowering in modern society?", ["mathematics", "logic", "analytical", "decisions", "data", "numbers"], "Math literacy"),
    ("Explain how deep listening resolves interpersonal conflicts.", ["listening", "empathy", "conflict", "understand", "respect", "resolution"], "Deep listening"),
    ("Describe the beauty of autumn foliage in a temperate forest.", ["autumn", "foliage", "colors", "gold", "red", "leaves", "forest"], "Autumn foliage")
]
for idx, (inp, kw, rub) in enumerate(coherence_prompts, 1):
    add_probe(f"COH_{idx:03d}", "Semantic Consistency", "English", inp, kw, rub)

print(f"Total frozen benchmark probes constructed: {len(probes)}")
cat_counts = {}
for p in probes:
    cat = p["category"]
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

for cat, count in cat_counts.items():
    print(f"  {cat:28s}: {count:2d} probes")

out_file = REPO_ROOT / "experiments/cognition/ucip_06/sealed_500_benchmark.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2, ensure_ascii=False)

# Compute authoritative SHA-256
with open(out_file, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()

print(f"\nSaved sealed benchmark to: {out_file}")
print(f"Authoritative SHA-256: {h}")
