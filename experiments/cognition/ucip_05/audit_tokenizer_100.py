import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import json
from scripts.run_m3_evaluations import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(Path('data/experiments/vocab_4096'))

# Generate 100+ representative strings across languages and domains
strings = [
    # English Greetings & Daily Chat
    "Hello! How are you doing today?",
    "Good morning! I hope you have a wonderful day ahead.",
    "Can you help me solve this problem?",
    "Thank you very much for your kind assistance.",
    "Nice to meet you, ChakrView.",
    "What is the weather like in your world?",
    "Could you tell me a little bit about yourself?",
    "I am an engineer working on local AI systems.",
    "Let us explore interesting topics together.",
    "See you later and take care!",
    
    # English Concepts & Technical
    "A central processing unit executes instructions fetched from computer memory.",
    "Random access memory provides high-speed volatile storage for active processes.",
    "Machine learning algorithms discover statistical patterns in numerical data.",
    "A compiler translates source code into executable binary machine instructions.",
    "An operating system manages hardware resources and coordinates software processes.",
    "The sky appears blue because molecules in Earth's atmosphere scatter sunlight.",
    "Photosynthesis converts solar light energy into chemical energy stored in glucose.",
    "Gravity is the fundamental force of attraction between masses with energy.",
    "Water boils at 100 degrees Celsius under standard atmospheric sea level pressure.",
    "A bicycle is a two-wheeled vehicle propelled by human pedal power.",
    "The solar system consists of the Sun and celestial objects gravitationally bound to it.",
    "Electric motors convert electrical energy into mechanical rotational kinetic energy.",
    "Clean architecture decouples core business domain logic from peripheral UI frameworks.",
    "Unit testing validates individual isolated software units against expected contracts.",
    "Continuous integration automatically builds and tests code commits across branches.",
    "Neural networks pass weighted linear combinations through non-linear activation functions.",
    "Backpropagation computes partial derivatives of the loss with respect to parameters.",
    "Autoregressive generation samples tokens sequentially conditioned on historical prefixes.",
    "A database index accelerates record retrieval at the cost of additional write overhead.",
    "Recursion is a technique where a function calls itself to solve smaller subproblems.",

    # Devanagari Hindi
    "नमस्ते दुनिया! आज का दिन बहुत सुंदर और सुखद है।",
    "आप कैसे हैं? क्या सब कुछ ठीक चल रहा है?",
    "मैं एक स्वतंत्र न्यूरल भाषा मॉडल हूँ जो स्थानीय रूप से काम करता है।",
    "कंप्यूटर एक इलेक्ट्रॉनिक उपकरण है जो सूचनाओं को प्रोसेस करता है।",
    "भारत एक प्राचीन, महान और सांस्कृतिक रूप से समृद्ध राष्ट्र है।",
    "शिक्षा जीवन का सबसे महत्वपूर्ण और अमूल्य उपहार है।",
    "ज्ञान हमें सही और गलत में अंतर करने का सामर्थ्य देता है।",
    "पेड़-पौधे हवा को शुद्ध करते हैं और ऑक्सीजन प्रदान करते हैं।",
    "नदियाँ पर्वतों से निकलकर मैदानों को उपजाऊ और हरा-भरा बनाती हैं।",
    "समय का सदुपयोग करना ही वास्तविक सफलता का रहस्य है।",
    "सूर्य सौरमंडल का केंद्र है और पृथ्वी उसके चारों ओर परिक्रमा करती है।",
    "हिंदी भाषा भारत की प्रमुख और आधिकारिक भाषाओं में से एक है।",
    "देवनागरी लिपि में अक्षरों का उच्चारण जैसा लिखा जाता है वैसा ही होता है।",
    "मेहनत और ईमानदारी से किया गया कार्य कभी व्यर्थ नहीं जाता।",
    "सत्य और अहिंसा भारतीय संस्कृति के मूल स्तंभ माने जाते हैं।",
    "पुस्तकें मनुष्य की सबसे अच्छी और सच्ची मित्र होती हैं।",
    "स्वास्थ्य ही सबसे बड़ा धन और जीवन की सबसे बड़ी पूंजी है।",
    "विज्ञान और प्रौद्योगिकी ने हमारे जीवन को सरल और सुगम बना दिया है।",
    "क्या आप मुझे गणित के कुछ आसान नियम समझा सकते हैं?",
    "धन्यवाद! आपका यह उत्तर मेरे लिए बहुत उपयोगी और ज्ञानवर्धक रहा।",

    # Roman Hindi
    "Aap kaise hain? Sab theek thaak hai na?",
    "Main ChakrView hoon, ek lightweight CPU-first conversational assistant.",
    "Bhai, machine learning ko simple language mein samjhao.",
    "Computer ka CPU instructions execute karta hai aur data flow coordinate karta hai.",
    "Yeh project ek dam local hardware par execute hota hai bina external API ke.",
    "Aaj ka mausam kaafi suhana aur thanda lag raha hai.",
    "Kya aap meri coding problem solve karne mein help kar sakte ho?",
    "Python language seekhna bahut aasan aur mazedaar hai.",
    "Mehnat karne se har mushkil aasan ho sakti hai.",
    "Chalo milkar naye concepts seekhte hain aur study karte hain.",
    "Shukriya bhai, aapki advice bahut helpful thi.",
    "Kal subah hum naya topic start karenge.",
    "ChakrView CPU par bina GPU ke efficiently execute ho sakta hai.",
    "Har step par verify karna aur code test karna best practice hai.",
    "Aapka bahut bahut dhanyawaad!",

    # Hinglish Code-Switched
    "Machine learning mein models training data se complex patterns learn karte hain.",
    "CPU instructions decode karta hai aur calculations execute karta hai.",
    "Neural network weights gradient descent ke through update hote hain.",
    "Yeh model completely local CPU environment mein lightweight inference deta hai.",
    "Agar loss decrease ho raha hai, toh validation set par check karna zaroori hai.",
    "Code debugging ke liye logs aur traces check karna sabse reliable method hai.",
    "Hum unit tests run karke pipeline integrity verify karte hain.",
    "Transformer architecture self-attention mechanism use karke tokens relate karta hai.",
    "Data preprocessing ke bina model performance improve nahi hoti.",
    "ChakrView external APIs ya pretrained weights par bilkul depend nahi karta.",
    "Python list comprehension code ko compact aur clean banati hai.",
    "Memory management operating system ka ek critical task hota hai.",
    "API endpoints securely handle karna software engineering mein essential hai.",
    "Bhai, simple explanation do taaki beginner bhi easily samajh sake.",
    "Thanks for the detailed clarification, ab doubt clearly clear ho gaya.",

    # Exact Copying / Formatting / Quotes
    "Say exactly: The sky is blue.",
    "Repeat after me: In the middle of difficulty lies opportunity.",
    "Copy this text: Python 3.12 is fast, reliable, and expressive.",
    "Echo verbatim: [DEBUG_LOG_001] System health status is OK.",
    "The quick brown fox jumps over the lazy dog.",

    # Arithmetic & Number Expressions
    "If I have 5 apples and give 2 to my friend, 3 apples remain.",
    "12 + 15 = 27",
    "100 - 37 = 63",
    "8 * 9 = 72",
    "144 / 12 = 12",
    "Calculate: 25 plus 17 equals 42.",
    "Solve: 50 minus 18 gives 32.",

    # Multi-turn formatting tokens
    "User: Hello!\nAssistant: Hi there! How can I help you today?",
    "User: What is 2 + 2?\nAssistant: 2 + 2 equals 4.",
    "User: नमस्ते!\nAssistant: नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?"
]

print(f"Total probe strings to test: {len(strings)}")
perfect_count = 0
mismatches = []

for idx, text in enumerate(strings):
    enc = tok.encode(text, add_bos=False, add_eos=False)
    dec = tok.decode(enc)
    if dec == text:
        perfect_count += 1
    else:
        mismatches.append((idx, text, dec, enc))

print(f"Round-trip results: {perfect_count}/{len(strings)} exact matches.")
if mismatches:
    print(f"Detected {len(mismatches)} mismatches:")
    for idx, orig, dec, enc in mismatches:
        print(f"[{idx}] Orig: {repr(orig)} != Dec: {repr(dec)}")
else:
    print("ALL 100+ STRINGS PASSED ROUND-TRIP WITH 100.0% FIDELITY!")
