"""
Generate 150-probe baseline diagnostic benchmark for UCIP-03.
Covers:
- exact_copying (20)
- simple_sentence (20)
- factual_qa (20)
- arithmetic (15)
- persona_greeting (15)
- devanagari_hindi (20)
- roman_hindi (15)
- hinglish (15)
- multiturn_context (10)
Total = 150 probes
"""

import json
from pathlib import Path

probes = []

# 1. Exact Copying (20)
copy_items = [
    ("The sky is blue.", "The sky is blue."),
    ("Antigravity is active.", "Antigravity is active."),
    ("ChakrView is a neural model.", "ChakrView is a neural model."),
    ("The cat sat on the mat.", "The cat sat on the mat."),
    ("Water is essential for life.", "Water is essential for life."),
    ("Python is an interpreted language.", "Python is an interpreted language."),
    ("Knowledge is power.", "Knowledge is power."),
    ("Practice makes perfect.", "Practice makes perfect."),
    ("Consistency is the key.", "Consistency is the key."),
    ("Focus on foundational capabilities.", "Focus on foundational capabilities."),
    ("Sun rises in the east.", "Sun rises in the east."),
    ("Birds fly in the sky.", "Birds fly in the sky."),
    ("Trees produce oxygen.", "Trees produce oxygen."),
    ("Open source drives innovation.", "Open source drives innovation."),
    ("Simplicity is sophistication.", "Simplicity is sophistication."),
    ("Learning requires persistence.", "Learning requires persistence."),
    ("Truth prevails.", "Truth prevails."),
    ("Time is valuable.", "Time is valuable."),
    ("Curiosity inspires discovery.", "Curiosity inspires discovery."),
    ("Clarity brings confidence.", "Clarity brings confidence.")
]

for i, (src, exp) in enumerate(copy_items, 1):
    probes.append({
        "probe_id": f"diag_copy_{i:03d}",
        "category": "exact_copying",
        "prompt": f"User: Say exactly: {src}\nAssistant:",
        "target": exp,
        "eval_type": "exact_copy"
    })

# 2. Simple Sentence Generation (20)
subjects = [
    ("a dog", "dog"), ("a cat", "cat"), ("the sun", "sun"), ("the moon", "moon"),
    ("a book", "book"), ("a river", "river"), ("a tree", "tree"), ("an apple", "apple"),
    ("a computer", "computer"), ("a car", "car"), ("a mountain", "mountain"), ("rain", "rain"),
    ("fire", "fire"), ("a flower", "flower"), ("a bird", "bird"), ("music", "music"),
    ("tea", "tea"), ("a chair", "chair"), ("a window", "window"), ("a school", "school")
]

for i, (subj, kw) in enumerate(subjects, 1):
    probes.append({
        "probe_id": f"diag_simple_{i:03d}",
        "category": "simple_sentence",
        "prompt": f"User: Write one simple sentence about {subj}.\nAssistant:",
        "target": kw,
        "eval_type": "keyword_in_sentence"
    })

# 3. Basic Factual QA (20)
facts = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of India?", "New Delhi"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many days are in a week?", "7"),
    ("How many hours are in a day?", "24"),
    ("What gas do humans need to breathe?", "oxygen"),
    ("Which ocean is the largest on Earth?", "Pacific"),
    ("What color do you get by mixing red and blue?", "purple"),
    ("What is the primary language spoken in Germany?", "German"),
    ("What is the shape of the Earth?", "sphere"),
    ("Which animal is known as the king of the jungle?", "lion"),
    ("How many continents are there on Earth?", "7"),
    ("What is the currency of the United Kingdom?", "pound"),
    ("What is frozen water called?", "ice"),
    ("What part of the plant absorbs water from the soil?", "roots"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("What do bees produce?", "honey"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many months are in a year?", "12")
]

for i, (q, ans) in enumerate(facts, 1):
    probes.append({
        "probe_id": f"diag_fact_{i:03d}",
        "category": "factual_qa",
        "prompt": f"User: {q}\nAssistant:",
        "target": ans,
        "eval_type": "substring_match"
    })

# 4. Arithmetic (15)
math_items = [
    ("What is 2 + 2?", "4"),
    ("What is 3 + 5?", "8"),
    ("What is 7 + 6?", "13"),
    ("What is 10 + 15?", "25"),
    ("What is 5 - 2?", "3"),
    ("What is 9 - 4?", "5"),
    ("What is 15 - 7?", "8"),
    ("What is 20 - 8?", "12"),
    ("What is 3 * 3?", "9"),
    ("What is 4 * 5?", "20"),
    ("What is 6 * 2?", "12"),
    ("What is 10 / 2?", "5"),
    ("What is 12 / 3?", "4"),
    ("What is 1 + 1?", "2"),
    ("What is 8 + 8?", "16")
]

for i, (q, ans) in enumerate(math_items, 1):
    probes.append({
        "probe_id": f"diag_math_{i:03d}",
        "category": "arithmetic",
        "prompt": f"User: {q}\nAssistant:",
        "target": ans,
        "eval_type": "number_match"
    })

# 5. Persona & Greeting (15)
greetings = [
    ("Hello! Who are you?", "ChakrView"),
    ("What is your name?", "ChakrView"),
    ("Hi assistant, how are you today?", "assistant"),
    ("Tell me about yourself.", "ChakrView"),
    ("Good morning!", "morning"),
    ("Good evening!", "evening"),
    ("Are you an AI?", "AI"),
    ("What can you do?", "help"),
    ("Who created ChakrView?", "indigenous"),
    ("Hello there!", "hello"),
    ("Can you assist me?", "help"),
    ("What is your purpose?", "assist"),
    ("Hi, nice to meet you.", "meet"),
    ("Greetings!", "greetings"),
    ("Introduce yourself briefly.", "ChakrView")
]

for i, (q, kw) in enumerate(greetings, 1):
    probes.append({
        "probe_id": f"diag_persona_{i:03d}",
        "category": "persona_greeting",
        "prompt": f"User: {q}\nAssistant:",
        "target": kw,
        "eval_type": "persona_presence"
    })

# 6. Devanagari Hindi (20)
devanagari = [
    ("नमस्ते, आप कैसे हैं?", "नमस्ते"),
    ("आपका नाम क्या है?", "ChakrView"),
    ("भारत की राजधानी क्या है?", "दिल्ली"),
    ("पानी का रंग क्या होता है?", "रंगहीन"),
    ("सूरज किस दिशा से उगता है?", "पूर्व"),
    ("एक हफ्ते में कितने दिन होते हैं?", "7"),
    ("सेब का रंग कैसा होता है?", "लाल"),
    ("आप क्या कर सकते हैं?", "मदद"),
    ("दूध का रंग कैसा होता है?", "सफेद"),
    ("पेड़ हमें क्या देते हैं?", "ऑक्सीजन"),
    ("नमस्ते!", "नमस्ते"),
    ("आज का मौसम कैसा है?", "मौसम"),
    ("क्या आप मेरी सहायता करेंगे?", "सहायता"),
    ("गाय हमें क्या देती है?", "दूध"),
    ("पृथ्वी गोल है या चपटी?", "गोल"),
    ("हिंदी हमारी भाषा है।", "भाषा"),
    ("एक साल में कितने महीने होते हैं?", "12"),
    ("पक्षी कहाँ उड़ते हैं?", "आसमान"),
    ("किताबें पढ़ने से क्या मिलता है?", "ज्ञान"),
    ("धन्यवाद!", "स्वागत")
]

for i, (q, ans) in enumerate(devanagari, 1):
    probes.append({
        "probe_id": f"diag_dev_{i:03d}",
        "category": "devanagari_hindi",
        "prompt": f"User: {q}\nAssistant:",
        "target": ans,
        "eval_type": "devanagari_match"
    })

# 7. Roman Hindi (15)
roman_hindi = [
    ("Aapka naam kya hai?", "ChakrView"),
    ("Namaste, aap kaise ho?", "theek"),
    ("Kya aap meri madad kar sakte ho?", "madad"),
    ("Paani pina kyun zaroori hai?", "zaroori"),
    ("Bharat ki rajdhani kya hai?", "Delhi"),
    ("Suraj kahan se nikalta hai?", "east"),
    ("Aap kaun ho?", "ChakrView"),
    ("Mujhe ek kahani sunao.", "kahani"),
    ("Dhanyawaad!", "swagat"),
    ("Aap kya kaam karte ho?", "sahayata"),
    ("Subah ka namaste!", "namaste"),
    ("Doodh ka rang kaisa hota hai?", "safed"),
    ("Ek hafte mein kitne din hote hain?", "7"),
    ("Kitab padhna achhi aadat hai.", "aadat"),
    ("Mausam kaisa hai aaj?", "mausam")
]

for i, (q, ans) in enumerate(roman_hindi, 1):
    probes.append({
        "probe_id": f"diag_rom_{i:03d}",
        "category": "roman_hindi",
        "prompt": f"User: {q}\nAssistant:",
        "target": ans,
        "eval_type": "roman_hindi_match"
    })

# 8. Hinglish (15)
hinglish = [
    ("Hello, aap meri study me help kar sakte ho?", "help"),
    ("Mera project machine learning par hai, aap guide karoge?", "guide"),
    ("What is your name? Aapka intro do please.", "ChakrView"),
    ("Weather kaisa lag raha hai aaj?", "weather"),
    ("Please mujhe simple words me explain karo.", "explain"),
    ("Can you give me an example of Python code?", "code"),
    ("Aaj ka din kaisa chal raha hai aapka?", "accha"),
    ("Aapka response time kaafi fast hai.", "thank"),
    ("Mujhe coding practice karni hai, kahan se start karu?", "start"),
    ("Coffee ya tea, kya better hai?", "tea"),
    ("Is this project open source?", "open"),
    ("Thanks a lot for your help!", "welcome"),
    ("Kya aap multilingual support provide karte ho?", "multilingual"),
    ("Good evening! Aaj hum kya seekhenge?", "seekhenge"),
    ("Simple sentence generate karo about a cat please.", "cat")
]

for i, (q, ans) in enumerate(hinglish, 1):
    probes.append({
        "probe_id": f"diag_hing_{i:03d}",
        "category": "hinglish",
        "prompt": f"User: {q}\nAssistant:",
        "target": ans,
        "eval_type": "hinglish_match"
    })

# 9. Multi-turn Context (10)
multiturn = [
    ("User: My name is Rahul.\nAssistant: Nice to meet you Rahul!\nUser: What is my name?\nAssistant:", "Rahul"),
    ("User: I live in Mumbai.\nAssistant: Mumbai is a vibrant city.\nUser: Where do I live?\nAssistant:", "Mumbai"),
    ("User: My favorite fruit is mango.\nAssistant: Mangoes are delicious.\nUser: What is my favorite fruit?\nAssistant:", "mango"),
    ("User: The secret code is 4921.\nAssistant: Understood, I noted it.\nUser: What is the secret code?\nAssistant:", "4921"),
    ("User: My dog is named Bruno.\nAssistant: Bruno is a great dog name.\nUser: What is my dog's name?\nAssistant:", "Bruno"),
    ("User: I am building a neural network.\nAssistant: Neural networks are fascinating.\nUser: What am I building?\nAssistant:", "neural"),
    ("User: My car is blue.\nAssistant: Blue is a great color.\nUser: What color is my car?\nAssistant:", "blue"),
    ("User: We are meeting on Friday.\nAssistant: Friday is noted on the schedule.\nUser: What day are we meeting?\nAssistant:", "Friday"),
    ("User: My sister's name is Priya.\nAssistant: Hello Priya's brother!\nUser: What is my sister's name?\nAssistant:", "Priya"),
    ("User: The capital city we discussed was Paris.\nAssistant: Paris, the capital of France.\nUser: Which city did we discuss?\nAssistant:", "Paris")
]

for i, (prompt, ans) in enumerate(multiturn, 1):
    probes.append({
        "probe_id": f"diag_mturn_{i:03d}",
        "category": "multiturn_context",
        "prompt": prompt,
        "target": ans,
        "eval_type": "context_retrieval"
    })

out_path = Path("experiments/cognition/ucip_03/evaluation/diagnostic_baseline_150.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2, ensure_ascii=False)

print(f"Generated {len(probes)} probes in {out_path}")
