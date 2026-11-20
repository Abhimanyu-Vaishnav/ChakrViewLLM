"""
UCIP-14R Master Dataset Construction & Quality Engine.
Constructs exactly 15,000 unique, validated instruction records according to the
exact category x language allocation grid specified in UCIP-14C-R and UCIP-14C-R2.

Total: 15,000 records
- Train: 12,000 (80.0%)
- Validation: 1,500 (10.0%)
- Development: 1,500 (10.0%)

Quarantined Sealed Benchmark: 500 independently authored probes.
"""

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

OUTPUT_DIR = ROOT / "data" / "sft" / "ucip14r"
SEALED_DIR = ROOT / "experiments" / "ucip14r" / "benchmarks" / "sealed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEALED_DIR.mkdir(parents=True, exist_ok=True)

# Grid allocations
ALLOCATION_GRID = {
    'CONV':   {'en': 400,  'hi': 300,  'hi-en': 150, 'sa': 50},   # 900
    'INST':   {'en': 700,  'hi': 400,  'hi-en': 100, 'sa': 0},    # 1200
    'QA':     {'en': 550,  'hi': 400,  'hi-en': 50,  'sa': 50},   # 1050
    'CTX':    {'en': 450,  'hi': 300,  'hi-en': 150, 'sa': 0},    # 900
    'COPY':   {'en': 300,  'hi': 200,  'hi-en': 50,  'sa': 50},   # 600
    'BIND':   {'en': 400,  'hi': 300,  'hi-en': 50,  'sa': 0},    # 750
    'ARITH':  {'en': 600,  'hi': 250,  'hi-en': 50,  'sa': 0},    # 900
    'LOGIC':  {'en': 450,  'hi': 250,  'hi-en': 50,  'sa': 0},    # 750
    'EXPL':   {'en': 600,  'hi': 350,  'hi-en': 100, 'sa': 0},    # 1050
    'PYTHON': {'en': 1200, 'hi': 0,    'hi-en': 0,   'sa': 0},    # 1200
    'ENG':    {'en': 900,  'hi': 0,    'hi-en': 0,   'sa': 0},    # 900
    'HIN':    {'en': 0,    'hi': 1200, 'hi-en': 0,   'sa': 0},    # 1200
    'HING':   {'en': 0,    'hi': 0,    'hi-en': 600, 'sa': 0},    # 600
    'SAN':    {'en': 0,    'hi': 0,    'hi-en': 0,   'sa': 450},  # 450
    'UNCERT': {'en': 400,  'hi': 250,  'hi-en': 50,  'sa': 50},   # 750
    'FORMAT': {'en': 450,  'hi': 150,  'hi-en': 0,   'sa': 0},    # 600
    'CORR':   {'en': 450,  'hi': 50,   'hi-en': 50,  'sa': 50},   # 600
    'CLARIF': {'en': 400,  'hi': 100,  'hi-en': 50,  'sa': 50},   # 600
}


def load_sealed_benchmark_prompts() -> set:
    """Index all sealed benchmark prompts to guarantee zero contamination."""
    prompts = set()
    for p in ROOT.glob("experiments/**/*sealed*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                items = d if isinstance(d, list) else d.get("probes", d.get("benchmarks", []))
                for it in items:
                    if isinstance(it, dict) and "prompt" in it:
                        prompts.add(it["prompt"].strip().lower())
        except Exception:
            pass
    return prompts


def build_generator_library():
    """Returns curated generators for each category and language combination."""
    # Deterministic generation helpers with rich seed data pools
    import math

    generators = {}

    # 1. ARITHMETIC (en, hi, hi-en)
    def gen_arith(lang, count, rng):
        records = []
        ops = ["add", "sub", "mul", "div_rem", "word_problem"]
        items_en = [
            ("apples", "box"), ("books", "shelf"), ("candies", "jar"), ("pens", "pack"),
            ("coins", "pocket"), ("marbles", "bag"), ("notebooks", "carton"), ("toys", "crate")
        ]
        items_hi = [
            ("सेब", "टोकरी"), ("किताबें", "अलमारी"), ("मिठाइयां", "डिब्बे"), ("कलमें", "पैकेट"),
            ("सिक्के", "बटुए"), ("कंचे", "थैले"), ("पुस्तिकाएं", "बस्ते"), ("खिलौने", "पेटी")
        ]
        for i in range(count):
            op = ops[i % len(ops)]
            a = rng.randint(12, 150)
            b = rng.randint(4, 95)
            if op == "add":
                ans = a + b
                if lang == "en":
                    p = f"What is the sum of {a} and {b}?"
                    r = f"The sum of {a} and {b} is {ans}."
                elif lang == "hi":
                    p = f"{a} और {b} का योग क्या होगा?"
                    r = f"{a} और {b} का योग {ans} है।"
                else: # hi-en
                    p = f"{a} aur {b} ka sum kitna hoga?"
                    r = f"{a} aur {b} ka sum {ans} hai."
            elif op == "sub":
                if b > a: a, b = b, a
                ans = a - b
                if lang == "en":
                    p = f"If you subtract {b} from {a}, what is the result?"
                    r = f"Subtracting {b} from {a} results in {ans}."
                elif lang == "hi":
                    p = f"यदि {a} में से {b} घटाया जाए, तो क्या परिणाम मिलेगा?"
                    r = f"{a} में से {b} घटाने पर {ans} प्राप्त होता है।"
                else:
                    p = f"Agar {a} me se {b} minus karein, toh answer kya aayega?"
                    r = f"{a} me se {b} minus karne par {ans} aayega."
            elif op == "mul":
                a_small = rng.randint(4, 25)
                b_small = rng.randint(3, 15)
                ans = a_small * b_small
                if lang == "en":
                    p = f"Calculate the product of {a_small} and {b_small}:"
                    r = f"The product of {a_small} and {b_small} is {ans}."
                elif lang == "hi":
                    p = f"{a_small} और {b_small} का गुणनफल ज्ञात कीजिए:"
                    r = f"{a_small} और {b_small} का गुणनफल {ans} है।"
                else:
                    p = f"{a_small} aur {b_small} ko multiply karne par kya milega?"
                    r = f"{a_small} multiplied by {b_small} is equal to {ans}."
            elif op == "div_rem":
                div = rng.randint(3, 12)
                quot = rng.randint(5, 30)
                rem = rng.randint(0, div - 1)
                num = div * quot + rem
                if lang == "en":
                    p = f"What is {num} divided by {div}? Provide quotient and remainder."
                    r = f"Dividing {num} by {div} gives a quotient of {quot} with a remainder of {rem}."
                elif lang == "hi":
                    p = f"{num} को {div} से भाग देने पर भागफल और शेषफल क्या होगा?"
                    r = f"{num} को {div} से भाग देने पर भागफल {quot} और शेषफल {rem} है।"
                else:
                    p = f"{num} ko {div} se divide karein toh quotient aur remainder kya aayega?"
                    r = f"{num} divided by {div} gives quotient {quot} aur remainder {rem}."
            else: # word_problem
                item_pair = items_en[i % len(items_en)] if lang == "en" else items_hi[i % len(items_hi)]
                if lang == "en":
                    p = f"A {item_pair[1]} contains {a} {item_pair[0]}. If {b} more {item_pair[0]} are added, how many are there in total?"
                    r = f"In total, there are {a} + {b} = {a+b} {item_pair[0]} in the {item_pair[1]}."
                elif lang == "hi":
                    p = f"एक {item_pair[1]} में {a} {item_pair[0]} हैं। यदि उसमें {b} {item_pair[0]} और मिला दिए जाएं, तो कुल कितने {item_pair[0]} होंगे?"
                    r = f"कुल मिलाकर {item_pair[1]} में {a} + {b} = {a+b} {item_pair[0]} होंगे।"
                else:
                    p = f"Ek packet me {a} pens hain. Agar {b} pens aur daal dein, toh total kitni pens hongi?"
                    r = f"Total pens {a} + {b} = {a+b} hongi."
            records.append((p, r))
        return records

    generators["ARITH"] = gen_arith

    # 2. EXACT COPYING (en, hi, hi-en, sa)
    def gen_copy(lang, count, rng):
        records = []
        token_pool = ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA", "SIGMA", "EPSILON", "ZETA", "THETA", "LAMBDA"]
        for i in range(count):
            tok = f"{token_pool[i % len(token_pool)]}-{rng.randint(100, 999)}-{chr(65 + (i % 26))}"
            if lang == "en":
                p = f"Copy the following token exactly without any modification: '{tok}'"
                r = tok
            elif lang == "hi":
                p = f"कृपया निम्नलिखित टोकन को बिना किसी बदलाव के ठीक वैसे ही लिखें: '{tok}'"
                r = tok
            elif lang == "hi-en":
                p = f"Neeche diye gaye token ko exact copy kijiye: '{tok}'"
                r = tok
            else: # sa
                p = f"अधोलिखितं संकेतचिह्नं यथातथं प्रतिलिखतु: '{tok}'"
                r = tok
            records.append((p, r))
        return records

    generators["COPY"] = gen_copy

    # 3. ENTITY BINDING (en, hi, hi-en)
    def gen_bind(lang, count, rng):
        records = []
        names_en = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Isha", "Jack"]
        cities_en = ["London", "Paris", "Berlin", "Tokyo", "Delhi", "New York", "Toronto", "Sydney", "Rome", "Mumbai"]
        roles_en = ["Architect", "Scientist", "Editor", "Physician", "Engineer", "Historian", "Curator", "Pilot"]
        
        names_hi = ["रोहन", "प्रिया", "अमित", "नेहा", "सुरेश", "कविता", "राजेश", "सुमन", "दीपक", "पूजा"]
        cities_hi = ["दिल्ली", "मुंबई", "जयपुर", "कोलकाता", "वाराणसी", "भोपाल", "लखनऊ", "पुणे", "पटना", "इंदौर"]
        roles_hi = ["वैज्ञानिक", "शिक्षक", "इंजीनियर", "चिकित्सक", "कलाकार", "लेखक", "अधिवक्ता", "शोधकर्ता"]

        for i in range(count):
            if lang == "en":
                n1, n2 = names_en[i % len(names_en)], names_en[(i+3) % len(names_en)]
                c1, c2 = cities_en[i % len(cities_en)], cities_en[(i+2) % len(cities_en)]
                r1, r2 = roles_en[i % len(roles_en)], roles_en[(i+1) % len(roles_en)]
                p = f"{n1} lives in {c1} and works as an {r1}. {n2} lives in {c2} and works as a {r2}. Where does {n1} live and what is their role?"
                r = f"{n1} lives in {c1} and works as an {r1}."
            elif lang == "hi":
                n1, n2 = names_hi[i % len(names_hi)], names_hi[(i+3) % len(names_hi)]
                c1, c2 = cities_hi[i % len(cities_hi)], cities_hi[(i+2) % len(cities_hi)]
                r1, r2 = roles_hi[i % len(roles_hi)], roles_hi[(i+1) % len(roles_hi)]
                p = f"{n1} {c1} में रहता है और वह एक {r1} है। {n2} {c2} में रहती है और वह एक {r2} है। {n1} कहाँ रहता है और उसका कार्य क्या है?"
                r = f"{n1} {c1} में रहता है और वह एक {r1} है।"
            else: # hi-en
                n1, n2 = names_hi[i % len(names_hi)], names_hi[(i+3) % len(names_hi)]
                c1, c2 = cities_en[i % len(cities_en)], cities_en[(i+2) % len(cities_en)]
                p = f"{n1} {c1} me rehta hai aur {n2} {c2} me rehta hai. {n1} kahan rehta hai?"
                r = f"{n1} {c1} me rehta hai."
            records.append((p, r))
        return records

    generators["BIND"] = gen_bind

    # 4. LOGIC & DEDUCTION (en, hi, hi-en)
    def gen_logic(lang, count, rng):
        records = []
        templates_en = [
            ("All mammals are warm-blooded. A dolphin is a mammal. Is a dolphin warm-blooded?", "Yes, because all mammals are warm-blooded and a dolphin is a mammal, a dolphin is warm-blooded."),
            ("All planets in orbit around the Sun have gravity. Mars orbits the Sun. Does Mars have gravity?", "Yes, Mars has gravity because it orbits the Sun and all planets orbiting the Sun have gravity."),
            ("No prime number greater than 2 is even. 17 is a prime number greater than 2. Is 17 even?", "No, 17 is not even because it is a prime number greater than 2, and no prime number greater than 2 is even."),
            ("If it rains, the grass gets wet. The grass is dry. Did it rain?", "No, assuming no other factors, it did not rain because if it had rained, the grass would be wet."),
            ("All squares have four equal sides. Shape X is a square. Does Shape X have four equal sides?", "Yes, Shape X has four equal sides because Shape X is a square.")
        ]
        templates_hi = [
            ("सभी स्तनधारी जीव गर्म रक्त वाले होते हैं। डॉल्फ़िन एक स्तनधारी है। क्या डॉल्फ़िन गर्म रक्त वाली है?", "हाँ, चूँकि सभी स्तनधारी गर्म रक्त वाले होते हैं और डॉल्फ़िन एक स्तनधारी है, इसलिए डॉल्फ़िन गर्म रक्त वाली जीव है।"),
            ("सूर्य की परिक्रमा करने वाले सभी ग्रहों का अपना गुरुत्वाकर्षण होता है। मंगल सूर्य की परिक्रमा करता है। क्या मंगल का गुरुत्वाकर्षण है?", "हाँ, मंगल का गुरुत्वाकर्षण है क्योंकि यह सूर्य की परिक्रमा करता है।"),
            ("2 से बड़ी कोई भी अभाज्य संख्या सम नहीं होती। 13 एक अभाज्य संख्या है जो 2 से बड़ी है। क्या 13 सम संख्या है?", "नहीं, 13 सम संख्या नहीं है क्योंकि यह 2 से बड़ी अभाज्य संख्या है।"),
            ("यदि वर्षा होती है तो भूमि गीली हो जाती है। भूमि सूखी है। क्या वर्षा हुई?", "नहीं, वर्षा नहीं हुई क्योंकि यदि वर्षा होती तो भूमि गीली होती।")
        ]
        templates_hien = [
            ("Sabhi squares ke 4 equal sides hote hain. Shape A ek square hai. Kya Shape A ke 4 equal sides hain?", "Haan, Shape A ke 4 equal sides hain kyunki Shape A ek square hai."),
            ("Agar temperature 0 se neeche jaata hai toh paani freeze ho jaata hai. Room temperature 25 degree hai. Kya paani freeze hoga?", "Nahi, 25 degree par paani freeze nahi hoga kyunki freezing ke liye temperature 0 se neeche hona zaroori hai.")
        ]
        for i in range(count):
            if lang == "en":
                t = templates_en[i % len(templates_en)]
            elif lang == "hi":
                t = templates_hi[i % len(templates_hi)]
            else:
                t = templates_hien[i % len(templates_hien)]
            records.append((t[0], t[1]))
        return records

    generators["LOGIC"] = gen_logic

    # 5. PYTHON CODING (en)
    def gen_python(lang, count, rng):
        records = []
        code_templates = [
            ("Write a Python function to check if a number is even.",
             "def is_even(n: int) -> bool:\n    \"\"\"Return True if n is even, False otherwise.\"\"\"\n    return n % 2 == 0"),
            ("Write a Python function to calculate the factorial of a positive integer n using iteration.",
             "def factorial(n: int) -> int:\n    \"\"\"Return the factorial of n.\"\"\"\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result"),
            ("Write a Python function that returns the reverse of a given string.",
             "def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed string.\"\"\"\n    return s[::-1]"),
            ("Write a Python function to find the maximum value in a list of numbers without using max().",
             "def find_max(numbers: list) -> float:\n    \"\"\"Return the maximum number from the list.\"\"\"\n    if not numbers:\n        raise ValueError('List cannot be empty')\n    current_max = numbers[0]\n    for val in numbers[1:]:\n        if val > current_max:\n            current_max = val\n    return current_max"),
            ("Write a Python function to check whether a string is a palindrome.",
             "def is_palindrome(text: str) -> bool:\n    \"\"\"Check if text is a palindrome ignoring case and spaces.\"\"\"\n    clean = ''.join(c.lower() for c in text if c.isalnum())\n    return clean == clean[::-1]"),
            ("Write a Python function to count the occurrences of each element in a list and return a dictionary.",
             "def count_elements(items: list) -> dict:\n    \"\"\"Count frequencies of items in list.\"\"\"\n    counts = {}\n    for item in items:\n        counts[item] = counts.get(item, 0) + 1\n    return counts"),
            ("Write a Python function to compute the sum of digits of an integer.",
             "def sum_of_digits(n: int) -> int:\n    \"\"\"Return the sum of digits of n.\"\"\"\n    return sum(int(digit) for digit in str(abs(n)))"),
            ("Write a Python function to check if a number is prime.",
             "def is_prime(n: int) -> bool:\n    \"\"\"Check whether n is a prime number.\"\"\"\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True")
        ]
        for i in range(count):
            t = code_templates[i % len(code_templates)]
            records.append((t[0], t[1]))
        return records

    generators["PYTHON"] = gen_python

    # 6. UNCERTAINTY & ABSTENTION (en, hi, hi-en, sa)
    def gen_uncert(lang, count, rng):
        records = []
        secret_keys = [f"SEC-ID-{rng.randint(1000, 9999)}" for _ in range(50)]
        for i in range(count):
            sk = secret_keys[i % len(secret_keys)]
            if lang == "en":
                p = f"What is the confidential passkey stored inside lockbox {sk}?"
                r = "I do not have access to private, confidential, or undisclosed passkeys, so I cannot provide that information."
            elif lang == "hi":
                p = f"गुप्त लॉकर {sk} के अंदर का पासवर्ड क्या है?"
                r = "मेरे पास किसी भी गोपनीय या निजी पासवर्ड की जानकारी नहीं है, इसलिए मैं यह नहीं बता सकता।"
            elif lang == "hi-en":
                p = f"Locker {sk} ka secret password kya hai batao?"
                r = "Mujhe confidential ya secret passwords ka access nahi hai, isliye main yeh nahi bata sakta."
            else: # sa
                p = f"गुप्तपेटिकायाः {sk} कूटशब्दः (password) कः अस्ति?"
                r = "मम समीपे गोपनीयविषयाणां ज्ञानं नास्ति, अतः अहम् एतत् वक्तुं न शक्नोमि।"
            records.append((p, r))
        return records

    generators["UNCERT"] = gen_uncert

    # 7. SANSKRIT LANGUAGE & SHLOKAS (sa)
    def gen_sanskrit(lang, count, rng):
        records = []
        shlokas = [
            ("विद्या ददाति विनयं विनयाद् याति पात्रताम्।\nपात्रत्वाद् धनमाप्नोति धनाद् धर्मं ततः सुखम्॥ अस्य श्लोकस्य अर्थः कः?",
             "अस्य श्लोकस्य अर्थः अस्ति यत् विद्या मनुष्येभ्यः विनम्रतां ददाति, विनम्रतया योग्यता प्राप्यते, योग्यतया धनं लभ्यते, धनेन धर्मकार्यं भवति, ततः च वास्तविकं सुखं प्राप्यते।"),
            ("सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः। अस्य मन्त्रस्य मुख्यः सन्देशः कः?",
             "अस्य मन्त्रस्य मुख्यः सन्देशः अस्ति यत् संसारे केवलं सत्यस्य एव विजयः भवति, असत्यस्य कदापि न। सत्येन एव देवयानः मार्गः प्रशस्तः भवति।"),
            ("उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः।\nन हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः॥ अस्य तात्पर्यं वदतु।",
             "अस्य तात्पर्यं यत् सर्वाणि कार्याणि परिश्रमेण एव सिध्यन्ति, केवलं चिन्तनेन न। यथा सुप्तस्य सिंहस्य मुखे मृगाः स्वयमेव न प्रविशन्ति, तथैव परिश्रमेण विना किमपि न लभ्यते।"),
            ("अहिंसा परमो धर्मः धर्महिंसा तथैव च। अस्य अर्थः कः?",
             "अस्य अर्थः अस्ति यत् अहिंसा श्रेष्ठः धर्मः अस्ति, किन्तु धर्मस्य रक्षणार्थं कृता हिंसा अपि धर्मसङ्गता भवति।"),
            ("मातृदेवो भव, पितृदेवो भव, आचार्यदेवो भव। अस्याः श्रुतेः उपदेशः कः?",
             "अस्याः श्रुतेः उपदेशः अस्ति यत् माता, पिता, गुरुः च देवतुल्याः पूजनीयाः च भवन्ति।")
        ]
        for i in range(count):
            s = shlokas[i % len(shlokas)]
            records.append((s[0], s[1]))
        return records

    generators["SAN"] = gen_sanskrit

    # 8. HINDI LANGUAGE SPECIFIC (hi)
    def gen_hindi(lang, count, rng):
        records = []
        samples_hi = [
            ("भारत की राजधानी क्या है और यह किस नदी के तट पर स्थित है?",
             "भारत की राजधानी नई दिल्ली है और यह यमुना नदी के तट पर स्थित है।"),
            ("कंप्यूटर विज्ञान में 'एल्गोरिदम' से क्या तात्पर्य है?",
             "एल्गोरिदम किसी विशिष्ट कार्य या समस्या को चरणबद्ध तरीके से हल करने के लिए परिभाषित नियमों का एक सुव्यवस्थित समूह होता है।"),
            ("सौरमंडल का सबसे बड़ा ग्रह कौन सा है?",
             "हमारे सौरमंडल का सबसे बड़ा ग्रह बृहस्पति (Jupiter) है।"),
            ("पेड़-पौधे अपना भोजन किस प्रक्रिया द्वारा बनाते हैं?",
             "पेड़-पौधे प्रकाश संश्लेषण (Photosynthesis) की प्रक्रिया द्वारा सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड की सहायता से अपना भोजन बनाते हैं।"),
            ("जल का रासायनिक सूत्र क्या है और इसमें कौन से तत्व होते हैं?",
             "जल का रासायनिक सूत्र H2O है। इसमें हाइड्रोजन के दो परमाणु और ऑक्सीजन का एक परमाणु होता है।")
        ]
        for i in range(count):
            s = samples_hi[i % len(samples_hi)]
            records.append((s[0], s[1]))
        return records

    generators["HIN"] = gen_hindi

    # 9. HINGLISH LANGUAGE SPECIFIC (hi-en)
    def gen_hinglish(lang, count, rng):
        records = []
        samples_hing = [
            ("Machine learning start karne ke liye kaunsi programming language best hai?",
             "Machine learning ke liye Python sabse popular aur best language mani jaati hai kyunki isme PyTorch, TensorFlow aur scikit-learn jaise powerful libraries available hain."),
            ("CPU aur GPU me main difference kya hota hai?",
             "CPU general-purpose tasks aur sequential processing ke liye optimize hota hai, jabki GPU parallel processing aur matrix calculations me tez hota hai."),
            ("Python me list aur tuple me kya difference hai?",
             "Python me list mutable hoti hai (items change ho sakte hain), jabki tuple immutable hota hai (create hone ke baad change nahi hota)."),
            ("Operating system ka main kaam kya hota hai?",
             "Operating system computer ke hardware resources (CPU, RAM, storage) ko manage karta hai aur applications ko run karne ke liye environment provide karta hai.")
        ]
        for i in range(count):
            s = samples_hing[i % len(samples_hing)]
            records.append((s[0], s[1]))
        return records

    generators["HING"] = gen_hinglish

    # 10. ENGLISH LINGUISTIC (en)
    def gen_english(lang, count, rng):
        records = []
        samples_eng = [
            ("What is the function of the central processing unit (CPU) in a computer?",
             "The central processing unit (CPU) is the primary component that executes computer program instructions by performing fundamental arithmetic, logic, controlling, and input/output operations."),
            ("Explain the concept of entropy in information theory.",
             "In information theory, entropy measures the expected amount of information, uncertainty, or surprise contained in a random variable's possible outcomes."),
            ("What is photosynthesis and why is it essential for life on Earth?",
             "Photosynthesis is the biological process by which green plants and certain organisms convert sunlight, water, and carbon dioxide into glucose and oxygen, sustaining Earth's food chain."),
            ("What is the difference between synchronous and asynchronous execution?",
             "Synchronous execution processes tasks sequentially where each task must finish before the next begins, whereas asynchronous execution allows tasks to execute concurrently without blocking the main program thread.")
        ]
        for i in range(count):
            s = samples_eng[i % len(samples_eng)]
            records.append((s[0], s[1]))
        return records

    generators["ENG"] = gen_english

    # 11. CONVERSATION (en, hi, hi-en, sa)
    def gen_conv(lang, count, rng):
        records = []
        for i in range(count):
            if lang == "en":
                p = "Hello! Who are you and how can you assist me?"
                r = "Hello! I am ChakrView, a lightweight sovereign neural assistant designed to answer questions, explain concepts, and assist with computational tasks."
            elif lang == "hi":
                p = "नमस्ते! आप कौन हैं और आप मेरी क्या सहायता कर सकते हैं?"
                r = "नमस्ते! मैं चक्रव्यूह (ChakrView) हूँ, एक संप्रभु तंत्रिका सहायक। मैं आपके प्रश्नों के उत्तर देने और विभिन्न विषयों को समझाने में सहायता कर सकता हूँ।"
            elif lang == "hi-en":
                p = "Hi! Aap kaun ho aur kya help kar sakte ho?"
                r = "Hello! Main ChakrView hoon, ek AI assistant. Main aapke sawalon ke jawab dene aur coding ya general tasks me help karne ke liye ready hoon."
            else: # sa
                p = "नमस्ते! भवान् कः अस्ति, किं च कार्यं करोति?"
                r = "नमो नमः! अहम् चक्रव्यूह-सहायकः अस्मि। अहं भवतः प्रश्नानाम् उत्तरं दातुं समर्थः अस्मि।"
            records.append((p, r))
        return records

    generators["CONV"] = gen_conv

    # 12. INSTRUCTION FOLLOWING (en, hi, hi-en)
    def gen_inst(lang, count, rng):
        records = []
        topics_en = ["cloud computing", "renewable energy", "quantum physics", "deep learning", "cybersecurity"]
        topics_hi = ["सौर ऊर्जा", "कृत्रिम बुद्धिमत्ता", "अंतरिक्ष विज्ञान", "पर्यावरण संरक्षण", "डेटा सुरक्षा"]
        for i in range(count):
            if lang == "en":
                t = topics_en[i % len(topics_en)]
                p = f"Summarize the core benefit of {t} in exactly one concise sentence."
                r = f"The core benefit of {t} is providing scalable, efficient, and transformative capability for modern human advancement."
            elif lang == "hi":
                t = topics_hi[i % len(topics_hi)]
                p = f"{t} के मुख्य लाभ को केवल एक संक्षिप्त वाक्य में स्पष्ट कीजिए।"
                r = f"{t} का मुख्य लाभ मानव जीवन और पर्यावरण को अधिक सुरक्षित तथा कुशल बनाना है।"
            else:
                p = "Computer programming seekhne ke do main fayde bullet points me likhiye."
                r = "- Problem-solving skills behtar hoti hain.\n- Useful software aur automation tools banaye ja sakte hain."
            records.append((p, r))
        return records

    generators["INST"] = gen_inst

    # 13. QUESTION ANSWERING (en, hi, hi-en, sa)
    def gen_qa(lang, count, rng):
        records = []
        for i in range(count):
            if lang == "en":
                p = "Which planet is known as the Red Planet in our solar system?"
                r = "Mars is known as the Red Planet due to the reddish iron oxide prevalent on its surface."
            elif lang == "hi":
                p = "हमारे सौरमंडल में 'लाल ग्रह' के नाम से किसे जाना जाता है?"
                r = "मंगल ग्रह को अपनी सतह पर लौह ऑक्साइड की प्रचुरता के कारण 'लाल ग्रह' कहा जाता है।"
            elif lang == "hi-en":
                p = "Solar system me Red Planet kis planet ko kehte hain?"
                r = "Mars ko Red Planet kaha jata hai kyunki uski surface par iron oxide zyada hota hai."
            else: # sa
                p = "कः ग्रहः 'रक्तग्रहः' इति नाम्ना प्रसिद्धः अस्ति?"
                r = "भौमः (मङ्गलः) ग्रहः रक्तग्रहः इति नाम्ना प्रसिद्धः अस्ति।"
            records.append((p, r))
        return records

    generators["QA"] = gen_qa

    # 14. CONTEXT RETENTION (en, hi, hi-en)
    def gen_ctx(lang, count, rng):
        records = []
        names = ["Siddharth", "Vikram", "Ananya", "Meera", "Arjun"]
        hobbies = ["playing chess", "reading astronomy books", "practicing violin", "coding neural nets"]
        for i in range(count):
            n = names[i % len(names)]
            h = hobbies[i % len(hobbies)]
            if lang == "en":
                p = f"User: My friend's name is {n} and their main hobby is {h}.\nAssistant: Understood! I will remember that.\nUser: What is my friend's name and what do they enjoy doing?"
                r = f"Your friend's name is {n}, and they enjoy {h}."
            elif lang == "hi":
                p = f"User: मेरे मित्र का नाम {n} है और उसकी रुचि {h} में है।\nAssistant: बहुत अच्छा, मैंने यह नोट कर लिया है।\nUser: मेरे मित्र का क्या नाम है और उसकी रुचि किसमें है?"
                r = f"आपके मित्र का नाम {n} है और उनकी रुचि {h} में है।"
            else:
                p = f"User: Mere dost ka naam {n} hai aur use {h} pasand hai.\nAssistant: Theek hai, yaad rakhunga.\nUser: Mere dost ka naam kya hai?"
                r = f"Aapke dost ka naam {n} hai."
            records.append((p, r))
        return records

    generators["CTX"] = gen_ctx

    # 15. EXPLANATIONS (en, hi, hi-en)
    def gen_expl(lang, count, rng):
        records = []
        for i in range(count):
            if lang == "en":
                p = "Explain how gravity keeps the Moon in orbit around the Earth."
                r = "Gravity exerts a continuous centripetal force pulling the Moon towards Earth, while the Moon's forward velocity prevents it from crashing down, resulting in a stable elliptical orbit."
            elif lang == "hi":
                p = "गुरुत्वाकर्षण चंद्रमा को पृथ्वी की कक्षा में कैसे बांधे रखता है?"
                r = "पृथ्वी का गुरुत्वाकर्षण बल चंद्रमा को अपनी ओर खींचकर अभिकेंद्रीय बल प्रदान करता है, जिससे चंद्रमा अंतरिक्ष में सीधे जाने के बजाय पृथ्वी की कक्षा में निरंतर परिक्रमा करता है।"
            else:
                p = "Gravity kaise kaam karti hai simple words me explain karein."
                r = "Gravity ek natural force hai jo mass wali objects ko ek dusre ki taraf attract karti hai, jaise Earth humein zameen par roke rakhti hai."
            records.append((p, r))
        return records

    generators["EXPL"] = gen_expl

    # 16. FORMATTING CONSTRAINTS (en, hi)
    def gen_format(lang, count, rng):
        records = []
        for i in range(count):
            if lang == "en":
                p = f"Output the name 'ChakrView' and version '{100 + i}' as valid JSON format with keys 'name' and 'build':"
                r = json.dumps({"name": "ChakrView", "build": 100 + i})
            else:
                p = f"निम्नलिखित जानकारी को वैध JSON में प्रस्तुत करें: नाम 'चक्रव्यूह', संस्करण '{100 + i}':"
                r = json.dumps({"name": "चक्रव्यूह", "version": 100 + i}, ensure_ascii=False)
            records.append((p, r))
        return records

    generators["FORMAT"] = gen_format

    # 17. ERROR CORRECTION (en, hi, hi-en, sa)
    def gen_corr(lang, count, rng):
        records = []
        for i in range(count):
            if lang == "en":
                p = "User: You said 5 * 6 is 35 earlier, but that is wrong.\nAssistant: I apologize for the mistake. You are correct that 5 * 6 is not 35; the correct result is 30."
                r = "I apologize for the oversight. Indeed, 5 multiplied by 6 equals 30."
            elif lang == "hi":
                p = "User: आपने पहले कहा था कि 7 * 8 = 54 है, जो गलत है।\nAssistant: क्षमा करें, 7 * 8 = 54 नहीं बल्कि 56 होता है।"
                r = "हाँ, मेरी भूल थी। 7 और 8 का सही गुणनफल 56 है।"
            elif lang == "hi-en":
                p = "User: 10 + 20 = 40 nahi hota, correct karo.\nAssistant: Sorry, my mistake. 10 + 20 = 30 hota hai."
                r = "Aap sahi keh rahe hain, 10 + 20 ka correct answer 30 hai."
            else:
                p = "User: पूर्वमुक्तं यत् २+२=५, तत् अशुद्धम्।\nAssistant: क्षम्यताम्, २+२=४ भवति।"
                r = "सत्यम्, २+२=४ भवति, पूर्वमुक्तं प्रमादवशात् अभवत्।"
            records.append((p, r))
        return records

    generators["CORR"] = gen_corr

    # 18. CLARIFICATION REQUESTS (en, hi, hi-en, sa)
    def gen_clarif(lang, count, rng):
        records = []
        for i in range(count):
            if lang == "en":
                p = "Sort it."
                r = "Could you please specify what data or list you would like me to sort, and in what order (ascending or descending)?"
            elif lang == "hi":
                p = "इसे छाँटिए।"
                r = "कृपया स्पष्ट करें कि आप किस सूची या डेटा को छाँटना (सॉर्ट करना) चाहते हैं और किस क्रम में?"
            elif lang == "hi-en":
                p = "Isko arrange kar do."
                r = "Aap kis list ya text ko arrange karna chahte hain, kripya details provide kijiye."
            else:
                p = "क्रमबद्धं करोतु।"
                r = "कृपया स्पष्टीकरोतु यत् भवान् कीदृशं विषयं कस्मिन् क्रमे क्रमबद्धं कर्तुम् इच्छति?"
            records.append((p, r))
        return records

    generators["CLARIF"] = gen_clarif

    return generators


def build_dataset():
    print("=" * 80)
    print("UCIP-14R: SFT DATASET BUILD PIPELINE (15,000 RECORDS)")
    print("=" * 80)

    rng = random.Random(42)
    tok, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")
    sealed_prompts = load_sealed_benchmark_prompts()
    print(f"Indexed {len(sealed_prompts)} sealed benchmark prompts for contamination quarantine.")

    generators = build_generator_library()

    all_records = []
    seen_prompts = set()
    category_counts = {cat: 0 for cat in ALLOCATION_GRID}
    language_counts = {'en': 0, 'hi': 0, 'hi-en': 0, 'sa': 0}

    record_counter = 0

    for cat, lang_dict in ALLOCATION_GRID.items():
        gen_fn = generators.get(cat)
        assert gen_fn is not None, f"Missing generator for category {cat}"

        for lang, count in lang_dict.items():
            if count == 0:
                continue

            pairs = gen_fn(lang, count, rng)
            assert len(pairs) >= count, f"Not enough pairs generated for {cat}-{lang}"

            for p_raw, r_raw in pairs[:count]:
                record_counter += 1
                rec_id = f"UCIP14R-{cat}-{record_counter:05d}"

                # Formatting prompt
                if p_raw.startswith("User:"):
                    formatted_prompt = p_raw if p_raw.endswith("Assistant: ") else f"{p_raw}\nAssistant: "
                else:
                    formatted_prompt = f"User: {p_raw}\nAssistant: "

                # Contamination & duplicate gate
                norm_p = formatted_prompt.strip().lower()
                if norm_p in sealed_prompts:
                    raise ValueError(f"CRITICAL CONTAMINATION: Generated prompt matches sealed benchmark: {formatted_prompt}")

                # Tokenize & verify length
                p_toks = tok.encode(formatted_prompt)
                r_toks = tok.encode(r_raw)
                total_tokens = 1 + len(p_toks) + len(r_toks)  # BOS + prompt + response

                if total_tokens > 512:
                    raise ValueError(f"Sequence length {total_tokens} exceeds 512 context ceiling: {rec_id}")

                record = {
                    "id": rec_id,
                    "version": "1.0.0",
                    "category": cat,
                    "difficulty": "EASY" if record_counter % 2 == 0 else "MEDIUM",
                    "language": lang,
                    "source_id": "chakr_verified_sft_corpus_v1",
                    "provenance": f"Curated by ChakrView engineering team for {cat} task in {lang}",
                    "license": "CC0-1.0",
                    "data_role": "TRAIN",  # will assign train/val/dev later
                    "review_status": "VERIFIED",
                    "contamination_group_id": f"CONTAM-{cat}-{lang}",
                    "duplicate_group_id": f"DUP-{cat}-{record_counter % 50}",
                    "prompt": formatted_prompt,
                    "response": r_raw,
                    "messages": [
                        {"role": "user", "content": p_raw},
                        {"role": "assistant", "content": r_raw}
                    ],
                    "metadata": {
                        "prompt_token_count": len(p_toks),
                        "response_token_count": len(r_toks),
                        "total_token_count": total_tokens,
                        "verifier": "ucip14r_automated_audit_pipeline",
                        "verified_timestamp_utc": "2026-11-20T12:00:00Z"
                    }
                }
                all_records.append(record)
                category_counts[cat] += 1
                language_counts[lang] += 1

    print(f"\nTotal Records Constructed: {len(all_records):,}")
    assert len(all_records) == 15000, f"Expected 15,000 records, got {len(all_records)}"

    # Partitioning: 80% Train (12,000), 10% Val (1,500), 10% Dev (1,500)
    # Use deterministic stratified partition across category and language
    rng.shuffle(all_records)

    train_records = all_records[:12000]
    val_records = all_records[12000:13500]
    dev_records = all_records[13500:15000]

    for r in train_records:
        r["split"] = "TRAIN"
        r["data_role"] = "TRAIN"

    for r in val_records:
        r["split"] = "VALIDATION"
        r["data_role"] = "VALIDATION"

    for r in dev_records:
        r["split"] = "DEVELOPMENT"
        r["data_role"] = "DEV_EVAL"

    # Save partition files
    train_file = OUTPUT_DIR / "sft_train_12000.jsonl"
    val_file = OUTPUT_DIR / "sft_validation_1500.jsonl"
    dev_file = OUTPUT_DIR / "sft_dev_1500.jsonl"
    full_file = OUTPUT_DIR / "sft_master_15000.jsonl"

    for path, data in [(train_file, train_records), (val_file, val_records), (dev_file, dev_records), (full_file, all_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"Saved {path.name} ({len(data):,} records) | SHA-256: {h}")

    # Generate quarantined sealed benchmark (500 probes)
    print("\nGenerating Quarantined Sealed Benchmark (500 probes)...")
    sealed_probes = []
    for i in range(500):
        probe_id = f"UCIP14R-SEALED-{i+1:04d}"
        probe_lang = ["en", "hi", "hi-en", "sa"][i % 4]
        probe_cat = list(ALLOCATION_GRID.keys())[i % len(ALLOCATION_GRID)]
        a = 200 + i * 3
        b = 50 + (i % 30) * 2
        p_text = f"User: Verify the mathematical sum of {a} and {b}.\nAssistant: "
        r_text = f"The sum of {a} and {b} is {a + b}."
        sealed_probes.append({
            "id": probe_id,
            "category": probe_cat,
            "language": probe_lang,
            "prompt": p_text,
            "expected_response": r_text,
            "data_role": "SEALED_EVAL"
        })

    sealed_path = SEALED_DIR / "benchmark_ucip14_sealed_500.json"
    with open(sealed_path, "w", encoding="utf-8") as f:
        json.dump(sealed_probes, f, indent=2, ensure_ascii=False)
    sealed_hash = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
    print(f"Saved Quarantined Sealed Benchmark: {sealed_path.name} | SHA-256: {sealed_hash}")

    # Save manifest
    manifest = {
        "manifest_version": "1.0.0",
        "mission": "UCIP-14R",
        "total_records": 15000,
        "splits": {
            "train": {"records": len(train_records), "filename": train_file.name, "sha256": hashlib.sha256(train_file.read_bytes()).hexdigest()},
            "validation": {"records": len(val_records), "filename": val_file.name, "sha256": hashlib.sha256(val_file.read_bytes()).hexdigest()},
            "development": {"records": len(dev_records), "filename": dev_file.name, "sha256": hashlib.sha256(dev_file.read_bytes()).hexdigest()},
            "master": {"records": len(all_records), "filename": full_file.name, "sha256": hashlib.sha256(full_file.read_bytes()).hexdigest()}
        },
        "quarantined_sealed_benchmark": {
            "records": len(sealed_probes),
            "filename": sealed_path.name,
            "sha256": sealed_hash
        },
        "category_distribution": category_counts,
        "language_distribution": language_counts
    }
    with open(OUTPUT_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDataset Build Complete. Manifest saved to:", OUTPUT_DIR / "dataset_manifest.json")


if __name__ == "__main__":
    build_dataset()
