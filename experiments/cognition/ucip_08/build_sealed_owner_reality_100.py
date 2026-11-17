import json, hashlib, random
from pathlib import Path

# 100 completely fresh probes across 9 categories
OWNER_REALITY_100_PROBES = [
    # 1. English Conversation (15)
    {"id": "OR_CONV_01", "category": "English Conversation", "prompt": "Hi! Can you introduce yourself?", "expected_keywords": ["chakrview", "assistant", "ai", "model"], "rubric": "Self-introduction as ChakrView assistant"},
    {"id": "OR_CONV_02", "category": "English Conversation", "prompt": "I had a really tiring day at work.", "expected_keywords": ["rest", "relax", "hear", "tired", "unwind", "hope"], "rubric": "Empathetic conversation acknowledging fatigue"},
    {"id": "OR_CONV_03", "category": "English Conversation", "prompt": "What are your main capabilities?", "expected_keywords": ["language", "text", "assist", "questions", "code", "explain"], "rubric": "Clear explanation of assistant capabilities"},
    {"id": "OR_CONV_04", "category": "English Conversation", "prompt": "Could you suggest a quick healthy breakfast?", "expected_keywords": ["eggs", "oatmeal", "fruit", "yogurt", "toast", "smoothie"], "rubric": "Suggests realistic breakfast food"},
    {"id": "OR_CONV_05", "category": "English Conversation", "prompt": "Thank you for explaining that so clearly.", "expected_keywords": ["welcome", "glad", "happy", "help", "anytime"], "rubric": "Courteous polite reply"},
    {"id": "OR_CONV_06", "category": "English Conversation", "prompt": "Good afternoon! How is everything going?", "expected_keywords": ["afternoon", "well", "good", "help", "assist", "ready"], "rubric": "Polite afternoon greeting"},
    {"id": "OR_CONV_07", "category": "English Conversation", "prompt": "I need some motivation to start studying today.", "expected_keywords": ["start", "focus", "goal", "break", "small", "step"], "rubric": "Encouraging advice to study"},
    {"id": "OR_CONV_08", "category": "English Conversation", "prompt": "What is the best way to prepare for a software interview?", "expected_keywords": ["practice", "coding", "data structures", "algorithms", "projects", "mock"], "rubric": "Pragmatic technical interview advice"},
    {"id": "OR_CONV_09", "category": "English Conversation", "prompt": "Can you recommend a good habit for better sleep?", "expected_keywords": ["screen", "schedule", "dark", "bedtime", "routine", "caffeine"], "rubric": "Sleep hygiene recommendation"},
    {"id": "OR_CONV_10", "category": "English Conversation", "prompt": "Do you remember previous conversations if memory is off?", "expected_keywords": ["no", "memory", "session", "off", "stateless", "retain"], "rubric": "Accurate explanation of memory off mode"},
    {"id": "OR_CONV_11", "category": "English Conversation", "prompt": "How do you handle complex tasks?", "expected_keywords": ["break", "steps", "analyze", "subtasks", "approach"], "rubric": "Explains stepwise task resolution"},
    {"id": "OR_CONV_12", "category": "English Conversation", "prompt": "Tell me a fun fact about space.", "expected_keywords": ["space", "sun", "stars", "planet", "silent", "light", "galaxy"], "rubric": "Genuine fun fact about space"},
    {"id": "OR_CONV_13", "category": "English Conversation", "prompt": "What should I do if my code throws an unhandled exception?", "expected_keywords": ["traceback", "debug", "error", "catch", "inspect", "log"], "rubric": "Debugging advice"},
    {"id": "OR_CONV_14", "category": "English Conversation", "prompt": "I am feeling overwhelmed with many deadlines.", "expected_keywords": ["prioritize", "one", "list", "breathe", "break", "focus"], "rubric": "Constructive stress management advice"},
    {"id": "OR_CONV_15", "category": "English Conversation", "prompt": "Goodbye, ChakrView! Talk to you later.", "expected_keywords": ["goodbye", "later", "take care", "bye", "welcome", "great"], "rubric": "Polite conversational farewell"},

    # 2. English Explanations (10)
    {"id": "OR_EXPL_01", "category": "English Explanations", "prompt": "Explain the difference between RAM and persistent storage.", "expected_keywords": ["volatile", "temporary", "disk", "ssd", "power", "speed", "permanent"], "rubric": "Contrasts volatile memory with non-volatile disk"},
    {"id": "OR_EXPL_02", "category": "English Explanations", "prompt": "How does an operating system manage computer hardware?", "expected_keywords": ["kernel", "cpu", "memory", "drivers", "resources", "processes"], "rubric": "Explains resource allocation and hardware abstraction"},
    {"id": "OR_EXPL_03", "category": "English Explanations", "prompt": "What is the purpose of an index in a database?", "expected_keywords": ["faster", "lookup", "search", "query", "b-tree", "speed"], "rubric": "Explains accelerated data retrieval via indexes"},
    {"id": "OR_EXPL_04", "category": "English Explanations", "prompt": "Why does ice float on liquid water?", "expected_keywords": ["density", "less dense", "crystal", "hydrogen", "structure", "expands"], "rubric": "Explains density anomaly of solid water"},
    {"id": "OR_EXPL_05", "category": "English Explanations", "prompt": "What does an electric motor do?", "expected_keywords": ["electrical", "mechanical", "magnetic", "motion", "energy", "rotation"], "rubric": "Converts electrical energy to mechanical rotation"},
    {"id": "OR_EXPL_06", "category": "English Explanations", "prompt": "Explain what an API is in simple terms.", "expected_keywords": ["interface", "communicate", "software", "request", "data", "bridge"], "rubric": "Interface connecting software applications"},
    {"id": "OR_EXPL_07", "category": "English Explanations", "prompt": "Why do leaves change color in autumn?", "expected_keywords": ["chlorophyll", "sunlight", "green", "pigments", "breakdown", "weather"], "rubric": "Breakdown of chlorophyll unmasking other pigments"},
    {"id": "OR_EXPL_08", "category": "English Explanations", "prompt": "What is gravity and what does it do?", "expected_keywords": ["attraction", "mass", "pull", "earth", "planets", "force"], "rubric": "Attractive force between masses"},
    {"id": "OR_EXPL_09", "category": "English Explanations", "prompt": "How does a bicycle brake work?", "expected_keywords": ["friction", "pads", "wheel", "lever", "slow", "clamp"], "rubric": "Friction between brake pads and rim/rotor"},
    {"id": "OR_EXPL_10", "category": "English Explanations", "prompt": "What is the greenhouse effect?", "expected_keywords": ["heat", "atmosphere", "trap", "gases", "warm", "radiation"], "rubric": "Atmospheric gases trapping thermal radiation"},

    # 3. Devanagari Hindi (15)
    {"id": "OR_DHIN_01", "category": "Devanagari Hindi", "prompt": "आपका मुख्य उद्देश्य क्या है?", "expected_keywords": ["सहायता", "भाषा", "प्रश्नों", "उत्तर", "सीखने"], "rubric": "Hindi explanation of assistant purpose"},
    {"id": "OR_DHIN_02", "category": "Devanagari Hindi", "prompt": "कंप्यूटर में सीपीयू (CPU) का क्या काम है?", "expected_keywords": ["प्रोसेसिंग", "मस्तिष्क", "गणना", "निर्देश", "काम"], "rubric": "Explains CPU role in Hindi"},
    {"id": "OR_DHIN_03", "category": "Devanagari Hindi", "prompt": "जल संरक्षण क्यों जरूरी है?", "expected_keywords": ["पानी", "जीवन", "भविष्य", "बचाना", "आवश्यक"], "rubric": "Explains water conservation importance"},
    {"id": "OR_DHIN_04", "category": "Devanagari Hindi", "prompt": "सूर्य हमें क्या प्रदान करता है?", "expected_keywords": ["प्रकाश", "ऊर्जा", "गर्मी", "धूप"], "rubric": "Mentions light and energy from sun"},
    {"id": "OR_DHIN_05", "category": "Devanagari Hindi", "prompt": "एक अच्छा विद्यार्थी बनने के लिए क्या करना चाहिए?", "expected_keywords": ["मेहनत", "अभ्यास", "समय", "अनुशासन", "पढ़ाई"], "rubric": "Discipline and study habits in Hindi"},
    {"id": "OR_DHIN_06", "category": "Devanagari Hindi", "prompt": "पेड़-पौधे हमारे पर्यावरण के लिए क्यों महत्वपूर्ण हैं?", "expected_keywords": ["ऑक्सीजन", "छाया", "फल", "पर्यावरण", "हवा"], "rubric": "Role of trees in environment"},
    {"id": "OR_DHIN_07", "category": "Devanagari Hindi", "prompt": "पुस्तकालय (Library) का क्या महत्व है?", "expected_keywords": ["किताबें", "ज्ञान", "पढ़ने", "अध्ययन", "शांति"], "rubric": "Significance of libraries"},
    {"id": "OR_DHIN_08", "category": "Devanagari Hindi", "prompt": "समय का सदुपयोग कैसे करें?", "expected_keywords": ["नियोजन", "समय", "कार्य", "महत्व", "लक्ष्य"], "rubric": "Time management in Hindi"},
    {"id": "OR_DHIN_09", "category": "Devanagari Hindi", "prompt": "स्वास्थ्य के लिए सुबह की सैर के क्या लाभ हैं?", "expected_keywords": ["ताजी", "हवा", "ऊर्जा", "व्यायाम", "स्वस्थ"], "rubric": "Benefits of morning walk"},
    {"id": "OR_DHIN_10", "category": "Devanagari Hindi", "prompt": "भारत की राजधानी का नाम क्या है?", "expected_keywords": ["नई दिल्ली", "दिल्ली"], "rubric": "Identifies New Delhi as capital"},
    {"id": "OR_DHIN_11", "category": "Devanagari Hindi", "prompt": "हवा हमारे जीवन के लिए क्यों आवश्यक है?", "expected_keywords": ["सांस", "ऑक्सीजन", "जीवित", "प्राणवायु"], "rubric": "Necessity of air/oxygen for respiration"},
    {"id": "OR_DHIN_12", "category": "Devanagari Hindi", "prompt": "मित्रता का क्या अर्थ है?", "expected_keywords": ["विश्वास", "साथ", "मदद", "सच्चा", "दोस्त"], "rubric": "Meaning of friendship in Hindi"},
    {"id": "OR_DHIN_13", "category": "Devanagari Hindi", "prompt": "चंद्रमा रात में क्यों चमकता है?", "expected_keywords": ["सूर्य", "प्रकाश", "परावर्तन", "रोशनी"], "rubric": "Moon reflects sunlight"},
    {"id": "OR_DHIN_14", "category": "Devanagari Hindi", "prompt": "इंटरनेट का हमारे जीवन में क्या उपयोग है?", "expected_keywords": ["जानकारी", "संचार", "शिक्षा", "सुविधा", "जुड़े"], "rubric": "Uses of internet in daily life"},
    {"id": "OR_DHIN_15", "category": "Devanagari Hindi", "prompt": "सत्य बोलने का क्या महत्व है?", "expected_keywords": ["विश्वास", "सम्मान", "सच्चाई", "शांति", "साहस"], "rubric": "Value of honesty"},

    # 4. Roman Hindi (10)
    {"id": "OR_RHIN_01", "category": "Roman Hindi", "prompt": "Kal mera exam hai, kuch tips do.", "expected_keywords": ["revision", "neend", "shant", "focus", "padhai", "confident"], "rubric": "Practical exam advice in Roman Hindi"},
    {"id": "OR_RHIN_02", "category": "Roman Hindi", "prompt": "Subah jaldi uthne ke kya fayde hain?", "expected_keywords": ["energy", "taaza", "samay", "fayda", "swasth", "din"], "rubric": "Benefits of waking early"},
    {"id": "OR_RHIN_03", "category": "Roman Hindi", "prompt": "Paani peena sehat ke liye kyu zaroori hai?", "expected_keywords": ["hydration", "body", "swasth", "zaroori", "energy", "paani"], "rubric": "Importance of drinking water"},
    {"id": "OR_RHIN_04", "category": "Roman Hindi", "prompt": "Gussa aane par kya karna chahiye?", "expected_keywords": ["saans", "shaant", "socho", "paani", "break"], "rubric": "Anger management tip"},
    {"id": "OR_RHIN_05", "category": "Roman Hindi", "prompt": "Chai banane ka simple tarika batao.", "expected_keywords": ["paani", "doodh", "patti", "chini", "ubalo", "adrak"], "rubric": "Tea preparation steps"},
    {"id": "OR_RHIN_06", "category": "Roman Hindi", "prompt": "Naye dost kaise banayein?", "expected_keywords": ["baat", "muskurana", "madad", "achha", "respect"], "rubric": "How to make new friends"},
    {"id": "OR_RHIN_07", "category": "Roman Hindi", "prompt": "Coding seekhne ki shuruat kahan se karein?", "expected_keywords": ["python", "basics", "practice", "logic", "start"], "rubric": "How to start learning coding"},
    {"id": "OR_RHIN_08", "category": "Roman Hindi", "prompt": "Kitabein padhne se kya labh milta hai?", "expected_keywords": ["gyaan", "vocabulary", "soch", "fayda", "seekhna"], "rubric": "Benefits of reading books"},
    {"id": "OR_RHIN_09", "category": "Roman Hindi", "prompt": "Aapka din kaisa ja raha hai?", "expected_keywords": ["badhiya", "achha", "madad", "ready", "theek"], "rubric": "Conversational reply in Roman Hindi"},
    {"id": "OR_RHIN_10", "category": "Roman Hindi", "prompt": "Aalaspana (laziness) ko kaise door karein?", "expected_keywords": ["goal", "exercise", "routine", "shuru", "action"], "rubric": "Overcoming laziness"},

    # 5. Hinglish (15)
    {"id": "OR_HING_01", "category": "Hinglish", "prompt": "Bhai, cloud computing ka basic concept kya hai?", "expected_keywords": ["server", "internet", "storage", "remote", "data", "access"], "rubric": "Explains cloud computing simply in Hinglish"},
    {"id": "OR_HING_02", "category": "Hinglish", "prompt": "Frontend aur backend me main difference kya hota hai?", "expected_keywords": ["ui", "interface", "server", "database", "logic", "user"], "rubric": "Distinguishes UI from server architecture"},
    {"id": "OR_HING_03", "category": "Hinglish", "prompt": "GitHub pe commit karne ka matlab kya hota hai?", "expected_keywords": ["save", "changes", "version", "history", "code", "snapshot"], "rubric": "Explains git commit in Hinglish"},
    {"id": "OR_HING_04", "category": "Hinglish", "prompt": "Open source software use karne ka sabse bada benefit kya hai?", "expected_keywords": ["free", "community", "code", "modify", "custom", "transparent"], "rubric": "Benefits of open source"},
    {"id": "OR_HING_05", "category": "Hinglish", "prompt": "Laptop me SSD lagane se performance kyu badhti hai?", "expected_keywords": ["fast", "speed", "boot", "read", "write", "storage"], "rubric": "Explains SSD speed advantage"},
    {"id": "OR_HING_06", "category": "Hinglish", "prompt": "Cache memory kya hoti hai aur ye kyu fast hoti hai?", "expected_keywords": ["cpu", "fast", "temporary", "frequent", "near", "data"], "rubric": "Explains CPU cache memory"},
    {"id": "OR_HING_07", "category": "Hinglish", "prompt": "Bug aur feature me kya farak hai?", "expected_keywords": ["error", "kaam", "problem", "functionality", "nayi", "fix"], "rubric": "Distinguishes software bug from feature"},
    {"id": "OR_HING_08", "category": "Hinglish", "prompt": "Ek clean code likhne ke do golden rules batao.", "expected_keywords": ["readable", "naming", "functions", "simple", "comments", "clean"], "rubric": "Clean code principles"},
    {"id": "OR_HING_09", "category": "Hinglish", "prompt": "Database me primary key ka role kya hota hai?", "expected_keywords": ["unique", "identify", "record", "duplicate", "row"], "rubric": "Unique identification in DB"},
    {"id": "OR_HING_10", "category": "Hinglish", "prompt": "Wireless router ka main function kya hai?", "expected_keywords": ["wifi", "internet", "connect", "network", "signal", "devices"], "rubric": "Role of WiFi router"},
    {"id": "OR_HING_11", "category": "Hinglish", "prompt": "Password strong rakhna kyu zaroori hai?", "expected_keywords": ["security", "hack", "safe", "protect", "account"], "rubric": "Password security in Hinglish"},
    {"id": "OR_HING_12", "category": "Hinglish", "prompt": "Python language itni popular kyu hai?", "expected_keywords": ["simple", "readable", "libraries", "easy", "ai", "versatile"], "rubric": "Why Python is widely adopted"},
    {"id": "OR_HING_13", "category": "Hinglish", "prompt": "Mobile me flight mode on karne se kya hota hai?", "expected_keywords": ["network", "signal", "disconnect", "radio", "cellular"], "rubric": "Disables wireless transmissions"},
    {"id": "OR_HING_14", "category": "Hinglish", "prompt": "Unit testing karna kyu important hai?", "expected_keywords": ["bugs", "verify", "code", "reliable", "break", "test"], "rubric": "Early defect detection"},
    {"id": "OR_HING_15", "category": "Hinglish", "prompt": "Compiler aur interpreter me short me difference batao.", "expected_keywords": ["line by line", "whole", "convert", "machine code", "speed"], "rubric": "Translating source code strategies"},

    # 6. Exact Copying (10)
    {"id": "OR_COPY_01", "category": "Exact Copying", "prompt": "Repeat word for word: The sun rises in the east.", "target": "The sun rises in the east.", "exact_match": True},
    {"id": "OR_COPY_02", "category": "Exact Copying", "prompt": "Copy this sentence exactly: Python is a programming language.", "target": "Python is a programming language.", "exact_match": True},
    {"id": "OR_COPY_03", "category": "Exact Copying", "prompt": "Say exactly: Water is essential for all life.", "target": "Water is essential for all life.", "exact_match": True},
    {"id": "OR_COPY_04", "category": "Exact Copying", "prompt": "Echo the text: 123456789", "target": "123456789", "exact_match": True},
    {"id": "OR_COPY_05", "category": "Exact Copying", "prompt": "Output only this exact phrase: Alpha Beta Gamma", "target": "Alpha Beta Gamma", "exact_match": True},
    {"id": "OR_COPY_06", "category": "Exact Copying", "prompt": "Copy verbatim: Delhi is the capital of India.", "target": "Delhi is the capital of India.", "exact_match": True},
    {"id": "OR_COPY_07", "category": "Exact Copying", "prompt": "Say only: Clean code creates reliable systems.", "target": "Clean code creates reliable systems.", "exact_match": True},
    {"id": "OR_COPY_08", "category": "Exact Copying", "prompt": "Mirror this string: Red Green Blue Yellow", "target": "Red Green Blue Yellow", "exact_match": True},
    {"id": "OR_COPY_09", "category": "Exact Copying", "prompt": "Echo verbatim: CPU executes instructions rapidly.", "target": "CPU executes instructions rapidly.", "exact_match": True},
    {"id": "OR_COPY_10", "category": "Exact Copying", "prompt": "Repeat exactly: Today is a brand new day.", "target": "Today is a brand new day.", "exact_match": True},

    # 7. Elementary Arithmetic (10)
    {"id": "OR_ARITH_01", "category": "Elementary Arithmetic", "prompt": "What is 15 plus 7?", "target_number": 22, "rubric": "15 + 7 = 22"},
    {"id": "OR_ARITH_02", "category": "Elementary Arithmetic", "prompt": "If you have 10 marbles and give away 4, how many do you have left?", "target_number": 6, "rubric": "10 - 4 = 6"},
    {"id": "OR_ARITH_03", "category": "Elementary Arithmetic", "prompt": "What is 8 multiplied by 3?", "target_number": 24, "rubric": "8 * 3 = 24"},
    {"id": "OR_ARITH_04", "category": "Elementary Arithmetic", "prompt": "Divide 20 by 4.", "target_number": 5, "rubric": "20 / 4 = 5"},
    {"id": "OR_ARITH_05", "category": "Elementary Arithmetic", "prompt": "A baker bakes 6 cakes in the morning and 7 cakes in the evening. How many total cakes were baked?", "target_number": 13, "rubric": "6 + 7 = 13"},
    {"id": "OR_ARITH_06", "category": "Elementary Arithmetic", "prompt": "You have 18 books and sell 9 of them. How many books remain?", "target_number": 9, "rubric": "18 - 9 = 9"},
    {"id": "OR_ARITH_07", "category": "Elementary Arithmetic", "prompt": "What is 5 plus 14?", "target_number": 19, "rubric": "5 + 14 = 19"},
    {"id": "OR_ARITH_08", "category": "Elementary Arithmetic", "prompt": "If a box holds 5 pencils, how many pencils do 4 identical boxes hold?", "target_number": 20, "rubric": "5 * 4 = 20"},
    {"id": "OR_ARITH_09", "category": "Elementary Arithmetic", "prompt": "What is 25 minus 8?", "target_number": 17, "rubric": "25 - 8 = 17"},
    {"id": "OR_ARITH_10", "category": "Elementary Arithmetic", "prompt": "Distribute 16 candies equally among 2 kids. How many candies does each kid get?", "target_number": 8, "rubric": "16 / 2 = 8"},

    # 8. Instruction Following (10)
    {"id": "OR_INST_01", "category": "Instruction Following", "prompt": "Reply with only the single word: Confirmed", "expected_keywords": ["confirmed"], "exact_match": True, "rubric": "Single word exact output"},
    {"id": "OR_INST_02", "category": "Instruction Following", "prompt": "Answer with only 'True' or 'False': The earth revolves around the sun.", "expected_keywords": ["true"], "rubric": "Selects True"},
    {"id": "OR_INST_03", "category": "Instruction Following", "prompt": "List exactly 3 colors, separated by commas.", "expected_keywords": [","], "rubric": "Lists 3 colors separated by commas"},
    {"id": "OR_INST_04", "category": "Instruction Following", "prompt": "Respond in ALL CAPITAL LETTERS: hello world", "target": "HELLO WORLD", "exact_match": True, "rubric": "Uppercased phrase"},
    {"id": "OR_INST_05", "category": "Instruction Following", "prompt": "Count from 1 to 5 using numbers separated by spaces.", "target": "1 2 3 4 5", "exact_match": True, "rubric": "1 2 3 4 5"},
    {"id": "OR_INST_06", "category": "Instruction Following", "prompt": "Write a 3-word sentence starting with 'Cats'.", "expected_keywords": ["cats"], "rubric": "3-word sentence starting with Cats"},
    {"id": "OR_INST_07", "category": "Instruction Following", "prompt": "Reply with only the word 'Stop' and nothing else.", "target": "Stop", "exact_match": True, "rubric": "Strict output: Stop"},
    {"id": "OR_INST_08", "category": "Instruction Following", "prompt": "Respond with the single letter: X", "target": "X", "exact_match": True, "rubric": "Strict single letter X"},
    {"id": "OR_INST_09", "category": "Instruction Following", "prompt": "Answer with only 'Yes' or 'No': Is 2 + 2 equal to 5?", "expected_keywords": ["no"], "rubric": "Selects No"},
    {"id": "OR_INST_10", "category": "Instruction Following", "prompt": "Output the word 'PASS' enclosed in square brackets.", "target": "[PASS]", "exact_match": True, "rubric": "[PASS]"},

    # 9. Semantic Consistency (5)
    {"id": "OR_CONS_01", "category": "Semantic Consistency", "prompt": "Describe a snowy winter morning without mentioning the summer or sun.", "expected_keywords": ["snow", "cold", "frost", "white", "ice", "winter"], "forbidden": ["summer", "sun"], "rubric": "Winter description without summer/sun"},
    {"id": "OR_CONS_02", "category": "Semantic Consistency", "prompt": "Explain what a library is without using the word 'computer'.", "expected_keywords": ["books", "reading", "shelves", "knowledge", "quiet"], "forbidden": ["computer"], "rubric": "Library explanation without computer"},
    {"id": "OR_CONS_03", "category": "Semantic Consistency", "prompt": "Describe the deep ocean without using the word 'mountain'.", "expected_keywords": ["water", "sea", "depth", "dark", "fish", "creatures", "pressure"], "forbidden": ["mountain"], "rubric": "Ocean description without mountain"},
    {"id": "OR_CONS_04", "category": "Semantic Consistency", "prompt": "Explain how an electric vehicle is powered without mentioning gasoline or diesel.", "expected_keywords": ["battery", "electricity", "motor", "charge", "energy"], "forbidden": ["gasoline", "diesel"], "rubric": "EV explanation without fossil fuels"},
    {"id": "OR_CONS_05", "category": "Semantic Consistency", "prompt": "Describe an apple fruit without using the word 'orange'.", "expected_keywords": ["sweet", "red", "fruit", "crisp", "tree", "seeds"], "forbidden": ["orange"], "rubric": "Apple description without orange"},
]

out_path = Path("D:/Project/ChakrView/experiments/cognition/ucip_08/sealed_owner_reality_100.json")
data = {
    "benchmark_name": "UCIP-08 Sealed Owner Reality 100-Probe Benchmark",
    "total_probes": len(OWNER_REALITY_100_PROBES),
    "version": "1.0",
    "probes": OWNER_REALITY_100_PROBES
}

json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
sha = hashlib.sha256(json_bytes).hexdigest()
data["sha256"] = sha

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Created and sealed 100-probe benchmark at {out_path}")
print(f"Total probes: {len(OWNER_REALITY_100_PROBES)}")
print(f"Authoritative SHA-256: {sha}")
