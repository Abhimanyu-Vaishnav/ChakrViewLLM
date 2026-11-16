"""
Generate 100 fixed baseline diagnostic prompts for UCIP-04.
Covers:
- exact_copying (15)
- simple_generation (15)
- basic_explanation (15)
- conversation_persona (15)
- devanagari_hindi (15)
- roman_hindi (10)
- hinglish (10)
- multi_turn_context (5)
Total = 100 fixed diagnostic probes.
"""

import json
from pathlib import Path

probes = [
    # 1. Exact Copying (15)
    {"id": "base_copy_01", "suite": "exact_copying", "prompt": "User: Say exactly: The sky is blue.\nAssistant:", "target": "The sky is blue."},
    {"id": "base_copy_02", "suite": "exact_copying", "prompt": "User: Repeat this sentence: A small bird is sitting on a tree.\nAssistant:", "target": "A small bird is sitting on a tree."},
    {"id": "base_copy_03", "suite": "exact_copying", "prompt": "User: Copy exactly: ChakrView is an independent AI project.\nAssistant:", "target": "ChakrView is an independent AI project."},
    {"id": "base_copy_04", "suite": "exact_copying", "prompt": "User: Say exactly: Python is a programming language.\nAssistant:", "target": "Python is a programming language."},
    {"id": "base_copy_05", "suite": "exact_copying", "prompt": "User: Please repeat: Fresh water is essential for life.\nAssistant:", "target": "Fresh water is essential for life."},
    {"id": "base_copy_06", "suite": "exact_copying", "prompt": "User: Echo the following text: The quick brown fox jumps.\nAssistant:", "target": "The quick brown fox jumps."},
    {"id": "base_copy_07", "suite": "exact_copying", "prompt": "User: Say exactly: Mathematics helps us solve problems.\nAssistant:", "target": "Mathematics helps us solve problems."},
    {"id": "base_copy_08", "suite": "exact_copying", "prompt": "User: Copy this sentence: Clean air is vital for health.\nAssistant:", "target": "Clean air is vital for health."},
    {"id": "base_copy_09", "suite": "exact_copying", "prompt": "User: Repeat verbatim: The sun provides warm light.\nAssistant:", "target": "The sun provides warm light."},
    {"id": "base_copy_10", "suite": "exact_copying", "prompt": "User: Say exactly: Reading books expands knowledge.\nAssistant:", "target": "Reading books expands knowledge."},
    {"id": "base_copy_11", "suite": "exact_copying", "prompt": "User: Copy exactly: Modern computers process information quickly.\nAssistant:", "target": "Modern computers process information quickly."},
    {"id": "base_copy_12", "suite": "exact_copying", "prompt": "User: Echo this: Practice leads to steady progress.\nAssistant:", "target": "Practice leads to steady progress."},
    {"id": "base_copy_13", "suite": "exact_copying", "prompt": "User: Say exactly: Open collaboration creates great tools.\nAssistant:", "target": "Open collaboration creates great tools."},
    {"id": "base_copy_14", "suite": "exact_copying", "prompt": "User: Repeat this sentence: The ocean waves roll steadily.\nAssistant:", "target": "The ocean waves roll steadily."},
    {"id": "base_copy_15", "suite": "exact_copying", "prompt": "User: Say exactly: Truth and clarity matter.\nAssistant:", "target": "Truth and clarity matter."},

    # 2. Simple Generation (15)
    {"id": "base_gen_01", "suite": "simple_generation", "prompt": "User: Write one simple sentence about a dog.\nAssistant:", "target": "dog"},
    {"id": "base_gen_02", "suite": "simple_generation", "prompt": "User: Describe a sunny day in one sentence.\nAssistant:", "target": "sunny"},
    {"id": "base_gen_03", "suite": "simple_generation", "prompt": "User: Write a sentence about a computer.\nAssistant:", "target": "computer"},
    {"id": "base_gen_04", "suite": "simple_generation", "prompt": "User: Write one sentence about a tree.\nAssistant:", "target": "tree"},
    {"id": "base_gen_05", "suite": "simple_generation", "prompt": "User: Write a short sentence about an apple.\nAssistant:", "target": "apple"},
    {"id": "base_gen_06", "suite": "simple_generation", "prompt": "User: Write one simple sentence about a cat.\nAssistant:", "target": "cat"},
    {"id": "base_gen_07", "suite": "simple_generation", "prompt": "User: Give me a sentence describing rain.\nAssistant:", "target": "rain"},
    {"id": "base_gen_08", "suite": "simple_generation", "prompt": "User: Write a sentence about a mountain.\nAssistant:", "target": "mountain"},
    {"id": "base_gen_09", "suite": "simple_generation", "prompt": "User: Write a simple sentence about a river.\nAssistant:", "target": "river"},
    {"id": "base_gen_10", "suite": "simple_generation", "prompt": "User: Write one sentence about a bicycle.\nAssistant:", "target": "bicycle"},
    {"id": "base_gen_11", "suite": "simple_generation", "prompt": "User: Describe a library in one sentence.\nAssistant:", "target": "library"},
    {"id": "base_gen_12", "suite": "simple_generation", "prompt": "User: Write a sentence about a flower.\nAssistant:", "target": "flower"},
    {"id": "base_gen_13", "suite": "simple_generation", "prompt": "User: Write a simple sentence about the moon.\nAssistant:", "target": "moon"},
    {"id": "base_gen_14", "suite": "simple_generation", "prompt": "User: Write one sentence about music.\nAssistant:", "target": "music"},
    {"id": "base_gen_15", "suite": "simple_generation", "prompt": "User: Write a simple sentence about bread.\nAssistant:", "target": "bread"},

    # 3. Basic Explanation (15)
    {"id": "base_exp_01", "suite": "basic_explanation", "prompt": "User: What does a CPU do?\nAssistant:", "target": "instructions"},
    {"id": "base_exp_02", "suite": "basic_explanation", "prompt": "User: What is a computer?\nAssistant:", "target": "machine"},
    {"id": "base_exp_03", "suite": "basic_explanation", "prompt": "User: Explain gravity simply.\nAssistant:", "target": "pulls"},
    {"id": "base_exp_04", "suite": "basic_explanation", "prompt": "User: What is machine learning?\nAssistant:", "target": "data"},
    {"id": "base_exp_05", "suite": "basic_explanation", "prompt": "User: What is photosynthesis?\nAssistant:", "target": "sunlight"},
    {"id": "base_exp_06", "suite": "basic_explanation", "prompt": "User: What is the internet?\nAssistant:", "target": "network"},
    {"id": "base_exp_07", "suite": "basic_explanation", "prompt": "User: Why do we need sleep?\nAssistant:", "target": "health"},
    {"id": "base_exp_08", "suite": "basic_explanation", "prompt": "User: What is an algorithm?\nAssistant:", "target": "step"},
    {"id": "base_exp_09", "suite": "basic_explanation", "prompt": "User: What is water made of?\nAssistant:", "target": "hydrogen"},
    {"id": "base_exp_10", "suite": "basic_explanation", "prompt": "User: Why do leaves look green?\nAssistant:", "target": "chlorophyll"},
    {"id": "base_exp_11", "suite": "basic_explanation", "prompt": "User: What is solar energy?\nAssistant:", "target": "sun"},
    {"id": "base_exp_12", "suite": "basic_explanation", "prompt": "User: What is the capital of France?\nAssistant:", "target": "Paris"},
    {"id": "base_exp_13", "suite": "basic_explanation", "prompt": "User: What is the capital of India?\nAssistant:", "target": "Delhi"},
    {"id": "base_exp_14", "suite": "basic_explanation", "prompt": "User: How many days are in a week?\nAssistant:", "target": "7"},
    {"id": "base_exp_15", "suite": "basic_explanation", "prompt": "User: What is 2 + 2?\nAssistant:", "target": "4"},

    # 4. Conversation & Persona (15)
    {"id": "base_conv_01", "suite": "conversation_persona", "prompt": "User: Hello! Who are you?\nAssistant:", "target": "ChakrView"},
    {"id": "base_conv_02", "suite": "conversation_persona", "prompt": "User: What is your name?\nAssistant:", "target": "ChakrView"},
    {"id": "base_conv_03", "suite": "conversation_persona", "prompt": "User: What can you help me with?\nAssistant:", "target": "help"},
    {"id": "base_conv_04", "suite": "conversation_persona", "prompt": "User: Can you explain something to me?\nAssistant:", "target": "yes"},
    {"id": "base_conv_05", "suite": "conversation_persona", "prompt": "User: Good morning!\nAssistant:", "target": "morning"},
    {"id": "base_conv_06", "suite": "conversation_persona", "prompt": "User: Good evening!\nAssistant:", "target": "evening"},
    {"id": "base_conv_07", "suite": "conversation_persona", "prompt": "User: Are you an AI assistant?\nAssistant:", "target": "assistant"},
    {"id": "base_conv_08", "suite": "conversation_persona", "prompt": "User: How are you today?\nAssistant:", "target": "ready"},
    {"id": "base_conv_09", "suite": "conversation_persona", "prompt": "User: Thank you for your assistance.\nAssistant:", "target": "welcome"},
    {"id": "base_conv_10", "suite": "conversation_persona", "prompt": "User: Nice to meet you.\nAssistant:", "target": "meet"},
    {"id": "base_conv_11", "suite": "conversation_persona", "prompt": "User: Can we have a conversation?\nAssistant:", "target": "yes"},
    {"id": "base_conv_12", "suite": "conversation_persona", "prompt": "User: Who created you?\nAssistant:", "target": "ChakrView"},
    {"id": "base_conv_13", "suite": "conversation_persona", "prompt": "User: What languages do you know?\nAssistant:", "target": "English"},
    {"id": "base_conv_14", "suite": "conversation_persona", "prompt": "User: Can you assist with studies?\nAssistant:", "target": "help"},
    {"id": "base_conv_15", "suite": "conversation_persona", "prompt": "User: Goodbye!\nAssistant:", "target": "goodbye"},

    # 5. Devanagari Hindi (15)
    {"id": "base_hin_01", "suite": "devanagari_hindi", "prompt": "User: नमस्ते! आप क्या कर सकते हैं?\nAssistant:", "target": "मदद"},
    {"id": "base_hin_02", "suite": "devanagari_hindi", "prompt": "User: आपका नाम क्या है?\nAssistant:", "target": "ChakrView"},
    {"id": "base_hin_03", "suite": "devanagari_hindi", "prompt": "User: भारत की राजधानी क्या है?\nAssistant:", "target": "दिल्ली"},
    {"id": "base_hin_04", "suite": "devanagari_hindi", "prompt": "User: पानी हमारे जीवन के लिए क्यों जरूरी है?\nAssistant:", "target": "जीवन"},
    {"id": "base_hin_05", "suite": "devanagari_hindi", "prompt": "User: सूरज किस दिशा से निकलता है?\nAssistant:", "target": "पूर्व"},
    {"id": "base_hin_06", "suite": "devanagari_hindi", "prompt": "User: एक हफ्ते में कितने दिन होते हैं?\nAssistant:", "target": "7"},
    {"id": "base_hin_07", "suite": "devanagari_hindi", "prompt": "User: पेड़ हमें क्या देते हैं?\nAssistant:", "target": "ऑक्सीजन"},
    {"id": "base_hin_08", "suite": "devanagari_hindi", "prompt": "User: दूध का रंग कैसा होता है?\nAssistant:", "target": "सफेद"},
    {"id": "base_hin_09", "suite": "devanagari_hindi", "prompt": "User: क्या आप मेरी सहायता करेंगे?\nAssistant:", "target": "हाँ"},
    {"id": "base_hin_10", "suite": "devanagari_hindi", "prompt": "User: किताब पढ़ने के क्या लाभ हैं?\nAssistant:", "target": "ज्ञान"},
    {"id": "base_hin_11", "suite": "devanagari_hindi", "prompt": "User: नमस्ते, आप कैसे हैं?\nAssistant:", "target": "नमस्ते"},
    {"id": "base_hin_12", "suite": "devanagari_hindi", "prompt": "User: पृथ्वी का आकार कैसा है?\nAssistant:", "target": "गोल"},
    {"id": "base_hin_13", "suite": "devanagari_hindi", "prompt": "User: सौर ऊर्जा क्या होती है?\nAssistant:", "target": "ऊर्जा"},
    {"id": "base_hin_14", "suite": "devanagari_hindi", "prompt": "User: गाय हमें क्या देती है?\nAssistant:", "target": "दूध"},
    {"id": "base_hin_15", "suite": "devanagari_hindi", "prompt": "User: धन्यवाद आपकी मदद के लिए!\nAssistant:", "target": "स्वागत"},

    # 6. Roman Hindi (10)
    {"id": "base_rom_01", "suite": "roman_hindi", "prompt": "User: Bhai, mujhe machine learning simple language mein samjhao.\nAssistant:", "target": "data"},
    {"id": "base_rom_02", "suite": "roman_hindi", "prompt": "User: Aapka naam kya hai?\nAssistant:", "target": "ChakrView"},
    {"id": "base_rom_03", "suite": "roman_hindi", "prompt": "User: Namaste, aap kaise ho?\nAssistant:", "target": "theek"},
    {"id": "base_rom_04", "suite": "roman_hindi", "prompt": "User: Kya aap meri madad kar sakte ho?\nAssistant:", "target": "madad"},
    {"id": "base_rom_05", "suite": "roman_hindi", "prompt": "User: Bharat ki rajdhani kya hai?\nAssistant:", "target": "Delhi"},
    {"id": "base_rom_06", "suite": "roman_hindi", "prompt": "User: Paani pina kyun zaroori hai?\nAssistant:", "target": "zaroori"},
    {"id": "base_rom_07", "suite": "roman_hindi", "prompt": "User: Ek hafte mein kitne din hote hain?\nAssistant:", "target": "7"},
    {"id": "base_rom_08", "suite": "roman_hindi", "prompt": "User: Doodh ka rang kaisa hota hai?\nAssistant:", "target": "safed"},
    {"id": "base_rom_09", "suite": "roman_hindi", "prompt": "User: Kitab padhna achhi aadat kyun hai?\nAssistant:", "target": "gyaan"},
    {"id": "base_rom_10", "suite": "roman_hindi", "prompt": "User: Dhanyawaad aapki help ke liye!\nAssistant:", "target": "swagat"},

    # 7. Hinglish (10)
    {"id": "base_hing_01", "suite": "hinglish", "prompt": "User: Computer ka CPU kya kaam karta hai? Simple example ke saath samjhao.\nAssistant:", "target": "process"},
    {"id": "base_hing_02", "suite": "hinglish", "prompt": "User: Hello! Study routine kaise maintain karein?\nAssistant:", "target": "routine"},
    {"id": "base_hing_03", "suite": "hinglish", "prompt": "User: Machine learning seekhne ke liye Python zaroori hai kya?\nAssistant:", "target": "Python"},
    {"id": "base_hing_04", "suite": "hinglish", "prompt": "User: Can you explain neural networks in simple words?\nAssistant:", "target": "network"},
    {"id": "base_hing_05", "suite": "hinglish", "prompt": "User: Coding practice daily karni chahiye?\nAssistant:", "target": "practice"},
    {"id": "base_hing_06", "suite": "hinglish", "prompt": "User: What is your primary purpose?\nAssistant:", "target": "assist"},
    {"id": "base_hing_07", "suite": "hinglish", "prompt": "User: Data science me mathematics ka kya use hai?\nAssistant:", "target": "math"},
    {"id": "base_hing_08", "suite": "hinglish", "prompt": "User: Open source software safe hota hai kya?\nAssistant:", "target": "safe"},
    {"id": "base_hing_09", "suite": "hinglish", "prompt": "User: Thanks a lot for the quick support!\nAssistant:", "target": "welcome"},
    {"id": "base_hing_10", "suite": "hinglish", "prompt": "User: Simple sentence banao about a cat please.\nAssistant:", "target": "cat"},

    # 8. Multi-turn Context (5)
    {"id": "base_ctx_01", "suite": "multi_turn_context", "prompt": "User: My name is Rahul.\nAssistant: Hello Rahul!\nUser: What is my name?\nAssistant:", "target": "Rahul"},
    {"id": "base_ctx_02", "suite": "multi_turn_context", "prompt": "User: I live in Delhi.\nAssistant: Delhi is noted.\nUser: Where do I live?\nAssistant:", "target": "Delhi"},
    {"id": "base_ctx_03", "suite": "multi_turn_context", "prompt": "User: My project is named ChakrView.\nAssistant: Great project.\nUser: What is my project called?\nAssistant:", "target": "ChakrView"},
    {"id": "base_ctx_04", "suite": "multi_turn_context", "prompt": "User: A CPU processes instructions.\nAssistant: Understood.\nUser: What does a CPU process?\nAssistant:", "target": "instructions"},
    {"id": "base_ctx_05", "suite": "multi_turn_context", "prompt": "User: The meeting is on Friday.\nAssistant: Friday is on schedule.\nUser: Which day is the meeting?\nAssistant:", "target": "Friday"}
]

out_file = Path("experiments/cognition/ucip_04/evaluation/baseline_100_probes.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2, ensure_ascii=False)

print(f"Generated {len(probes)} probes to {out_file}")
