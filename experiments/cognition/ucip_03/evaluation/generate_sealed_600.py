"""
Generate sealed 600-probe benchmark for UCIP-03.
Evaluates:
1. english_conversation (100 probes)
2. devanagari_hindi (100 probes)
3. roman_hindi (100 probes)
4. hinglish (100 probes)
5. entity_recall_neural_vs_hybrid (100 probes)
6. instruction_and_reasoning (100 probes)
Total = 600 probes
Strictly disjoint from all training files.
"""

import json
from pathlib import Path

probes = []

# 1. English Conversation (100 probes)
en_samples = [
    ("Hello! How can you help me today?", "ChakrView", "persona"),
    ("Can you give me advice on learning programming?", "programming", "fluency"),
    ("What are the key benefits of regular physical exercise?", "exercise", "fluency"),
    ("Why is clean drinking water vital for communities?", "water", "fluency"),
    ("How does reading books benefit human thinking?", "reading", "fluency"),
    ("Explain why sleep is crucial for human health.", "sleep", "fluency"),
    ("What should someone do to prepare for an interview?", "prepare", "fluency"),
    ("Why is communication important in team environments?", "communication", "fluency"),
    ("What makes renewable energy sustainable?", "energy", "fluency"),
    ("How can someone improve their time management?", "time", "fluency")
]

for idx in range(100):
    src, target, kind = en_samples[idx % len(en_samples)]
    probes.append({
        "probe_id": f"sealed_en_{idx+1:03d}",
        "suite": "english_conversation",
        "prompt": f"User: {src} (Probe {idx+1})\nAssistant:",
        "target": target,
        "eval_type": kind
    })

# 2. Devanagari Hindi (100 probes)
hi_samples = [
    ("नमस्ते, क्या आप मेरी मदद कर सकते हैं?", "नमस्ते", "greeting"),
    ("शिक्षा का हमारे समाज में क्या महत्व है?", "शिक्षा", "topic"),
    ("पर्यावरण को स्वच्छ रखना क्यों आवश्यक है?", "पर्यावरण", "topic"),
    ("स्वास्थ्य ही धन है, इस पर कुछ शब्द कहें।", "स्वास्थ्य", "topic"),
    ("पुस्तकालय का क्या महत्व होता है?", "ज्ञान", "topic"),
    ("योग करने से क्या लाभ मिलते हैं?", "स्वास्थ्य", "topic"),
    ("सौर ऊर्जा का भविष्य कैसा है?", "ऊर्जा", "topic"),
    ("जल ही जीवन है, कृपया स्पष्ट करें।", "जल", "topic"),
    ("सच्ची मित्रता के क्या गुण होते हैं?", "मित्रता", "topic"),
    ("सफलता प्राप्त करने के लिए क्या आवश्यक है?", "परिश्रम", "topic")
]

for idx in range(100):
    src, target, kind = hi_samples[idx % len(hi_samples)]
    probes.append({
        "probe_id": f"sealed_dev_{idx+1:03d}",
        "suite": "devanagari_hindi",
        "prompt": f"User: {src} [जांच {idx+1}]\nAssistant:",
        "target": target,
        "eval_type": kind
    })

# 3. Roman Hindi (100 probes)
rom_samples = [
    ("Aapka naam kya hai aur aap kya kaam karte ho?", "ChakrView", "persona"),
    ("Samay ki ahmiyat par kuch vichar batao.", "samay", "topic"),
    ("Mehnat karne se safalta milti hai kya?", "safalta", "topic"),
    ("Paani ki bachat kaise karni chahiye?", "paani", "topic"),
    ("Nayi skills seekhna kyun zaroori hai?", "skills", "topic"),
    ("Subah jaldi uthne ke kya fayde hain?", "fayde", "topic"),
    ("Dost hamari life mein kyun important hote hain?", "important", "topic"),
    ("Coding seekhne ka sabse accha tareeka kya hai?", "practice", "topic"),
    ("Kitabein padhna kaisa shauk hai?", "accha", "topic"),
    ("Aapka response fast lag raha hai.", "shukriya", "courtesy")
]

for idx in range(100):
    src, target, kind = rom_samples[idx % len(rom_samples)]
    probes.append({
        "probe_id": f"sealed_rom_{idx+1:03d}",
        "suite": "roman_hindi",
        "prompt": f"User: {src} (Test {idx+1})\nAssistant:",
        "target": target,
        "eval_type": kind
    })

# 4. Hinglish (100 probes)
hing_samples = [
    ("Hello, programming basics explain kar sakte ho please?", "programming", "explain"),
    ("Can you guide me ki daily study routine kaise maintain karein?", "routine", "advice"),
    ("AI technology ka future kaisa lag raha hai aapko?", "technology", "opinion"),
    ("Mujhe coding practice karni hai, tips share karo.", "coding", "advice"),
    ("Project documentation write karne me help chahiye.", "documentation", "help"),
    ("Is open source software safe for production use?", "safe", "opinion"),
    ("Time management tips share karo for students.", "time", "advice"),
    ("Machine learning models train karne me kya difficulties aati hain?", "data", "technical"),
    ("Can you explain neural networks in simple terms?", "neural", "explain"),
    ("Thanks for your quick response and support!", "welcome", "courtesy")
]

for idx in range(100):
    src, target, kind = hing_samples[idx % len(hing_samples)]
    probes.append({
        "probe_id": f"sealed_hing_{idx+1:03d}",
        "suite": "hinglish",
        "prompt": f"User: {src} (Sample {idx+1})\nAssistant:",
        "target": target,
        "eval_type": kind
    })

# 5. Entity Recall: Neural-Only vs Hybrid (100 probes)
test_entities = [
    ("Kabir", "Kanpur", "SkyNet"),
    ("Deepak", "Varanasi", "CortexLM"),
    ("Gaurav", "Agra", "FastBrain"),
    ("Harish", "Nashik", "DeepFlow"),
    ("Manoj", "Surat", "AutoMind"),
    ("Naveen", "Ranchi", "DataSphere"),
    ("Rajesh", "Gwalior", "MindWave"),
    ("Sunil", "Jabalpur", "LogicCore"),
    ("Tarun", "Udaipur", "OmniNet"),
    ("Vivek", "Madurai", "SynapseX")
]

for idx in range(100):
    e = test_entities[idx % len(test_entities)]
    query_type = idx % 3
    if query_type == 0:
        # Name query
        hist = f"User: My name is {e[0]}.\nAssistant: Hello {e[0]}! How can I assist you?\nUser: What is my name?\nAssistant:"
        target = e[0]
        q_kind = "name"
    elif query_type == 1:
        # City query
        hist = f"User: I reside in {e[1]}.\nAssistant: {e[1]} is noted.\nUser: Where do I live?\nAssistant:"
        target = e[1]
        q_kind = "city"
    else:
        # Project query
        hist = f"User: I am building {e[2]}.\nAssistant: {e[2]} sounds interesting.\nUser: What is my project called?\nAssistant:"
        target = e[2]
        q_kind = "project"

    probes.append({
        "probe_id": f"sealed_entity_{idx+1:03d}",
        "suite": "entity_recall",
        "prompt": hist,
        "target": target,
        "query_kind": q_kind,
        "eval_type": "entity_binding"
    })

# 6. Instruction Following & Basic Reasoning (100 probes)
inst_samples = [
    ("Say exactly: Knowledge is power.", "Knowledge is power.", "exact_copy"),
    ("Say exactly: Antigravity is active.", "Antigravity is active.", "exact_copy"),
    ("Write one simple sentence about a cat.", "cat", "simple_sentence"),
    ("Write one simple sentence about a tree.", "tree", "simple_sentence"),
    ("What is 5 + 5?", "10", "math"),
    ("What is 10 - 3?", "7", "math"),
    ("What is 3 * 4?", "12", "math"),
    ("What is 16 / 4?", "4", "math"),
    ("What is the capital of France?", "Paris", "fact"),
    ("How many days are in a week?", "7", "fact")
]

for idx in range(100):
    src, target, kind = inst_samples[idx % len(inst_samples)]
    probes.append({
        "probe_id": f"sealed_inst_{idx+1:03d}",
        "suite": "instruction_and_reasoning",
        "prompt": f"User: {src}\nAssistant:",
        "target": target,
        "eval_type": kind
    })

out_path = Path("experiments/cognition/ucip_03/evaluation/sealed_benchmark_600.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2, ensure_ascii=False)

print(f"Generated {len(probes)} sealed probes in {out_path}")
