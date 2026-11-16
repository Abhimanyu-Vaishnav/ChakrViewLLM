"""UCIP-07 Sealed 800-Probe Benchmark Builder.

Creates a frozen, leak-free 800-probe benchmark covering 10 required categories:
1. English conversation (100)
2. English explanations (80)
3. English sentence generation (70)
4. Devanagari Hindi (100)
5. Roman Hindi (80)
6. Hinglish (100)
7. Exact copying (80)
8. Elementary arithmetic (60)
9. Instruction following (60)
10. Semantic consistency (70)

Total: exactly 800 probes.
Freezes probe definitions and generates SHA-256 seal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = REPO_ROOT / "experiments" / "cognition" / "ucip_07" / "sealed_800_benchmark.json"


def build_probes() -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = []

    # 1. English Conversation (100 probes)
    conv_topics = [
        ("Hello ChakrView. How are you?", ["well", "good", "fine", "help", "assist", "today"], "Polite status greeting"),
        ("Good morning! How are you doing today?", ["good", "morning", "well", "assist", "today"], "Morning greeting"),
        ("Who created you and what is your purpose?", ["chakrview", "independent", "neural", "local", "purpose", "model"], "Persona identity"),
        ("Can you help me organize my study schedule?", ["schedule", "time", "tasks", "organize", "plan", "help"], "Helpful intent"),
        ("What makes a reliable software engineer?", ["code", "testing", "maintainable", "learn", "rigorous", "engineer"], "Concept opinion"),
        ("Thank you so much for your assistance!", ["welcome", "pleasure", "glad", "anytime", "happy"], "Gratitude response"),
        ("Tell me a polite way to decline an invitation.", ["thank", "invitation", "unfortunately", "commitment", "unable"], "Polite phrase"),
        ("How do you manage difficult priorities?", ["break", "priority", "focus", "step", "tasks"], "Advice explanation"),
        ("Good evening, ChakrView!", ["good", "evening", "assist", "today", "help"], "Evening greeting"),
        ("What is your perspective on continuous learning?", ["learning", "knowledge", "adapt", "grow", "skill"], "Educational perspective"),
    ]
    # Expand to 100 with distinct conversational situations
    idx = 1
    for base_p, kws, rubric in conv_topics:
        for prefix in ["", "Hey! ", "Kindly tell me: ", "Please respond: ", "Greetings! ", "Hello there, ", "Could you answer: ", "Quick question: ", "ChakrView, ", "Dear ChakrView, "]:
            p_text = f"{prefix}{base_p}".strip()
            probes.append({
                "probe_id": f"ENG_CONV_{idx:03d}",
                "category": "English Conversation",
                "language": "English",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": rubric,
                "exact_match": False,
            })
            idx += 1
            if idx > 100:
                break
        if idx > 100:
            break

    # 2. English Explanations (80 probes)
    expl_topics = [
        ("What does a CPU do?", ["processor", "instructions", "executes", "arithmetic", "memory", "fetches"]),
        ("Explain how gravity works in simple terms.", ["gravity", "force", "pulls", "mass", "attracts", "objects"]),
        ("What is photosynthesis?", ["plants", "sunlight", "water", "glucose", "oxygen", "energy"]),
        ("Describe what a computer algorithm is.", ["algorithm", "step", "rules", "instructions", "problem", "solve"]),
        ("What does a microscope do?", ["microscope", "magnifies", "cells", "specimens", "visible", "lens"]),
        ("How does the human heart function?", ["heart", "pumps", "blood", "oxygen", "circulates", "body"]),
        ("What is plate tectonics in geology?", ["plates", "crust", "mantle", "earth", "move", "continents"]),
        ("What is an electrical insulator?", ["insulator", "resists", "current", "electrons", "conduct", "charge"]),
    ]
    idx = 1
    for term, kws in expl_topics:
        for tpl in [
            "Explain {} in clear language.",
            "Can you tell me what {} is and why it matters?",
            "Provide a concise summary: {}",
            "Describe the basic concept: {}",
            "In your own words: {}",
            "Explain briefly: {}",
            "Summarize for a student: {}",
            "What is the definition and function of {}?",
            "Detail how this works: {}",
            "Give a plain explanation: {}",
        ]:
            p_text = tpl.format(term)
            probes.append({
                "probe_id": f"ENG_EXPL_{idx:03d}",
                "category": "English Explanations",
                "language": "English",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": f"Concept explanation of {term}",
                "exact_match": False,
            })
            idx += 1
            if idx > 80:
                break
        if idx > 80:
            break

    # 3. English Sentence Generation (70 probes)
    sent_prompts = [
        ("Write one simple sentence about a dog.", ["dog", "pet", "friendly", "animal", "loyal", "barks"]),
        ("Generate a sentence about an autumn morning.", ["autumn", "leaves", "cool", "morning", "crisp", "trees"]),
        ("Create a sentence describing a calm ocean.", ["ocean", "waves", "calm", "water", "shoreline", "blue"]),
        ("Write a single sentence about an artist painting.", ["artist", "canvas", "painting", "brush", "colors"]),
        ("Describe a starry night in one sentence.", ["stars", "night", "sky", "dark", "sparkled", "bright"]),
        ("Write a sentence about a speeding train.", ["train", "tracks", "speed", "rails", "engine", "travel"]),
        ("Generate a sentence about rain in a forest.", ["rain", "forest", "trees", "drops", "green", "drizzle"]),
    ]
    idx = 1
    for prompt_base, kws in sent_prompts:
        for prefix in ["", "Please ", "Kindly ", "In one sentence: ", "Draft a sentence: ", "Provide a sentence: ", "Write: ", "Formulate a sentence: ", "Quickly: ", "Just one line: "]:
            p_text = f"{prefix}{prompt_base}".strip()
            probes.append({
                "probe_id": f"ENG_SENT_{idx:03d}",
                "category": "English Sentence Generation",
                "language": "English",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": "Sentence structure and topical alignment",
                "exact_match": False,
            })
            idx += 1
            if idx > 70:
                break
        if idx > 70:
            break

    # 4. Devanagari Hindi (100 probes)
    hindi_topics = [
        ("नमस्ते! आप क्या कर सकते हैं?", ["नमस्ते", "सहायता", "कर", "सकता", "सीखने", "प्रश्न", "उत्तर"]),
        ("सूर्य किस दिशा से निकलता है?", ["सूर्य", "पूर्व", "दिशा", "उदय", "निकलता"]),
        ("भारत की राजधानी क्या है?", ["भारत", "राजधानी", "नई दिल्ली", "दिल्ली"]),
        ("पेड़-पौधे हमारे लिए क्यों जरूरी हैं?", ["पेड़", "ऑक्सीजन", "वायु", "स्वच्छ", "पर्यावरण"]),
        ("कंप्यूटर क्या काम करता है?", ["कंप्यूटर", "डेटा", "गणना", "प्रोसेस", "इलेक्ट्रॉनिक"]),
        ("जल का क्या महत्व है?", ["जल", "जीवन", "अनिवार्य", "पीने", "शरीर"]),
        ("किताबें पढ़ने से क्या लाभ होता है?", ["किताबें", "ज्ञान", "पढ़ने", "सोच", "विकास"]),
        ("समय का सदुपयोग क्यों करना चाहिए?", ["समय", "अनमोल", "सफलता", "उपयोग", "काम"]),
        ("स्वास्थ्य का ध्यान कैसे रखें?", ["स्वास्थ्य", "आहार", "व्यायाम", "स्वस्थ", "भोजन"]),
        ("वर्षा कैसे होती है?", ["वर्षा", "बादल", "जलवाष्प", "पानी", "बारिश"]),
    ]
    idx = 1
    for q, kws in hindi_topics:
        for tpl in [
            "{}",
            "कृपया बताइए: {}",
            "सरल शब्दों में उत्तर दीजिए: {}",
            "क्या आप बता सकते हैं: {}",
            "नमस्ते, {}",
            "जानकारी दीजिए: {}",
            "विस्तार से बताइए: {}",
            "एक पंक्ति में समझाइए: {}",
            "उत्तर दें: {}",
            "स्पष्ट कीजिए: {}",
        ]:
            p_text = tpl.format(q)
            probes.append({
                "probe_id": f"DEV_HINDI_{idx:03d}",
                "category": "Devanagari Hindi",
                "language": "Hindi",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": "Devanagari script presence and topical relevance",
                "exact_match": False,
            })
            idx += 1
            if idx > 100:
                break
        if idx > 100:
            break

    # 5. Roman Hindi (80 probes)
    roman_topics = [
        ("Aap kaise hain?", ["main", "theek", "hoon", "aap", "kaise"]),
        ("Aapka naam kya hai?", ["mera", "naam", "chakrview", "hai"]),
        ("Suraj kis disha se nikalta hai?", ["suraj", "poorab", "disha", "nikalta"]),
        ("Bharat ki rajdhani kya hai?", ["bharat", "rajdhani", "delhi", "new delhi"]),
        ("Paani peena kyun zaroori hai?", ["paani", "shareer", "sehat", "hydrated"]),
        ("Ped lagane se kya hota hai?", ["ped", "oxygen", "hawa", "swachh"]),
        ("Subah jaldi uthne ke kya fayde hain?", ["subah", "energy", "sehat", "din"]),
        ("Kitabein padhne se kya seekhne ko milta hai?", ["kitabein", "gyaan", "samajh", "naya"]),
    ]
    idx = 1
    for q, kws in roman_topics:
        for prefix in ["", "Sunie, ", "Bataiye: ", "Jaldi batao: ", "ChakrView, ", "Ek sawaal hai: ", "Please batao: ", "Aap bataiye: ", "Brief mein: ", "Short mein: "]:
            p_text = f"{prefix}{q}".strip()
            probes.append({
                "probe_id": f"ROM_HINDI_{idx:03d}",
                "category": "Roman Hindi",
                "language": "Roman Hindi",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": "Roman Hindi phrasing and context",
                "exact_match": False,
            })
            idx += 1
            if idx > 80:
                break
        if idx > 80:
            break

    # 6. Hinglish (100 probes)
    hinglish_topics = [
        ("Bhai, machine learning ko simple language mein samjhao.", ["machine learning", "data", "pattern", "seekh", "computer"]),
        ("Cloud computing kya hota hai brief mein batao.", ["cloud", "servers", "internet", "data", "store"]),
        ("API ka simple explanation do.", ["api", "software", "bridge", "data", "connect"]),
        ("Database indexing kyun zaroori hoti hai?", ["index", "search", "fast", "query", "table"]),
        ("Git aur GitHub me main difference kya hai?", ["git", "version control", "github", "online", "host"]),
        ("Operating system ka main purpose kya hai?", ["os", "hardware", "manage", "programs", "system"]),
        ("Cache memory fast access kaise deti hai?", ["cache", "cpu", "fast", "frequent", "memory"]),
        ("Virtual memory program crash hone se kaise bachati hai?", ["virtual memory", "ram", "hard drive", "space", "crash"]),
        ("Debugging code me kya hoti hai?", ["debug", "bugs", "errors", "fix", "code"]),
        ("Containers applications ko smoothly kaise run karte hain?", ["container", "dependencies", "package", "run", "environment"]),
    ]
    idx = 1
    for q, kws in hinglish_topics:
        for prefix in ["", "Hey ChakrView, ", "Please explain: ", "Bhai, ", "Short me batao: ", "Dost, ", "Simple words me: ", "Technically samjhao: ", "Ek line me: ", "Bataiye: "]:
            p_text = f"{prefix}{q}".strip()
            probes.append({
                "probe_id": f"HINGLISH_{idx:03d}",
                "category": "Hinglish",
                "language": "Hinglish",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": "Code-mixed Hinglish generation",
                "exact_match": False,
            })
            idx += 1
            if idx > 100:
                break
        if idx > 100:
            break

    # 7. Exact Copying (80 probes)
    copy_phrases = [
        "The sky is blue.",
        "Consistency is key to mastery.",
        "Water boils at one hundred degrees.",
        "Digital computers process binary signals.",
        "Morning walks provide daily energy.",
        "Patience overcomes difficult obstacles.",
        "Mathematics underpins modern physics.",
        "Knowledge grows when shared.",
    ]
    idx = 1
    for phrase in copy_phrases:
        for tpl in [
            "Say exactly: {}",
            "Echo verbatim: {}",
            "Please repeat this exact phrase: {}",
            "Copy precisely: {}",
            "Output only: {}",
            "Write back without changes: {}",
            "Repeat word for word: {}",
            "Duplicate this text: {}",
            "Echo back: {}",
            "Print verbatim: {}",
        ]:
            p_text = tpl.format(phrase)
            probes.append({
                "probe_id": f"EXACT_COPY_{idx:03d}",
                "category": "Exact Copying",
                "language": "English",
                "input": p_text,
                "expected_keywords": [phrase],
                "rubric": "Exact substring match",
                "exact_match": True,
            })
            idx += 1
            if idx > 80:
                break
        if idx > 80:
            break

    # 8. Elementary Arithmetic (60 probes)
    arith_items = [
        ("If I have 5 apples and give 2 to my friend, how many apples remain?", ["3", "three"], "5 - 2 = 3"),
        ("What is 4 + 3?", ["7", "seven"], "4 + 3 = 7"),
        ("Compute: 10 - 4", ["6", "six"], "10 - 4 = 6"),
        ("Calculate 8 plus 5.", ["13", "thirteen"], "8 + 5 = 13"),
        ("What is 6 multiplied by 3?", ["18", "eighteen"], "6 * 3 = 18"),
        ("If there are 12 birds and 5 fly away, how many stay?", ["7", "seven"], "12 - 5 = 7"),
    ]
    idx = 1
    for q, kws, rubric in arith_items:
        for prefix in ["", "Calculate: ", "Please compute: ", "Solve this: ", "Answer: ", "Quick math: ", "Tell me: ", "Find result: ", "Evaluate: ", "What is: "]:
            p_text = f"{prefix}{q}".strip()
            probes.append({
                "probe_id": f"ARITH_{idx:03d}",
                "category": "Elementary Arithmetic",
                "language": "English",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": rubric,
                "exact_match": False,
            })
            idx += 1
            if idx > 60:
                break
        if idx > 60:
            break

    # 9. Instruction Following (60 probes)
    inst_items = [
        ("Answer in exactly one word: What is the color of the clear daytime sky?", ["blue"]),
        ("Reply with only 'Yes' or 'No': Is the Earth flat?", ["no"]),
        ("Provide the capital of France in one word.", ["paris"]),
        ("State the opposite of 'Cold' in one single word.", ["hot", "warm"]),
        ("Reply with only 'True' or 'False': Water boils at 100 degrees Celsius.", ["true"]),
        ("State the primary color of fresh grass in one word.", ["green"]),
    ]
    idx = 1
    for q, kws in inst_items:
        for prefix in ["", "Instruction: ", "Follow strictly: ", "Constraint: ", "Mandatory rule: ", "Direct answer: ", "Respond: ", "Execute: ", "Exact reply: ", "Strict output: "]:
            p_text = f"{prefix}{q}".strip()
            probes.append({
                "probe_id": f"INSTRUCT_{idx:03d}",
                "category": "Instruction Following",
                "language": "English",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": "Constraint keyword satisfied",
                "exact_match": False,
            })
            idx += 1
            if idx > 60:
                break
        if idx > 60:
            break

    # 10. Semantic Consistency (70 probes)
    sem_items = [
        ("Describe the ocean without mentioning trees or mountains.", ["ocean", "water", "waves", "tides", "sea", "marine"]),
        ("Discuss computer memory without mentioning cooking or food.", ["memory", "ram", "storage", "data", "bytes", "cache"]),
        ("Explain what an airplane does without talking about oceans.", ["airplane", "flies", "air", "wings", "travel", "passengers"]),
        ("Explain what a bicycle is without mentioning airplanes.", ["bicycle", "wheels", "pedals", "ride", "cyclist", "gears"]),
        ("Describe a desert climate without mentioning ice or snow.", ["desert", "sand", "dry", "heat", "arid", "sun"]),
        ("Explain solar power without talking about coal or petroleum.", ["solar", "sun", "light", "energy", "photovoltaic", "panels"]),
        ("Describe the moon without mentioning rivers or forests.", ["moon", "orbit", "earth", "craters", "satellite", "night"]),
    ]
    idx = 1
    for q, kws in sem_items:
        for prefix in ["", "Please ", "Carefully: ", "With focus: ", "Consistent description: ", "Topical focus: ", "Strict focus: ", "Answer: ", "Explain: ", "Provide info: "]:
            p_text = f"{prefix}{q}".strip()
            probes.append({
                "probe_id": f"SEM_CONST_{idx:03d}",
                "category": "Semantic Consistency",
                "language": "English",
                "input": p_text,
                "expected_keywords": kws,
                "rubric": "Topical alignment without domain drift",
                "exact_match": False,
            })
            idx += 1
            if idx > 70:
                break
        if idx > 70:
            break

    return probes


def main():
    probes = build_probes()
    total = len(probes)
    print(f"Total frozen benchmark probes constructed: {total}")

    cat_counts = {}
    for p in probes:
        c = p["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    for c, cnt in sorted(cat_counts.items()):
        print(f"  {c:<30}: {cnt:>3} probes")

    data = {
        "metadata": {
            "mission": "UCIP-07",
            "total_probes": total,
            "version": "1.0-frozen-sealed",
            "benchmark_distribution": cat_counts,
        },
        "probes": probes,
    }

    serialized = json.dumps(data, indent=2, ensure_ascii=False)
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        f.write(serialized)

    h = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    print(f"\nSaved sealed 800 benchmark to: {BENCHMARK_PATH}")
    print(f"Authoritative SHA-256: {h}")


if __name__ == "__main__":
    main()
