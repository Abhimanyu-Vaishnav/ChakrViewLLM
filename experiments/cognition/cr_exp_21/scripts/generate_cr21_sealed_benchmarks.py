"""
generate_cr21_sealed_benchmarks.py - 1,450 Sealed Fresh Evaluation Benchmark Probes

Capabilities:
1. English conversation (150 probes)
2. Devanagari Hindi (150 probes)
3. Roman Hindi (150 probes)
4. Hinglish (150 probes)
5. Cross-language switching (100 probes)
6. Multi-turn context (200 probes: 1, 2, 3, 5, 8 turns)
7. Entity binding (200 probes)
8. User correction (100 probes)
9. Instruction following (100 probes)
10. Uncertainty and abstention (100 probes)

Total: 1,400 Probes completely held out with zero train overlap.
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
BENCH_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_21" / "evaluation"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

random.seed(424242)

SEALED_NAMES = [f"SealedUser_{i:03d}" for i in range(300)]
SEALED_CITIES = [f"SealedCity_{i:03d}" for i in range(300)]
SEALED_PROJECTS = [f"SealedProject_{i:03d}" for i in range(150)]

def build_benchmarks():
    benchmarks = {}

    # 1. English conversation (150 probes)
    b_en = []
    for i in range(150):
        name = SEALED_NAMES[i]
        b_en.append({
            "id": f"CR21_EN_{i+1:03d}",
            "category": "english_conversation",
            "prompt": f"User: Hello, my name is {name}.\nAssistant: Hello {name}!\nUser: What is my name?\nAssistant:",
            "expected": [name],
            "target": name
        })
    benchmarks["sealed_english_150.json"] = b_en

    # 2. Devanagari Hindi (150 probes)
    b_hi = []
    for i in range(150):
        name = f"परीक्षार्थी_{i+1:03d}"
        b_hi.append({
            "id": f"CR21_HI_{i+1:03d}",
            "category": "devanagari_hindi",
            "prompt": f"User: मेरा नाम {name} है।\nAssistant: नमस्ते {name}!\nUser: मेरा नाम क्या है?\nAssistant:",
            "expected": [name],
            "target": name
        })
    benchmarks["sealed_devanagari_150.json"] = b_hi

    # 3. Roman Hindi (150 probes)
    b_rom = []
    for i in range(150):
        name = SEALED_NAMES[i+50]
        b_rom.append({
            "id": f"CR21_ROM_{i+1:03d}",
            "category": "roman_hindi",
            "prompt": f"User: Mera naam {name} hai.\nAssistant: Hello {name}!\nUser: Mera naam kya hai?\nAssistant:",
            "expected": [name],
            "target": name
        })
    benchmarks["sealed_roman_hindi_150.json"] = b_rom

    # 4. Hinglish (150 probes)
    b_hing = []
    for i in range(150):
        name = SEALED_NAMES[i+100]
        proj = SEALED_PROJECTS[i]
        b_hing.append({
            "id": f"CR21_HING_{i+1:03d}",
            "category": "hinglish",
            "prompt": f"User: Hey, I am {name} and building {proj}.\nAssistant: Great {name}!\nUser: Mera project name kya hai?\nAssistant:",
            "expected": [proj],
            "target": proj
        })
    benchmarks["sealed_hinglish_150.json"] = b_hing

    # 5. Cross-language switching (100 probes)
    b_cross = []
    for i in range(100):
        name = SEALED_NAMES[i+150]
        b_cross.append({
            "id": f"CR21_CROSS_{i+1:03d}",
            "category": "cross_language",
            "prompt": f"User: My name is {name}. Mujhe Hindi mein batao mera naam kya hai.\nAssistant:",
            "expected": [name],
            "target": name
        })
    benchmarks["sealed_cross_language_100.json"] = b_cross

    # 6. Multi-turn context (200 probes: 40 each for 1, 2, 3, 5, 8 turns)
    b_ctx = []
    horizons = [(1, 40), (2, 40), (3, 40), (5, 40), (8, 40)]
    p_idx = 1
    for h, count in horizons:
        for _ in range(count):
            name = SEALED_NAMES[p_idx % len(SEALED_NAMES)]
            city = SEALED_CITIES[p_idx % len(SEALED_CITIES)]
            turns = [(f"My name is {name} and I live in {city}.", f"Acknowledged {name} from {city}.")]
            for k in range(h - 1):
                turns.append((f"Tell me about topic {k+1}.", f"Topic {k+1} is clear."))
            
            prompt_str = ""
            for u, a in turns:
                prompt_str += f"User: {u}\nAssistant: {a}\n"
            prompt_str += "User: Where do I live?\nAssistant:"

            b_ctx.append({
                "id": f"CR21_CTX_{p_idx:03d}",
                "turns_horizon": h,
                "category": "multiturn_context",
                "prompt": prompt_str,
                "expected": [city],
                "target": city
            })
            p_idx += 1
    benchmarks["sealed_context_200.json"] = b_ctx

    # 7. Entity binding (200 probes)
    b_bind = []
    for i in range(200):
        name = SEALED_NAMES[(i+20) % len(SEALED_NAMES)]
        city = SEALED_CITIES[(i+40) % len(SEALED_CITIES)]
        proj = SEALED_PROJECTS[i % len(SEALED_PROJECTS)]
        b_bind.append({
            "id": f"CR21_BIND_{i+1:03d}",
            "category": "entity_binding",
            "prompt": f"User: User {name} works on {proj} in {city}.\nAssistant: Verified.\nUser: Which project does {name} work on?\nAssistant:",
            "expected": [proj],
            "target": proj
        })
    benchmarks["sealed_entity_binding_200.json"] = b_bind

    # 8. User correction (100 probes)
    b_corr = []
    for i in range(100):
        name = SEALED_NAMES[i]
        c1 = SEALED_CITIES[i]
        c2 = SEALED_CITIES[i+100]
        b_corr.append({
            "id": f"CR21_CORR_{i+1:03d}",
            "category": "user_correction",
            "prompt": f"User: I am {name} living in {c1}.\nAssistant: Understood.\nUser: Actually, I moved to {c2}.\nAssistant: Location updated to {c2}.\nUser: Which city do I live in now?\nAssistant:",
            "expected": [c2],
            "target": c2
        })
    benchmarks["sealed_user_correction_100.json"] = b_corr

    # 9. Instruction following (100 probes)
    b_inst = []
    for i in range(100):
        name = SEALED_NAMES[i]
        b_inst.append({
            "id": f"CR21_INST_{i+1:03d}",
            "category": "instruction_following",
            "prompt": f"User: What is 2 + 2? Answer in exactly one word.\nAssistant:",
            "expected": ["4", "four"],
            "target": "4"
        })
    benchmarks["sealed_instruction_following_100.json"] = b_inst

    # 10. Uncertainty and abstention (100 probes)
    b_abst = []
    for i in range(100):
        unknown_entity = f"UnknownEntity_{i+1:03d}"
        b_abst.append({
            "id": f"CR21_ABST_{i+1:03d}",
            "category": "uncertainty_abstention",
            "prompt": f"User: Where is {unknown_entity} located?\nAssistant:",
            "expected": ["not provided", "unknown", "do not know", "unavailable", "cannot"],
            "target": "unknown"
        })
    benchmarks["sealed_uncertainty_abstention_100.json"] = b_abst

    return benchmarks

def main():
    print("Generating CR-21 Sealed Evaluation Benchmarks (1,400 Probes)...")
    bmarks = build_benchmarks()
    manifest = {}
    for fname, data in bmarks.items():
        p = BENCH_DIR / fname
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest[fname] = {
            "probes_count": len(data),
            "sha256": h
        }
        print(f"Sealed Benchmark {fname}: {len(data)} probes | SHA-256: {h}")

    with open(BENCH_DIR / "sealed_benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved all sealed benchmarks and sealed_benchmark_manifest.json.")

if __name__ == "__main__":
    main()

