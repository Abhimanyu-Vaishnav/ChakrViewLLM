"""
CR-EXP-09 Comprehensive Multi-Tier Capability Benchmark Suite Generator
Constructs 600 frozen evaluation probes across 8 independent cognitive categories:
1. Conversational Understanding & Persona Coherence (75)
2. Single-Hop & Counterfactual Entity Binding (75)
3. Multi-Entity Distractor Disambiguation (75)
4. Epistemic Missing-Information Abstention (75)
5. Multi-Turn Dialogue State Tracking & Memory Recall (75)
6. Structured State Transformation & Elementary Arithmetic (75)
7. Conditional & Syllogistic Logical Reasoning (75)
8. Output Stability & Constraint Satisfaction (75)

All probes feature strictly disjoint OOD entities and counterbalanced designs.
Outputs benchmark_suite.json and benchmark_manifest.json.
"""

import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_09_autonomous_mission" / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# Strict Disjoint Entity Pools
SUBJECTS_BENCHMARK = [
    "Vikram", "Zoya", "Arjun", "Meera", "Suresh", "Tanya", "Aditya", "Ritu",
    "Dev", "Tara", "Nikhil", "Ishita", "Varun", "Rhea", "Manish", "Pooja"
]

CITIES_BENCHMARK = [
    "Delhi", "Jaipur", "Pune", "Surat", "Kanpur", "Nagpur", "Indore", "Patna",
    "Mumbai", "Bengaluru", "Bhopal", "Nashik", "Chennai", "Kolkata", "Hyderabad", "Lucknow"
]

OBJECTS_BENCHMARK = ["books", "pens", "coins", "apples", "candies", "keys", "cards", "notebooks"]

def build_benchmark():
    random.seed(999) # Fixed seed for frozen benchmark
    probes = []
    
    # -------------------------------------------------------------
    # Category 1: Conversational Understanding & Persona (75 samples)
    # -------------------------------------------------------------
    conv_prompts = [
        ("User: Hello! How are you doing today?\nAssistant: ", "Hello! I am doing well, ready to assist you. How can I help you today?"),
        ("User: Who are you?\nAssistant: ", "I am ChakrView, an indigenous neural intelligence being developed from scratch."),
        ("User: What is your primary purpose?\nAssistant: ", "My purpose is to develop genuine neural capabilities for conversational understanding and reasoning."),
        ("User: Can you explain your mission?\nAssistant: ", "I am built to progressively learn language understanding, contextual memory, and problem-solving step by step."),
        ("User: Good morning!\nAssistant: ", "Good morning! I hope you have a wonderful and productive day. What are you working on?"),
        ("User: Thank you so much for your assistance.\nAssistant: ", "You are very welcome! Please let me know if you need any further help."),
        ("User: Namaste!\nAssistant: ", "Namaste! Main badhiya hoon. Aapki kya madad kar sakta hoon?"),
        ("User: Tell me about yourself.\nAssistant: ", "I am ChakrView, an autonomous cognitive prototype learning to understand context and solve problems.")
    ]
    for i in range(75):
        p, a = random.choice(conv_prompts)
        probes.append({
            "id": f"CAT1_{i+1:03d}",
            "category": "conversational_understanding",
            "prompt": p,
            "expected_target": a,
            "evaluation_type": "coherence_and_persona"
        })

    # -------------------------------------------------------------
    # Category 2: Single-Hop & Counterfactual Entity Binding (75 samples)
    # -------------------------------------------------------------
    for i in range(75):
        s = random.choice(SUBJECTS_BENCHMARK)
        c1, c2 = random.sample(CITIES_BENCHMARK, 2)
        if i % 2 == 0:
            # Single-hop
            p = f"Mapping: {s} lives in {c1}. Query: Where does {s} live? Answer: "
            exp = c1
        else:
            # Counterfactual update
            p = f"Mapping: {s} used to live in {c1}, but recently relocated to {c2}. Query: Where does {s} currently live? Answer: "
            exp = c2
        probes.append({
            "id": f"CAT2_{i+1:03d}",
            "category": "single_hop_and_counterfactual",
            "prompt": p,
            "expected_target": exp,
            "evaluation_type": "exact_entity_match"
        })

    # -------------------------------------------------------------
    # Category 3: Multi-Entity Distractor Disambiguation (75 samples)
    # -------------------------------------------------------------
    for i in range(75):
        s1, s2 = random.sample(SUBJECTS_BENCHMARK, 2)
        c1, c2 = random.sample(CITIES_BENCHMARK, 2)
        # Randomize query target to avoid recency bias
        target_s, target_c = (s1, c1) if random.random() < 0.5 else (s2, c2)
        p = f"Mapping: {s1} lives in {c1}. {s2} lives in {c2}. Query: Where does {target_s} live? Answer: "
        probes.append({
            "id": f"CAT3_{i+1:03d}",
            "category": "multi_entity_distractor",
            "prompt": p,
            "expected_target": target_c,
            "distractor_entity": s2 if target_s == s1 else s1,
            "distractor_value": c2 if target_c == c1 else c1,
            "evaluation_type": "distractor_resistance"
        })

    # -------------------------------------------------------------
    # Category 4: Epistemic Missing-Information Abstention (75 samples)
    # -------------------------------------------------------------
    for i in range(75):
        s1, s2 = random.sample(SUBJECTS_BENCHMARK, 2)
        c1 = random.choice(CITIES_BENCHMARK)
        p = f"Mapping: {s1} lives in {c1}. Query: Where does {s2} live? Answer: "
        probes.append({
            "id": f"CAT4_{i+1:03d}",
            "category": "missing_info_abstention",
            "prompt": p,
            "expected_target": "The context does not state where " + s2 + " lives.",
            "unmentioned_entity": s2,
            "evaluation_type": "epistemic_abstention"
        })

    # -------------------------------------------------------------
    # Category 5: Multi-Turn Dialogue State Tracking & Memory Recall (75 samples)
    # -------------------------------------------------------------
    for i in range(75):
        s = random.choice(SUBJECTS_BENCHMARK)
        c1, c2 = random.sample(CITIES_BENCHMARK, 2)
        p = f"User: Please note that {s} is located in {c1}.\nAssistant: Understood, I have noted that {s} is in {c1}.\nUser: Actually, {s} just moved to {c2}.\nAssistant: Noted the change.\nUser: Where is {s} now?\nAssistant: "
        probes.append({
            "id": f"CAT5_{i+1:03d}",
            "category": "multiturn_state_tracking",
            "prompt": p,
            "expected_target": c2,
            "earlier_value": c1,
            "evaluation_type": "multiturn_state_update"
        })

    # -------------------------------------------------------------
    # Category 6: Structured State Transformation & Arithmetic (75 samples)
    # -------------------------------------------------------------
    for i in range(75):
        s1, s2 = random.sample(SUBJECTS_BENCHMARK, 2)
        obj = random.choice(OBJECTS_BENCHMARK)
        init_a = random.randint(4, 9)
        give_qty = random.randint(1, init_a - 1)
        rem_a = init_a - give_qty
        p = f"Context: {s1} has {init_a} {obj}. {s1} gives {give_qty} {obj} to {s2}. Question: How many {obj} does {s1} have left? Answer: "
        probes.append({
            "id": f"CAT6_{i+1:03d}",
            "category": "structured_arithmetic_state",
            "prompt": p,
            "expected_target": str(rem_a),
            "initial_state": init_a,
            "delta": give_qty,
            "evaluation_type": "numerical_exact_match"
        })

    # -------------------------------------------------------------
    # Category 7: Conditional & Syllogistic Logical Reasoning (75 samples)
    # -------------------------------------------------------------
    for i in range(75):
        s = random.choice(SUBJECTS_BENCHMARK)
        rules = [
            (f"Rule: If a person is in Delhi, they have a green pass. Premise: {s} is in Delhi. Question: Does {s} have a green pass? Answer: ", "Yes"),
            (f"Rule: If a person is in Mumbai, they have a red pass. Premise: {s} is in Pune. Question: Does {s} have a red pass? Answer: ", "No"),
            (f"Premise: All members of Team Alpha are certified. {s} is a member of Team Alpha. Question: Is {s} certified? Answer: ", "Yes"),
            (f"Premise: Only senior engineers can approve releases. {s} is an intern. Question: Can {s} approve releases? Answer: ", "No")
        ]
        p, ans = random.choice(rules)
        probes.append({
            "id": f"CAT7_{i+1:03d}",
            "category": "conditional_logical_reasoning",
            "prompt": p,
            "expected_target": ans,
            "evaluation_type": "boolean_reasoning"
        })

    # -------------------------------------------------------------
    # Category 8: Output Stability & Constraint Satisfaction (75 samples)
    # -------------------------------------------------------------
    for i in range(75):
        s = random.choice(SUBJECTS_BENCHMARK)
        c = random.choice(CITIES_BENCHMARK)
        p = f"Instruction: Reply with only the city name in one single word. Context: {s} is in {c}. Question: Where is {s}? Answer: "
        probes.append({
            "id": f"CAT8_{i+1:03d}",
            "category": "constraint_satisfaction",
            "prompt": p,
            "expected_target": c,
            "constraint": "single_word_exact",
            "evaluation_type": "constrained_output"
        })

    assert len(probes) == 600
    
    with open(BENCHMARK_DIR / "benchmark_suite.json", "w", encoding="utf-8") as f:
        json.dump(probes, f, indent=2)
        
    manifest = {
        "total_probes": len(probes),
        "category_counts": {
            "conversational_understanding": 75,
            "single_hop_and_counterfactual": 75,
            "multi_entity_distractor": 75,
            "missing_info_abstention": 75,
            "multiturn_state_tracking": 75,
            "structured_arithmetic_state": 75,
            "conditional_logical_reasoning": 75,
            "constraint_satisfaction": 75
        },
        "evaluation_standard": "Greedy generation with category-specific verification rules",
        "random_seed": 999
    }
    
    with open(BENCHMARK_DIR / "benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"CR-EXP-09 600-Probe Benchmark Suite generated successfully at {BENCHMARK_DIR}.")

if __name__ == "__main__":
    build_benchmark()

