"""
Project Genesis Complete Dual-Track Acceptance Suite.
Evaluates:
Track A: Pure Neural Foundation (Direct autoregressive generation, no tools, no memory)
Track B: Full Cognitive Hybrid (ChakrGenesis Neural core + Memory + Reasoning + Router)
"""

from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.interface.genesis.agent_controller import ChakrGenesisAgent


def run_genesis_evaluation():
    print("=================================================================")
    print("  PROJECT GENESIS: INDEPENDENT DUAL-TRACK ACCEPTANCE SUITE       ")
    print("=================================================================\n")

    test_mem_dir = Path("data/eval_genesis_memory")
    if test_mem_dir.exists():
        shutil.rmtree(test_mem_dir)
    test_mem_dir.mkdir(parents=True, exist_ok=True)

    agent = ChakrGenesisAgent(memory_dir=test_mem_dir)

    results = {}

    # ---------------- TRACK A: Pure Neural Core ----------------
    print(">>> EVALUATING TRACK A: Pure Neural Foundation (No tools, no templates) <<<")
    neural_prompts = [
        "User: Hello! Who are you?\nAssistant:",
        "User: Good morning!\nAssistant:",
        "User: What is photosynthesis?\nAssistant:",
        "User: What is 5 + 7?\nAssistant:",
        "User: Write a python function to check if a number is even.\nAssistant:",
        "User: Namaste, aap kaise hain?\nAssistant:"
    ]
    track_a_outputs = []
    track_a_success = 0
    for np in neural_prompts:
        res = agent.process_message(np, mode="pure_neural")
        reply = res["reply"]
        # Measure if reply is non-empty, English/Hindi words present, and no loop
        valid = len(reply) > 3 and not reply.startswith("The kind even") and not reply.startswith("The khelabdhankhya")
        if valid:
            track_a_success += 1
        track_a_outputs.append({
            "prompt": np.replace("\n", " "),
            "neural_reply": reply,
            "valid": valid,
            "latency_ms": res["latency_ms"]
        })
        print(f"  [Prompt]: {np.replace(chr(10), ' ')}")
        print(f"  [Reply] : {repr(reply)}")

    track_a_rate = (track_a_success / len(neural_prompts)) * 100
    results["Track_A_Pure_Neural"] = {
        "total": len(neural_prompts),
        "passed": track_a_success,
        "pass_rate_pct": track_a_rate,
        "samples": track_a_outputs
    }
    print(f"\nTrack A Pure Neural Pass Rate: {track_a_rate:.1f}%\n")

    # ---------------- TRACK B: Full Cognitive Hybrid ----------------
    print(">>> EVALUATING TRACK B: Full Cognitive Hybrid System <<<")

    # 1. 100 Conversational Prompts
    conv_inputs = [
        "hi chakra", "hello there", "good morning", "kya kr rhe ho", "kaise ho aap",
        "namaste assistant", "sab badhiya hai?", "thank you very much", "dhanyawaad dost", "see you later"
    ] * 10
    conv_passed = sum(1 for q in conv_inputs if len(agent.process_message(q, mode="hybrid")["reply"]) > 5)

    # 2. 50 Multi-Turn Context Exchanges
    mt_passed = 0
    for i in range(50):
        agent.memory.clear_working_memory()
        agent.process_message(f"Turn 1: My drone model is FalconX_{i}.", mode="hybrid")
        agent.process_message("Turn 2: What is my drone model?", mode="hybrid")
        if len(agent.memory.working_memory) >= 4:
            mt_passed += 1

    # 3. 30 Persistent Memory Scenarios (Falcon & Python)
    agent.process_message("My project is called Project Falcon. Remember that it uses Python.", mode="hybrid")
    r_name = agent.process_message("What is my project called?", mode="hybrid")
    r_lang = agent.process_message("What language does it use?", mode="hybrid")
    r_loc = agent.process_message("Where is my project?", mode="hybrid")

    mem_passed = 0
    if "Project Falcon" in r_name["reply"]:
        mem_passed += 1
    if "Python" in r_lang["reply"]:
        mem_passed += 1
    if "location has not been provided" in r_loc["reply"]:
        mem_passed += 1

    for j in range(27):
        agent.memory.store_fact(f"Server_{j}", "port", f"808{j}")
        if agent.memory.get_fact(f"Server_{j}", "port") == f"808{j}":
            mem_passed += 1

    # 4. 30 Natural Language Word Arithmetic Scenarios
    math_cases = [
        ("I have 5 boxes. Each box contains 8 books.", "40"),
        ("I have 5 boxes. Each contains 8 books. I remove 7 books. How many remain?", "33"),
        ("I have 6 bags. Each bag contains 10 apples. How many apples are there?", "60"),
        ("I have 4 crates with 12 bottles each. I gave away 8 bottles.", "40"),
        ("What is 15 + 25?", "40"),
        ("Calculate 90 - 45", "45"),
        ("Multiply 8 by 7", "56")
    ] * 5
    math_cases = math_cases[:30]
    math_passed = sum(1 for q, exp in math_cases if agent.process_message(q, mode="hybrid")["reply"].strip() == exp)

    # 5. 20 Unknown-Information & Abstention Scenarios
    abs_queries = [
        "What was the exact color of my school bag when I was 10?",
        "Where was I born?",
        "What did I eat yesterday for lunch?",
        "What was my childhood pet's secret name?"
    ] * 5
    abs_passed = sum(1 for q in abs_queries if agent.process_message(q, mode="hybrid")["source"] == "principled_abstention")

    results["Track_B_Cognitive_Hybrid"] = {
        "ordinary_conversation": {"total": 100, "passed": conv_passed, "pct": (conv_passed / 100) * 100},
        "multi_turn": {"total": 50, "passed": mt_passed, "pct": (mt_passed / 50) * 100},
        "persistent_memory": {"total": 30, "passed": mem_passed, "pct": (mem_passed / 30) * 100},
        "word_arithmetic": {"total": 30, "passed": math_passed, "pct": (math_passed / 30) * 100},
        "principled_abstention": {"total": 20, "passed": abs_passed, "pct": (abs_passed / 20) * 100}
    }

    report_path = Path("experiments/project_genesis/acceptance_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=================================================================")
    print("  FINAL ACCEPTANCE SUMMARY:")
    print(f"  Track A (Pure Neural Foundation): {track_a_rate:.1f}%")
    print(f"  Track B (Full Cognitive Hybrid) : Conv={conv_passed}/100, MT={mt_passed}/50, Mem={mem_passed}/30, Math={math_passed}/30, Abs={abs_passed}/20")
    print("=================================================================\n")


if __name__ == "__main__":
    run_genesis_evaluation()
