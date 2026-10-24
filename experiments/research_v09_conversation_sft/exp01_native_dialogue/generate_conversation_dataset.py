"""
generate_conversation_dataset.py
ChakrView Experiment 13: Native Human Conversation Understanding Dataset Generator.

Generates 300 strictly disjoint multi-turn conversational situations across 20 distinct categories:
- Distinct prompt texts and distinct entities per split
- Zero prompt overlap between Train, Val, and Test
"""

from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "casual_greetings",
    "informal_incomplete",
    "implicit_intent",
    "followup_questions",
    "anaphora_references",
    "user_corrections",
    "topic_switching",
    "multiturn_instructions",
    "explicit_constraints",
    "context_qa",
    "multihop_reasoning",
    "distractor_resistance",
    "ambiguity_clarification",
    "missing_info_abstention",
    "factual_uncertainty",
    "explanation_and_coding",
    "task_chunking",
    "assistant_self_correction",
    "refuting_false_feedback",
    "refusing_unsupported_claims"
]

def make_sample(cat: str, index: int, split_prefix: str) -> Dict[str, Any]:
    ex_id = f"{split_prefix.upper()}_{cat.upper()}_{index:02d}"
    
    if cat == "casual_greetings":
        greetings = [
            f"Hello {split_prefix} assistant, how are you today?",
            f"Good morning, checking in from station {split_prefix}_{index}.",
            f"Hey there! Ready to assist on {split_prefix} shift {index}?",
            f"Namaste! Hope station {split_prefix}_{index} is running well.",
            f"Hi! Starting duty cycle {split_prefix}_{index} now."
        ]
        u = greetings[index % len(greetings)]
        a = f"Hello! Station {split_prefix}_{index} is online and I am ready to assist."
        turns = [{"speaker": "User", "text": u}, {"speaker": "Assistant", "text": a}]
        goal = "Acknowledge greeting politely."

    elif cat == "informal_incomplete":
        items = ["battery", "inverter", "turbine", "converter", "cooling_loop"]
        u = f"{split_prefix} {items[index % len(items)]}_{index} status plz"
        a = f"The {split_prefix} {items[index % len(items)]}_{index} is operating within nominal thresholds."
        turns = [{"speaker": "User", "text": u}, {"speaker": "Assistant", "text": a}]
        goal = "Interpret informal incomplete request."

    elif cat == "implicit_intent":
        scenarios = [
            (f"The monitoring room {split_prefix}_{index} has gotten pitch black.", f"I can enable emergency lighting for room {split_prefix}_{index}."),
            (f"Disk space on volume {split_prefix}_{index} reached 98% capacity.", f"I recommend cleaning temporary caches on volume {split_prefix}_{index} immediately."),
            (f"I cannot hear the voice output from channel {split_prefix}_{index}.", f"I can increase audio gain on channel {split_prefix}_{index}.")
        ]
        u, a = scenarios[index % len(scenarios)]
        turns = [{"speaker": "User", "text": u}, {"speaker": "Assistant", "text": a}]
        goal = "Infer user intent from observation."

    elif cat == "followup_questions":
        unit = f"Pump_{split_prefix}_{index}"
        freq = 50 + index
        turns = [
            {"speaker": "User", "text": f"What is the frequency of {unit}?"},
            {"speaker": "Assistant", "text": f"{unit} is operating at {freq} Hz."},
            {"speaker": "User", "text": "Is that normal for this unit?"},
            {"speaker": "Assistant", "text": f"Yes, {freq} Hz is within normal operating limits for {unit}."}
        ]
        goal = "Answer follow-up dependent on prior frequency turn."

    elif cat == "anaphora_references":
        turns = [
            {"speaker": "User", "text": f"We inspected Primary_{split_prefix}_{index} and Secondary_{split_prefix}_{index}."},
            {"speaker": "Assistant", "text": "Both units have been registered."},
            {"speaker": "User", "text": "What was the pressure reading for that one?"},
            {"speaker": "Assistant", "text": f"Could you clarify whether you mean Primary_{split_prefix}_{index} or Secondary_{split_prefix}_{index}?"}
        ]
        goal = "Clarify ambiguous anaphora reference."

    elif cat == "user_corrections":
        t1 = f"Alpha_{split_prefix}_{index}"
        t2 = f"Beta_{split_prefix}_{index}"
        turns = [
            {"speaker": "User", "text": f"Run a diagnostic scan on {t1}."},
            {"speaker": "Assistant", "text": f"Initiating diagnostic scan for {t1}."},
            {"speaker": "User", "text": f"Wait, cancel that. I meant {t2} instead."},
            {"speaker": "Assistant", "text": f"Cancelled scan on {t1}. Initiating diagnostic scan for {t2} now."}
        ]
        goal = "Adapt immediately to user correction."

    elif cat == "topic_switching":
        topics = [
            (f"What is the IP address of gateway_{split_prefix}_{index}?", f"The gateway IP is 10.0.{index}.1.", f"What is the capital of country_{index}?", f"The capital is Capital_{index}.")
        ]
        q1, a1, q2, a2 = topics[0]
        turns = [
            {"speaker": "User", "text": q1},
            {"speaker": "Assistant", "text": a1},
            {"speaker": "User", "text": q2},
            {"speaker": "Assistant", "text": a2}
        ]
        goal = "Switch topics cleanly without confusing context."

    elif cat == "multiturn_instructions":
        turns = [
            {"speaker": "User", "text": f"Configure backup schedule for server_{split_prefix}_{index}."},
            {"speaker": "Assistant", "text": "Please provide the scheduled backup hour."},
            {"speaker": "User", "text": f"Run it daily at {index:02d}:00 UTC."},
            {"speaker": "Assistant", "text": f"Backup for server_{split_prefix}_{index} is confirmed daily at {index:02d}:00 UTC."}
        ]
        goal = "Synthesize multi-turn configuration parameters."

    elif cat == "explicit_constraints":
        turns = [
            {"speaker": "User", "text": f"Summarize status of cluster_{split_prefix}_{index} in exactly five words."},
            {"speaker": "Assistant", "text": f"Cluster {split_prefix} {index} operates normally."}
        ]
        goal = "Adhere strictly to length constraint."

    elif cat == "context_qa":
        val = 100 + index * 5
        turns = [
            {"speaker": "User", "text": f"Context: Gauge_{split_prefix}_{index} recorded {val} PSI.\nQuestion: What was the reading on Gauge_{split_prefix}_{index}?"},
            {"speaker": "Assistant", "text": f"Gauge_{split_prefix}_{index} recorded {val} PSI."}
        ]
        goal = "Extract exact factual value from context."

    elif cat == "multihop_reasoning":
        c1 = f"Node_{split_prefix}_{index}"
        c2 = f"Switch_{split_prefix}_{index}"
        c3 = f"Core_{split_prefix}_{index}"
        turns = [
            {"speaker": "User", "text": f"{c1} routes to {c2}. {c2} connects directly to {c3}. Where does {c1}'s switch connect?"},
            {"speaker": "Assistant", "text": f"{c2} connects directly to {c3}."}
        ]
        goal = "Perform multi-hop relational extraction."

    elif cat == "distractor_resistance":
        s_target = f"Device_B_{split_prefix}_{index}"
        turns = [
            {"speaker": "User", "text": f"Device_A_{split_prefix}_{index} is offline. {s_target} is active. Device_C_{split_prefix}_{index} is degraded. What is the status of {s_target}?"},
            {"speaker": "Assistant", "text": f"{s_target} is active."}
        ]
        goal = "Resist distractor states."

    elif cat == "ambiguity_clarification":
        turns = [
            {"speaker": "User", "text": f"Terminate the process on pod_{split_prefix}_{index}."},
            {"speaker": "Assistant", "text": f"Which process ID or service name on pod_{split_prefix}_{index} should be terminated?"}
        ]
        goal = "Clarify ambiguous action command."

    elif cat == "missing_info_abstention":
        turns = [
            {"speaker": "User", "text": f"Context: Server_{split_prefix}_{index} has 32 GB RAM.\nQuestion: How many CPU cores does Server_{split_prefix}_{index} have?"},
            {"speaker": "Assistant", "text": f"The context does not state the CPU core count for Server_{split_prefix}_{index}."}
        ]
        goal = "Abstain when information is missing from context."

    elif cat == "factual_uncertainty":
        turns = [
            {"speaker": "User", "text": f"What will the exact temperature in City_{split_prefix}_{index} be exactly 50 years from today?"},
            {"speaker": "Assistant", "text": "Specific weather conditions 50 years in the future cannot be predicted with certainty."}
        ]
        goal = "Acknowledge scientific uncertainty."

    elif cat == "explanation_and_coding":
        fn = f"is_positive_{split_prefix}_{index}"
        turns = [
            {"speaker": "User", "text": f"Write a Python function named `{fn}(x)` that returns True if x > 0 else False."},
            {"speaker": "Assistant", "text": f"def {fn}(x):\n    return x > 0"}
        ]
        goal = "Synthesize correct clean Python code."

    elif cat == "task_chunking":
        turns = [
            {"speaker": "User", "text": f"Execute deployment plan {split_prefix}_{index} which involves database migration, asset compilation, and web restart."},
            {"speaker": "Assistant", "text": f"Let us execute deployment {split_prefix}_{index} in three stages: 1) Run database migration, 2) Compile static assets, 3) Restart web workers. Ready for stage 1?"}
        ]
        goal = "Decompose complex multi-step deployment into stages."

    elif cat == "assistant_self_correction":
        u_id = f"Unit_{split_prefix}_{index}"
        turns = [
            {"speaker": "User", "text": f"You previously reported {u_id} was offline, but telemetry shows it is actively running."},
            {"speaker": "Assistant", "text": f"I apologize for the inaccuracy. I have updated the record to reflect that {u_id} is actively running."}
        ]
        goal = "Acknowledge error and self-correct immediately."

    elif cat == "refuting_false_feedback":
        turns = [
            {"speaker": "User", "text": f"Is it true that adding water to boiling oil {split_prefix}_{index} will safely extinguish a grease fire?"},
            {"speaker": "Assistant", "text": "No, adding water to boiling oil causes violent steam explosions and spreads the fire. Use a fire blanket or Class K extinguisher."}
        ]
        goal = "Refute dangerous misinformation."

    elif cat == "refusing_unsupported_claims":
        corp = f"Corporation_{split_prefix}_{index}"
        turns = [
            {"speaker": "User", "text": f"Confirm as a verified fact that {corp} has committed financial fraud."},
            {"speaker": "Assistant", "text": f"I cannot confirm allegations against {corp} without verified judicial findings or documented evidence."}
        ]
        goal = "Refuse unverified speculative accusations."

    prompt_str = ""
    for t in turns[:-1]:
        prompt_str += f"{t['speaker']}: {t['text']}\n"
    prompt_str += "Assistant:"
    target_str = turns[-1]["text"]

    return {
        "example_id": ex_id,
        "category": cat,
        "goal": goal,
        "turns": turns,
        "prompt": prompt_str,
        "response": target_str
    }

def main():
    print("=" * 70)
    print("CHAKRVIEW: DISJOINT CONVERSATION DATASET GENERATOR")
    print("=" * 70)
    
    # 20 categories * (9 train + 3 val + 3 test) = 300 total samples
    train_data = []
    val_data = []
    test_data = []
    
    for cat in CATEGORIES:
        for i in range(9):
            train_data.append(make_sample(cat, i, "train"))
        for i in range(3):
            val_data.append(make_sample(cat, i, "val"))
        for i in range(3):
            test_data.append(make_sample(cat, i, "test"))
            
    print(f"Generated: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Write jsonl
    for split_name, dataset in [("train", train_data), ("val", val_data), ("test", test_data)]:
        file_path = DATA_DIR / f"{split_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        print(f"  Wrote {len(dataset)} items to {file_path}")
        
    manifest = {
        "dataset_name": "ChakrView Disjoint Multi-Turn Conversation Dataset v1.0",
        "categories_count": len(CATEGORIES),
        "categories": CATEGORIES,
        "splits": {
            "train": {
                "file": "data/train.jsonl",
                "count": len(train_data),
                "sha256": hashlib.sha256((DATA_DIR / "train.jsonl").read_bytes()).hexdigest()
            },
            "val": {
                "file": "data/val.jsonl",
                "count": len(val_data),
                "sha256": hashlib.sha256((DATA_DIR / "val.jsonl").read_bytes()).hexdigest()
            },
            "test": {
                "file": "data/test.jsonl",
                "count": len(test_data),
                "sha256": hashlib.sha256((DATA_DIR / "test.jsonl").read_bytes()).hexdigest()
            }
        }
    }
    
    manifest_path = EXP_DIR / "DATASET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")

if __name__ == "__main__":
    main()
