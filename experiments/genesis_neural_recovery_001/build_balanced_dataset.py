"""
Balanced & Audited Conversational SFT Dataset Generator for Genesis Recovery.
Assembles a clean, balanced conversational curriculum:
1. Diverse English dialogue & explanations
2. Hindi & Hinglish natural conversations
3. Technical concepts & Python explanations
4. Grounded factual Q&A
5. Standardized prompt formatting with balanced response distributions (no single template > 2% of dataset)
"""

from __future__ import annotations
import json
from pathlib import Path
import random
from collections import Counter


def build_balanced_recovery_sft_dataset(
    output_path: str = "experiments/genesis_neural_recovery_001/recovery_sft_dataset.jsonl",
    max_duplicate_target: int = 4
) -> list[dict]:
    examples = []

    # 1. Load ucip09s balanced dialogues
    p_09 = Path("experiments/ucip09s/datasets/ucip09s_balanced_train.jsonl")
    if p_09.exists():
        with open(p_09, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                p = d.get("prompt", "").strip()
                t = d.get("target", "").strip()
                if p and t and len(t) > 3:
                    examples.append({"prompt": p, "response": t, "source": "ucip09s"})

    # 2. Load ultimate_brain dataset
    p_ub = Path("experiments/ultimate_brain/datasets/train.jsonl")
    if p_ub.exists():
        with open(p_ub, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                p = d.get("prompt", "").strip()
                t = d.get("response", "").strip()
                if p and t and len(t) > 3:
                    examples.append({"prompt": p, "response": t, "source": "ultimate_brain"})

    # 3. Add diverse open-ended conversational exchanges
    synthetic_conversations = [
        ("User: Hello, how are you?\nAssistant:", "Hello! I am doing well and ready to assist you today."),
        ("User: Namaste, tum kaise ho?\nAssistant:", "Namaste! Main badhiya hoon. Aap bataiye, aaj kya kaam karna hai?"),
        ("User: I am building a small AI model.\nAssistant:", "That is an exciting project! Are you focusing on the neural architecture, tokenizer, or the training dataset?"),
        ("User: Explain how a computer stores information.\nAssistant:", "Computers store information as binary digits—bits of 0 and 1—using transistors in RAM and magnetic or solid-state cells in storage drives."),
        ("User: The capital of Australia is Canberra.\nAssistant:", "That is correct. Canberra is the federal capital of Australia."),
        ("User: What is photosynthesis?\nAssistant:", "Photosynthesis is the biological process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."),
        ("User: Write a python function to check if a number is even.\nAssistant:", "def is_even(n: int) -> bool:\n    return n % 2 == 0"),
        ("User: What is 5 + 7?\nAssistant:", "5 + 7 = 12."),
        ("User: Who created you?\nAssistant:", "I am ChakrView, an independent indigenous AI system engineered from scratch."),
        ("User: Good morning!\nAssistant:", "Good morning! I hope you have a productive and inspiring day ahead."),
        ("User: Tell me a fact about space.\nAssistant:", "One day on Venus is longer than one year on Venus because of its extremely slow axial rotation."),
        ("User: How does RAM work?\nAssistant:", "RAM provides fast, temporary read and write storage directly accessible to the processor while applications are actively running.")
    ]
    for p, r in synthetic_conversations:
        examples.append({"prompt": p, "response": r, "source": "curated_conversations"})

    # Filter out single-word answers and excessive templates (like "Green.", "Acknowledged.", "You will have X remaining.")
    target_counts = Counter()
    filtered_examples = []
    
    # Shuffle for fair sampling
    random.seed(42)
    random.shuffle(examples)

    for ex in examples:
        resp = ex["response"].strip()
        # Cap repetitive targets
        if target_counts[resp] >= max_duplicate_target:
            continue
        
        # Avoid short repetitive token traps
        if resp in ["Green.", "Acknowledged.", "Yes.", "Affirmative.", "3"]:
            continue
        if resp.startswith("You will have") and "remaining" in resp and target_counts["you_will_have"] >= 5:
            continue
        if resp.startswith("You will have"):
            target_counts["you_will_have"] += 1

        target_counts[resp] += 1
        filtered_examples.append(ex)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in filtered_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated balanced SFT dataset with {len(filtered_examples)} examples at {output_path}")
    print(f"Unique response count: {len(target_counts)}")
    return filtered_examples


if __name__ == "__main__":
    build_balanced_recovery_sft_dataset()
