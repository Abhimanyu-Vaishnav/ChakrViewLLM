"""
NR-015 Phase 5: Curriculum Learning Controlled Experiments
Compares:
Condition A: Randomized Mixed Training (Uniform shuffling of all tasks)
Condition B: Easy-to-Hard Curriculum (Simple arithmetic -> multi-variable binding -> multi-step reasoning)
Condition C: Concept-Focused Curriculum (Blocked by task family)
Condition D: Balanced Multi-Domain (Interleaved conversation + coding + reasoning)

Evaluates on:
1. Convergence loss
2. Generalization accuracy on held-out tasks
"""

import sys
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M

def generate_curriculum_datasets():
    random.seed(42)
    # Stage 1: Easy (simple single-digit arithmetic)
    easy = []
    for _ in range(300):
        a, b = random.randint(1, 9), random.randint(1, 9)
        ans = a + b
        easy.append({"text": f"<user>\nCalculate {a} + {b}.\n<assistant>\n{ans}", "level": 1, "ans": str(ans)})
    
    # Stage 2: Medium (2-digit arithmetic and simple variable binding)
    medium = []
    for _ in range(300):
        a, b = random.randint(10, 50), random.randint(10, 50)
        ans = a + b
        medium.append({"text": f"<user>\nLet x = {a} and y = {b}. What is x + y?\n<assistant>\n{ans}", "level": 2, "ans": str(ans)})

    # Stage 3: Hard (multi-step or multiplication)
    hard = []
    for _ in range(300):
        a, b, c = random.randint(2, 9), random.randint(2, 9), random.randint(2, 9)
        ans = a * b + c
        hard.append({"text": f"<user>\nCalculate {a} * {b} + {c}.\n<assistant>\n{ans}", "level": 3, "ans": str(ans)})

    # Held-out Test Set (Unseen ranges and operations)
    test = []
    for _ in range(100):
        a, b = random.randint(55, 95), random.randint(55, 95)
        ans = a + b
        test.append({"prompt": f"<user>\nCalculate {a} + {b}.\n<assistant>\n", "ans": str(ans)})

    out_dir = Path("data/curriculum_015")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "easy.json", 'w') as f: json.dump(easy, f)
    with open(out_dir / "medium.json", 'w') as f: json.dump(medium, f)
    with open(out_dir / "hard.json", 'w') as f: json.dump(hard, f)
    with open(out_dir / "test.json", 'w') as f: json.dump(test, f)
    print("Curriculum datasets generated successfully.")

def run_curriculum_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Curriculum Experiment on: {device}")
    generate_curriculum_datasets()

    with open("data/curriculum_015/easy.json") as f: easy = json.load(f)
    with open("data/curriculum_015/medium.json") as f: medium = json.load(f)
    with open("data/curriculum_015/hard.json") as f: hard = json.load(f)
    with open("data/curriculum_015/test.json") as f: test_set = json.load(f)

    # Condition A: Randomized Mixed
    all_mixed = easy + medium + hard
    random.seed(123)
    random.shuffle(all_mixed)

    # Condition B: Easy-to-Hard
    easy_to_hard = easy + medium + hard

    conditions = {
        "Randomized_Mixed": all_mixed,
        "Easy_to_Hard": easy_to_hard
    }

    results = {}

    for name, data in conditions.items():
        print(f"\n--- Testing Condition: {name} ---")
        engine = ChakrViewInferenceEngine17M(
            checkpoint_path='experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
            device=device
        )
        model = engine.model
        tokenizer = engine.tokenizer
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Train on sequences
        model.train()
        total_loss = 0.0
        steps = 0
        for item in data[:300]: # 300 steps
            prompt, ans = item['text'].split("<assistant>\n")
            p_ids = tokenizer.encode(prompt + "<assistant>\n")
            a_ids = tokenizer.encode(ans) + [getattr(tokenizer, 'eos_id', 1) or 1]
            input_ids = torch.tensor([p_ids + a_ids], dtype=torch.long, device=device)
            labels = torch.tensor([[-100]*len(p_ids) + a_ids], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            s_logits = logits[..., :-1, :].contiguous().view(-1, 4096)
            s_labels = labels[..., 1:].contiguous().view(-1)
            loss = loss_fn(s_logits, s_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        avg_train_loss = total_loss / steps

        # Evaluate on Held-Out Test Set
        model.eval()
        correct = 0
        for item in test_set:
            p_ids = torch.tensor(tokenizer.encode(item['prompt']), dtype=torch.long, device=device).unsqueeze(0)
            curr = p_ids
            with torch.no_grad():
                for _ in range(6):
                    out, _ = model(curr)
                    nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr = torch.cat([curr, nxt], dim=1)
                    if nxt.item() == 1: break
            gen_text = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()
            if item['ans'] in gen_text:
                correct += 1

        test_acc = (correct / len(test_set)) * 100
        print(f"Condition: {name} | Final Train Loss: {avg_train_loss:.4f} | Held-out Test Acc: {correct}/{len(test_set)} ({test_acc:.1f}%)")
        results[name] = {"train_loss": avg_train_loss, "test_acc": test_acc, "correct": correct, "total": len(test_set)}

    with open("data/curriculum_015/curriculum_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\nCurriculum experiment completed and written to data/curriculum_015/curriculum_results.json")

if __name__ == '__main__':
    run_curriculum_experiment()
