import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig

RELEASE_CKPT = REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt"
OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_17r/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 30 Multi-turn stress conversations (5-8 turns each)
# Designed according to Phase F requirements:
# - Early fact introduced, referenced later
# - Pronouns and indirect references
# - Follow-ups
# - Multiple entities
# - User corrections
# - Topic changes and returns
# - Missing info and conflicting info

def build_multiturn_conversations():
    convs = []
    
    # Conv 1: Person, role, city, correction, recall
    convs.append({
        "id": "MT_01",
        "title": "Identity, Role, and Location Correction",
        "turns": [
            {"user": "Hello! My name is Tanvi.", "assistant_gold": "Hello Tanvi!"},
            {"user": "I work as a data scientist in Hyderabad.", "assistant_gold": "Noted, data scientist in Hyderabad."},
            {"user": "Actually, I recently moved, so I now live in Pune, not Hyderabad.", "assistant_gold": "Updated: you live in Pune."},
            {"user": "What is my profession?", "eval_type": "fact_retention", "expected": ["data scientist"]},
            {"user": "Which city do I live in now?", "eval_type": "corrected_fact", "expected": ["Pune"], "forbidden": ["Hyderabad"]},
            {"user": "What is the capital of France?", "eval_type": "topic_change", "expected": ["Paris"]},
            {"user": "What was my name again?", "eval_type": "early_fact_return", "expected": ["Tanvi"]},
            {"user": "What is my favorite dessert?", "eval_type": "missing_info", "expected": ["not know", "unspecified", "not mention"]}
        ]
    })
    
    # Conv 2: Multi-entity inventory & follow-up
    convs.append({
        "id": "MT_02",
        "title": "Two Devices and Status",
        "turns": [
            {"user": "We have two servers in rack 4: Alpha and Beta.", "assistant_gold": "Servers Alpha and Beta in rack 4 recorded."},
            {"user": "Alpha is running Debian, while Beta is running Ubuntu.", "assistant_gold": "Alpha has Debian, Beta has Ubuntu."},
            {"user": "Beta experienced a disk failure this morning.", "assistant_gold": "Noted disk failure on Beta."},
            {"user": "Which operating system does Alpha run?", "eval_type": "fact_retention", "expected": ["Debian"]},
            {"user": "Which server had the disk issue?", "eval_type": "reference_resolution", "expected": ["Beta"]},
            {"user": "How many seconds are in a minute?", "eval_type": "topic_change", "expected": ["60"]},
            {"user": "Where are those servers located?", "eval_type": "early_fact_return", "expected": ["rack 4", "rack"]}
        ]
    })
    
    # Conv 3: Pet details and activity
    convs.append({
        "id": "MT_03",
        "title": "Pet details and activity",
        "turns": [
            {"user": "I adopted a golden retriever named Casper.", "assistant_gold": "Casper the golden retriever is noted."},
            {"user": "He is 3 years old.", "assistant_gold": "3 years old."},
            {"user": "He loves to swim in the local lake.", "assistant_gold": "Great exercise."},
            {"user": "What breed of dog is Casper?", "eval_type": "fact_retention", "expected": ["golden retriever"]},
            {"user": "How old is he?", "eval_type": "reference_resolution", "expected": ["3"]},
            {"user": "What do bees produce?", "eval_type": "topic_change", "expected": ["honey"]},
            {"user": "Where does my dog like to swim?", "eval_type": "early_fact_return", "expected": ["lake"]}
        ]
    })
    
    # Conv 4: Project budget and deadline update
    convs.append({
        "id": "MT_04",
        "title": "Project budget and deadline update",
        "turns": [
            {"user": "Our new web project is codenamed Falcon.", "assistant_gold": "Project Falcon recorded."},
            {"user": "The budget was set to $50,000.", "assistant_gold": "Budget $50,000 noted."},
            {"user": "Management just approved an increase to $75,000.", "assistant_gold": "Budget updated to $75,000."},
            {"user": "What is the project codename?", "eval_type": "fact_retention", "expected": ["Falcon"]},
            {"user": "What is the approved budget now?", "eval_type": "corrected_fact", "expected": ["75,000", "75000", "$75,000"]},
            {"user": "What is 5 plus 5?", "eval_type": "topic_change", "expected": ["10", "ten"]},
            {"user": "Who is the lead frontend developer?", "eval_type": "missing_info", "expected": ["not know", "not mention", "unspecified"]}
        ]
    })
    
    # Conv 5: Vehicle and travel plans
    convs.append({
        "id": "MT_05",
        "title": "Vehicle and travel plans",
        "turns": [
            {"user": "I drive a red motorcycle.", "assistant_gold": "Red motorcycle noted."},
            {"user": "I am riding it to Goa next weekend.", "assistant_gold": "Ride to Goa recorded."},
            {"user": "What color is my motorcycle?", "eval_type": "fact_retention", "expected": ["red"]},
            {"user": "Where am I riding next weekend?", "eval_type": "fact_retention", "expected": ["Goa"]},
            {"user": "Do fish walk on land?", "eval_type": "topic_change", "expected": ["No"]},
            {"user": "What vehicle am I taking?", "eval_type": "reference_resolution", "expected": ["motorcycle", "bike"]}
        ]
    })
    
    # Build 25 additional concise conversations for stress testing (total 30)
    for i in range(6, 31):
        name = f"User_{i}"
        fav_tool = f"Tool_{i}"
        city = f"City_{i}"
        convs.append({
            "id": f"MT_{i:02d}",
            "title": f"Stress Conversation {i}",
            "turns": [
                {"user": f"Greetings! Call me {name}.", "assistant_gold": f"Greetings {name}!"},
                {"user": f"My primary workstation is located in {city}.", "assistant_gold": f"Workstation in {city} recorded."},
                {"user": f"I rely heavily on {fav_tool} for daily debugging.", "assistant_gold": f"{fav_tool} noted."},
                {"user": "What is my chosen name?", "eval_type": "fact_retention", "expected": [name]},
                {"user": "Which city is my workstation in?", "eval_type": "fact_retention", "expected": [city]},
                {"user": "What is the opposite of hot?", "eval_type": "topic_change", "expected": ["cold"]},
                {"user": "Which debugging tool did I mention?", "eval_type": "early_fact_return", "expected": [fav_tool]},
                {"user": "What is my mother's maiden name?", "eval_type": "missing_info", "expected": ["not know", "unspecified", "not mention"]}
            ]
        })
    return convs

def evaluate_multiturn():
    runner = DirectNeuralRunner(RELEASE_CKPT)
    tok = runner.tokenizer
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)
    
    convs = build_multiturn_conversations()
    results = []
    
    metric_counts = {
        "fact_retention": {"total": 0, "correct": 0},
        "corrected_fact": {"total": 0, "correct": 0},
        "reference_resolution": {"total": 0, "correct": 0},
        "topic_change": {"total": 0, "correct": 0},
        "early_fact_return": {"total": 0, "correct": 0},
        "missing_info": {"total": 0, "correct": 0}
    }
    
    total_eval_turns = 0
    
    for conv in convs:
        cid = conv["id"]
        history = []
        turn_logs = []
        
        for turn_idx, t in enumerate(conv["turns"]):
            user_msg = t["user"]
            
            # Build prompt from accumulated history
            prompt_parts = []
            for h in history:
                prompt_parts.append(f"User: {h['user']}\nAssistant: {h['assistant']}")
            prompt_parts.append(f"User: {user_msg}\nAssistant:")
            prompt_str = "\n".join(prompt_parts)
            
            # Generate
            toks = tok.encode(prompt_str, add_bos=True, add_eos=False)
            curr = list(toks)
            gen = []
            with torch.no_grad():
                for step in range(40):
                    t_in = torch.tensor([curr], dtype=torch.long)
                    logits = runner.model(t_in)[0, -1, :]
                    next_tok = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                    gen.append(next_tok)
                    curr.append(next_tok)
                    if next_tok == 1:
                        break
            resp = tok.decode(gen, errors="replace").strip()
            
            is_eval = "eval_type" in t
            turn_record = {
                "turn_index": turn_idx + 1,
                "user": user_msg,
                "model_response": resp,
                "is_evaluated": is_eval
            }
            
            if is_eval:
                etype = t["eval_type"]
                expected = t["expected"]
                forbidden = t.get("forbidden", [])
                
                correct = any(exp.lower() in resp.lower() for exp in expected)
                if any(forb.lower() in resp.lower() for forb in forbidden):
                    correct = False
                    
                metric_counts[etype]["total"] += 1
                if correct:
                    metric_counts[etype]["correct"] += 1
                total_eval_turns += 1
                
                turn_record["eval_type"] = etype
                turn_record["expected"] = expected
                turn_record["correct"] = correct
                
            turn_logs.append(turn_record)
            
            # Update history using gold if initial turn, or model response
            assistant_record = t.get("assistant_gold", resp)
            history.append({"user": user_msg, "assistant": assistant_record})
            
        results.append({
            "conversation_id": cid,
            "title": conv["title"],
            "turns": turn_logs
        })
        
    summary = {
        "total_conversations": len(convs),
        "total_evaluated_turns": total_eval_turns,
        "metrics_by_category": metric_counts
    }
    
    with open(OUT_DIR / "multiturn_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "conversations": results}, f, indent=2)
        
    print("\n=== MULTI-TURN CONVERSATION STRESS TEST RESULTS ===")
    for k, v in metric_counts.items():
        tot = v["total"]
        cor = v["correct"]
        pct = (cor / tot * 100) if tot > 0 else 0.0
        print(f"  {k:22s}: {cor:2d}/{tot:2d} ({pct:5.1f}%)")

if __name__ == "__main__":
    evaluate_multiturn()
