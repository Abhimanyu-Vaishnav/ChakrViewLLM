"""
Multi-turn Context Handling and Conversational Retention Evaluation for Mission CV-GENESIS-NR-005.

Evaluates how well Genesis SFT Best handles preceding dialogue history without external memory:
Tests:
1. Entity recall across conversational turns (e.g. User shares name/project, asks about it 2 turns later)
2. Correction following (e.g. User corrects an entity, model accepts correction)
3. Topic continuity (e.g. follow-up questions without repeating the subject)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


MULTITURN_SCENARIOS = [
    {
        "id": "multiturn_entity_recall_01",
        "category": "entity_recall",
        "language": "en",
        "turns": [
            ("User: My name is Alice and I am designing a compiler in Rust.\nAssistant:", "Hello Alice! Building a compiler in Rust is a fantastic project. What part are you currently working on?"),
            ("User: What programming language am I using for my compiler?\nAssistant:", "Rust")
        ],
        "target_keyword": "Rust"
    },
    {
        "id": "multiturn_entity_recall_02",
        "category": "entity_recall",
        "language": "hinglish",
        "turns": [
            ("User: Bhai humara naya office Bangalore me open hone wala hai.\nAssistant:", "Bohot badhiya! Bangalore tech hubs ke liye famous hai. Office kab start hoga?"),
            ("User: Humara office kis shahar me open hone wala hai?\nAssistant:", "Bangalore")
        ],
        "target_keyword": "Bangalore"
    },
    {
        "id": "multiturn_entity_recall_03",
        "category": "entity_recall",
        "language": "hi",
        "turns": [
            ("User: मेरे पास तीन सफेद बिल्लियाँ और दो काले कुत्ते हैं।\nAssistant:", "यह बहुत अच्छा है! आपके पास कुल पाँच प्यारे पालतू जानवर हैं।"),
            ("User: मेरे पास कुल कितने पालतू जानवर हैं?\nAssistant:", "पाँच")
        ],
        "target_keyword": "पाँच"
    },
    {
        "id": "multiturn_correction_01",
        "category": "correction",
        "language": "en",
        "turns": [
            ("User: The project deadline is Friday.\nAssistant:", "Understood. The project deadline is noted for Friday."),
            ("User: Actually, the client rescheduled it to next Tuesday. When is the deadline now?\nAssistant:", "Tuesday")
        ],
        "target_keyword": "Tuesday"
    },
    {
        "id": "multiturn_topic_continuity_01",
        "category": "topic_continuity",
        "language": "en",
        "turns": [
            ("User: What is an index in a database?\nAssistant:", "An index is a data structure, such as a B-Tree, that improves the speed of data retrieval operations on a database table."),
            ("User: What is its primary disadvantage?\nAssistant:", "It consumes additional disk storage and slows down write operations such as INSERT, UPDATE, and DELETE.")
        ],
        "target_keyword": "write"
    }
]


def evaluate_multiturn(
    checkpoint_path: str = "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt",
    tokenizer_dir: str = "data/experiments/vocab_4096",
    output_path: str = "experiments/genesis_conversational_foundation_005/evaluations/multiturn_evaluation_results.json",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
):
    print("=== MULTI-TURN CONVERSATIONAL CONTEXT EVALUATION ===")
    tok, _ = load_tokenizer_artifacts(tokenizer_dir)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ChakrGenesis(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    
    for sc in MULTITURN_SCENARIOS:
        context_prompt = ""
        # Build multi-turn context
        for turn_idx, (user_turn, assistant_ref) in enumerate(sc["turns"]):
            if turn_idx < len(sc["turns"]) - 1:
                context_prompt += f"{user_turn} {assistant_ref}\n"
            else:
                context_prompt += f"{user_turn}"

        p_ids = tok.encode(context_prompt)
        idx = torch.tensor([p_ids], dtype=torch.long, device=device)
        generated = []
        with torch.no_grad():
            for _ in range(35):
                idx_cond = idx[:, -512:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / 0.7
                v, _ = torch.topk(logits, min(20, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                if idx_next.item() == 1:
                    break
                generated.append(idx_next.item())
                idx = torch.cat((idx, idx_next), dim=1)

        try:
            resp = tok.decode(generated, errors="replace").strip()
        except:
            resp = tok.decode_bytes(generated).decode("utf-8", errors="replace").strip()

        keyword_recalled = sc["target_keyword"].lower() in resp.lower()
        print(f"Scenario: {sc['id']} ({sc['language']})")
        print(f"  Context Input: {repr(context_prompt)}")
        print(f"  Generated Response: {repr(resp)}")
        print(f"  Target Keyword '{sc['target_keyword']}': {'FOUND' if keyword_recalled else 'NOT FOUND'}\n")

        results.append({
            "id": sc["id"],
            "category": sc["category"],
            "language": sc["language"],
            "prompt": context_prompt,
            "response": resp,
            "target_keyword": sc["target_keyword"],
            "keyword_recalled": keyword_recalled
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    evaluate_multiturn()
