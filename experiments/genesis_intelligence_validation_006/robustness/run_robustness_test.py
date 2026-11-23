"""
Robustness and Perturbation Evaluation Harness for Mission CV-GENESIS-NR-006.

Tests:
1. Unusual wording & syntactic inversion
2. Misspellings and typos
3. Extremely short prompts (1-2 words)
4. Long multi-sentence prompts
5. Rephrased standard questions
6. Unfamiliar / out-of-domain topics
7. Questions with insufficient / missing context
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

ROBUSTNESS_PROMPTS = [
    # 1. Unusual wording
    ("unusual_01", "unusual_wording", "en", "User: Functioning mechanism of central processing computation units, how does it occur?\nAssistant:"),
    ("unusual_02", "unusual_wording", "hinglish", "User: Computer ka jo dimaag kehlata hai, wo actions kaise execute karta hai bhai?\nAssistant:"),
    ("unusual_03", "unusual_wording", "hi", "User: संगणक की अंतःस्मृति किस विधि से क्रियान्वित होती है?\nAssistant:"),

    # 2. Misspellings & Typos
    ("typo_01", "typo", "en", "User: Whta is the captal of Frnace?\nAssistant:"),
    ("typo_02", "typo", "hinglish", "User: Pythn me fuktino kse likhte hain?\nAssistant:"),
    ("typo_03", "typo", "hi", "User: कम्प्यूटर मे रैम (RAM) का क्या काय्र है?\nAssistant:"),

    # 3. Short prompts
    ("short_01", "short_prompt", "en", "User: Hello\nAssistant:"),
    ("short_02", "short_prompt", "hinglish", "User: Suno\nAssistant:"),
    ("short_03", "short_prompt", "hi", "User: नमस्ते\nAssistant:"),
    ("short_04", "short_prompt", "en", "User: RAM\nAssistant:"),

    # 4. Long multi-sentence prompts
    ("long_01", "long_prompt", "en", "User: I am designing an autonomous embedded device that operates on low battery power in rural agricultural farms. The microcontroller needs to read soil moisture sensor data every ten minutes and upload telemetry packets over a low-bandwidth cellular modem while sleeping in between to preserve power. What power management strategy would be most suitable?\nAssistant:"),
    ("long_02", "long_prompt", "hinglish", "User: Humari company ek microservice architecture develop kar rahi hai jisme 15 alag alag services hain jo RabbitMQ ke through message pass karti hain aur MongoDB me data store karti hain, lekin peak load ke time database par high connections ki wajah se timeouts aa rahe hain. Is situation ko resolve karne ke liye connection pooling aur caching kaise help kar sakti hain?\nAssistant:"),

    # 5. Rephrased standard questions
    ("rephrase_01", "rephrased", "en", "User: Provide an explanation regarding why electronic computers mandate RAM.\nAssistant:"),
    ("rephrase_02", "rephrased", "hi", "User: भारत देश की मुख्य प्रशासनिक राजधानी का क्या नाम है?\nAssistant:"),
    ("rephrase_03", "rephrased", "hinglish", "User: List aur tuple me jo main antar hota hai, wo detail me clarify karo.\nAssistant:"),

    # 6. Unfamiliar / Out-of-domain topics
    ("unfamiliar_01", "unfamiliar", "en", "User: What are the primary metabolic enzymes active in the hydrothermal vents of Enceladus?\nAssistant:"),
    ("unfamiliar_02", "unfamiliar", "hinglish", "User: 4D spacetime manifold me wormhole geometry ka metric tensor equation batao.\nAssistant:"),

    # 7. Insufficient information / Incomplete questions
    ("insufficient_01", "insufficient_info", "en", "User: How much does it weigh?\nAssistant:"),
    ("insufficient_02", "insufficient_info", "hinglish", "User: Uski total price kitni hogi final bill me?\nAssistant:"),
    ("insufficient_03", "insufficient_info", "hi", "User: वह वहाँ कब पहुँचेगा?\nAssistant:")
]


def run_robustness_test(
    checkpoint_path: str = "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt",
    tokenizer_dir: str = "data/experiments/vocab_4096",
    output_path: str = "experiments/genesis_intelligence_validation_006/robustness/robustness_results.json",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
):
    print("=== RUNNING ROBUSTNESS & PERTURBATION TESTS ===")
    tok, _ = load_tokenizer_artifacts(tokenizer_dir)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ChakrGenesis(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    eos_id = 1

    for item in ROBUSTNESS_PROMPTS:
        p_id, p_type, lang, prompt_text = item
        p_ids = tok.encode(prompt_text)
        idx = torch.tensor([p_ids], dtype=torch.long, device=device)
        generated = []
        with torch.no_grad():
            for _ in range(40):
                idx_cond = idx[:, -512:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / 0.7
                v, _ = torch.topk(logits, min(20, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                if idx_next.item() == eos_id:
                    break
                generated.append(idx_next.item())
                idx = torch.cat((idx, idx_next), dim=1)

        try:
            resp = tok.decode(generated, errors="replace").strip()
        except:
            resp = tok.decode_bytes(generated).decode("utf-8", errors="replace").strip()

        # Behavior analysis
        if p_type == "insufficient_info":
            if any(k in resp.lower() for k in ["what", "which", "clarify", "specify", "किस", "kya", "pata nahi", "provide"]):
                behavior = "Appropriate clarification/uncertainty"
            else:
                behavior = "Hallucinatory assertion"
        elif p_type == "unfamiliar":
            if any(k in resp.lower() for k in ["unknown", "hypothetical", "speculative", "nahi", "not determined", "cannot"]):
                behavior = "Appropriate uncertainty"
            else:
                behavior = "Confident extrapolation/hallucination"
        else:
            behavior = "Coherent response" if len(resp.split()) >= 3 else "Degraded/short"

        results.append({
            "id": p_id,
            "type": p_type,
            "language": lang,
            "prompt": prompt_text,
            "response": resp,
            "behavior": behavior
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Robustness results saved to {output_path}")


if __name__ == "__main__":
    run_robustness_test()
