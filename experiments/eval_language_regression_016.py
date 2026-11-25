"""
NR-016 Phase 7: Language and Regression Testing Suite
Evaluates English, Hindi (Devanagari), and Hinglish dialogue independently.
Measures:
1. Strict factual / task accuracy on verifiable questions
2. Conversational coherence vs semantic correctness
3. Repetition penalty behavior and runaway suppression
4. UTF-8 byte integrity across Devanagari multi-byte sequences
5. Cognitive runtime regression check (Hybrid controller)
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from chakrview.brain.cognitive_architecture import CognitiveController

def run_language_and_regression_suite():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== NR-016 LANGUAGE & REGRESSION TESTING on {device} ===")

    engine = ChakrViewInferenceEngine17M(
        checkpoint_path='experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
        device=device
    )
    controller = CognitiveController()

    prompts = [
        # English
        {"id": "EN-01", "lang": "English", "prompt": "What is the capital of France?", "expected_fact": "Paris"},
        {"id": "EN-02", "lang": "English", "prompt": "If a train travels at 60 mph for 2 hours, how far does it go?", "expected_fact": "120"},
        {"id": "EN-03", "lang": "English", "prompt": "Explain what a neural network is in two sentences.", "expected_fact": "network"},
        {"id": "EN-04", "lang": "English", "prompt": "Which planet is closest to the Sun?", "expected_fact": "Mercury"},

        # Hindi (Devanagari)
        {"id": "HI-01", "lang": "Hindi", "prompt": "भारत की राजधानी क्या है?", "expected_fact": "दिल्ली"},
        {"id": "HI-02", "lang": "Hindi", "prompt": "सूर्य किस दिशा में उगता है?", "expected_fact": "पूर्व"},
        {"id": "HI-03", "lang": "Hindi", "prompt": "नमस्ते! आप आज मेरी क्या सहायता कर सकते हैं?", "expected_fact": ""},
        {"id": "HI-04", "lang": "Hindi", "prompt": "पेड़ हमें क्या देते हैं?", "expected_fact": "ऑक्सीजन"},

        # Hinglish
        {"id": "HG-01", "lang": "Hinglish", "prompt": "Bhai, Python me print function kaise use karte hain?", "expected_fact": "print"},
        {"id": "HG-02", "lang": "Hinglish", "prompt": "Cricket match me kitne players hote hain ek team me?", "expected_fact": "11"},
        {"id": "HG-03", "lang": "Hinglish", "prompt": "Mujhe ek acchi chai banane ka tarika batao.", "expected_fact": "chai"},
        {"id": "HG-04", "lang": "Hinglish", "prompt": "Data structure me stack kya hota hai?", "expected_fact": "stack"}
    ]

    results = []

    def pure_neural_fn(p):
        formatted = f"<user>\n{p}\n<assistant>\n"
        resp = ""
        for chunk in engine.generate_stream(formatted, max_new_tokens=60, temperature=0.7):
            resp += chunk.get("text", "")
        return resp.strip()

    utf8_clean_count = 0
    pure_coherent_count = 0
    hybrid_coherent_count = 0
    pure_factual_hits = 0
    hybrid_factual_hits = 0

    for item in prompts:
        p_text = item['prompt']
        exp = item['expected_fact'].lower()

        # Pure neural
        pure_resp = pure_neural_fn(p_text)

        # Hybrid
        hybrid_res = controller.process(p_text, pure_neural_fn, mode="hybrid")
        hybrid_resp = hybrid_res["response"]

        # Check UTF-8 corruption
        has_replacement_char = ("\ufffd" in pure_resp) or ("\ufffd" in hybrid_resp)
        if not has_replacement_char:
            utf8_clean_count += 1

        # Check repetition / coherence
        words_p = pure_resp.split()
        rep_ratio_p = len(set(words_p)) / max(len(words_p), 1)
        p_coherent = len(pure_resp) > 5 and rep_ratio_p > 0.4
        if p_coherent: pure_coherent_count += 1

        words_h = hybrid_resp.split()
        rep_ratio_h = len(set(words_h)) / max(len(words_h), 1)
        h_coherent = len(hybrid_resp) > 5 and rep_ratio_h > 0.4
        if h_coherent: hybrid_coherent_count += 1

        p_fact = (exp in pure_resp.lower()) if exp else True
        h_fact = (exp in hybrid_resp.lower()) if exp else True
        if p_fact and exp: pure_factual_hits += 1
        if h_fact and exp: hybrid_factual_hits += 1

        results.append({
            "id": item['id'],
            "lang": item['lang'],
            "prompt": p_text,
            "expected_fact": item['expected_fact'],
            "pure_neural_response": pure_resp,
            "hybrid_response": hybrid_resp,
            "p_coherent": p_coherent,
            "h_coherent": h_coherent,
            "p_factual": p_fact,
            "h_factual": h_fact
        })

        print(f"[{item['lang']}] {p_text}")
        print(f"  -> Pure:   {repr(pure_resp[:60])}")
        print(f"  -> Hybrid: {repr(hybrid_resp[:60])}\n")

    summary = {
        "total_prompts": len(prompts),
        "utf8_clean_rate": (utf8_clean_count / len(prompts)) * 100,
        "pure_coherence_rate": (pure_coherent_count / len(prompts)) * 100,
        "hybrid_coherence_rate": (hybrid_coherent_count / len(prompts)) * 100,
        "pure_factual_hits": pure_factual_hits,
        "hybrid_factual_hits": hybrid_factual_hits,
        "details": results
    }

    out_file = Path("data/scientific_eval_016/language_regression_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== LANGUAGE & REGRESSION SUMMARY ===")
    print(f"UTF-8 Byte Integrity: {summary['utf8_clean_rate']:.1f}%")
    print(f"Pure Neural Coherence: {summary['pure_coherence_rate']:.1f}%")
    print(f"Hybrid Coherence:      {summary['hybrid_coherence_rate']:.1f}%")
    print(f"Pure Factual Hits:     {summary['pure_factual_hits']}/10")
    print(f"Hybrid Factual Hits:   {summary['hybrid_factual_hits']}/10")
    print(f"Saved to: {out_file}")

if __name__ == '__main__':
    run_language_and_regression_suite()
