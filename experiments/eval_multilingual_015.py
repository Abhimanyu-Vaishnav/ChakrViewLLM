"""
NR-015 Phase 8: Multilingual and Conversational Evaluation
Evaluates ChakrGenesis-17M Stage-D across English, Hindi, and Hinglish prompts.
Measures:
1. Pure-neural conversational coherence
2. Language identification fidelity
3. Repetition penalty behavior
4. Hybrid Cognitive Architecture evaluation
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from chakrview.brain.cognitive_architecture import CognitiveController

def run_multilingual_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Multilingual Eval on: {device}")

    # Pure neural engine
    pure_engine = ChakrViewInferenceEngine17M(
        checkpoint_path='experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
        device=device
    )

    # Cognitive controller
    controller = CognitiveController()

    def pure_neural_fn(prompt):
        formatted = f"<user>\n{prompt}\n<assistant>\n"
        resp = ""
        for chunk in pure_engine.generate_stream(formatted, max_new_tokens=60, temperature=0.7):
            resp += chunk.get("text", "")
        return resp.strip()

    prompts = [
        # English Conversation
        {"id": "LANG-EN-01", "lang": "English", "prompt": "What is machine learning in simple terms?"},
        {"id": "LANG-EN-02", "lang": "English", "prompt": "Hello! How can you help me today?"},
        {"id": "LANG-EN-03", "lang": "English", "prompt": "Explain the difference between a list and a tuple in Python."},
        {"id": "LANG-EN-04", "lang": "English", "prompt": "Tell me a short fact about planets."},
        
        # Hindi Conversation (Devanagari)
        {"id": "LANG-HI-01", "lang": "Hindi", "prompt": "नमस्ते! आप कैसे हैं?"},
        {"id": "LANG-HI-02", "lang": "Hindi", "prompt": "कंप्यूटर क्या होता है? संक्षेप में बताएं।"},
        {"id": "LANG-HI-03", "lang": "Hindi", "prompt": "भारत की राजधानी क्या है?"},
        {"id": "LANG-HI-04", "lang": "Hindi", "prompt": "गणित सीखने के क्या लाभ हैं?"},

        # Hinglish Conversation
        {"id": "LANG-HG-01", "lang": "Hinglish", "prompt": "Bhai ek simple Python code likh kar samjhao."},
        {"id": "LANG-HG-02", "lang": "Hinglish", "prompt": "Mujhe data science sikhna hai, kahan se shuru karu?"},
        {"id": "LANG-HG-03", "lang": "Hinglish", "prompt": "Aaj ka weather kaisa lag raha hai?"},
        {"id": "LANG-HG-04", "lang": "Hinglish", "prompt": "AI aur human intelligence me kya farak hai?"}
    ]

    results = []
    print("\n--- Running Pure-Neural and Hybrid Comparisons ---")

    for item in prompts:
        p_text = item['prompt']
        
        # Pure neural generation
        pure_resp = pure_neural_fn(p_text)

        # Hybrid generation
        hybrid_res = controller.process(p_text, pure_neural_fn, mode="hybrid")
        hybrid_resp = hybrid_res["response"]

        # Check basic coherence: non-empty, non-trivial, no runaway repetition
        words = pure_resp.split()
        rep_ratio = len(set(words)) / max(len(words), 1)
        is_coherent = len(pure_resp) > 5 and rep_ratio > 0.4

        results.append({
            "id": item['id'],
            "lang": item['lang'],
            "prompt": p_text,
            "pure_neural_response": pure_resp,
            "hybrid_response": hybrid_resp,
            "repetition_ratio": round(rep_ratio, 3),
            "is_coherent": is_coherent
        })
        print(f"[{item['lang']}] {p_text[:30]}...")
        print(f"  -> Pure Neural: {repr(pure_resp[:60])}")
        print(f"  -> Hybrid:      {repr(hybrid_resp[:60])}\n")

    out_file = Path("data/sealed_eval_015/multilingual_eval_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    coherent_count = sum(1 for r in results if r['is_coherent'])
    pct = (coherent_count / len(results)) * 100
    print(f"Multilingual evaluation completed: {coherent_count}/{len(results)} ({pct:.1f}% coherent)")
    print(f"Saved to: {out_file}")

if __name__ == '__main__':
    run_multilingual_eval()
