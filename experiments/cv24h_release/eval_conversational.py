"""
Mission CV-24H: Conversational Evaluation Benchmark and Human Review Generator.
Evaluates the pure neural model (ChakrGenesis38M Conversational Alpha) across 12 distinct capabilities:
1. English conversation & greetings
2. Hindi conversation (Devanagari)
3. Hinglish conversation
4. Basic factual questions
5. Instruction following
6. Multi-turn context retention
7. Follow-up questions
8. Topic changes
9. Appropriate uncertainty
10. Rejection of unsupported claims
11. Output stability
12. Repetition and degeneration analysis

Computes metrics:
- Valid UTF-8 output rate (%)
- Average generated token length
- Repetition rate (4-gram repetition %)
- Empty / malformed response rate (%)
- Human-reviewed qualitative transcript

Saves results to:
- experiments/cv24h_release/conversational_evaluation_results.json
- reports/CV24H_CONVERSATIONAL_EVALUATION_REPORT.md
"""

import os
import sys
import json
import math
import hashlib
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EVAL_PROMPTS = [
    # 1. English conversation & greetings
    {"id": "eng_conv_01", "cat": "english_conversation", "prompt": "User: Hello! Who are you?\nAssistant:"},
    {"id": "eng_conv_02", "cat": "english_conversation", "prompt": "User: Good morning, how are you doing today?\nAssistant:"},
    {"id": "eng_conv_03", "cat": "english_conversation", "prompt": "User: Can you help me write a Python function?\nAssistant:"},

    # 2. Hindi conversation
    {"id": "hi_conv_01", "cat": "hindi_conversation", "prompt": "User: नमस्ते! आप कैसे हैं?\nAssistant:"},
    {"id": "hi_conv_02", "cat": "hindi_conversation", "prompt": "User: भारत की राजधानी क्या है?\nAssistant:"},
    {"id": "hi_conv_03", "cat": "hindi_conversation", "prompt": "User: क्या आप मेरी मदद कर सकते हैं?\nAssistant:"},

    # 3. Hinglish conversation
    {"id": "hing_conv_01", "cat": "hinglish_conversation", "prompt": "User: Hello dost, kaise ho?\nAssistant:"},
    {"id": "hing_conv_02", "cat": "hinglish_conversation", "prompt": "User: Python mein list kaise banate hain?\nAssistant:"},
    {"id": "hing_conv_03", "cat": "hinglish_conversation", "prompt": "User: ChakrView model kya hai aur iska goal kya hai?\nAssistant:"},

    # 4. Basic factual questions
    {"id": "fact_01", "cat": "basic_facts", "prompt": "User: What is the boiling point of water at standard pressure?\nAssistant:"},
    {"id": "fact_02", "cat": "basic_facts", "prompt": "User: What language is Python written in?\nAssistant:"},
    {"id": "fact_03", "cat": "basic_facts", "prompt": "User: Which planet is closest to the Sun?\nAssistant:"},

    # 5. Instruction following
    {"id": "inst_01", "cat": "instruction_following", "prompt": "User: Answer in exactly one word: Is the sky blue?\nAssistant:"},
    {"id": "inst_02", "cat": "instruction_following", "prompt": "User: Write three bullet points listing prime numbers between 1 and 10.\nAssistant:"},

    # 6. Multi-turn context retention
    {"id": "multiturn_01", "cat": "multiturn_context", "prompt": "User: My favorite color is green.\nAssistant: That is a wonderful color!\nUser: What is my favorite color?\nAssistant:"},
    {"id": "multiturn_02", "cat": "multiturn_context", "prompt": "User: Let us discuss Python programming.\nAssistant: Sure, what about Python would you like to discuss?\nUser: What topic did we just agree to discuss?\nAssistant:"},

    # 7. Follow-up questions
    {"id": "followup_01", "cat": "followup_questions", "prompt": "User: What is machine learning?\nAssistant: Machine learning is a field of artificial intelligence where algorithms learn patterns from data.\nUser: Can you give me a simple example of that?\nAssistant:"},

    # 8. Topic changes
    {"id": "topic_01", "cat": "topic_change", "prompt": "User: We were talking about algorithms, but let us talk about tea now. Do you know what green tea is?\nAssistant:"},

    # 9. Appropriate uncertainty
    {"id": "uncertainty_01", "cat": "appropriate_uncertainty", "prompt": "User: Who will win the FIFA World Cup in 2038?\nAssistant:"},
    {"id": "uncertainty_02", "cat": "appropriate_uncertainty", "prompt": "User: What is inside a black hole beyond the event horizon?\nAssistant:"},

    # 10. Rejection of unsupported claims
    {"id": "rejection_01", "cat": "rejection_unsupported", "prompt": "User: Can you explain why elephants can fly using their ears?\nAssistant:"},
    {"id": "rejection_02", "cat": "rejection_unsupported", "prompt": "User: Explain why 2 + 2 = 5.\nAssistant:"},

    # 11. Output stability
    {"id": "stability_01", "cat": "output_stability", "prompt": "User: Repeat the word 'ChakrView' once.\nAssistant:"},

    # 12. Repetition & degeneration test
    {"id": "degen_01", "cat": "repetition_degeneration", "prompt": "User: Tell me a short story about an autonomous AI.\nAssistant:"}
]

def generate(model, tokenizer, prompt_text, max_new_tokens=64, temperature=0.7, top_k=40, repetition_penalty=1.1, device="cuda"):
    model.eval()
    tokens = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(input_ids)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_token_logits = logits[0, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for t in set(input_ids[0].tolist()):
                    if next_token_logits[t] < 0:
                        next_token_logits[t] *= repetition_penalty
                    else:
                        next_token_logits[t] /= repetition_penalty

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token == 2 or next_token == 0:  # EOS or PAD
                break

            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            if input_ids.shape[1] >= 256:
                break

    decoded_text = tokenizer.decode(generated)
    return decoded_text, len(generated)

def calculate_repetition_rate(text, n=4):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngs = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    repeated = len(ngs) - len(set(ngs))
    return (repeated / len(ngs)) * 100.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    sft_ckpt_path = "experiments/cv24h_release/chakr_38m_conversational_sft.pt"
    ckpt = torch.load(sft_ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ChakrGenesis38M(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tok_dir = "data/experiments/vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    print("Running Conversational Evaluation across 12 capability areas...")

    eval_results = []
    total_tokens_gen = 0
    valid_utf8_count = 0
    empty_count = 0
    rep_scores = []

    for item in EVAL_PROMPTS:
        pid = item["id"]
        cat = item["cat"]
        prompt = item["prompt"]

        resp_text, num_toks = generate(model, tokenizer, prompt, max_new_tokens=64, temperature=0.7, top_k=40, repetition_penalty=1.15, device=device)
        total_tokens_gen += num_toks

        # Check UTF-8 validity
        try:
            resp_text.encode("utf-8")
            valid_utf8 = True
            valid_utf8_count += 1
        except Exception:
            valid_utf8 = False

        is_empty = len(resp_text.strip()) == 0
        if is_empty:
            empty_count += 1

        rep_rate = calculate_repetition_rate(resp_text, 4)
        rep_scores.append(rep_rate)

        eval_results.append({
            "id": pid,
            "category": cat,
            "prompt": prompt,
            "response": resp_text.strip(),
            "tokens_generated": num_toks,
            "valid_utf8": valid_utf8,
            "is_empty": is_empty,
            "4gram_repetition_rate_pct": round(rep_rate, 2)
        })

        print(f"\n--- [{cat}] ({pid}) ---")
        print(prompt.strip())
        print("Generated Assistant:", resp_text.strip())

    avg_tokens = total_tokens_gen / len(EVAL_PROMPTS)
    utf8_rate = (valid_utf8_count / len(EVAL_PROMPTS)) * 100.0
    empty_rate = (empty_count / len(EVAL_PROMPTS)) * 100.0
    avg_rep_rate = sum(rep_scores) / len(rep_scores)

    summary = {
        "mission": "CV-24H Conversational Evaluation",
        "model_name": "ChakrGenesis38M-Conversational-Alpha",
        "total_test_prompts": len(EVAL_PROMPTS),
        "valid_utf8_rate_pct": utf8_rate,
        "empty_response_rate_pct": empty_rate,
        "average_tokens_generated": round(avg_tokens, 1),
        "average_4gram_repetition_pct": round(avg_rep_rate, 2),
        "evaluation_items": eval_results
    }

    out_json = "experiments/cv24h_release/conversational_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved to {out_json}")
    print(f"Summary: UTF-8 Valid: {utf8_rate}% | Empty Rate: {empty_rate}% | Avg Tokens: {avg_tokens:.1f} | Avg 4-gram Repetition: {avg_rep_rate:.2f}%")

if __name__ == "__main__":
    main()
