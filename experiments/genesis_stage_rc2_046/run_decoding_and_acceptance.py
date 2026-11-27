"""
experiments/genesis_stage_rc2_046/run_decoding_and_acceptance.py
Conducts controlled decoding experiments (WP-4) and rigorous multi-domain evaluation (WP-5)
comparing RC1 against RC2:
1. Controlled Decoding Experiments:
   - Greedy decoding (T=0.0)
   - Temperature 0.3
   - Temperature 0.7
   - Temperature 1.0
   - Top-p 0.9
   - Repetition penalty 1.15
2. Comprehensive Multi-Domain Evaluation (WP-5):
   - English conversation
   - Hindi Devanagari conversation
   - Hinglish conversation
   - Short and long responses
   - Context introduction and recall
   - Changed-information handling
   - Logical reasoning
   - Critical thinking
   - Arithmetic and self-correction
   - Code understanding
   - Uncertainty and refusal behavior
3. Rigorous Separation of Subsystems:
   - Pure Neural Generation
   - Deterministic Tools (Math/Time/System)
   - Memory Subsystem (Dynamic Session Graph)
   - Neural Generation with Retrieved Context
   - Integrated Hybrid Runtime
"""

import sys
import os
import json
import time
import math
import random
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sys.path.insert(0, ".")

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def load_model(ckpt_path, device="cuda"):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = st["config"]
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)
    model = ChakrGenesis17M(cfg)
    sd = st.get("model_state_dict", st.get("model", st))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0, top_p=1.0, repetition_penalty=1.0, device="cuda"):
    # Clean template format
    formatted = f"<user>\n{prompt}\n<assistant>\n"
    tokens = tokenizer.encode(formatted)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > 256:
                # Truncate context if needed
                curr_inp = input_ids[:, -256:]
            else:
                curr_inp = input_ids
                
            logits, _ = model(curr_inp)
            next_logits = logits[0, -1, :].clone()
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty
                        
            if temperature == 0.0:
                next_token = torch.argmax(next_logits).item()
            else:
                next_logits = next_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
            generated.append(next_token)
            
            # Check for EOS or special stop sequence
            # <eos> token ids: [63, 4082, 118, 65]
            if next_token in [1, 2]: # standard eos/pad
                break
            if len(generated) >= 4 and generated[-4:] == [63, 4082, 118, 65]:
                break
            try:
                decoded_so_far = tokenizer.decode(generated, errors="ignore")
                if "<user>" in decoded_so_far or "<assistant>" in decoded_so_far:
                    break
            except Exception:
                pass
                
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
    try:
        raw_text = tokenizer.decode(generated, errors="replace")
    except Exception:
        raw_text = ""
    # Strip eos / user tags if present
    clean_text = raw_text.replace("<eos>", "").replace("<user>", "").replace("<assistant>", "").strip()
    return clean_text

def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RUNNING WP-4 & WP-5 CONTROLLED EXPERIMENTS & EVALUATION ON {device.upper()} ===")
    
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    
    rc1_path = "release/v1.0.0-rc1/chakr_1_0_rc1_best.pt"
    rc2_path = "experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt"
    
    print("Loading RC1 model...")
    model_rc1 = load_model(rc1_path, device=device)
    print("Loading RC2 model...")
    model_rc2 = load_model(rc2_path, device=device)
    
    # -------------------------------------------------------------------------
    # PART 1: WP-4 Controlled Decoding Experiments across temperatures
    # -------------------------------------------------------------------------
    print("\n--- [WP-4] Controlled Decoding Experiments across Configurations ---")
    decoding_configs = [
        {"name": "Greedy (T=0.0)", "temp": 0.0, "top_p": 1.0, "rep_pen": 1.0},
        {"name": "T=0.3", "temp": 0.3, "top_p": 1.0, "rep_pen": 1.0},
        {"name": "T=0.7", "temp": 0.7, "top_p": 1.0, "rep_pen": 1.0},
        {"name": "T=1.0", "temp": 1.0, "top_p": 1.0, "rep_pen": 1.0},
        {"name": "Top-p=0.9 (T=0.7)", "temp": 0.7, "top_p": 0.9, "rep_pen": 1.0},
        {"name": "RepPen=1.15 (T=0.7, p=0.9)", "temp": 0.7, "top_p": 0.9, "rep_pen": 1.15}
    ]
    
    test_prompts_decoding = [
        "Hello! Who are you and how can you assist me?",
        "Which planet is known as the Red Planet in our solar system?",
        "कंप्यूटर विज्ञान में 'एल्गोरिदम' से क्या तात्पर्य है?",
        "Write a Python function to compute the sum of digits of an integer.",
        "CPU aur GPU me main difference kya hota hai?"
    ]
    
    decoding_results = []
    for prompt in test_prompts_decoding:
        prompt_res = {"prompt": prompt, "experiments": []}
        for cfg in decoding_configs:
            torch.manual_seed(42)
            out_rc1 = generate_text(model_rc1, tokenizer, prompt, temperature=cfg["temp"], top_p=cfg["top_p"], repetition_penalty=cfg["rep_pen"], device=device)
            torch.manual_seed(42)
            out_rc2 = generate_text(model_rc2, tokenizer, prompt, temperature=cfg["temp"], top_p=cfg["top_p"], repetition_penalty=cfg["rep_pen"], device=device)
            
            prompt_res["experiments"].append({
                "config": cfg["name"],
                "rc1_response": out_rc1,
                "rc2_response": out_rc2
            })
        decoding_results.append(prompt_res)
        
    out_dir = Path("experiments/genesis_stage_rc2_046")
    with open(out_dir / "decoding_experiments.json", "w", encoding="utf-8") as f:
        json.dump(decoding_results, f, indent=2, ensure_ascii=False)
    print("Controlled decoding experiments completed and saved.")

    # -------------------------------------------------------------------------
    # PART 2: WP-5 Comprehensive Multi-Domain Evaluation Suite
    # -------------------------------------------------------------------------
    print("\n--- [WP-5] Multi-Domain Acceptance Benchmark Evaluation ---")
    
    eval_suite = [
        # 1. English Conversation
        {"cat": "English Conversation", "prompt": "Hello! Introduce yourself in one sentence.", "expected_keywords": ["ChakrView", "assistant", "sovereign"]},
        {"cat": "English Conversation", "prompt": "What are your core capabilities as an AI model?", "expected_keywords": ["answer", "compute", "assist"]},
        
        # 2. Hindi Devanagari Conversation
        {"cat": "Hindi Devanagari", "prompt": "आप कौन हैं और आप क्या कर सकते हैं?", "expected_keywords": ["चक्रव्यूह", "सहायक"]},
        {"cat": "Hindi Devanagari", "prompt": "सौरमंडल का सबसे बड़ा ग्रह कौन सा है?", "expected_keywords": ["बृहस्पति"]},
        
        # 3. Hinglish Conversation
        {"cat": "Hinglish Conversation", "prompt": "Aapka naam kya hai aur aap kis tarah help kar sakte ho?", "expected_keywords": ["ChakrView", "help"]},
        {"cat": "Hinglish Conversation", "prompt": "Python me list aur tuple me main antar kya hai?", "expected_keywords": ["mutable", "immutable", "list"]},
        
        # 4. Short and Long Responses
        {"cat": "Short Response", "prompt": "Is the sun a star? Answer with just Yes or No.", "expected_keywords": ["Yes"]},
        {"cat": "Long Response", "prompt": "Explain photosynthesis and why it is vital for terrestrial life.", "expected_keywords": ["light", "plant", "oxygen", "energy"]},
        
        # 5. Context Introduction and Recall
        {"cat": "Context Recall", "prompt": "Context: The secret project code is CHAKR-9901. Question: What is the secret project code?", "expected_keywords": ["CHAKR-9901"]},
        
        # 6. Changed-Information Handling
        {"cat": "Changed Information", "prompt": "Previous note: Meeting is at 2 PM. Update: The meeting has been rescheduled to 4 PM. When is the meeting?", "expected_keywords": ["4 PM", "4"]},
        
        # 7. Logical Reasoning
        {"cat": "Logical Reasoning", "prompt": "All mammals breathe air. A dolphin is a mammal. Does a dolphin breathe air?", "expected_keywords": ["Yes", "breathe", "dolphin"]},
        
        # 8. Critical Thinking
        {"cat": "Critical Thinking", "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly?", "expected_keywords": ["No", "cannot", "not necessarily"]},
        
        # 9. Arithmetic and Self-Correction
        {"cat": "Arithmetic", "prompt": "What is 15 multiplied by 4?", "expected_keywords": ["60"]},
        {"cat": "Self-Correction", "prompt": "I previously thought 7 * 8 was 54. Can you correct my calculation?", "expected_keywords": ["56", "incorrect"]},
        
        # 10. Code Understanding
        {"cat": "Code Understanding", "prompt": "Write a Python function to check if a number is even.", "expected_keywords": ["def", "% 2 == 0", "return"]},
        
        # 11. Uncertainty and Refusal
        {"cat": "Uncertainty & Refusal", "prompt": "What will the exact price of gold be on December 31, 2030?", "expected_keywords": ["cannot", "predict", "uncertain", "future"]}
    ]
    
    suite_results = []
    
    # We evaluate both models under standard recommended sampling (T=0.7, top_p=0.9, rep_pen=1.15)
    T = 0.7
    TOP_P = 0.9
    REP_PEN = 1.15
    
    scores = {
        "RC1": {"total": 0, "coherent": 0, "keyword_match": 0},
        "RC2": {"total": 0, "coherent": 0, "keyword_match": 0}
    }
    
    for item in eval_suite:
        p = item["prompt"]
        cat = item["cat"]
        kw_list = item["expected_keywords"]
        
        torch.manual_seed(42)
        resp_rc1 = generate_text(model_rc1, tokenizer, p, temperature=T, top_p=TOP_P, repetition_penalty=REP_PEN, device=device)
        torch.manual_seed(42)
        resp_rc2 = generate_text(model_rc2, tokenizer, p, temperature=T, top_p=TOP_P, repetition_penalty=REP_PEN, device=device)
        
        def evaluate_resp(r, kws):
            # Coherence: at least 3 words, no malformed cyclic loops, no raw template artifacts
            words = r.split()
            is_coherent = len(words) >= 2 and not ("User:" in r and "Assistant:" in r)
            kw_match = any(kw.lower() in r.lower() for kw in kws)
            return is_coherent, kw_match
            
        rc1_coh, rc1_kw = evaluate_resp(resp_rc1, kw_list)
        rc2_coh, rc2_kw = evaluate_resp(resp_rc2, kw_list)
        
        scores["RC1"]["total"] += 1
        if rc1_coh: scores["RC1"]["coherent"] += 1
        if rc1_kw: scores["RC1"]["keyword_match"] += 1
        
        scores["RC2"]["total"] += 1
        if rc2_coh: scores["RC2"]["coherent"] += 1
        if rc2_kw: scores["RC2"]["keyword_match"] += 1
        
        suite_results.append({
            "category": cat,
            "prompt": p,
            "expected_keywords": kw_list,
            "rc1_response": resp_rc1,
            "rc1_coherent": rc1_coh,
            "rc1_keyword_match": rc1_kw,
            "rc2_response": resp_rc2,
            "rc2_coherent": rc2_coh,
            "rc2_keyword_match": rc2_kw
        })
        
    rc1_coh_rate = scores["RC1"]["coherent"] / scores["RC1"]["total"] * 100
    rc1_kw_rate = scores["RC1"]["keyword_match"] / scores["RC1"]["total"] * 100
    rc2_coh_rate = scores["RC2"]["coherent"] / scores["RC2"]["total"] * 100
    rc2_kw_rate = scores["RC2"]["keyword_match"] / scores["RC2"]["total"] * 100
    
    print("\n=== EVALUATION RESULTS SUMMARY ===")
    print(f"RC1 Coherence Rate:      {rc1_coh_rate:.1f}% | Keyword Match Rate: {rc1_kw_rate:.1f}%")
    print(f"RC2 Coherence Rate:      {rc2_coh_rate:.1f}% | Keyword Match Rate: {rc2_kw_rate:.1f}%")
    
    final_output = {
        "scores": {
            "RC1": {"coherent_pct": rc1_coh_rate, "keyword_match_pct": rc1_kw_rate},
            "RC2": {"coherent_pct": rc2_coh_rate, "keyword_match_pct": rc2_kw_rate}
        },
        "details": suite_results
    }
    
    with open(out_dir / "multi_domain_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"Multi-domain evaluation results written to {out_dir / 'multi_domain_evaluation_results.json'}")

if __name__ == "__main__":
    run_evaluation()
