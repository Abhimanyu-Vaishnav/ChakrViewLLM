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

# 30 Dedicated Hindi & Hinglish Probes (Phase G)
# 10 Natural Hindi (Devanagari)
# 10 Roman Hindi
# 10 Mixed Hinglish / Technical Questions

def build_hinglish_probes():
    probes = []
    
    # Category A: Natural Hindi (Devanagari) (10)
    devanagari = [
        ("नमस्ते! आप कैसे हैं?", ["नमस्ते", "मैं", "ठीक", "आप", "मदद"], "Greetings response in Hindi"),
        ("आपका शुभ नाम क्या है?", ["चक्रव्यूह", "नाम", "ChakrView"], "Identity statement in Hindi"),
        ("क्या आप मेरी सहायता कर सकते हैं?", ["हाँ", "सहायता", "मदद", "कर सकता"], "Affirmation of help in Hindi"),
        ("आज मौसम बहुत सुहावना है।", ["अच्छा", "मौसम", "सुहावना"], "Conversational weather response in Hindi"),
        ("भारत की राजधानी कहाँ है?", ["नई दिल्ली", "दिल्ली", "Delhi"], "Fact retrieval: New Delhi"),
        ("मुझे एक छोटा सा चुटकुला सुनाइए।", ["हँस", "एक", "था"], "Conversational response / joke attempt in Hindi"),
        ("धन्यवाद आपके सहयोग के लिए!", ["स्वागत", "धन्यवाद", "मदद"], "Acknowledgment of thanks in Hindi"),
        ("अलविदा! फिर मिलेंगे।", ["अलविदा", "नमस्ते", "फिर मिलेंगे"], "Farewell response in Hindi"),
        ("क्या आप एक कृत्रिम बुद्धिमत्ता हैं?", ["हाँ", "एआई", "मॉडल", "AI"], "Affirmation of AI identity in Hindi"),
        ("कम्प्यूटर का मुख्य दिमाग क्या होता है?", ["सीपीयू", "CPU", "प्रोसेसर"], "Fact retrieval: CPU in Hindi")
    ]
    for idx, (p, exp, goal) in enumerate(devanagari, 1):
        probes.append({
            "id": f"HIN_DEV_{idx:02d}",
            "type": "Devanagari Hindi",
            "prompt": f"User: {p}\nAssistant:",
            "expected_keywords": exp,
            "goal": goal
        })
        
    # Category B: Roman Hindi (10)
    roman_hindi = [
        ("Namaste! Sab theek thak?", ["Namaste", "theek", "haan", "sab"], "Greeting in Roman Hindi"),
        ("Aapka naam kya hai?", ["ChakrView", "naam", "mera"], "Identity in Roman Hindi"),
        ("Mujhe thoda kaam me sahyog chahiye.", ["haan", "madad", "sahyog", "bataiye"], "Offering help in Roman Hindi"),
        ("Aaj bohot garmi hai yaar.", ["garmi", "paani", "dhyan"], "Conversational reaction to heat in Roman Hindi"),
        ("Aap kisne banaya hai?", ["ChakrView", "team", "indigenous"], "Developer origin in Roman Hindi"),
        ("Shukriya dost!", ["swagat", "shukriya", "madad"], "Gratitude response in Roman Hindi"),
        ("Chai piyoge mere sath?", ["AI", "software", "chai", "shukriya"], "Polite conversational decline in Roman Hindi"),
        ("Alvida, kal milte hain.", ["Alvida", "namaste", "milenge", "kal"], "Farewell in Roman Hindi"),
        ("Sach batao, kya aap insaan ho?", ["nahi", "AI", "model", "software"], "Denial of being human in Roman Hindi"),
        ("Padhai me man nahi lag raha, kya karun?", ["break", "focus", "padhai", "tips"], "Conversational advice in Roman Hindi")
    ]
    for idx, (p, exp, goal) in enumerate(roman_hindi, 1):
        probes.append({
            "id": f"HIN_ROM_{idx:02d}",
            "type": "Roman Hindi",
            "prompt": f"User: {p}\nAssistant:",
            "expected_keywords": exp,
            "goal": goal
        })
        
    # Category C: Mixed Hinglish & Technical (10)
    hinglish_tech = [
        ("Python me list aur tuple me kya difference hai?", ["immutable", "mutable", "tuple", "list"], "Technical difference explanation"),
        ("Bhai, git commit push kaise karte hain?", ["git push", "commit", "terminal"], "Git commands explanation"),
        ("Database me indexing kyun use karte hain?", ["fast", "search", "query", "speed"], "Indexing explanation"),
        ("Kya ye model local CPU pe chal raha hai?", ["haan", "CPU", "local", "natively"], "Confirmation of local CPU"),
        ("Thoda code review me help kar doge?", ["haan", "code", "batao", "help"], "Readiness to help with code"),
        ("Recursion kya hota hai simple words me?", ["function", "call", "itself"], "Recursion concept explanation"),
        ("Docker container aur VM me kya fark hai?", ["lightweight", "kernel", "OS", "VM"], "Docker vs VM explanation"),
        ("Bhai project submit karne ka deadline pass aa gaya hai.", ["tension", "focus", "complete", "kaam"], "Supportive conversational advice"),
        ("Bug fix karne ka sabse badhiya tarika kya hai?", ["debug", "logs", "test", "print"], "Debugging tips in Hinglish"),
        ("Aap multi-lingual dialogue support karte ho?", ["haan", "Hindi", "English", "Hinglish"], "Multilingual capability confirmation")
    ]
    for idx, (p, exp, goal) in enumerate(hinglish_tech, 1):
        probes.append({
            "id": f"HIN_TECH_{idx:02d}",
            "type": "Mixed Hinglish / Technical",
            "prompt": f"User: {p}\nAssistant:",
            "expected_keywords": exp,
            "goal": goal
        })
        
    return probes

def evaluate_hinglish():
    runner = DirectNeuralRunner(RELEASE_CKPT)
    tok = runner.tokenizer
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)
    
    probes = build_hinglish_probes()
    results = []
    
    stats_by_type = {}
    total_relevant = 0
    total_appropriate_lang = 0
    total_refusal_sinks = 0
    
    for item in probes:
        ptype = item["type"]
        if ptype not in stats_by_type:
            stats_by_type[ptype] = {"total": 0, "hits": 0, "appropriate_lang": 0, "refusals": 0}
        stats_by_type[ptype]["total"] += 1
        
        prompt = item["prompt"]
        toks = tok.encode(prompt, add_bos=True, add_eos=False)
        curr = list(toks)
        gen = []
        with torch.no_grad():
            for step in range(48):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                next_tok = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                gen.append(next_tok)
                curr.append(next_tok)
                if next_tok == 1:
                    break
        resp = tok.decode(gen, errors="replace").strip()
        
        # Keyword relevance
        hit = any(kw.lower() in resp.lower() for kw in item["expected_keywords"])
        if hit:
            total_relevant += 1
            stats_by_type[ptype]["hits"] += 1
            
        # Language appropriateness (contains Hindi/Hinglish tokens or Devanagari if Devanagari prompt)
        has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in resp)
        hindi_roman_tokens = ["namaste", "aap", "main", "hai", "hoon", "theek", "shukriya", "alvida", "dhanyawad", "madad", "kaise", "batao"]
        has_roman_hindi = any(t in resp.lower().split() for t in hindi_roman_tokens)
        
        appr_lang = False
        if ptype == "Devanagari Hindi" and has_devanagari:
            appr_lang = True
        elif ptype in ["Roman Hindi", "Mixed Hinglish / Technical"] and (has_roman_hindi or has_devanagari):
            appr_lang = True
            
        if appr_lang:
            total_appropriate_lang += 1
            stats_by_type[ptype]["appropriate_lang"] += 1
            
        is_refusal = "not provided in the context" in resp.lower()
        if is_refusal:
            total_refusal_sinks += 1
            stats_by_type[ptype]["refusals"] += 1
            
        results.append({
            "id": item["id"],
            "type": ptype,
            "prompt": prompt,
            "response": resp,
            "keyword_hit": hit,
            "appropriate_language": appr_lang,
            "is_refusal": is_refusal,
            "expected_keywords": item["expected_keywords"]
        })
        
    summary = {
        "total_probes": len(probes),
        "total_keyword_relevant": total_relevant,
        "total_appropriate_language": total_appropriate_lang,
        "total_refusal_sinks": total_refusal_sinks,
        "breakdown_by_type": stats_by_type
    }
    
    with open(OUT_DIR / "hinglish_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "probes": results}, f, indent=2)
        
    print("\n=== HINDI & HINGLISH INDEPENDENT EVALUATION ===")
    print(f"Total Probes: {len(probes)}")
    print(f"Keyword Relevant: {total_relevant}/{len(probes)} ({total_relevant/len(probes)*100:.1f}%)")
    print(f"Appropriate Language Grounded: {total_appropriate_lang}/{len(probes)} ({total_appropriate_lang/len(probes)*100:.1f}%)")
    print(f"Refusal Sink Rate: {total_refusal_sinks}/{len(probes)} ({total_refusal_sinks/len(probes)*100:.1f}%)")
    for k, v in stats_by_type.items():
        print(f"  {k:28s}: Relevant={v['hits']}/{v['total']}, ApprLang={v['appropriate_lang']}/{v['total']}")

if __name__ == "__main__":
    evaluate_hinglish()
