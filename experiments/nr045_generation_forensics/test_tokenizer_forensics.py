"""
experiments/nr045_generation_forensics/test_tokenizer_forensics.py
Phase 2: Comprehensive Tokenizer Forensics
Tests:
- English sentences
- Hindi Devanagari sentences
- Hinglish sentences
- Python code
- Numbers & math expressions
- Special tokens (BOS=0, EOS=1, PAD=2, user/assistant tags)
- Empty strings & unusual Unicode characters
Records: original text, token IDs, decoded text, round-trip fidelity, byte-level behavior.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def run_tokenizer_forensics():
    tokenizer, meta = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    test_cases = [
        {"cat": "English", "text": "The quick brown fox jumps over the lazy dog."},
        {"cat": "English Tech", "text": "ChakrView is an independent sovereign neural language model."},
        {"cat": "Hindi", "text": "चक्रव्यूह एक संप्रभु न्यूरल भाषा मॉडल है।"},
        {"cat": "Hindi Grammar", "text": "ऊर्जा संरक्षण का सिद्धांत क्या है?"},
        {"cat": "Hinglish", "text": "ChakrView model local GPU par fast run karta hai."},
        {"cat": "Hinglish Code-switch", "text": "Is function me recursion use karke factorial nikalna hai."},
        {"cat": "Python Code", "text": "def calculate_factorial(n: int) -> int:\n    return 1 if n <= 1 else n * calculate_factorial(n - 1)"},
        {"cat": "Math Expression", "text": "x^2 + 2xy + y^2 = (x + y)^2 and 145 + 355 = 500"},
        {"cat": "Devanagari Numerals", "text": "संख्या १२५ और ३५५ का योग ४८० होता है।"},
        {"cat": "Special Formatting", "text": "<user>\nHello! Who are you?\n<assistant>\nI am ChakrView.<eos>"},
        {"cat": "Rare Unicode & Emoji", "text": "Math: ∀x ∈ ℝ. Symbols: ॐ नमस्ते! Rare: 𝄞 𠮷. Emojis: 🚀🤖✨"},
        {"cat": "Empty String", "text": ""},
        {"cat": "Whitespace & Indents", "text": "    \t\n  indentation test  \n\n"}
    ]

    results = []
    print("=== EXECUTING TOKENIZER FORENSICS ===")

    for tc in test_cases:
        txt = tc["text"]
        toks = tokenizer.encode(txt)
        dec_strict = tokenizer.decode(toks, errors="strict")
        dec_replace = tokenizer.decode(toks, errors="replace")
        match = (txt == dec_strict)
        
        rec = {
            "category": tc["cat"],
            "original_text": txt,
            "token_count": len(toks),
            "byte_count": len(txt.encode('utf-8')),
            "token_ids": toks[:25], # preview
            "decoded_strict_matches": match,
            "round_trip_identical": match
        }
        results.append(rec)
        print(f"[{tc['cat']:22s}] Bytes: {len(txt.encode('utf-8')):3d} | Toks: {len(toks):3d} | Match: {match}")

    out_file = Path("experiments/nr045_generation_forensics/tokenizer_forensics_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nTokenizer forensics saved to {out_file}")

if __name__ == "__main__":
    run_tokenizer_forensics()
