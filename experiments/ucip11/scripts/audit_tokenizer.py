"""
Phase C: Tokenizer & Representation Study Script.
Handles multi-byte byte-level BPE safely with errors='replace'.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)

test_corpora = {
    "english_conversational": [
        "Hello, how can I help you today?",
        "I am an artificial intelligence assistant created to run locally on your computer.",
        "The weather in Delhi is pleasant this afternoon.",
        "Can you explain the difference between RAM and storage in simple terms?"
    ],
    "english_technical": [
        "What is the function of the instruction register inside a CPU?",
        "Describe how pipelining improves instruction throughput in modern processors.",
        "Explain how virtual memory provides process isolation using page tables.",
        "Quicksort operates by selecting a pivot element and partitioning the array recursively."
    ],
    "hindi_devanagari": [
        "नमस्ते! मैं चक्रव्यूह हूँ, आपका स्थानीय कृत्रिम बुद्धिमत्ता सहायक।",
        "कंप्यूटर विज्ञान में डेटा संरचनाएं और एल्गोरिदम अत्यंत महत्वपूर्ण हैं।",
        "केंद्रीय प्रसंस्करण इकाई यानी सीपीयू कंप्यूटर का मस्तिष्क कहलाता है।",
        "डेटाबेस अनुक्रमणिका तेजी से प्रश्नों को हल करने में मदद करती है।"
    ],
    "roman_hindi": [
        "Namaste! Aap kaise hain? Main aapki kya madad kar sakta hoon?",
        "Computer science mein data structures aur algorithms bohot zaroori hain.",
        "Mujhe coding seekhni hai aur ek accha project banana hai."
    ],
    "hinglish": [
        "Bhai, CPU ka main kaam kya hota hai? Thoda detail mein explain karo.",
        "Virtual memory kaise process isolation provide karti hai?",
        "Microservices architecture monolithic se kaise behtar hoti hai scalability ke liye?"
    ],
    "code_python": [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])",
        "import torch\nimport torch.nn as nn\n\nclass Attention(nn.Module):\n    def __init__(self, d_model):\n        super().__init__()"
    ],
    "math_logic": [
        "Solve the following: 67 + 54 = 121. Compute 356 - 200 = 156.",
        "All squares are rectangles. Shape S is a square. Is Shape S a rectangle? Yes."
    ]
}

results = {}

for cat, sentences in test_corpora.items():
    cat_chars = 0
    cat_bytes = 0
    cat_tokens = 0
    cat_words = 0
    round_trip_success = True
    
    for s in sentences:
        s_bytes = len(s.encode("utf-8"))
        s_chars = len(s)
        s_words = len(s.split())
        tokens = tokenizer.encode(s, add_bos=False, add_eos=False)
        decoded = tokenizer.decode(tokens, errors="replace")
        
        if decoded.strip() != s.strip():
            round_trip_success = False
            
        cat_chars += s_chars
        cat_bytes += s_bytes
        cat_tokens += len(tokens)
        cat_words += s_words
            
    chars_per_token = cat_chars / cat_tokens if cat_tokens > 0 else 0
    bytes_per_token = cat_bytes / cat_tokens if cat_tokens > 0 else 0
    tokens_per_word = cat_tokens / cat_words if cat_words > 0 else 0
    
    results[cat] = {
        "total_chars": cat_chars,
        "total_bytes": cat_bytes,
        "total_tokens": cat_tokens,
        "total_words": cat_words,
        "fertility_tokens_per_word": round(tokens_per_word, 2),
        "chars_per_token": round(chars_per_token, 2),
        "bytes_per_token": round(bytes_per_token, 2),
        "round_trip_exact": round_trip_success
    }

out_p = REPO_ROOT / "experiments" / "ucip11" / "data" / "manifests" / "tokenizer_audit_results.json"
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Tokenizer representation audit saved to", out_p)
for k, v in results.items():
    print(f"[{k}] Fertility: {v['fertility_tokens_per_word']} toks/word | Chars/tok: {v['chars_per_token']} | Bytes/tok: {v['bytes_per_token']} | Round-trip: {v['round_trip_exact']}")
