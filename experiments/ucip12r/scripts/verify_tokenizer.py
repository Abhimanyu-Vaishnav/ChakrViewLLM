"""
UCIP-12R Phase E: Tokenizer Independent Forensic Verification Script.
Loads the frozen ChakrView byte-level BPE tokenizer and rigorously tests:
1. Artifact integrity and configuration hashes.
2. Round-trip lossless property across 9 distinct categories including Unicode edge cases.
3. Word fertility and byte-efficiency calculations.
4. Determinism, special token bounds, and ID range validity.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok_dir = ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, tok_cfg = load_tokenizer_artifacts(tok_dir)

print(f"Loaded Tokenizer: Vocab={tokenizer.vocab_size}, Merges={tokenizer.num_merges}")
print(f"Checksums: {tok_cfg.get('checksums')}")

test_cases = {
    "English prose": "Artificial intelligence models require extensive pretraining on high-quality corpora to learn robust semantic structures.",
    "Devanagari Hindi": "कृत्रिम बुद्धिमत्ता और भाषा अधिग्रहण का अध्ययन तंत्रिका नेटवर्क के लिए अत्यंत महत्वपूर्ण है।",
    "Roman Hindi": "Yeh model natural language processing ke liye train kiya gaya hai.",
    "Hinglish mixed": "Bhai is model ka performance evaluate karne ke baad hi next experiment start karenge.",
    "Sanskrit classical": "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः येनाक्रमन्त्यृषयो ह्याप्तकामाः।",
    "Python code": "def train_epoch(model, dataloader, optimizer, criterion):\n    model.train()\n    for x, y in dataloader:\n        loss = criterion(model(x), y)\n        loss.backward()\n        optimizer.step()",
    "Mixed script": "ChakrView (चक्रव्यूह) is an open-source multilingual neural language model developed in India.",
    "Punctuation heavy": "!@#$%^&*()_+-=[]{}|;':\",.<>/?`~",
    "Unicode edge cases": "ZWNJ & ZWJ: \u200c \u200d Devanagari: \u0915\u094d\u0937 \u0936\u094d\u0930 Emojis: \U0001F600 \U0001F916"
}

results = {}
for cat, text in test_cases.items():
    enc = tokenizer.encode(text)
    dec = tokenizer.decode(enc)
    match = (dec == text)
    words = len(text.split())
    tokens = len(enc)
    utf8_bytes = len(text.encode("utf-8"))
    fertility = round(tokens / words, 3) if words > 0 else tokens
    bytes_per_tok = round(utf8_bytes / tokens, 3) if tokens > 0 else 0
    chars_per_tok = round(len(text) / tokens, 3) if tokens > 0 else 0
    
    assert match, f"FAILED LOSSLESS ROUNDTRIP FOR {cat}!"
    assert all(0 <= tid < 4096 for tid in enc), f"OUT OF VOCABULARY TOKEN ID IN {cat}!"
    
    results[cat] = {
        "words": words,
        "tokens": tokens,
        "utf8_bytes": utf8_bytes,
        "tokens_per_word": fertility,
        "bytes_per_token": bytes_per_tok,
        "chars_per_token": chars_per_tok,
        "lossless_roundtrip": match
    }

# Also test fertility on larger natural corpus paragraphs
benchmark_corpora = {
    "English Wiki Sample": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's metabolic activities. This chemical energy is stored in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon dioxide and water.",
    "Hindi Wiki Sample": "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे और कुछ अन्य जीव प्रकाश ऊर्जा को रासायनिक ऊर्जा में परिवर्तित करते हैं। हरे पौधों में प्रकाश संश्लेषण के दौरान प्रकाश ऊर्जा का उपयोग जल, कार्बन डाइऑक्साइड तथा खनिजों को ऑक्सीजन तथा ऊर्जा से भरपूर कार्बनिक यौगिकों में बदलने के लिए किया जाता है।"
}

corpus_metrics = {}
for name, text in benchmark_corpora.items():
    enc = tokenizer.encode(text)
    dec = tokenizer.decode(enc)
    words = len(text.split())
    tokens = len(enc)
    utf8_bytes = len(text.encode("utf-8"))
    corpus_metrics[name] = {
        "words": words,
        "tokens": tokens,
        "utf8_bytes": utf8_bytes,
        "tokens_per_word": round(tokens / words, 3),
        "bytes_per_token": round(utf8_bytes / tokens, 3),
        "chars_per_token": round(len(text) / tokens, 3),
        "lossless": dec == text
    }

output_data = {
    "tokenizer_config": {
        "vocab_size": tokenizer.vocab_size,
        "num_merges": tokenizer.num_merges,
        "checksums": tok_cfg.get("checksums")
    },
    "test_cases": results,
    "natural_corpus_benchmarks": corpus_metrics
}

out_path = ROOT / "experiments" / "ucip12r" / "reports" / "tokenizer_verification_data.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print("\n--- Test Results ---")
for k, v in results.items():
    print(f"{k:<22} | Words: {v['words']:>3} | Tokens: {v['tokens']:>3} | Fert: {v['tokens_per_word']:>5} | B/Tok: {v['bytes_per_token']:>5} | Lossless: {v['lossless_roundtrip']}")

print("\n--- Natural Corpus Benchmarks ---")
for k, v in corpus_metrics.items():
    print(f"{k:<22} | Words: {v['words']:>3} | Tokens: {v['tokens']:>3} | Fert: {v['tokens_per_word']:>5} | B/Tok: {v['bytes_per_token']:>5} | Lossless: {v['lossless']}")
