"""
UCIP-12 Tokenizer Fertility and Representation Audit Script.
Evaluates tokenization fertility, word fragmentation, and compression efficiency across
English, Hindi, Sanskrit, Code, Hinglish, and Mathematics.
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

test_samples = {
    "english": [
        "Artificial intelligence is a branch of computer science focused on building smart machines.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Photosynthesis is the process used by plants, algae and certain bacteria to turn sunlight into chemical energy.",
        "Neural language models learn statistical representations of grammar and semantics from massive corpora."
    ],
    "hindi": [
        "कृत्रिम बुद्धिमत्ता कंप्यूटर विज्ञान की एक शाखा है जो बुद्धिमान मशीनों के निर्माण पर केंद्रित है।",
        "भारत एक विविध और प्राचीन संस्कृति वाला देश है जहां कई भाषाएं बोली जाती हैं।",
        "विज्ञान और प्रौद्योगिकी ने मानव जीवन को बहुत आसान और सुविधाजनक बना दिया है।",
        "भाषा अधिग्रहण तंत्रिका नेटवर्क के लिए एक महत्वपूर्ण चुनौती है।"
    ],
    "sanskrit": [
        "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः।",
        "धर्मो रक्षति रक्षितः तस्माद् धर्मं न त्यजेत्।",
        "विद्या ददाति विनयं विनयाद् याति पात्रताम्।",
        "वसुधैव कुटुम्बकम् उदारचरितानां तु वसुधैव कुटुम्बकम्।"
    ],
    "code": [
        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "class NeuralNetwork(nn.Module):\n    def __init__(self, vocab_size, d_model):\n        super().__init__()",
        "import os\nimport sys\nfrom pathlib import Path\nprint('System ready')"
    ],
    "hinglish": [
        "Bhai kal ka match dekh ke maza aa gaya, kya mast bowling ki thi.",
        "Main abhi market ja raha hoon, kuch samaan lana hai kya?",
        "Mujhe lagta hai ki AI models ko Hindi aur English dono samajh aani chahiye.",
        "Aapka model bahut accha perform kar raha hai training ke baad."
    ],
    "mathematics": [
        "Let f(x) = 3x^2 + 2x - 5. Find the derivative f'(x) and its roots.",
        "Solve the system of linear equations: 2x + 3y = 12 and x - y = 1.",
        "The integral of 1/x with respect to x is ln|x| + C.",
        "If a matrix A is orthogonal, then A^T * A = I."
    ]
}

fertility_results = {}
for domain, texts in test_samples.items():
    tot_chars = sum(len(t) for t in texts)
    tot_words = sum(len(t.split()) for t in texts)
    byte_lens = sum(len(t.encode("utf-8")) for t in texts)
    
    encoded_list = [tokenizer.encode(t) for t in texts]
    tot_tokens = sum(len(enc) for enc in encoded_list)
    
    # Check lossless roundtrip
    roundtrip_ok = all(tokenizer.decode(enc) == t for enc, t in zip(encoded_list, texts))
    
    fertility_results[domain] = {
        "sample_count": len(texts),
        "total_chars": tot_chars,
        "total_words": tot_words,
        "total_utf8_bytes": byte_lens,
        "total_tokens": tot_tokens,
        "tokens_per_word": round(tot_tokens / tot_words, 3),
        "chars_per_token": round(tot_chars / tot_tokens, 3),
        "bytes_per_token": round(byte_lens / tot_tokens, 3),
        "lossless_roundtrip": roundtrip_ok
    }

output_path = ROOT / "experiments" / "ucip12" / "manifests" / "tokenizer_fertility_audit.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "vocab_size": tokenizer.vocab_size,
        "num_merges": tokenizer.num_merges,
        "checksums": tok_cfg.get("checksums", {}),
        "domains": fertility_results
    }, f, indent=2)

print(f"Saved fertility audit to {output_path}")
print(json.dumps(fertility_results, indent=2))
