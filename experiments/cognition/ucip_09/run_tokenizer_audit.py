import sys
import json
import re
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

# Multilingual benchmark corpus for tokenization audit
CORPUS = {
    "English": [
        "The quick brown fox jumps over the lazy dog.",
        "A central processing unit executes instructions using the fetch-decode-execute cycle.",
        "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients.",
        "Artificial intelligence and machine learning enable computers to learn from empirical data.",
        "Good morning! I am doing well and ready to assist you with your questions today."
    ],
    "Devanagari_Hindi": [
        "नमस्ते! मैं चक्रव्यूह हूँ, आपका कृत्रिम बुद्धिमत्ता सहायक।",
        "सूर्य का प्रकाश पेड़-पौधों में प्रकाश संश्लेषण के लिए अत्यंत आवश्यक है।",
        "कंप्यूटर में सीपीयू मुख्य मस्तिष्क की तरह कार्य करता है।",
        "किताबें पढ़ने से सोचने की क्षमता और एकाग्रता बढ़ती है।",
        "भारत की राजधानी नई दिल्ली है और यहाँ विभिन्न संस्कृतियों का संगम है।"
    ],
    "Roman_Hindi": [
        "Namaste! Main ChakrView hoon, aapka AI assistant.",
        "Subah jaldi uthne se mann shaant rehta hai aur dinbhar energy milti hai.",
        "Machine learning ek aisi technique hai jisme computer data se seekhta hai.",
        "Unit testing karne se code maintainable aur bug-free rehta hai.",
        "Agar main aapko 5 apples doon aur 2 wapas le loon to kitne bachenge?"
    ],
    "Hinglish": [
        "Python language itni popular kyu hai software development mein?",
        "Cache memory RAM se fast hoti hai because ye CPU chip ke paas hoti hai.",
        "Neural networks deep learning model architecture ko train karte hain.",
        "Bhai, database index create karne se query speed kaafi badh jaati hai.",
        "Code refactoring se software clean aur scalable banta hai bina functionality change kiye."
    ],
    "Technical_Arithmetic": [
        "15 + 7 = 22",
        "100 - 45 = 55",
        "def compute_hash(data: bytes) -> str: return hashlib.sha256(data).hexdigest()",
        "SELECT user_id, count(*) FROM sessions WHERE status = 'active' GROUP BY user_id;",
        "Memory footprint: 51.8 MB RSS, latency 7.2 ms per token."
    ]
}

results = {}
print("=== CHAKRVIEW BPE TOKENIZER AUDIT (VOCAB 4096) ===")

for lang, samples in CORPUS.items():
    total_chars = 0
    total_words = 0
    total_tokens = 0
    roundtrip_failures = 0
    
    for s in samples:
        tokens = tokenizer.encode(s, add_bos=False, add_eos=False)
        decoded = tokenizer.decode(tokens, errors="strict")
        if decoded != s:
            roundtrip_failures += 1
            
        words = len(s.split())
        chars = len(s)
        t_count = len(tokens)
        
        total_words += words
        total_chars += chars
        total_tokens += t_count
        
    tokens_per_word = total_tokens / total_words
    tokens_per_char = total_tokens / total_chars
    
    results[lang] = {
        "samples": len(samples),
        "total_chars": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_word": round(tokens_per_word, 2),
        "tokens_per_char": round(tokens_per_char, 2),
        "roundtrip_fidelity": "100%" if roundtrip_failures == 0 else f"{roundtrip_failures} failed"
    }
    print(f"{lang:<22} | Tokens/Word: {tokens_per_word:>4.2f} | Tokens/Char: {tokens_per_char:>4.2f} | Fidelity: {results[lang]['roundtrip_fidelity']}")

out_file = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "tokenizer_audit_stats.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
