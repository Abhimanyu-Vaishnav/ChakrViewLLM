import sys
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

test_cases = [
    ("English", "User: What does a CPU do?\nAssistant:", " A CPU acts as the brain."),
    ("Devanagari", "User: नमस्ते!\nAssistant:", " नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("Roman Hindi", "User: Subah jaldi uthne ke kya fayde hain?\nAssistant:", " Subah jaldi uthne se mann shaant rehta hai."),
    ("Hinglish", "User: Machine learning ko simple language mein samjhao.\nAssistant:", " Machine learning ek aisi technique hai.")
]

print("=== TOKEN BOUNDARY INVESTIGATION ===")

for lang, p_str, t_str in test_cases:
    # Path A: Separate encoding (Prompt and Target encoded independently)
    p_ids = tokenizer.encode(p_str, add_bos=True, add_eos=False)
    t_ids = tokenizer.encode(t_str, add_bos=False, add_eos=True)
    sep_input = p_ids + t_ids
    
    # Path B: Unified concatenated encoding
    full_str = p_str + t_str
    unified_input = tokenizer.encode(full_str, add_bos=True, add_eos=True)
    
    print(f"\n--- Domain: {lang} ---")
    print(f"Prompt: '{p_str.replace(chr(10), '\\n')}'")
    print(f"Target: '{t_str}'")
    
    match = (sep_input == unified_input)
    print(f"Exact Token Sequence Match? {'YES' if match else 'NO (BOUNDARY DISCREPANCY)'}")
    
    if not match:
        print(f"  Separate Enc (len {len(sep_input)}):   Last 3 of prompt = {p_ids[-3:]}, First 3 of target = {t_ids[:3]}")
        print(f"    Target tid 0 piece: {t_ids[0]} -> {repr(tokenizer.decode([t_ids[0]]))}")
        print(f"  Unified Enc  (len {len(unified_input)}):   Around boundary = {unified_input[len(p_ids)-2:len(p_ids)+3]}")
        for tid in unified_input[len(p_ids)-2:len(p_ids)+3]:
            print(f"    Unified piece: {tid} -> {repr(tokenizer.decode([tid]))}")
    else:
        print(f"  Unified len: {len(unified_input)}")
