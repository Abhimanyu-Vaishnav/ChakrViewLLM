"""
CR-EXP-11 Phase 3: Tokenizer and Entity Representation Diagnosis
Tests whether unfamiliar entity names create a representation bottleneck.
Analyzes:
- Familiar Indian names vs. Unseen Indian names
- Short vs. Long names
- Names with uncommon character combinations
- Capitalization variants
- Token IDs, token counts, subword fragments, roundtrip reconstruction
- Byte fallback behavior
- Correlation between token fragmentation and model errors
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_11_entity_binding"

def diagnose_names():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    test_names = [
        # Familiar training names
        ("Aarav", "familiar"), ("Meera", "familiar"), ("Kabir", "familiar"), ("Neha", "familiar"),
        ("Rohan", "familiar"), ("Ananya", "familiar"), ("Priya", "familiar"), ("Arjun", "familiar"),
        # Common unseen Indian names
        ("Kavita", "unseen_common"), ("Manish", "unseen_common"), ("Siddharth", "unseen_common"), ("Divya", "unseen_common"),
        ("Tanvi", "unseen_common"), ("Gaurav", "unseen_common"), ("Deepak", "unseen_common"), ("Lakshmi", "unseen_common"),
        # Short names
        ("Dev", "short"), ("Uma", "short"), ("Jay", "short"), ("Om", "short"), ("Ali", "short"),
        # Long / Compound names
        ("Harshvardhan", "long"), ("Chandrashekhar", "long"), ("Madhusudhan", "long"), ("Bhubaneshwar", "long"),
        # Uncommon character combinations / Regional names
        ("Jyotsna", "uncommon"), ("Yashwant", "uncommon"), ("Mridula", "uncommon"), ("Trimbak", "uncommon"),
        # Case variations
        ("aarav", "lowercase"), ("meera", "lowercase"), ("MEERA", "uppercase"), ("AARAV", "uppercase")
    ]
    
    results = []
    for name, group in test_names:
        ids = tok.encode(name)
        # Check roundtrip
        try:
            decoded = tok.decode(ids, errors="replace")
        except Exception:
            decoded = "".join([tok.decode([tid], errors="replace") for tid in ids])
            
        preserved = (decoded == name)
        
        # Check subword pieces
        pieces = []
        for tid in ids:
            try:
                piece_str = tok.decode([tid], errors="replace")
            except Exception:
                piece_str = f"[{tid}]"
            pieces.append((tid, piece_str))
            
        results.append({
            "name": name,
            "group": group,
            "token_ids": ids,
            "num_tokens": len(ids),
            "pieces": pieces,
            "decoded": decoded,
            "roundtrip_preserved": preserved
        })
        
    return results

def main():
    res = diagnose_names()
    out_file = OUT_DIR / "entity_tokenization_diagnostic.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved entity tokenization analysis to {out_file}")
    
    # Print summary
    print("\n%-18s | %-14s | %-10s | %s" % ("Name", "Group", "Num Tokens", "Fragments"))
    print("-" * 65)
    for r in res:
        frags = " + ".join([repr(p[1]) for p in r["pieces"]])
        print("%-18s | %-14s | %-10d | %s" % (r["name"], r["group"], r["num_tokens"], frags))

if __name__ == "__main__":
    main()

