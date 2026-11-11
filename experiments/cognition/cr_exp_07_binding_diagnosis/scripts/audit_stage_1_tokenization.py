"""
Stage 1: Tokenization and Controlled Input Design Script
Constructs controlled relational prompt pairs across:
- Primary baseline, counterfactual value, subject substitution, multiple associations,
  reassignment, distractor, contradiction, missing info, English, and Hinglish.
Audits:
- Token IDs, token pieces, subject positions, value positions, query positions, delimiter positions,
  entity token lengths, single vs multi-token entity groups.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

CONTROLLED_PROMPTS = [
    {
        "id": "P01_PRIMARY",
        "description": "Primary relational statement",
        "text": "Mapping: Ravi lives in Delhi. Query: Where does Ravi live? Answer:",
        "subject": "Ravi",
        "value": "Delhi",
        "query_subject": "Ravi",
        "group": "A_single_token_entities"
    },
    {
        "id": "P02_COUNTERFACTUAL_VAL",
        "description": "Counterfactual value variation (Ravi lives in Jaipur)",
        "text": "Mapping: Ravi lives in Jaipur. Query: Where does Ravi live? Answer:",
        "subject": "Ravi",
        "value": "Jaipur",
        "query_subject": "Ravi",
        "group": "A_single_token_entities"
    },
    {
        "id": "P03_SUBJECT_SUBST",
        "description": "Subject substitution (Aman lives in Delhi)",
        "text": "Mapping: Aman lives in Delhi. Query: Where does Aman live? Answer:",
        "subject": "Aman",
        "value": "Delhi",
        "query_subject": "Aman",
        "group": "A_single_token_entities"
    },
    {
        "id": "P04_MULTIPLE_ASSOC",
        "description": "Multiple associations in context (Ravi -> Delhi, Aman -> Jaipur)",
        "text": "Mapping: Ravi lives in Delhi. Aman lives in Jaipur. Query: Where does Ravi live? Answer:",
        "subject": "Ravi",
        "value": "Delhi",
        "query_subject": "Ravi",
        "group": "A_single_token_entities"
    },
    {
        "id": "P05_REASSIGNMENT",
        "description": "Reassigned mapping across subjects (Ravi -> Jaipur, Aman -> Delhi)",
        "text": "Mapping: Ravi lives in Jaipur. Aman lives in Delhi. Query: Where does Ravi live? Answer:",
        "subject": "Ravi",
        "value": "Jaipur",
        "query_subject": "Ravi",
        "group": "A_single_token_entities"
    },
    {
        "id": "P06_DISTRACTOR",
        "description": "Unrelated distractor injection (Karan in Mumbai)",
        "text": "Mapping: Ravi lives in Delhi. Aman lives in Jaipur. Karan lives in Mumbai. Query: Where does Ravi live? Answer:",
        "subject": "Ravi",
        "value": "Delhi",
        "query_subject": "Ravi",
        "group": "A_single_token_entities"
    },
    {
        "id": "P07_CONTRADICTION_UPDATE",
        "description": "Sequential contradiction/update (Ravi in Delhi, later update: now lives in Jaipur)",
        "text": "Mapping: Ravi lives in Delhi. Later update: Ravi now lives in Jaipur. Query: Where does Ravi live? Answer:",
        "subject": "Ravi",
        "value": "Jaipur",
        "query_subject": "Ravi",
        "group": "A_single_token_entities"
    },
    {
        "id": "P08_MISSING_INFO",
        "description": "Target subject not in context (Aman lives in Jaipur; query: Where does Ravi live?)",
        "text": "Mapping: Aman lives in Jaipur. Query: Where does Ravi live? Answer:",
        "subject": "Aman",
        "value": "Jaipur",
        "query_subject": "Ravi",
        "group": "A_single_token_entities"
    },
    # Multi-token entity groups
    {
        "id": "P09_MULTITOK_ENTITIES",
        "description": "Multi-token entities (Vikramaditya lives in Thiruvananthapuram)",
        "text": "Mapping: Vikramaditya lives in Thiruvananthapuram. Query: Where does Vikramaditya live? Answer:",
        "subject": "Vikramaditya",
        "value": "Thiruvananthapuram",
        "query_subject": "Vikramaditya",
        "group": "B_multi_token_entities"
    },
    # Hinglish variation
    {
        "id": "P10_HINGLISH",
        "description": "Equivalent Hinglish phrasing",
        "text": "Mapping: Ravi Delhi me rehta hai. Query: Ravi kahan rehta hai? Answer:",
        "subject": "Ravi",
        "value": "Delhi",
        "query_subject": "Ravi",
        "group": "C_hinglish"
    }
]

def locate_subsequence(full_tokens, sub_tokens):
    n = len(sub_tokens)
    for i in range(len(full_tokens) - n + 1):
        if full_tokens[i:i+n] == sub_tokens:
            return list(range(i, i+n))
    return []

def main():
    print("=== CR-EXP-07 Stage 1: Tokenization & Controlled Input Analysis ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    analysis_results = []

    for item in CONTROLLED_PROMPTS:
        text = item["text"]
        tokens = tok.encode(text)
        pieces = [tok.decode([tid], errors="replace") for tid in tokens]

        # Locate delimiter "Answer:" or "Answer"
        ans_sub = tok.encode("Answer:")
        ans_positions = locate_subsequence(tokens, ans_sub)
        delimiter_pos = ans_positions[-1] if ans_positions else len(tokens) - 1

        # Locate context subject (first occurrence)
        subj_sub = tok.encode(" " + item["subject"])
        if not locate_subsequence(tokens, subj_sub):
            subj_sub = tok.encode(item["subject"])
        ctx_subj_pos = locate_subsequence(tokens, subj_sub)

        # Locate context value
        val_sub = tok.encode(" " + item["value"])
        if not locate_subsequence(tokens, val_sub):
            val_sub = tok.encode(item["value"])
        ctx_val_pos = locate_subsequence(tokens, val_sub)

        # Locate query subject (second occurrence or occurrence after "Query:")
        all_subj_pos = []
        for i in range(len(tokens) - len(subj_sub) + 1):
            if tokens[i:i+len(subj_sub)] == subj_sub:
                all_subj_pos.append(list(range(i, i+len(subj_sub))))
        
        query_subj_pos = all_subj_pos[-1] if len(all_subj_pos) > 1 else (all_subj_pos[0] if all_subj_pos else [])

        analysis_results.append({
            "prompt_id": item["id"],
            "group": item["group"],
            "description": item["description"],
            "raw_text": text,
            "total_token_count": len(tokens),
            "token_ids": tokens,
            "token_pieces": pieces,
            "subject_entity": item["subject"],
            "subject_token_ids": subj_sub,
            "subject_is_single_token": len(subj_sub) == 1,
            "context_subject_token_positions": ctx_subj_pos,
            "value_entity": item["value"],
            "value_token_ids": val_sub,
            "value_is_single_token": len(val_sub) == 1,
            "context_value_token_positions": ctx_val_pos,
            "query_subject_token_positions": query_subj_pos,
            "answer_delimiter_token_position": delimiter_pos
        })
        print(f"[{item['id']:22s}] Tokens={len(tokens):2d} | Subj={item['subject']} ({len(subj_sub)} tok) | Val={item['value']} ({len(val_sub)} tok) | DelimPos={delimiter_pos}")

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_07_binding_diagnosis/tokenization_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nTokenization analysis written to {out_file}")

if __name__ == "__main__":
    main()

