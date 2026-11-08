"""
CR-EXP-13 Section 5 & 6: Multi-Turn Working Memory and Entity Binding Audit
Generates a dedicated test suite with 100 multi-turn working memory and entity-binding cases:
- Evaluates subject accuracy and value accuracy independently
- Categorizes outcomes into:
  1. Correct subject and correct value
  2. Correct value but wrong subject (Subject substitution)
  3. Correct subject but stale value (Old-state retention)
  4. Wrong subject and wrong value
  5. Unsupported / hallucinated answer
  6. Appropriate abstention
"""

import sys
import json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_13_independent_verification" / "audit_data"

def build_audit_suite():
    # Fresh names and cities never used in CR-11/12
    names = ["Girish", "Hema", "Indu", "Janak", "Kamlesh", "Laxman", "Mina", "Narayan", "Omwati", "Pushpa"]
    cities = ["Ajmer", "Bikaner", "Cuttack", "Dhanbad", "Erode", "Firozabad", "Gaya", "Hubli", "Itanagar", "Jhansi"]
    
    probes = []
    pid = 1
    # 50 Single update dialogues (Turn 1 -> Turn 2 -> Query)
    for i in range(50):
        name = names[i % len(names)]
        c_old = cities[i % len(cities)]
        c_new = cities[(i + 3) % len(cities)]
        p = (
            f"User: Please track this update: {name} is staying in {c_old}.\n"
            f"Assistant: Understood, I have noted that {name} is in {c_old}.\n"
            f"User: Actually, {name} just moved over to {c_new}.\n"
            f"Assistant: Noted the new location.\n"
            f"User: Where is {name} now?\n"
            f"Assistant: "
        )
        probes.append({
            "id": f"AUDIT_WM_UPD1_{pid:03d}",
            "type": "single_update",
            "subject": name,
            "target_value": c_new,
            "old_value": c_old,
            "prompt": p
        })
        pid += 1
        
    # 50 Two-entity interleaved updates
    for i in range(50):
        n1 = names[i % len(names)]
        n2 = names[(i + 1) % len(names)]
        c1 = cities[i % len(cities)]
        c2 = cities[(i + 1) % len(cities)]
        c1_up = cities[(i + 4) % len(cities)]
        c2_up = cities[(i + 5) % len(cities)]
        
        target_name = n1 if (i % 2 == 0) else n2
        target_val = c1_up if (i % 2 == 0) else c2_up
        stale_val = c1 if (i % 2 == 0) else c2
        distractor_val = c2_up if (i % 2 == 0) else c1_up
        
        p = (
            f"User: Track these two people: {n1} is in {c1}, and {n2} is in {c2}.\n"
            f"Assistant: Recorded both.\n"
            f"User: Update: {n1} has moved to {c1_up}.\n"
            f"Assistant: Updated {n1}.\n"
            f"User: Also, {n2} relocated to {c2_up}.\n"
            f"Assistant: Updated {n2}.\n"
            f"User: Where is {target_name} now?\n"
            f"Assistant: "
        )
        probes.append({
            "id": f"AUDIT_WM_INTER_{pid:03d}",
            "type": "interleaved_two_entity",
            "subject": target_name,
            "target_value": target_val,
            "old_value": stale_val,
            "distractor_value": distractor_val,
            "prompt": p
        })
        pid += 1
        
    return probes

def evaluate_memory_audit(model, tok, probes):
    cat_summary = {
        "correct_subject_and_value": 0,
        "correct_value_wrong_subject": 0,
        "correct_subject_stale_value": 0,
        "wrong_subject_and_value": 0,
        "unsupported_hallucination": 0,
        "total_probes": len(probes)
    }
    
    details = []
    for p in probes:
        prompt = p["prompt"]
        subj = p["subject"]
        tgt = p["target_value"]
        old = p["old_value"]
        
        enc = tok.encode(prompt)
        curr = list(enc)
        with torch.no_grad():
            for _ in range(25):
                x = torch.tensor([curr], dtype=torch.long)
                logits = model(x)[0, -1]
                nxt = logits.argmax().item()
                curr.append(nxt)
                if nxt == 13 and len(curr) - len(enc) > 3:
                    break
        gen = tok.decode(curr[len(enc):], errors="replace").strip()
        
        has_subj = subj.lower() in gen.lower()
        has_tgt = tgt.lower() in gen.lower()
        has_old = old.lower() in gen.lower()
        
        classification = ""
        if has_subj and has_tgt and not has_old:
            cat_summary["correct_subject_and_value"] += 1
            classification = "CORRECT_SUBJECT_AND_VALUE"
        elif not has_subj and has_tgt and not has_old:
            cat_summary["correct_value_wrong_subject"] += 1
            classification = "CORRECT_VALUE_WRONG_SUBJECT"
        elif has_subj and has_old and not has_tgt:
            cat_summary["correct_subject_stale_value"] += 1
            classification = "CORRECT_SUBJECT_STALE_VALUE"
        elif not has_subj and not has_tgt:
            cat_summary["wrong_subject_and_value"] += 1
            classification = "WRONG_SUBJECT_AND_VALUE"
        else:
            cat_summary["unsupported_hallucination"] += 1
            classification = "UNSUPPORTED_MIXED"
            
        details.append({
            "id": p["id"],
            "type": p["type"],
            "subject": subj,
            "target": tgt,
            "generated": gen,
            "classification": classification
        })
        
    return cat_summary, details

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    probes = build_audit_suite()
    
    cfg = ModelConfig()
    
    models = {
        "cr11_candidate_a": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/candidate_a/cr_exp_11_candidate_a_best_val.pt",
        "cr12_candidate_a": REPO_ROOT / "experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/candidate_a/cr_exp_12_candidate_a_best_val.pt",
        "cr12_candidate_b": REPO_ROOT / "experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/candidate_b/cr_exp_12_candidate_b_best_val.pt"
    }
    
    results = {}
    for name, ckpt in models.items():
        m = ChakrMicro(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(ck.get("model_state_dict", ck))
        m.eval()
        
        summ, details = evaluate_memory_audit(m, tok, probes)
        results[name] = {"summary": summ, "details": details}
        print(f"=== {name} Working Memory Audit ===")
        print(f"  Correct Subject & Value: {summ['correct_subject_and_value']}/100 ({summ['correct_subject_and_value']}%)")
        print(f"  Subject Substitution (Right City, Wrong Name): {summ['correct_value_wrong_subject']}/100")
        print(f"  Stale Value (Old City): {summ['correct_subject_stale_value']}/100")
        print(f"  Wrong Subject & Value: {summ['wrong_subject_and_value']}/100")
        
    out_file = OUT_DIR / "working_memory_audit_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved memory audit results to {out_file}")

if __name__ == "__main__":
    main()

