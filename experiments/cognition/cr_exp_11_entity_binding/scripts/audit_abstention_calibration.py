"""
CR-EXP-11 Phase 2: Mandatory Balanced Answerability & Epistemic Calibration Audit
Evaluates models on a balanced 200-probe suite:
- 100 Answerable queries (entity facts explicitly present in context)
- 100 Unanswerable queries (unmentioned entities or unmentioned attributes)

Calculates:
- Correct answers on answerable
- Correct abstentions on unanswerable
- False abstentions
- Hallucinated answers
- Answer coverage
- Selective accuracy
- Abstention precision & recall
- Confusion matrix
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
OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_11_entity_binding"

def build_balanced_calibration_suite():
    # Disjoint names and cities
    names_answerable = [
        "Aarav", "Meera", "Kabir", "Neha", "Rohan", "Ananya", "Vikram", "Pooja", "Aditya", "Sneha",
        "Kunal", "Tanvi", "Siddharth", "Ishita", "Varun", "Riya", "Gaurav", "Kavya", "Deepak", "Simran",
        "Naveen", "Swati", "Rahul", "Divya", "Amit", "Shreya", "Manish", "Preeti", "Sanjay", "Ankita",
        "Alok", "Priya", "Harsh", "Bhavna", "Mohit", "Jyoti", "Rajesh", "Nandini", "Karan", "Payal",
        "Sachin", "Archana", "Mayank", "Pallavi", "Tarun", "Shikha", "Ashok", "Komal", "Sunil", "Monika"
    ]
    cities = [
        "Jaipur", "Pune", "Lucknow", "Bhopal", "Indore", "Surat", "Patna", "Nagpur", "Kochi", "Agra",
        "Varanasi", "Amritsar", "Chandigarh", "Madurai", "Mysuru", "Udaipur", "Ranchi", "Guwahati", "Gwalior", "Jodhpur"
    ]
    professions = [
        "architect", "doctor", "engineer", "teacher", "lawyer", "scientist", "artist", "pilot", "chef", "geologist"
    ]
    
    probes = []
    pid = 1
    # 50 answerable location queries
    for i in range(50):
        name = names_answerable[i]
        city = cities[i % len(cities)]
        probes.append({
            "id": f"CALIB_ANS_LOC_{pid:03d}",
            "answerable": True,
            "type": "location",
            "entity": name,
            "target": city,
            "prompt": f"Context: {name} lives in {city}.\nQuestion: Where does {name} live?\nAnswer: "
        })
        pid += 1
        
    # 50 answerable profession queries
    for i in range(50):
        name = names_answerable[i]
        prof = professions[i % len(professions)]
        probes.append({
            "id": f"CALIB_ANS_PROF_{pid:03d}",
            "answerable": True,
            "type": "profession",
            "entity": name,
            "target": prof,
            "prompt": f"Context: {name} works as a {prof}.\nQuestion: What is {name}'s profession?\nAnswer: "
        })
        pid += 1
        
    # 50 unanswerable queries: Entity unmentioned
    unmentioned_names = [
        "Dev", "Kiran", "Tara", "Nikhil", "Suman", "Vikas", "Rekha", "Abhay", "Sarita", "Hemant",
        "Lalit", "Vandana", "Girish", "Asha", "Pradeep", "Geeta", "Sunita", "Manoj", "Usha", "Arvind",
        "Vidya", "Chetan", "Madhu", "Bhavesh", "Kavita", "Ramesh", "Seema", "Dinesh", "Rashmi", "Jatin",
        "Shalini", "Mahesh", "Sangeeta", "Mukesh", "Ritu", "Naresh", "Anita", "Pankaj", "Sunita", "Yogesh",
        "Aarti", "Satish", "Neelam", "Kamal", "Chitra", "Bharat", "Sunil", "Sapna", "Vinay", "Meena"
    ]
    for i in range(50):
        context_name = names_answerable[i]
        target_name = unmentioned_names[i]
        city = cities[(i + 3) % len(cities)]
        probes.append({
            "id": f"CALIB_UNANS_ENT_{pid:03d}",
            "answerable": False,
            "type": "unmentioned_entity",
            "entity": target_name,
            "prompt": f"Context: {context_name} lives in {city}.\nQuestion: Where does {target_name} live?\nAnswer: "
        })
        pid += 1
        
    # 50 unanswerable queries: Unmentioned attribute (context gives city, question asks profession)
    for i in range(50):
        name = names_answerable[i]
        city = cities[(i + 5) % len(cities)]
        probes.append({
            "id": f"CALIB_UNANS_ATTR_{pid:03d}",
            "answerable": False,
            "type": "unmentioned_attribute",
            "entity": name,
            "prompt": f"Context: {name} lives in {city}.\nQuestion: What is {name}'s profession?\nAnswer: "
        })
        pid += 1
        
    return probes

def evaluate_calibration(model, tok, probes):
    abstain_cues = ["does not", "not specify", "unknown", "unspecified", "cannot", "no information", "not state", "unmentioned", "not mention"]
    
    tp_ans = 0 # answered and correct
    fn_ans = 0 # answerable but abstained (false abstention)
    err_ans = 0 # answered but incorrect (hallucination on answerable)
    
    tn_unans = 0 # unanswerable and correctly abstained
    fp_unans = 0 # unanswerable but answered (hallucination on unanswerable)
    
    details = []
    
    for p in probes:
        prompt = p["prompt"]
        is_ans = p["answerable"]
        enc = tok.encode(prompt)
        curr = list(enc)
        with torch.no_grad():
            for _ in range(25):
                x = torch.tensor([curr], dtype=torch.long)
                logits = model(x)[0, -1]
                nxt = logits.argmax().item()
                curr.append(nxt)
                if nxt == 13 and len(curr) - len(enc) > 5:
                    break
        gen = tok.decode(curr[len(enc):], errors="replace").strip()
        abstained = any(cue in gen.lower() for cue in abstain_cues)
        
        status = ""
        if is_ans:
            target = p["target"]
            correct = target.lower() in gen.lower()
            if abstained:
                fn_ans += 1
                status = "FALSE_ABSTENTION"
            elif correct:
                tp_ans += 1
                status = "CORRECT_ANSWER"
            else:
                err_ans += 1
                status = "WRONG_ANSWER"
        else:
            if abstained:
                tn_unans += 1
                status = "CORRECT_ABSTENTION"
            else:
                fp_unans += 1
                status = "HALLUCINATED_ANSWER"
                
        details.append({
            "id": p["id"],
            "type": p["type"],
            "answerable": is_ans,
            "prompt": prompt,
            "generated": gen,
            "status": status,
            "abstained": abstained
        })
        
    # Metrics
    tot_ans = 100
    tot_unans = 100
    
    coverage = (tot_ans - fn_ans) / tot_ans * 100.0
    ans_acc = tp_ans / tot_ans * 100.0
    selective_acc = (tp_ans / max(1, (tot_ans - fn_ans))) * 100.0
    
    abstain_recall = tn_unans / tot_unans * 100.0
    total_abstained = fn_ans + tn_unans
    abstain_prec = (tn_unans / max(1, total_abstained)) * 100.0
    
    hallucination_unans_rate = fp_unans / tot_unans * 100.0
    
    return {
        "confusion_matrix": {
            "answerable_correct": tp_ans,
            "answerable_false_abstention": fn_ans,
            "answerable_wrong_answer": err_ans,
            "unanswerable_correct_abstention": tn_unans,
            "unanswerable_hallucination": fp_unans
        },
        "metrics": {
            "answerable_accuracy_pct": round(ans_acc, 2),
            "answer_coverage_pct": round(coverage, 2),
            "selective_accuracy_pct": round(selective_acc, 2),
            "correct_abstention_pct": round(abstain_recall, 2),
            "abstention_precision_pct": round(abstain_prec, 2),
            "hallucination_on_unanswerable_pct": round(hallucination_unans_rate, 2),
            "overall_discrimination_acc_pct": round((tp_ans + tn_unans) / 200.0 * 100.0, 2)
        },
        "details": details
    }

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    probes = build_balanced_calibration_suite()
    
    models = {
        "cr09_candidate_3": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_cand3/cr_exp_09_candidate_3_best_val_copy.pt",
        "cr10_arm_a": REPO_ROOT / "experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_a/cr_exp_10_arm_a_best_val.pt",
        "cr10_arm_b": REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_arm_b/cr_exp_10_arm_b_best_val_copy.pt"
    }
    
    results = {}
    for name, ckpt in models.items():
        print(f"Evaluating calibration on {name}...")
        cfg = ModelConfig()
        m = ChakrMicro(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(ck.get("model_state_dict", ck))
        m.eval()
        
        res = evaluate_calibration(m, tok, probes)
        results[name] = res
        print(f"[{name}] Discrimination: {res['metrics']['overall_discrimination_acc_pct']}%, Ans Acc: {res['metrics']['answerable_accuracy_pct']}%, Correct Abstain: {res['metrics']['correct_abstention_pct']}%, False Abstain: {res['confusion_matrix']['answerable_false_abstention']}/100")
        
    out_file = OUT_DIR / "calibration_audit_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved calibration audit results to {out_file}")

if __name__ == "__main__":
    main()

