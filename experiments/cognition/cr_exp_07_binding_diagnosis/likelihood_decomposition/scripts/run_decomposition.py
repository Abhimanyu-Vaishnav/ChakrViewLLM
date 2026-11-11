"""
CR-EXP-07.2: Master Likelihood Decomposition Diagnostic Script
Executes Stages 1 through 7:
1. Verifies checkpoint identities.
2. Constructs matched counterfactual & neutral prompt manifests across subjects and entity loads.
3. Computes complete concatenated candidate tokenizations.
4. Scores candidates with teacher-forced conditional log probabilities across all 4 checkpoints.
5. Decomposes contextual likelihood lift vs neutral context.
6. Computes symmetric assignment-following scores across all city pairs.
7. Measures static candidate preference and length normalization.
8. Evaluates subject robustness (Ravi, Aman, Karan, unseen subjects Vikram, Zoya) and distractors (2-entity, 3-entity).
9. Compares ordinary inference likelihoods against causal activation interventions.
Outputs all structured JSON deliverables.
"""

import sys
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
OUTPUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_07_binding_diagnosis" / "likelihood_decomposition"

CHECKPOINTS = {
    "parent": REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
    "pilot": REPO_ROOT / "experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt",
    "control_arm": REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt",
    "treatment_arm": REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/treatment_arm/checkpoints/treatment_arm_best_val.pt"
}

CANDIDATE_CITIES = ["Delhi", "Jaipur", "Mumbai", "Chennai", "Kolkata"]
SUBJECTS = ["Ravi", "Aman", "Karan", "Vikram", "Zoya"]

def load_model(ckpt_path):
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    m.eval()
    return m

def score_candidate(model, tok, prompt_text, candidate_str):
    enc_prompt = tok.encode(prompt_text)
    full_str = prompt_text + candidate_str
    enc_full = tok.encode(full_str)
    cand_ids = enc_full[len(enc_prompt):]
    cand_tokens = [tok.decode([tid]) for tid in cand_ids]
    
    curr = list(enc_prompt)
    token_lps = []
    token_probs = []
    token_ranks = []
    
    for tid in cand_ids:
        x = torch.tensor([curr], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0, -1]
            lp = F.log_softmax(logits, dim=-1)
            p = F.softmax(logits, dim=-1)
            
            token_lps.append(lp[tid].item())
            token_probs.append(p[tid].item())
            token_ranks.append((torch.argsort(logits, descending=True) == tid).nonzero().item() + 1)
        curr.append(tid)
        
    tot_lp = sum(token_lps)
    mean_lp = tot_lp / len(token_lps) if token_lps else 0.0
    
    return {
        "candidate": candidate_str,
        "token_ids": cand_ids,
        "token_strings": cand_tokens,
        "token_count": len(cand_ids),
        "per_token_log_probs": [round(val, 4) for val in token_lps],
        "per_token_probs": [round(val, 6) for val in token_probs],
        "per_token_ranks": token_ranks,
        "total_log_prob": round(tot_lp, 4),
        "mean_token_log_prob": round(mean_lp, 4),
        "first_token_log_prob": round(token_lps[0], 4) if token_lps else None,
        "first_token_prob": round(token_probs[0], 6) if token_probs else None,
        "first_token_rank": token_ranks[0] if token_ranks else None
    }

def main():
    print("=== CR-EXP-07.2: Contextual Likelihood Decomposition Execution ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    # 1. Preflight hash verification
    pre_hashes = {}
    for name, path in CHECKPOINTS.items():
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        pre_hashes[name] = h
        print(f"Verified {name:15s}: {h[:16]}...")
        
    # 2. Build candidate tokenization ground truth
    print("\nExtracting concatenated candidate tokenizations...")
    cand_tok_data = {}
    base_prompt = "Mapping: Ravi lives in a city. Query: Where does Ravi live? Answer: "
    enc_base = tok.encode(base_prompt)
    for c in CANDIDATE_CITIES:
        full = base_prompt + c
        enc_full = tok.encode(full)
        c_ids = enc_full[len(enc_base):]
        cand_tok_data[c] = {
            "token_ids": c_ids,
            "token_strings": [tok.decode([tid]) for tid in c_ids],
            "token_count": len(c_ids),
            "standalone_token_ids": tok.encode(c),
            "standalone_token_strings": [tok.decode([tid]) for tid in tok.encode(c)],
            "concatenated_matches_standalone": (c_ids == tok.encode(c))
        }
    with open(OUTPUT_DIR / "candidate_tokenization.json", "w", encoding="utf-8") as f:
        json.dump(cand_tok_data, f, indent=2)
    print("Saved candidate_tokenization.json")

    # 3. Build matched prompt manifests
    print("\nBuilding matched prompt manifests...")
    manifest = {
        "canonical_counterfactual_prompts": {},
        "neutral_prompts": {},
        "subject_robustness_prompts": {},
        "distractor_prompts": {}
    }
    
    for c in ["Delhi", "Jaipur", "Mumbai", "Chennai"]:
        manifest["canonical_counterfactual_prompts"][c] = f"Mapping: Ravi lives in {c}. Query: Where does Ravi live? Answer: "
    manifest["neutral_prompts"]["Ravi"] = "Mapping: Ravi lives in a city. Query: Where does Ravi live? Answer: "

    for s in SUBJECTS:
        manifest["neutral_prompts"][s] = f"Mapping: {s} lives in a city. Query: Where does {s} live? Answer: "
        manifest["subject_robustness_prompts"][s] = {}
        for c in ["Delhi", "Jaipur", "Mumbai"]:
            manifest["subject_robustness_prompts"][s][c] = f"Mapping: {s} lives in {c}. Query: Where does {s} live? Answer: "

    manifest["distractor_prompts"]["two_entity_Ravi_Delhi_Priya_Jaipur"] = "Mapping: Ravi lives in Delhi. Priya lives in Jaipur. Query: Where does Ravi live? Answer: "
    manifest["distractor_prompts"]["two_entity_Ravi_Jaipur_Priya_Delhi"] = "Mapping: Ravi lives in Jaipur. Priya lives in Delhi. Query: Where does Ravi live? Answer: "
    manifest["distractor_prompts"]["three_entity_Ravi_Delhi_Priya_Jaipur_Aman_Mumbai"] = "Mapping: Ravi lives in Delhi. Priya lives in Jaipur. Aman lives in Mumbai. Query: Where does Ravi live? Answer: "
    manifest["distractor_prompts"]["three_entity_Ravi_Mumbai_Priya_Delhi_Aman_Jaipur"] = "Mapping: Ravi lives in Mumbai. Priya lives in Delhi. Aman lives in Jaipur. Query: Where does Ravi live? Answer: "

    with open(OUTPUT_DIR / "matched_prompt_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved matched_prompt_manifest.json")

    # 4. Score all conditions across checkpoints
    print("\nScoring candidate sequences across all 4 checkpoints...")
    all_scores = {}
    lifts_data = {}
    assignment_data = {}
    static_pref_data = {}

    for m_name, m_path in CHECKPOINTS.items():
        print(f"\nProcessing model: {m_name}")
        model = load_model(m_path)
        all_scores[m_name] = {
            "canonical_counterfactuals": {},
            "neutral": {},
            "subjects": {},
            "distractors": {}
        }

        # Score neutral prompt
        neutral_p = manifest["neutral_prompts"]["Ravi"]
        neutral_scores = {}
        for cand in CANDIDATE_CITIES:
            neutral_scores[cand] = score_candidate(model, tok, neutral_p, cand)
        all_scores[m_name]["neutral"]["Ravi"] = neutral_scores

        # Static preference analysis under neutral prompt
        sorted_neutral_total = sorted(neutral_scores.items(), key=lambda x: x[1]["total_log_prob"], reverse=True)
        sorted_neutral_mean = sorted(neutral_scores.items(), key=lambda x: x[1]["mean_token_log_prob"], reverse=True)
        sorted_neutral_first = sorted(neutral_scores.items(), key=lambda x: x[1]["first_token_log_prob"], reverse=True)

        static_pref_data[m_name] = {
            "neutral_prompt": neutral_p,
            "ranked_by_total_log_prob": [{"candidate": k, "total_lp": v["total_log_prob"], "token_count": v["token_count"]} for k, v in sorted_neutral_total],
            "ranked_by_mean_log_prob": [{"candidate": k, "mean_lp": v["mean_token_log_prob"], "token_count": v["token_count"]} for k, v in sorted_neutral_mean],
            "ranked_by_first_token_log_prob": [{"candidate": k, "first_token_lp": v["first_token_log_prob"], "first_token_prob": v["first_token_prob"]} for k, v in sorted_neutral_first],
            "total_lp_gap_delhi_vs_jaipur": round(neutral_scores["Delhi"]["total_log_prob"] - neutral_scores["Jaipur"]["total_log_prob"], 4),
            "total_lp_gap_delhi_vs_mumbai": round(neutral_scores["Delhi"]["total_log_prob"] - neutral_scores["Mumbai"]["total_log_prob"], 4),
            "mean_lp_gap_delhi_vs_jaipur": round(neutral_scores["Delhi"]["mean_token_log_prob"] - neutral_scores["Jaipur"]["mean_token_log_prob"], 4),
            "mean_lp_gap_delhi_vs_mumbai": round(neutral_scores["Delhi"]["mean_token_log_prob"] - neutral_scores["Mumbai"]["mean_token_log_prob"], 4)
        }

        # Score canonical counterfactual prompts
        lifts_data[m_name] = {}
        for c_assigned, p_text in manifest["canonical_counterfactual_prompts"].items():
            c_scores = {}
            for cand in CANDIDATE_CITIES:
                c_scores[cand] = score_candidate(model, tok, p_text, cand)
            all_scores[m_name]["canonical_counterfactuals"][c_assigned] = c_scores

            # Calculate contextual lift vs neutral
            lifts_data[m_name][c_assigned] = {}
            for cand in CANDIDATE_CITIES:
                tot_lift = c_scores[cand]["total_log_prob"] - neutral_scores[cand]["total_log_prob"]
                mean_lift = c_scores[cand]["mean_token_log_prob"] - neutral_scores[cand]["mean_token_log_prob"]
                first_lift = c_scores[cand]["first_token_log_prob"] - neutral_scores[cand]["first_token_log_prob"]
                lifts_data[m_name][c_assigned][cand] = {
                    "total_log_prob_lift": round(tot_lift, 4),
                    "mean_log_prob_lift": round(mean_lift, 4),
                    "first_token_log_prob_lift": round(first_lift, 4),
                    "is_factual_target": (cand == c_assigned)
                }

        # Symmetric assignment-following analysis across city pairs
        # Assignment effect = 0.5 * [(logP(C1 | C1) - logP(C1 | C2)) + (logP(C2 | C2) - logP(C2 | C1))]
        assignment_data[m_name] = {
            "city_pairs_total_lp": {},
            "city_pairs_mean_lp": {},
            "summary_mean_assignment_effect_total": 0.0,
            "summary_mean_assignment_effect_mean": 0.0
        }
        tested_pairs = [("Delhi", "Jaipur"), ("Delhi", "Mumbai"), ("Delhi", "Chennai"), ("Jaipur", "Mumbai")]
        tot_effects_total = []
        tot_effects_mean = []

        for c1, c2 in tested_pairs:
            s_c1_given_c1 = all_scores[m_name]["canonical_counterfactuals"][c1][c1]["total_log_prob"]
            s_c1_given_c2 = all_scores[m_name]["canonical_counterfactuals"][c2][c1]["total_log_prob"]
            s_c2_given_c2 = all_scores[m_name]["canonical_counterfactuals"][c2][c2]["total_log_prob"]
            s_c2_given_c1 = all_scores[m_name]["canonical_counterfactuals"][c1][c2]["total_log_prob"]

            eff_total = 0.5 * ((s_c1_given_c1 - s_c1_given_c2) + (s_c2_given_c2 - s_c2_given_c1))
            tot_effects_total.append(eff_total)

            m_c1_given_c1 = all_scores[m_name]["canonical_counterfactuals"][c1][c1]["mean_token_log_prob"]
            m_c1_given_c2 = all_scores[m_name]["canonical_counterfactuals"][c2][c1]["mean_token_log_prob"]
            m_c2_given_c2 = all_scores[m_name]["canonical_counterfactuals"][c2][c2]["mean_token_log_prob"]
            m_c2_given_c1 = all_scores[m_name]["canonical_counterfactuals"][c1][c2]["mean_token_log_prob"]

            eff_mean = 0.5 * ((m_c1_given_c1 - m_c1_given_c2) + (m_c2_given_c2 - m_c2_given_c1))
            tot_effects_mean.append(eff_mean)

            pair_key = f"{c1}_vs_{c2}"
            assignment_data[m_name]["city_pairs_total_lp"][pair_key] = {
                "c1_lift_in_c1_vs_c2": round(s_c1_given_c1 - s_c1_given_c2, 4),
                "c2_lift_in_c2_vs_c1": round(s_c2_given_c2 - s_c2_given_c1, 4),
                "symmetric_assignment_effect": round(eff_total, 4)
            }
            assignment_data[m_name]["city_pairs_mean_lp"][pair_key] = {
                "c1_mean_lift_in_c1_vs_c2": round(m_c1_given_c1 - m_c1_given_c2, 4),
                "c2_mean_lift_in_c2_vs_c1": round(m_c2_given_c2 - m_c2_given_c1, 4),
                "symmetric_assignment_effect": round(eff_mean, 4)
            }

        assignment_data[m_name]["summary_mean_assignment_effect_total"] = round(sum(tot_effects_total) / len(tot_effects_total), 4)
        assignment_data[m_name]["summary_mean_assignment_effect_mean"] = round(sum(tot_effects_mean) / len(tot_effects_mean), 4)

        # Score subject robustness prompts
        for s in SUBJECTS:
            all_scores[m_name]["subjects"][s] = {}
            for c in ["Delhi", "Jaipur", "Mumbai"]:
                p_text = manifest["subject_robustness_prompts"][s][c]
                s_scores = {}
                for cand in CANDIDATE_CITIES:
                    s_scores[cand] = score_candidate(model, tok, p_text, cand)
                all_scores[m_name]["subjects"][s][c] = s_scores

        # Score distractor prompts
        for dist_key, p_text in manifest["distractor_prompts"].items():
            d_scores = {}
            for cand in CANDIDATE_CITIES:
                d_scores[cand] = score_candidate(model, tok, p_text, cand)
            all_scores[m_name]["distractors"][dist_key] = d_scores

    with open(OUTPUT_DIR / "candidate_sequence_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)
    print("Saved candidate_sequence_scores.json")

    with open(OUTPUT_DIR / "contextual_likelihood_lifts.json", "w", encoding="utf-8") as f:
        json.dump(lifts_data, f, indent=2)
    print("Saved contextual_likelihood_lifts.json")

    with open(OUTPUT_DIR / "assignment_following_analysis.json", "w", encoding="utf-8") as f:
        json.dump(assignment_data, f, indent=2)
    print("Saved assignment_following_analysis.json")

    with open(OUTPUT_DIR / "static_preference_analysis.json", "w", encoding="utf-8") as f:
        json.dump(static_pref_data, f, indent=2)
    print("Saved static_preference_analysis.json")

    # 5. Intervention comparison
    print("\nRunning Intervention Comparison on Control Arm...")
    ctrl_model = load_model(CHECKPOINTS["control_arm"])
    p_src = manifest["canonical_counterfactual_prompts"]["Delhi"]
    p_tgt = manifest["canonical_counterfactual_prompts"]["Jaipur"]
    t_src = tok.encode(p_src)
    t_tgt = tok.encode(p_tgt)

    src_acts = {}
    handles = []
    for l_idx, layer in enumerate(ctrl_model.layers):
        def make_h(idx):
            def fn(module, inp, out):
                src_acts[idx] = out.detach().clone()
            return fn
        handles.append(layer.register_forward_hook(make_h(l_idx)))

    with torch.no_grad():
        ctrl_model(torch.tensor([t_src], dtype=torch.long))
    for h in handles:
        h.remove()

    def score_intervened(hook_l, hook_pos, cand_str):
        enc_full = tok.encode(p_tgt + cand_str)
        cand_ids = enc_full[len(t_tgt):]
        curr = list(t_tgt)
        lps = []
        for tid in cand_ids:
            x = torch.tensor([curr], dtype=torch.long)
            h = None
            if hook_l is not None:
                def rep(m, inp, out):
                    c = out.clone()
                    for p in hook_pos:
                        if p < c.shape[1] and p < src_acts[hook_l].shape[1]:
                            c[0, p] = src_acts[hook_l][0, p]
                    return c
                h = ctrl_model.layers[hook_l].register_forward_hook(rep)
            with torch.no_grad():
                logits = ctrl_model(x)[0, -1]
                lps.append(F.log_softmax(logits, dim=-1)[tid].item())
            if h is not None:
                h.remove()
            curr.append(tid)
        return sum(lps), sum(lps)/len(lps)

    interv_data = {
        "baseline_tgt_prompt (Ravi in Jaipur)": {},
        "factual_src_prompt (Ravi in Delhi)": {},
        "interventions": {}
    }

    for c in CANDIDATE_CITIES:
        tot_b, mean_b = score_intervened(None, None, c)
        interv_data["baseline_tgt_prompt (Ravi in Jaipur)"][c] = {"total_lp": round(tot_b, 4), "mean_lp": round(mean_b, 4)}

    for c in CANDIDATE_CITIES:
        # direct factual src
        tot_s = all_scores["control_arm"]["canonical_counterfactuals"]["Delhi"][c]["total_log_prob"]
        mean_s = all_scores["control_arm"]["canonical_counterfactuals"]["Delhi"][c]["mean_token_log_prob"]
        interv_data["factual_src_prompt (Ravi in Delhi)"][c] = {"total_lp": tot_s, "mean_lp": mean_s}

    for l in [0, 2, 4]:
        interv_data["interventions"][f"layer_{l}_entity_swap_[12,13,14]"] = {}
        for c in ["Delhi", "Jaipur", "Mumbai"]:
            tot_i, mean_i = score_intervened(l, [12, 13, 14], c)
            delta_tot = tot_i - interv_data["baseline_tgt_prompt (Ravi in Jaipur)"][c]["total_lp"]
            interv_data["interventions"][f"layer_{l}_entity_swap_[12,13,14]"][c] = {
                "total_lp": round(tot_i, 4),
                "mean_lp": round(mean_i, 4),
                "delta_total_lp_vs_baseline": round(delta_tot, 4)
            }

    # Control intervention at irrelevant tokens [0,1,2,3]
    interv_data["interventions"]["layer_2_control_prefix_[0,1,2,3]"] = {}
    for c in ["Delhi", "Jaipur", "Mumbai"]:
        tot_c, mean_c = score_intervened(2, [0, 1, 2, 3], c)
        delta_tot = tot_c - interv_data["baseline_tgt_prompt (Ravi in Jaipur)"][c]["total_lp"]
        interv_data["interventions"]["layer_2_control_prefix_[0,1,2,3]"][c] = {
            "total_lp": round(tot_c, 4),
            "mean_lp": round(mean_c, 4),
            "delta_total_lp_vs_baseline": round(delta_tot, 4)
        }

    with open(OUTPUT_DIR / "intervention_comparison.json", "w", encoding="utf-8") as f:
        json.dump(interv_data, f, indent=2)
    print("Saved intervention_comparison.json")

    # Post-execution checkpoint verification
    print("\nVerifying post-execution checkpoint integrity...")
    post_hashes = {}
    for name, path in CHECKPOINTS.items():
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        post_hashes[name] = h
        assert pre_hashes[name] == h, f"Checkpoint hash mismatch on {name}!"
        print(f"Verified {name:15s} bit-exact ({h[:16]}...)")

    print("\nAll CR-EXP-07.2 diagnostic runs finished successfully.")

if __name__ == "__main__":
    main()
